package vector

import (
	"hash/fnv"
	"math"
	"math/rand"
	"strings"
	"unicode"
)

// projEntry is a single non-zero entry in the sparse projection matrix.
type projEntry struct {
	dim int     // output dimension index
	val float32 // ±scale
}

// RPOption configures a RandomProjections embedder. Options other than the
// defaults change embeddings; NewRandomProjections (no options) stays
// bit-identical to v1.3 (seed 42, unigrams, min token length 2, no hashing).
type RPOption func(*RandomProjections)

// WithSeed sets the projection RNG seed. Default 42.
func WithSeed(seed int64) RPOption {
	return func(rp *RandomProjections) {
		rp.seed = seed
		rp.hasSeed = true
	}
}

// WithNGrams includes 1..n word n-grams. Default 1 (unigrams only).
func WithNGrams(n int) RPOption {
	return func(rp *RandomProjections) {
		if n > 0 {
			rp.ngrams = n
		}
	}
}

// WithHashingTrick maps out-of-vocabulary tokens into a fixed number of
// extra projection rows so unseen tokens still contribute.
func WithHashingTrick(buckets int) RPOption {
	return func(rp *RandomProjections) {
		if buckets > 0 {
			rp.hashBuckets = buckets
		}
	}
}

// WithTFIDF weights projection rows by inverse document frequency computed
// at Fit time. FitMore after TF-IDF requires a full Fit to refresh IDF.
func WithTFIDF() RPOption {
	return func(rp *RandomProjections) { rp.tfidf = true }
}

// WithMinTokenLen sets the minimum token length in runes. Default 2.
func WithMinTokenLen(n int) RPOption {
	return func(rp *RandomProjections) {
		if n > 0 {
			rp.minTokenLen = n
		}
	}
}

// RandomProjections is a text embedder using sparse random projections
// (Achlioptas 2003). It builds a vocabulary from a corpus, generates a
// sparse projection matrix, and maps text to fixed-size float32 vectors.
//
// The projection approximately preserves cosine distances between documents
// (Johnson-Lindenstrauss lemma). Quality depends on output dimensionality —
// 128–256 dims works well for classification; 512+ for semantic search.
//
// Zero dependencies beyond math/rand. Not a neural embedding — no semantics,
// but fast, deterministic, and good enough for keyword-based similarity.
type RandomProjections struct {
	vocab     map[string]int // token → index
	tokens    []string       // index → token
	outputDim int
	proj      [][]projEntry // per-vocab-index (then hash-bucket) projection entries
	scale     float32

	seed        int64
	hasSeed     bool
	ngrams      int
	hashBuckets int
	tfidf       bool
	minTokenLen int
	idf         []float32
}

// NewRandomProjections creates an embedder with the given output dimensionality.
// Call Fit() to build a vocabulary and projection matrix before embedding.
func NewRandomProjections(outputDim int) *RandomProjections {
	return NewRandomProjectionsOpts(outputDim)
}

// NewRandomProjectionsOpts is NewRandomProjections with options.
func NewRandomProjectionsOpts(outputDim int, opts ...RPOption) *RandomProjections {
	rp := &RandomProjections{
		vocab:       make(map[string]int),
		outputDim:   outputDim,
		seed:        42,
		hasSeed:     true,
		ngrams:      1,
		minTokenLen: 2,
	}
	for _, opt := range opts {
		if opt != nil {
			opt(rp)
		}
	}
	if rp.ngrams <= 0 {
		rp.ngrams = 1
	}
	if rp.minTokenLen <= 0 {
		rp.minTokenLen = 2
	}
	return rp
}

// Dims returns the output dimensionality of the embedder.
func (rp *RandomProjections) Dims() int { return rp.outputDim }

// MustEmbed is Embed, panicking if Embed returns an error. RandomProjections
// currently never errors; this matches the README example.
func (rp *RandomProjections) MustEmbed(text string) Vector {
	return MustEmbed(rp, text)
}

// EmbedBatch embeds each text independently. Satisfies BatchEmbedder.
func (rp *RandomProjections) EmbedBatch(texts []string) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	out := make([]Vector, len(texts))
	for i, t := range texts {
		v, err := rp.Embed(t)
		if err != nil {
			return nil, err
		}
		out[i] = v
	}
	return out, nil
}

// Fit builds the vocabulary from a corpus and generates the projection matrix.
// Must be called before Embed(). Re-fitting overwrites the previous vocabulary.
func (rp *RandomProjections) Fit(corpus []string) {
	rp.vocab = make(map[string]int)
	rp.tokens = nil
	df := map[string]int{}
	nDocs := 0
	for _, doc := range corpus {
		seen := map[string]struct{}{}
		toks := rp.tokensOf(doc)
		if len(toks) > 0 {
			nDocs++
		}
		for _, tok := range toks {
			if _, ok := rp.vocab[tok]; !ok {
				rp.vocab[tok] = len(rp.tokens)
				rp.tokens = append(rp.tokens, tok)
			}
			if _, ok := seen[tok]; !ok {
				seen[tok] = struct{}{}
				df[tok]++
			}
		}
	}
	rp.idf = nil
	if rp.tfidf && nDocs > 0 {
		rp.idf = make([]float32, len(rp.tokens))
		for i, tok := range rp.tokens {
			rp.idf[i] = float32(math.Log(float64(1+nDocs)/float64(1+df[tok])) + 1)
		}
	}
	rp.generateProjection()
}

// FitMore adds tokens from corpus and extends the projection matrix. Existing
// token rows stay unchanged so previously embedded documents remain comparable
// when TF-IDF is off. With TF-IDF, call Fit instead to refresh IDF weights.
func (rp *RandomProjections) FitMore(corpus []string) {
	if rp.vocab == nil {
		rp.vocab = make(map[string]int)
	}
	start := len(rp.tokens)
	for _, doc := range corpus {
		for _, tok := range rp.tokensOf(doc) {
			if _, ok := rp.vocab[tok]; !ok {
				rp.vocab[tok] = len(rp.tokens)
				rp.tokens = append(rp.tokens, tok)
			}
		}
	}
	if start == 0 && len(rp.proj) == 0 {
		rp.generateProjection()
		return
	}
	if start == len(rp.tokens) {
		return
	}
	rp.extendProjection(start)
}

// generateProjection builds a sparse random projection matrix using the
// Achlioptas distribution: entries are {-1, 0, +1} * sqrt(3/D).
func (rp *RandomProjections) generateProjection() {
	V := len(rp.tokens)
	D := rp.outputDim
	if D == 0 {
		return
	}
	rp.scale = float32(math.Sqrt(3.0 / float64(D)))
	rng := rand.New(rand.NewSource(rp.effectiveSeed()))
	rp.proj = make([][]projEntry, 0, V+rp.hashBuckets)
	for i := 0; i < V; i++ {
		rp.proj = append(rp.proj, makeProjRow(rng, D, rp.scale))
	}
	rp.appendHashRows()
}

func (rp *RandomProjections) extendProjection(from int) {
	D := rp.outputDim
	if D == 0 {
		return
	}
	rp.scale = float32(math.Sqrt(3.0 / float64(D)))
	rng := rand.New(rand.NewSource(rp.effectiveSeed()))
	for i := 0; i < from; i++ {
		for j := 0; j < D; j++ {
			rng.Float64()
		}
	}
	// Drop stale hash rows (they are regenerated from an independent seed).
	if len(rp.proj) > from {
		rp.proj = rp.proj[:from]
	}
	for i := from; i < len(rp.tokens); i++ {
		rp.proj = append(rp.proj, makeProjRow(rng, D, rp.scale))
	}
	rp.appendHashRows()
}

func (rp *RandomProjections) appendHashRows() {
	if rp.hashBuckets <= 0 || rp.outputDim == 0 {
		return
	}
	hrng := rand.New(rand.NewSource(rp.effectiveSeed() ^ 0x9e3779b9))
	for i := 0; i < rp.hashBuckets; i++ {
		rp.proj = append(rp.proj, makeProjRow(hrng, rp.outputDim, rp.scale))
	}
}

func makeProjRow(rng *rand.Rand, D int, scale float32) []projEntry {
	entries := make([]projEntry, 0, D/3+1)
	for j := 0; j < D; j++ {
		r := rng.Float64()
		if r < 1.0/6.0 {
			entries = append(entries, projEntry{dim: j, val: -scale})
		} else if r < 2.0/6.0 {
			entries = append(entries, projEntry{dim: j, val: scale})
		}
	}
	return entries
}

func (rp *RandomProjections) effectiveSeed() int64 {
	if rp.hasSeed {
		return rp.seed
	}
	if rp.seed != 0 {
		return rp.seed
	}
	return 42
}

// Embed converts text to a vector using the random projection.
// Tokens not in the vocabulary are ignored (or hashed if hashing is on).
// If no vocabulary has been fit and hashing is off, returns a zero vector.
// The output vector is L2-normalized.
func (rp *RandomProjections) Embed(text string) (Vector, error) {
	tokens := rp.tokensOf(text)
	out := make(Vector, rp.outputDim)

	for _, tok := range tokens {
		idx, ok := rp.projIndex(tok)
		if !ok {
			continue
		}
		w := float32(1)
		if rp.tfidf && idx < len(rp.idf) {
			w = rp.idf[idx]
		}
		if idx >= len(rp.proj) {
			continue
		}
		for _, e := range rp.proj[idx] {
			out[e.dim] += e.val * w
		}
	}

	n := Norm(out)
	if n > 0 {
		for i := range out {
			out[i] /= n
		}
	}
	return out, nil
}

func (rp *RandomProjections) projIndex(tok string) (int, bool) {
	if idx, ok := rp.vocab[tok]; ok {
		return idx, true
	}
	if rp.hashBuckets > 0 && len(rp.tokens) < len(rp.proj) {
		h := hashToken(tok) % uint32(rp.hashBuckets)
		return len(rp.tokens) + int(h), true
	}
	return 0, false
}

func hashToken(tok string) uint32 {
	h := fnv.New32a()
	_, _ = h.Write([]byte(tok))
	return h.Sum32()
}

// VocabSize returns the number of unique tokens in the vocabulary.
func (rp *RandomProjections) VocabSize() int { return len(rp.tokens) }

func (rp *RandomProjections) tokensOf(text string) []string {
	minLen := rp.minTokenLen
	if minLen <= 0 {
		minLen = 2
	}
	uni := tokenizeMin(text, minLen)
	if rp.ngrams <= 1 {
		return uni
	}
	out := make([]string, 0, len(uni)*rp.ngrams)
	out = append(out, uni...)
	for g := 2; g <= rp.ngrams; g++ {
		for i := 0; i+g <= len(uni); i++ {
			out = append(out, strings.Join(uni[i:i+g], " "))
		}
	}
	return out
}

// tokenize splits text into lowercase tokens, filtering non-letters and
// tokens shorter than 2 characters.
func tokenize(text string) []string {
	return tokenizeMin(text, 2)
}

func tokenizeMin(text string, minLen int) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	tokens := make([]string, 0, len(fields))
	for _, token := range fields {
		if len(token) >= minLen {
			tokens = append(tokens, token)
		}
	}
	return tokens
}
