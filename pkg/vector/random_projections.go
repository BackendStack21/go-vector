package vector

import (
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
	proj      [][]projEntry // per-vocab-index projection entries
	scale     float32
}

// NewRandomProjections creates an embedder with the given output dimensionality.
// Call Fit() to build a vocabulary and projection matrix before embedding.
func NewRandomProjections(outputDim int) *RandomProjections {
	return &RandomProjections{
		vocab:     make(map[string]int),
		outputDim: outputDim,
	}
}

// Dims returns the output dimensionality of the embedder.
func (rp *RandomProjections) Dims() int { return rp.outputDim }

// Fit builds the vocabulary from a corpus and generates the projection matrix.
// Must be called before Embed(). Re-fitting overwrites the previous vocabulary.
func (rp *RandomProjections) Fit(corpus []string) {
	rp.vocab = make(map[string]int)
	rp.tokens = nil
	for _, doc := range corpus {
		for _, tok := range tokenize(doc) {
			if _, ok := rp.vocab[tok]; !ok {
				rp.vocab[tok] = len(rp.tokens)
				rp.tokens = append(rp.tokens, tok)
			}
		}
	}
	rp.generateProjection()
}

// generateProjection builds a sparse random projection matrix using the
// Achlioptas distribution: entries are {-1, 0, +1} * sqrt(3/D).
func (rp *RandomProjections) generateProjection() {
	V := len(rp.tokens)
	D := rp.outputDim
	if V == 0 || D == 0 {
		return
	}

	rp.scale = float32(math.Sqrt(3.0 / float64(D)))
	rp.proj = make([][]projEntry, V)

	rng := rand.New(rand.NewSource(42)) // deterministic seed for reproducibility

	for i := 0; i < V; i++ {
		// Expected non-zero entries per row: D/3. Preallocate to avoid the
		// append growth churn while filling each sparse row.
		entries := make([]projEntry, 0, D/3+1)
		for j := 0; j < D; j++ {
			r := rng.Float64()
			if r < 1.0/6.0 {
				entries = append(entries, projEntry{dim: j, val: -rp.scale})
			} else if r < 2.0/6.0 {
				entries = append(entries, projEntry{dim: j, val: rp.scale})
			}
			// 2/3 chance: zero, skip
		}
		rp.proj[i] = entries
	}
}

// Embed converts text to a vector using the random projection.
// Tokens not in the vocabulary are ignored. If no vocabulary has been fit,
// returns a zero vector. The output vector is L2-normalized.
func (rp *RandomProjections) Embed(text string) (Vector, error) {
	tokens := tokenize(text)
	out := make(Vector, rp.outputDim)

	for _, tok := range tokens {
		idx, ok := rp.vocab[tok]
		if !ok {
			continue
		}
		for _, e := range rp.proj[idx] {
			out[e.dim] += e.val
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

// VocabSize returns the number of unique tokens in the vocabulary.
func (rp *RandomProjections) VocabSize() int { return len(rp.tokens) }

// tokenize splits text into lowercase tokens, filtering non-letters and
// tokens shorter than 2 characters.
func tokenize(text string) []string {
	fields := strings.FieldsFunc(strings.ToLower(text), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	tokens := make([]string, 0, len(fields))
	for _, token := range fields {
		if len(token) >= 2 {
			tokens = append(tokens, token)
		}
	}
	return tokens
}
