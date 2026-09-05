package vector

import (
	"encoding/gob"
	"io"
	"os"
)

func init() {
	gob.Register(rpPersistData{})
	gob.Register(rpRow{})
}

// rpRow is a serializable non-zero entry in the sparse projection matrix.
type rpRow struct {
	Dim int     // output dimension index
	Val float32 // ±scale
}

// rpPersistData is the serializable representation of RandomProjections.
type rpPersistData struct {
	Vocab     map[string]int
	Tokens    []string
	OutputDim int
	Scale     float32
	Proj      [][]rpRow // per-vocab-index projection entries
	// Optional fields (zero on v1 files):
	Seed        int64
	HasSeed     bool
	NGrams      int
	HashBuckets int
	TFIDF       bool
	MinTokenLen int
	IDF         []float32
}

func (rp *RandomProjections) persist() rpPersistData {
	proj := make([][]rpRow, len(rp.proj))
	for i, row := range rp.proj {
		proj[i] = make([]rpRow, len(row))
		for j, e := range row {
			proj[i][j] = rpRow{Dim: e.dim, Val: e.val}
		}
	}
	return rpPersistData{
		Vocab:       rp.vocab,
		Tokens:      rp.tokens,
		OutputDim:   rp.outputDim,
		Scale:       rp.scale,
		Proj:        proj,
		Seed:        rp.seed,
		HasSeed:     rp.hasSeed,
		NGrams:      rp.ngrams,
		HashBuckets: rp.hashBuckets,
		TFIDF:       rp.tfidf,
		MinTokenLen: rp.minTokenLen,
		IDF:         rp.idf,
	}
}

func rpFromPersist(data rpPersistData) *RandomProjections {
	proj := make([][]projEntry, len(data.Proj))
	for i, row := range data.Proj {
		proj[i] = make([]projEntry, len(row))
		for j, e := range row {
			proj[i][j] = projEntry{dim: e.Dim, val: e.Val}
		}
	}
	ngrams := data.NGrams
	if ngrams <= 0 {
		ngrams = 1
	}
	minLen := data.MinTokenLen
	if minLen <= 0 {
		minLen = 2
	}
	seed := data.Seed
	hasSeed := data.HasSeed
	if !hasSeed && seed == 0 {
		seed = 42
		hasSeed = true
	}
	return &RandomProjections{
		vocab:       data.Vocab,
		tokens:      data.Tokens,
		outputDim:   data.OutputDim,
		proj:        proj,
		scale:       data.Scale,
		seed:        seed,
		hasSeed:     hasSeed,
		ngrams:      ngrams,
		hashBuckets: data.HashBuckets,
		tfidf:       data.TFIDF,
		minTokenLen: minLen,
		idf:         data.IDF,
	}
}

// SaveEmbedder writes the RandomProjections state to a gob file.
// The output dimension and vocabulary are preserved so that future
// Embed() calls produce the same vectors for the same text.
func (rp *RandomProjections) SaveEmbedder(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(rp.persist())
}

// WriteTo gob-encodes embedder state onto w.
func (rp *RandomProjections) WriteTo(w io.Writer) (int64, error) {
	cw := &countingWriter{w: w}
	err := gob.NewEncoder(cw).Encode(rp.persist())
	return cw.n, err
}

// LoadEmbedder reads RandomProjections state from a gob file and returns
// a ready-to-use embedder. Returns an error if the file cannot be read
// or decoded.
func LoadEmbedder(path string) (*RandomProjections, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ReadEmbedder(f)
}

// ReadEmbedder gob-decodes a RandomProjections embedder from r.
func ReadEmbedder(r io.Reader) (*RandomProjections, error) {
	var data rpPersistData
	if err := gob.NewDecoder(r).Decode(&data); err != nil {
		return nil, err
	}
	return rpFromPersist(data), nil
}
