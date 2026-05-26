package vector

import (
	"encoding/gob"
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
}

// SaveEmbedder writes the RandomProjections state to a gob file.
// The output dimension and vocabulary are preserved so that future
// Embed() calls produce the same vectors for the same text.
func (rp *RandomProjections) SaveEmbedder(path string) error {
	proj := make([][]rpRow, len(rp.proj))
	for i, row := range rp.proj {
		proj[i] = make([]rpRow, len(row))
		for j, e := range row {
			proj[i][j] = rpRow{Dim: e.dim, Val: e.val}
		}
	}
	data := rpPersistData{
		Vocab:     rp.vocab,
		Tokens:    rp.tokens,
		OutputDim: rp.outputDim,
		Scale:     rp.scale,
		Proj:      proj,
	}
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(data)
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
	var data rpPersistData
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return nil, err
	}
	// Reconstruct the internal proj matrix
	proj := make([][]projEntry, len(data.Proj))
	for i, row := range data.Proj {
		proj[i] = make([]projEntry, len(row))
		for j, e := range row {
			proj[i][j] = projEntry{dim: e.Dim, val: e.Val}
		}
	}
	return &RandomProjections{
		vocab:     data.Vocab,
		tokens:    data.Tokens,
		outputDim: data.OutputDim,
		proj:      proj,
		scale:     data.Scale,
	}, nil
}
