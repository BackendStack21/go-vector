package vector

import (
	"encoding/gob"
	"encoding/json"
	"io"
	"os"
)

// storeData is the serializable representation of a Store for persistence.
type storeData struct {
	Vectors  []Vector
	IDs      []string
	Metric   Metric
	Metadata []map[string]string // optional; nil/empty on v1 files
}

func (s *Store) snapshot() storeData {
	meta := s.meta
	if !hasAnyMeta(meta) {
		meta = nil
	}
	return storeData{
		Vectors:  s.vectors,
		IDs:      s.ids,
		Metric:   s.metric,
		Metadata: meta,
	}
}

func (s *Store) restore(data storeData) {
	s.vectors = data.Vectors
	s.ids = data.IDs
	s.metric = data.Metric
	s.meta = nil
	if len(data.Metadata) > 0 {
		s.meta = make([]map[string]string, len(s.ids))
		copy(s.meta, data.Metadata)
	}
	s.packed = nil
	s.packedOK = false
	s.dims = 0
	s.norms2 = nil
	s.byID = nil
	s.rebuildAux()
}

func hasAnyMeta(meta []map[string]string) bool {
	for _, m := range meta {
		if m != nil && len(m) > 0 {
			return true
		}
	}
	return false
}

type countingWriter struct {
	w io.Writer
	n int64
}

func (c *countingWriter) Write(p []byte) (int, error) {
	n, err := c.w.Write(p)
	c.n += int64(n)
	return n, err
}

type countingReader struct {
	r io.Reader
	n int64
}

func (c *countingReader) Read(p []byte) (int, error) {
	n, err := c.r.Read(p)
	c.n += int64(n)
	return n, err
}

// WriteTo gob-encodes the store onto w.
func (s *Store) WriteTo(w io.Writer) (int64, error) {
	cw := &countingWriter{w: w}
	err := gob.NewEncoder(cw).Encode(s.snapshot())
	return cw.n, err
}

// ReadFrom gob-decodes a store from r, replacing existing data.
func (s *Store) ReadFrom(r io.Reader) (int64, error) {
	cr := &countingReader{r: r}
	var data storeData
	if err := gob.NewDecoder(cr).Decode(&data); err != nil {
		return cr.n, err
	}
	s.restore(data)
	return cr.n, nil
}

// WriteJSONTo JSON-encodes the store onto w (indented, same as SaveJSON).
func (s *Store) WriteJSONTo(w io.Writer) error {
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(s.snapshot())
}

// ReadJSONFrom JSON-decodes a store from r, replacing existing data.
func (s *Store) ReadJSONFrom(r io.Reader) error {
	var data storeData
	if err := json.NewDecoder(r).Decode(&data); err != nil {
		return err
	}
	s.restore(data)
	return nil
}

// Save writes the store to a file using Go's gob encoder (compact binary format).
// Overwrites the file if it exists.
func (s *Store) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	_, err = s.WriteTo(f)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// SaveAtomic writes to path+".tmp", fsyncs, then renames onto path.
func (s *Store) SaveAtomic(path string) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	_, err = s.WriteTo(f)
	if err == nil {
		err = f.Sync()
	}
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		_ = os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// Load restores the store from a gob-encoded file. Existing data in the store
// is replaced. Returns an error if the file cannot be read or decoded.
func (s *Store) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	_, err = s.ReadFrom(f)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// SaveJSON writes the store to a file as human-readable JSON.
func (s *Store) SaveJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	err = s.WriteJSONTo(f)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}

// LoadJSON restores the store from a JSON file. Existing data is replaced.
func (s *Store) LoadJSON(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	err = s.ReadJSONFrom(f)
	if cerr := f.Close(); err == nil {
		err = cerr
	}
	return err
}
