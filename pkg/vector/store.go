package vector

import (
	"encoding/gob"
	"encoding/json"
	"os"
	"sort"
)

func init() {
	gob.Register(Vector{})
}

// SearchResult holds a single nearest-neighbor search result.
type SearchResult struct {
	ID       string
	Distance float32 // lower = more similar (except DotProductSimilarity where higher = more similar)
	Vector   Vector
}

// Store is an in-memory vector index with brute-force nearest-neighbor search.
// Zero-allocation on read paths; safe for concurrent reads but not concurrent
// read/write — guard with a sync.Mutex externally if needed.
type Store struct {
	vectors []Vector
	ids     []string
	metric  Metric
}

// NewStore creates a Store using the given distance metric.
func NewStore(metric Metric) *Store {
	return &Store{metric: metric}
}

// Add inserts a vector with the given id into the store.
func (s *Store) Add(id string, v Vector) {
	s.ids = append(s.ids, id)
	s.vectors = append(s.vectors, Clone(v))
}

// Len returns the number of vectors in the store.
func (s *Store) Len() int {
	return len(s.vectors)
}

// Search returns the k nearest neighbors to the query vector.
// If k > Len(), all vectors are returned.
// If k <= 0, returns nil.
func (s *Store) Search(query Vector, k int) []SearchResult {
	if k <= 0 || len(s.vectors) == 0 {
		return nil
	}

	// Compute all distances.
	distances := make([]float32, len(s.vectors))
	for i := range s.vectors {
		distances[i] = Distance(query, s.vectors[i], s.metric)
	}

	// Collect indices and sort by distance.
	indices := make([]int, len(s.vectors))
	for i := range indices {
		indices[i] = i
	}

	if s.metric.Ascending() {
		sort.Slice(indices, func(i, j int) bool {
			return distances[indices[i]] < distances[indices[j]]
		})
	} else {
		sort.Slice(indices, func(i, j int) bool {
			return distances[indices[i]] > distances[indices[j]]
		})
	}

	if k > len(indices) {
		k = len(indices)
	}

	results := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		idx := indices[i]
		results[i] = SearchResult{
			ID:       s.ids[idx],
			Distance: distances[idx],
			Vector:   Clone(s.vectors[idx]),
		}
	}
	return results
}

// Get returns the vector for the given id, or nil if not found.
func (s *Store) Get(id string) Vector {
	for i := range s.ids {
		if s.ids[i] == id {
			return Clone(s.vectors[i])
		}
	}
	return nil
}

// Remove deletes the vector with the given id. Returns true if found and removed.
func (s *Store) Remove(id string) bool {
	for i := range s.ids {
		if s.ids[i] == id {
			last := len(s.ids) - 1
			s.ids[i] = s.ids[last]
			s.vectors[i] = s.vectors[last]
			s.ids = s.ids[:last]
			s.vectors = s.vectors[:last]
			return true
		}
	}
	return false
}

// storeData is the serializable representation of a Store for persistence.
type storeData struct {
	Vectors []Vector
	IDs     []string
	Metric  Metric
}

// Save writes the store to a file using Go's gob encoder (compact binary format).
// Overwrites the file if it exists.
func (s *Store) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	data := storeData{
		Vectors: s.vectors,
		IDs:     s.ids,
		Metric:  s.metric,
	}
	return gob.NewEncoder(f).Encode(data)
}

// Load restores the store from a gob-encoded file. Existing data in the store
// is replaced. Returns an error if the file cannot be read or decoded.
func (s *Store) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var data storeData
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return err
	}

	s.vectors = data.Vectors
	s.ids = data.IDs
	s.metric = data.Metric
	return nil
}

// SaveJSON writes the store to a file as human-readable JSON.
func (s *Store) SaveJSON(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	data := storeData{
		Vectors: s.vectors,
		IDs:     s.ids,
		Metric:  s.metric,
	}
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(data)
}

// LoadJSON restores the store from a JSON file. Existing data is replaced.
func (s *Store) LoadJSON(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var data storeData
	if err := json.NewDecoder(f).Decode(&data); err != nil {
		return err
	}

	s.vectors = data.Vectors
	s.ids = data.IDs
	s.metric = data.Metric
	return nil
}
