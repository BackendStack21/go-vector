package vector

import (
	"encoding/gob"
	"encoding/json"
	"math"
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
	n := len(s.vectors)
	if k <= 0 || n == 0 {
		return nil
	}
	if k > n {
		k = n
	}

	asc := s.metric.Ascending()

	// scored pairs a stored-vector index with its distance/similarity score.
	type scored struct {
		idx   int
		score float32
	}

	// worse reports whether score a ranks below score b for this metric, i.e.
	// a is the better eviction candidate. For distances (ascending) a larger
	// score is worse; for similarities a smaller score is worse.
	worse := func(a, b float32) bool {
		if asc {
			return a > b
		}
		return a < b
	}

	// Maintain a bounded heap of the best k results seen so far. The root is
	// always the worst of the kept set, so a better candidate evicts it in
	// O(log k) — overall O(n·log k) versus a full O(n·log n) sort, and with no
	// reflection (sort.Slice) on the hot path.
	heap := make([]scored, 0, k)
	siftUp := func(i int) {
		for i > 0 {
			parent := (i - 1) / 2
			if !worse(heap[parent].score, heap[i].score) {
				break
			}
			heap[parent], heap[i] = heap[i], heap[parent]
			i = parent
		}
	}
	siftDown := func() {
		i := 0
		for {
			l, r, worst := 2*i+1, 2*i+2, i
			if l < len(heap) && worse(heap[worst].score, heap[l].score) {
				worst = l
			}
			if r < len(heap) && worse(heap[worst].score, heap[r].score) {
				worst = r
			}
			if worst == i {
				break
			}
			heap[i], heap[worst] = heap[worst], heap[i]
			i = worst
		}
	}

	scoreFn := s.scorer(query)
	for i := 0; i < n; i++ {
		score := scoreFn(s.vectors[i])
		if len(heap) < k {
			heap = append(heap, scored{i, score})
			siftUp(len(heap) - 1)
		} else if worse(heap[0].score, score) {
			// Candidate beats the current worst kept result.
			heap[0] = scored{i, score}
			siftDown()
		}
	}

	// Heap holds the top k unordered; sort best-first for the caller.
	sort.Slice(heap, func(i, j int) bool {
		return worse(heap[j].score, heap[i].score)
	})

	results := make([]SearchResult, len(heap))
	for i, h := range heap {
		results[i] = SearchResult{
			ID:       s.ids[h.idx],
			Distance: h.score,
			Vector:   Clone(s.vectors[h.idx]),
		}
	}
	return results
}

// scorer returns a closure that scores a stored vector against query under the
// store's metric. For CosineDistance the query's self–dot product is computed
// once here rather than re-derived for every stored vector inside Cosine.
func (s *Store) scorer(query Vector) func(Vector) float32 {
	if s.metric != CosineDistance {
		return func(v Vector) float32 { return Distance(query, v, s.metric) }
	}
	var qq float32
	for _, x := range query {
		qq += x * x
	}
	qqf := float64(qq)
	return func(v Vector) float32 {
		if len(v) != len(query) || qq == 0 {
			return 1 // CosineDist of a zero/mismatched vector: 1 - 0
		}
		var dot, vv float32
		for i := range query {
			dot += query[i] * v[i]
			vv += v[i] * v[i]
		}
		if vv == 0 {
			return 1
		}
		return 1 - dot/float32(math.Sqrt(qqf*float64(vv)))
	}
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
