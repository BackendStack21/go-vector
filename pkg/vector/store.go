package vector

import (
	"encoding/gob"
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

// SearchHit is a Search result without a cloned vector.
type SearchHit struct {
	ID       string
	Distance float32
}

// Store is an in-memory vector index with brute-force nearest-neighbor search.
// Safe for concurrent reads but not concurrent read/write — use LockedStore
// or an external sync.Mutex if writers are involved.
//
// Duplicate IDs are allowed: Add always appends. Get, Remove, Upsert, Has,
// SetMeta, and Meta act on the first matching id.
type Store struct {
	vectors []Vector
	ids     []string
	metric  Metric

	// packed is a row-major backing array used when every vector has the
	// same length. vectors[i] then aliases packed[i*dims:(i+1)*dims].
	packed   []float32
	dims     int
	packedOK bool

	norms2 []float32      // cached Σv², parallel to vectors
	byID   map[string]int // id → first index
	meta   []map[string]string
}

// NewStore creates a Store using the given distance metric.
func NewStore(metric Metric) *Store {
	return &Store{metric: metric, byID: make(map[string]int)}
}

// Metric returns the store's distance metric.
func (s *Store) Metric() Metric { return s.metric }

// Len returns the number of vectors in the store.
func (s *Store) Len() int { return len(s.vectors) }

// Dims returns the dimensionality of the first stored vector, or 0 if empty.
func (s *Store) Dims() int {
	if len(s.vectors) == 0 {
		return 0
	}
	if s.packedOK && s.dims > 0 {
		return s.dims
	}
	return len(s.vectors[0])
}

// Has reports whether id is in the store (first-match).
func (s *Store) Has(id string) bool {
	_, ok := s.firstIndex(id)
	return ok
}

// IDs returns a copy of the id list, including duplicates, in store order.
func (s *Store) IDs() []string {
	out := make([]string, len(s.ids))
	copy(out, s.ids)
	return out
}

// Clear removes every vector. The metric is unchanged.
func (s *Store) Clear() {
	s.vectors = nil
	s.ids = nil
	s.packed = nil
	s.packedOK = false
	s.dims = 0
	s.norms2 = nil
	s.byID = make(map[string]int)
	s.meta = nil
}

// Add inserts a vector with the given id into the store.
func (s *Store) Add(id string, v Vector) {
	idx := len(s.ids)
	s.ids = append(s.ids, id)
	cloned := s.appendVector(v)
	s.vectors = append(s.vectors, cloned)
	s.norms2 = append(s.norms2, dotSelf(cloned))
	if s.meta != nil {
		s.meta = append(s.meta, nil)
	}
	s.noteID(id, idx)
}

// AddUnique inserts only if id is not already present. Returns true if added.
func (s *Store) AddUnique(id string, v Vector) bool {
	if _, ok := s.firstIndex(id); ok {
		return false
	}
	s.Add(id, v)
	return true
}

// Upsert replaces the first vector with id, or Add if id is new.
func (s *Store) Upsert(id string, v Vector) {
	if i, ok := s.firstIndex(id); ok {
		s.replaceAt(i, v)
		return
	}
	s.Add(id, v)
}

// Get returns the vector for the given id, or nil if not found.
func (s *Store) Get(id string) Vector {
	i, ok := s.firstIndex(id)
	if !ok {
		return nil
	}
	return Clone(s.vectors[i])
}

// Remove deletes the first vector with the given id. Returns true if removed.
func (s *Store) Remove(id string) bool {
	i, ok := s.firstIndex(id)
	if !ok {
		return false
	}
	last := len(s.ids) - 1
	movedID := s.ids[last]

	if i != last {
		s.ids[i] = s.ids[last]
		if s.packedOK && s.dims > 0 {
			offI, offL := i*s.dims, last*s.dims
			copy(s.packed[offI:offI+s.dims], s.packed[offL:offL+s.dims])
			s.vectors[i] = Vector(s.packed[offI : offI+s.dims : offI+s.dims])
		} else {
			s.vectors[i] = s.vectors[last]
		}
		s.norms2[i] = s.norms2[last]
		if s.meta != nil {
			s.meta[i] = s.meta[last]
		}
	}

	s.ids = s.ids[:last]
	s.vectors = s.vectors[:last]
	s.norms2 = s.norms2[:last]
	if s.packedOK {
		if last == 0 {
			s.packed = s.packed[:0]
			s.dims = 0
		} else {
			s.packed = s.packed[:last*s.dims]
		}
	}
	if s.meta != nil {
		s.meta = s.meta[:last]
	}
	s.repairByID(id, movedID, i, last)
	return true
}

// SetMeta replaces metadata on the first matching id. meta is cloned.
// Passing nil clears metadata for that id. Returns false if id is missing.
func (s *Store) SetMeta(id string, meta map[string]string) bool {
	i, ok := s.firstIndex(id)
	if !ok {
		return false
	}
	s.ensureMeta()
	s.meta[i] = cloneMeta(meta)
	return true
}

// Meta returns a clone of the metadata for the first matching id, or nil.
func (s *Store) Meta(id string) map[string]string {
	i, ok := s.firstIndex(id)
	if !ok || s.meta == nil {
		return nil
	}
	return cloneMeta(s.meta[i])
}

func (s *Store) appendVector(v Vector) Vector {
	if len(s.vectors) == 0 {
		s.dims = len(v)
		s.packedOK = true
		s.packed = append(s.packed[:0], v...)
		if s.dims == 0 {
			return Vector{}
		}
		return Vector(s.packed[0:s.dims:s.dims])
	}
	if s.packedOK && len(v) == s.dims {
		if s.dims == 0 {
			return Vector{}
		}
		start := len(s.packed)
		oldCap := cap(s.packed)
		s.packed = append(s.packed, v...)
		if cap(s.packed) != oldCap {
			s.relinkPacked()
		}
		return Vector(s.packed[start : start+s.dims : start+s.dims])
	}
	if s.packedOK {
		s.unpack()
	}
	return Clone(v)
}

func (s *Store) unpack() {
	for i := range s.vectors {
		s.vectors[i] = Clone(s.vectors[i])
	}
	s.packed = nil
	s.packedOK = false
	s.dims = 0
}

func (s *Store) replaceAt(i int, v Vector) {
	if s.packedOK && len(v) == s.dims && s.dims > 0 {
		off := i * s.dims
		copy(s.packed[off:off+s.dims], v)
		s.vectors[i] = Vector(s.packed[off : off+s.dims : off+s.dims])
	} else if s.packedOK {
		s.unpack()
		s.vectors[i] = Clone(v)
	} else {
		s.vectors[i] = Clone(v)
	}
	s.norms2[i] = dotSelf(s.vectors[i])
}

func (s *Store) firstIndex(id string) (int, bool) {
	if s.byID != nil {
		i, ok := s.byID[id]
		return i, ok
	}
	for i, x := range s.ids {
		if x == id {
			return i, true
		}
	}
	return 0, false
}

func (s *Store) noteID(id string, idx int) {
	if s.byID == nil {
		s.byID = make(map[string]int)
	}
	if _, exists := s.byID[id]; !exists {
		s.byID[id] = idx
	}
}

func (s *Store) repairByID(removedID, movedID string, i, last int) {
	if s.byID == nil {
		return
	}
	delete(s.byID, removedID)
	for j, id := range s.ids {
		if id == removedID {
			s.byID[removedID] = j
			break
		}
	}
	if i != last && movedID != removedID {
		if prev, ok := s.byID[movedID]; ok && prev == last {
			s.byID[movedID] = i
		}
	}
}

func (s *Store) rebuildAux() {
	s.byID = make(map[string]int, len(s.ids))
	for i, id := range s.ids {
		if _, exists := s.byID[id]; !exists {
			s.byID[id] = i
		}
	}
	s.norms2 = make([]float32, len(s.vectors))
	for i, v := range s.vectors {
		s.norms2[i] = dotSelf(v)
	}
	s.repackIfUniform()
}

func (s *Store) repackIfUniform() {
	s.packed = nil
	s.packedOK = false
	s.dims = 0
	if len(s.vectors) == 0 {
		s.packedOK = true
		return
	}
	d := len(s.vectors[0])
	for i := 1; i < len(s.vectors); i++ {
		if len(s.vectors[i]) != d {
			return
		}
	}
	s.dims = d
	s.packedOK = true
	if d == 0 {
		return
	}
	s.packed = make([]float32, len(s.vectors)*d)
	for i, v := range s.vectors {
		copy(s.packed[i*d:(i+1)*d], v)
		s.vectors[i] = Vector(s.packed[i*d : (i+1)*d : (i+1)*d])
	}
}

func (s *Store) ensureMeta() {
	if s.meta == nil {
		s.meta = make([]map[string]string, len(s.ids))
	}
}

func (s *Store) relinkPacked() {
	if !s.packedOK || s.dims == 0 {
		return
	}
	for i := range s.vectors {
		s.vectors[i] = Vector(s.packed[i*s.dims : (i+1)*s.dims : (i+1)*s.dims])
	}
}

func (s *Store) row(i int) Vector {
	if s.packedOK && s.dims > 0 {
		return Vector(s.packed[i*s.dims : (i+1)*s.dims])
	}
	return s.vectors[i]
}

func cloneMeta(m map[string]string) map[string]string {
	if m == nil {
		return nil
	}
	out := make(map[string]string, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}
