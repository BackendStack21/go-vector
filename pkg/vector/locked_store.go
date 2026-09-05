package vector

import (
	"io"
	"sync"
)

// LockedStore is a Store guarded by an RWMutex. Reads take RLock, writes
// take Lock. The original Store type stays lock-free.
type LockedStore struct {
	mu sync.RWMutex
	s  Store
}

// NewLockedStore creates a LockedStore using the given distance metric.
func NewLockedStore(metric Metric) *LockedStore {
	return &LockedStore{s: *NewStore(metric)}
}

// Store returns the inner Store. The pointer is not safe to use without
// coordinating with LockedStore's own methods.
func (ls *LockedStore) Store() *Store { return &ls.s }

func (ls *LockedStore) Add(id string, v Vector) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.s.Add(id, v)
}

func (ls *LockedStore) AddUnique(id string, v Vector) bool {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.AddUnique(id, v)
}

func (ls *LockedStore) Upsert(id string, v Vector) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.s.Upsert(id, v)
}

func (ls *LockedStore) Remove(id string) bool {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.Remove(id)
}

func (ls *LockedStore) Clear() {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	ls.s.Clear()
}

func (ls *LockedStore) SetMeta(id string, meta map[string]string) bool {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.SetMeta(id, meta)
}

func (ls *LockedStore) Get(id string) Vector {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Get(id)
}

func (ls *LockedStore) Has(id string) bool {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Has(id)
}

func (ls *LockedStore) Meta(id string) map[string]string {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Meta(id)
}

func (ls *LockedStore) IDs() []string {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.IDs()
}

func (ls *LockedStore) Len() int {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Len()
}

func (ls *LockedStore) Dims() int {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Dims()
}

func (ls *LockedStore) Metric() Metric {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Metric()
}

func (ls *LockedStore) Search(query Vector, k int) []SearchResult {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Search(query, k)
}

func (ls *LockedStore) SearchIDs(query Vector, k int) []SearchHit {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.SearchIDs(query, k)
}

func (ls *LockedStore) SearchOpts(query Vector, k int, opts ...SearchOption) []SearchResult {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.SearchOpts(query, k, opts...)
}

func (ls *LockedStore) SearchBatch(queries []Vector, k int) [][]SearchResult {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.SearchBatch(queries, k)
}

func (ls *LockedStore) SearchRadius(query Vector, radius float32) []SearchResult {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.SearchRadius(query, radius)
}

func (ls *LockedStore) Save(path string) error {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.Save(path)
}

func (ls *LockedStore) SaveAtomic(path string) error {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.SaveAtomic(path)
}

func (ls *LockedStore) SaveJSON(path string) error {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.SaveJSON(path)
}

func (ls *LockedStore) Load(path string) error {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.Load(path)
}

func (ls *LockedStore) LoadJSON(path string) error {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.LoadJSON(path)
}

func (ls *LockedStore) WriteTo(w io.Writer) (int64, error) {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.WriteTo(w)
}

func (ls *LockedStore) ReadFrom(r io.Reader) (int64, error) {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.ReadFrom(r)
}

func (ls *LockedStore) WriteJSONTo(w io.Writer) error {
	ls.mu.RLock()
	defer ls.mu.RUnlock()
	return ls.s.WriteJSONTo(w)
}

func (ls *LockedStore) ReadJSONFrom(r io.Reader) error {
	ls.mu.Lock()
	defer ls.mu.Unlock()
	return ls.s.ReadJSONFrom(r)
}
