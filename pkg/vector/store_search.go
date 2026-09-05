package vector

import (
	"runtime"
	"sort"
	"sync"
)

// SearchOption configures SearchOpts.
type SearchOption func(*searchConfig)

type searchConfig struct {
	noVectors  bool
	pred       func(id string) bool
	metaKey    string
	metaVal    string
	hasMeta    bool
	maxDist    float32
	hasMaxDist bool
	parallel   int
}

// WithoutVectors leaves SearchResult.Vector nil (no clone).
func WithoutVectors() SearchOption {
	return func(c *searchConfig) { c.noVectors = true }
}

// WithPredicate keeps only ids for which fn returns true.
func WithPredicate(fn func(id string) bool) SearchOption {
	return func(c *searchConfig) { c.pred = fn }
}

// WithMetaEqual keeps only ids whose first-match metadata has key=value.
func WithMetaEqual(key, value string) SearchOption {
	return func(c *searchConfig) {
		c.metaKey = key
		c.metaVal = value
		c.hasMeta = true
	}
}

// WithMaxDistance keeps scores that beat d: score <= d for distances,
// score >= d for DotProductSimilarity.
func WithMaxDistance(d float32) SearchOption {
	return func(c *searchConfig) {
		c.maxDist = d
		c.hasMaxDist = true
	}
}

// WithParallel scores in parallel when Len() >= minN. minN <= 0 disables.
// Off by default. Results are deterministic (ties break by lower index).
func WithParallel(minN int) SearchOption {
	return func(c *searchConfig) { c.parallel = minN }
}

// Search returns the k nearest neighbors to the query vector.
// If k > Len(), all vectors are returned.
// If k <= 0, returns nil.
func (s *Store) Search(query Vector, k int) []SearchResult {
	return s.SearchOpts(query, k)
}

// SearchIDs is Search without cloning vectors.
func (s *Store) SearchIDs(query Vector, k int) []SearchHit {
	hits := s.search(query, k, searchConfig{noVectors: true})
	if hits == nil {
		return nil
	}
	out := make([]SearchHit, len(hits))
	for i, h := range hits {
		out[i] = SearchHit{ID: h.ID, Distance: h.Distance}
	}
	return out
}

// SearchOpts is Search with optional filters and result-shape controls.
// Search(query, k) is equivalent to SearchOpts(query, k).
func (s *Store) SearchOpts(query Vector, k int, opts ...SearchOption) []SearchResult {
	var cfg searchConfig
	for _, opt := range opts {
		if opt != nil {
			opt(&cfg)
		}
	}
	return s.search(query, k, cfg)
}

// SearchBatch runs Search for each query. A nil query still produces a
// (possibly empty) result slice so input and output lengths match.
func (s *Store) SearchBatch(queries []Vector, k int) [][]SearchResult {
	if queries == nil {
		return nil
	}
	out := make([][]SearchResult, len(queries))
	for i, q := range queries {
		out[i] = s.Search(q, k)
	}
	return out
}

// SearchRadius returns every stored vector whose score beats radius,
// sorted best-first. Unbounded — prefer SearchOpts with WithMaxDistance
// and a k cap when n is large.
func (s *Store) SearchRadius(query Vector, radius float32) []SearchResult {
	n := len(s.vectors)
	if n == 0 {
		return nil
	}
	return s.search(query, n, searchConfig{hasMaxDist: true, maxDist: radius})
}

// Rerank recomputes distances for hits that carry a Vector and returns
// the top k under metric, sorted best-first. Hits with a nil Vector are
// dropped. If k <= 0, returns nil.
func Rerank(query Vector, hits []SearchResult, metric Metric, k int) []SearchResult {
	if k <= 0 || len(hits) == 0 {
		return nil
	}
	type scored struct {
		h     SearchResult
		score float32
	}
	kept := make([]scored, 0, len(hits))
	for _, h := range hits {
		if h.Vector == nil {
			continue
		}
		kept = append(kept, scored{h: h, score: Distance(query, h.Vector, metric)})
	}
	if len(kept) == 0 {
		return nil
	}
	asc := metric.Ascending()
	sort.SliceStable(kept, func(i, j int) bool {
		if kept[i].score != kept[j].score {
			if asc {
				return kept[i].score < kept[j].score
			}
			return kept[i].score > kept[j].score
		}
		return i < j
	})
	if k > len(kept) {
		k = len(kept)
	}
	out := make([]SearchResult, k)
	for i := 0; i < k; i++ {
		h := kept[i].h
		h.Distance = kept[i].score
		h.Vector = Clone(h.Vector)
		out[i] = h
	}
	return out
}

type scored struct {
	idx   int
	score float32
}

func (s *Store) search(query Vector, k int, cfg searchConfig) []SearchResult {
	n := len(s.vectors)
	if k <= 0 || n == 0 {
		return nil
	}

	keep := s.keepFn(cfg)
	asc := s.metric.Ascending()
	scoreFn := s.scoreAtFunc(query)

	within := func(score float32) bool {
		if !cfg.hasMaxDist {
			return true
		}
		if asc {
			return score <= cfg.maxDist
		}
		return score >= cfg.maxDist
	}

	// worse reports whether a ranks below b (a is the better eviction).
	// Equal scores: higher index is worse, so lower index wins — deterministic
	// and matches “first seen” for exact ties.
	worse := func(a, b scored) bool {
		if a.score != b.score {
			if asc {
				return a.score > b.score
			}
			return a.score < b.score
		}
		return a.idx > b.idx
	}

	var heap []scored
	if cfg.parallel > 0 && n >= cfg.parallel {
		heap = s.searchParallel(n, k, keep, within, scoreFn, worse)
	} else {
		heap = selectTopK(n, k, keep, within, scoreFn, worse)
	}
	if len(heap) == 0 {
		return nil
	}

	sort.Slice(heap, func(i, j int) bool {
		return worse(heap[j], heap[i])
	})

	results := make([]SearchResult, len(heap))
	for i, h := range heap {
		results[i] = SearchResult{
			ID:       s.ids[h.idx],
			Distance: h.score,
		}
		if !cfg.noVectors {
			results[i].Vector = Clone(s.vectors[h.idx])
		}
	}
	return results
}

func (s *Store) keepFn(cfg searchConfig) func(int) bool {
	if cfg.pred == nil && !cfg.hasMeta {
		return nil
	}
	return func(i int) bool {
		if cfg.pred != nil && !cfg.pred(s.ids[i]) {
			return false
		}
		if cfg.hasMeta {
			if s.meta == nil || s.meta[i] == nil || s.meta[i][cfg.metaKey] != cfg.metaVal {
				return false
			}
		}
		return true
	}
}

func selectTopK(n, k int, keep func(int) bool, within func(float32) bool, scoreFn func(int) float32, worse func(a, b scored) bool) []scored {
	if k > n {
		k = n
	}
	heap := make([]scored, 0, k)
	// Max-heap on "worseness": root is the worst kept result so a better
	// candidate evicts it. The v1.3 sift compared parent-worse-than-child
	// and only looked correct when k >= n (no evictions).
	siftUp := func(i int) {
		for i > 0 {
			parent := (i - 1) / 2
			if !worse(heap[i], heap[parent]) {
				break
			}
			heap[parent], heap[i] = heap[i], heap[parent]
			i = parent
		}
	}
	siftDown := func() {
		i := 0
		for {
			l, r, w := 2*i+1, 2*i+2, i
			if l < len(heap) && worse(heap[l], heap[w]) {
				w = l
			}
			if r < len(heap) && worse(heap[r], heap[w]) {
				w = r
			}
			if w == i {
				break
			}
			heap[i], heap[w] = heap[w], heap[i]
			i = w
		}
	}
	for i := 0; i < n; i++ {
		if keep != nil && !keep(i) {
			continue
		}
		score := scoreFn(i)
		if !within(score) {
			continue
		}
		cand := scored{i, score}
		if len(heap) < k {
			heap = append(heap, cand)
			siftUp(len(heap) - 1)
		} else if worse(heap[0], cand) {
			heap[0] = cand
			siftDown()
		}
	}
	return heap
}

func (s *Store) searchParallel(n, k int, keep func(int) bool, within func(float32) bool, scoreFn func(int) float32, worse func(a, b scored) bool) []scored {
	workers := runtime.GOMAXPROCS(0)
	if workers < 2 {
		return selectTopK(n, k, keep, within, scoreFn, worse)
	}
	if workers > n {
		workers = n
	}
	var wg sync.WaitGroup
	partial := make([][]scored, workers)
	chunk := (n + workers - 1) / workers
	for w := 0; w < workers; w++ {
		lo := w * chunk
		hi := lo + chunk
		if lo >= n {
			break
		}
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(w, lo, hi int) {
			defer wg.Done()
			localKeep := keep
			if localKeep != nil {
				// keep already closes over store; read-only.
			}
			// Score a contiguous slice by wrapping scoreFn/keep with an offset.
			// selectTopK indexes 0..span-1, so we translate.
			span := hi - lo
			partial[w] = selectTopK(span, k,
				func(j int) bool {
					i := lo + j
					if keep != nil && !keep(i) {
						return false
					}
					return true
				},
				within,
				func(j int) float32 { return scoreFn(lo + j) },
				func(a, b scored) bool {
					a.idx += lo
					b.idx += lo
					return worse(a, b)
				},
			)
			for i := range partial[w] {
				partial[w][i].idx += lo
			}
		}(w, lo, hi)
	}
	wg.Wait()

	merged := make([]scored, 0, k)
	siftUp := func(i int) {
		for i > 0 {
			parent := (i - 1) / 2
			if !worse(merged[i], merged[parent]) {
				break
			}
			merged[parent], merged[i] = merged[i], merged[parent]
			i = parent
		}
	}
	siftDown := func() {
		i := 0
		for {
			l, r, w := 2*i+1, 2*i+2, i
			if l < len(merged) && worse(merged[l], merged[w]) {
				w = l
			}
			if r < len(merged) && worse(merged[r], merged[w]) {
				w = r
			}
			if w == i {
				break
			}
			merged[i], merged[w] = merged[w], merged[i]
			i = w
		}
	}
	for _, part := range partial {
		for _, cand := range part {
			if len(merged) < k {
				merged = append(merged, cand)
				siftUp(len(merged) - 1)
				continue
			}
			if worse(merged[0], cand) {
				merged[0] = cand
				siftDown()
			}
		}
	}
	return merged
}

// scoreAtFunc returns a closure that scores stored row i against query.
func (s *Store) scoreAtFunc(query Vector) func(int) float32 {
	m := s.metric
	return func(i int) float32 {
		return Distance(query, s.row(i), m)
	}
}
