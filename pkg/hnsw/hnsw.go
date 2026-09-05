// Package hnsw is a pure-Go Hierarchical Navigable Small World index
// for approximate nearest-neighbor search. It depends only on pkg/vector
// and the standard library.
package hnsw

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"sort"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

// Option configures an Index.
type Option func(*Index)

// WithM sets the max number of neighbors per node on layers > 0 (default 16).
func WithM(m int) Option {
	return func(idx *Index) {
		if m > 0 {
			idx.m = m
			idx.mMax0 = 2 * m
		}
	}
}

// WithEfConstruction sets the candidate list size during insert (default 64).
func WithEfConstruction(ef int) Option {
	return func(idx *Index) {
		if ef > 0 {
			idx.efC = ef
		}
	}
}

// WithEfSearch sets the candidate list size during search (default 64).
func WithEfSearch(ef int) Option {
	return func(idx *Index) {
		if ef > 0 {
			idx.efS = ef
		}
	}
}

// WithSeed sets the RNG seed used for random levels (default 42).
func WithSeed(seed int64) Option {
	return func(idx *Index) { idx.rng = rand.New(rand.NewSource(seed)) }
}

type node struct {
	id    string
	v     vector.Vector
	neigh [][]int
}

// Index is an HNSW graph over vector.Vector values.
type Index struct {
	dims   int
	metric vector.Metric
	m      int
	mMax0  int
	efC    int
	efS    int
	ml     float64
	nodes  []node
	byID   map[string]int
	entry  int
	maxL   int
	rng    *rand.Rand
}

// New creates an empty HNSW index. dims is recorded for documentation;
// mismatched inserts are skipped (they would score as zero under pkg/vector).
func New(dims int, metric vector.Metric, opts ...Option) *Index {
	idx := &Index{
		dims:   dims,
		metric: metric,
		m:      16,
		mMax0:  32,
		efC:    64,
		efS:    64,
		entry:  -1,
		byID:   make(map[string]int),
		rng:    rand.New(rand.NewSource(42)),
	}
	for _, opt := range opts {
		if opt != nil {
			opt(idx)
		}
	}
	idx.ml = 1.0 / math.Log(float64(idx.m))
	return idx
}

// Len returns the number of indexed vectors.
func (idx *Index) Len() int { return len(idx.nodes) }

// Add inserts id with vector v. Re-adding an existing id is ignored
// (graphs are not mutated in place). Empty or nil vectors are ignored.
func (idx *Index) Add(id string, v vector.Vector) {
	if len(v) == 0 {
		return
	}
	if _, exists := idx.byID[id]; exists {
		return
	}
	v = vector.Clone(v)
	level := idx.randomLevel()
	n := node{id: id, v: v, neigh: make([][]int, level+1)}
	cur := len(idx.nodes)
	idx.nodes = append(idx.nodes, n)
	idx.byID[id] = cur

	if idx.entry < 0 {
		idx.entry = cur
		idx.maxL = level
		return
	}

	// Exact neighbor selection among nodes present on layer l. Graph search
	// is used at query time; construction stays exact so the graph is a
	// true (capped) k-NN and recall does not collapse around the entry point.
	for l := min(level, idx.maxL); l >= 0; l-- {
		cands := make([]int, 0, len(idx.nodes))
		for i := range idx.nodes {
			if i == cur {
				continue
			}
			if l < len(idx.nodes[i].neigh) {
				cands = append(cands, i)
			}
		}
		m := idx.m
		if l == 0 {
			m = idx.mMax0
		}
		selected := idx.selectNeighborsHeuristic(v, cands, m)
		idx.nodes[cur].neigh[l] = append([]int(nil), selected...)
		for _, nb := range selected {
			idx.nodes[nb].neigh = growLevels(idx.nodes[nb].neigh, l)
			idx.nodes[nb].neigh[l] = append(idx.nodes[nb].neigh[l], cur)
			maxN := idx.m
			if l == 0 {
				maxN = idx.mMax0
			}
			if len(idx.nodes[nb].neigh[l]) > maxN {
				// Reverse edges: keep closest, so a node's true neighbors
				// are not evicted by later long-range links.
				idx.nodes[nb].neigh[l] = idx.selectClosest(idx.nodes[nb].v, idx.nodes[nb].neigh[l], maxN)
			}
		}
	}
	if level > idx.maxL {
		idx.maxL = level
		idx.entry = cur
	}
}

// Search returns up to k approximate nearest neighbors.
func (idx *Index) Search(query vector.Vector, k int) []vector.SearchResult {
	if k <= 0 || len(idx.nodes) == 0 || idx.entry < 0 {
		return nil
	}
	start := idx.entry
	for l := idx.maxL; l > 0; l-- {
		start = idx.greedy(query, start, l)
	}
	ef := idx.efS
	if ef < k {
		ef = k
	}
	hits := idx.bestFirst(query, start, ef, 0)
	if k > len(hits) {
		k = len(hits)
	}
	out := make([]vector.SearchResult, k)
	for i := 0; i < k; i++ {
		n := idx.nodes[hits[i]]
		out[i] = vector.SearchResult{
			ID:       n.id,
			Distance: vector.Distance(query, n.v, idx.metric),
			Vector:   vector.Clone(n.v),
		}
	}
	return out
}

// RecallAgainst reports the mean fraction of exact top-k IDs recovered
// by this index over the given queries.
func (idx *Index) RecallAgainst(exact *vector.Store, queries []vector.Vector, k int) float64 {
	if exact == nil || k <= 0 || len(queries) == 0 {
		return 0
	}
	var sum float64
	for _, q := range queries {
		want := exact.SearchIDs(q, k)
		got := idx.Search(q, k)
		set := make(map[string]struct{}, len(got))
		for _, h := range got {
			set[h.ID] = struct{}{}
		}
		var hit int
		for _, h := range want {
			if _, ok := set[h.ID]; ok {
				hit++
			}
		}
		if len(want) > 0 {
			sum += float64(hit) / float64(len(want))
		}
	}
	return sum / float64(len(queries))
}

// Save gob-encodes the index to path.
func (idx *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(idx.snapshot())
}

// Load replaces the index from a gob file written by Save.
func (idx *Index) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	var data indexData
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return err
	}
	idx.restore(data)
	return nil
}

type indexData struct {
	Dims   int
	Metric vector.Metric
	M      int
	MMax0  int
	EfC    int
	EfS    int
	Entry  int
	MaxL   int
	IDs    []string
	Vecs   []vector.Vector
	Neigh  [][][]int
}

func (idx *Index) snapshot() indexData {
	data := indexData{
		Dims: idx.dims, Metric: idx.metric, M: idx.m, MMax0: idx.mMax0,
		EfC: idx.efC, EfS: idx.efS, Entry: idx.entry, MaxL: idx.maxL,
		IDs: make([]string, len(idx.nodes)), Vecs: make([]vector.Vector, len(idx.nodes)),
		Neigh: make([][][]int, len(idx.nodes)),
	}
	for i, n := range idx.nodes {
		data.IDs[i] = n.id
		data.Vecs[i] = n.v
		data.Neigh[i] = n.neigh
	}
	return data
}

func (idx *Index) restore(data indexData) {
	idx.dims = data.Dims
	idx.metric = data.Metric
	idx.m = data.M
	idx.mMax0 = data.MMax0
	idx.efC = data.EfC
	idx.efS = data.EfS
	idx.entry = data.Entry
	idx.maxL = data.MaxL
	idx.ml = 1.0 / math.Log(math.Max(2, float64(idx.m)))
	idx.nodes = make([]node, len(data.IDs))
	idx.byID = make(map[string]int, len(data.IDs))
	for i := range data.IDs {
		idx.nodes[i] = node{id: data.IDs[i], v: data.Vecs[i], neigh: data.Neigh[i]}
		idx.byID[data.IDs[i]] = i
	}
	if idx.rng == nil {
		idx.rng = rand.New(rand.NewSource(42))
	}
}

func (idx *Index) randomLevel() int {
	// floor(-ln(U) * ml) with U in (0,1]
	u := idx.rng.Float64()
	if u <= 0 {
		u = 1e-9
	}
	return int(math.Floor(-math.Log(u) * idx.ml))
}

func (idx *Index) dist(a, b vector.Vector) float32 {
	return vector.Distance(a, b, idx.metric)
}

func (idx *Index) better(a, b float32) bool {
	if idx.metric.Ascending() {
		return a < b
	}
	return a > b
}

func (idx *Index) greedy(q vector.Vector, start, layer int) int {
	cur := start
	curD := idx.dist(q, idx.nodes[cur].v)
	for {
		best, bestD := cur, curD
		var neigh []int
		if layer < len(idx.nodes[cur].neigh) {
			neigh = idx.nodes[cur].neigh[layer]
		}
		for _, nb := range neigh {
			d := idx.dist(q, idx.nodes[nb].v)
			if idx.better(d, bestD) {
				best, bestD = nb, d
			}
		}
		if best == cur {
			return cur
		}
		cur, curD = best, bestD
	}
}

// bestFirst expands closest-first and always enqueues unvisited neighbors
// (unlike a pruned HNSW candidate list, which can stall on a k-NN graph).
func (idx *Index) bestFirst(q vector.Vector, start, ef, layer int) []int {
	type item struct {
		i int
		d float32
	}
	if ef < 1 {
		ef = 1
	}
	seen := map[int]struct{}{start: {}}
	frontier := []item{{start, idx.dist(q, idx.nodes[start].v)}}
	var expanded []item
	worstExp := func() float32 {
		w := expanded[0].d
		for i := 1; i < len(expanded); i++ {
			if idx.better(w, expanded[i].d) {
				w = expanded[i].d
			}
		}
		return w
	}
	for len(frontier) > 0 {
		bi := 0
		for i := 1; i < len(frontier); i++ {
			if idx.better(frontier[i].d, frontier[bi].d) {
				bi = i
			}
		}
		c := frontier[bi]
		frontier = append(frontier[:bi], frontier[bi+1:]...)
		if len(expanded) >= ef && !idx.better(c.d, worstExp()) {
			break
		}
		expanded = append(expanded, c)
		var neigh []int
		if layer < len(idx.nodes[c.i].neigh) {
			neigh = idx.nodes[c.i].neigh[layer]
		}
		for _, nb := range neigh {
			if _, ok := seen[nb]; ok {
				continue
			}
			seen[nb] = struct{}{}
			frontier = append(frontier, item{nb, idx.dist(q, idx.nodes[nb].v)})
		}
	}
	sort.Slice(expanded, func(i, j int) bool { return idx.better(expanded[i].d, expanded[j].d) })
	out := make([]int, 0, len(expanded))
	uniq := map[int]struct{}{}
	for _, it := range expanded {
		if _, ok := uniq[it.i]; ok {
			continue
		}
		uniq[it.i] = struct{}{}
		out = append(out, it.i)
	}
	return out
}

func (idx *Index) selectClosest(v vector.Vector, candidates []int, m int) []int {
	type item struct {
		i int
		d float32
	}
	seen := make(map[int]struct{}, len(candidates))
	items := make([]item, 0, len(candidates))
	for _, i := range candidates {
		if i < 0 || i >= len(idx.nodes) {
			continue
		}
		if _, ok := seen[i]; ok {
			continue
		}
		seen[i] = struct{}{}
		items = append(items, item{i, idx.dist(v, idx.nodes[i].v)})
	}
	sort.Slice(items, func(i, j int) bool { return idx.better(items[i].d, items[j].d) })
	if m > len(items) {
		m = len(items)
	}
	out := make([]int, m)
	for i := 0; i < m; i++ {
		out[i] = items[i].i
	}
	return out
}

func (idx *Index) selectNeighbors(v vector.Vector, candidates []int, m int) []int {
	return idx.selectNeighborsHeuristic(v, candidates, m)
}

// selectNeighborsHeuristic keeps spatially diverse neighbors (HNSW paper),
// so the graph retains long-range edges instead of collapsing to a local k-NN.
func (idx *Index) selectNeighborsHeuristic(v vector.Vector, candidates []int, m int) []int {
	type item struct {
		i int
		d float32
	}
	seen := make(map[int]struct{}, len(candidates))
	items := make([]item, 0, len(candidates))
	for _, i := range candidates {
		if i < 0 || i >= len(idx.nodes) {
			continue
		}
		if _, ok := seen[i]; ok {
			continue
		}
		seen[i] = struct{}{}
		items = append(items, item{i, idx.dist(v, idx.nodes[i].v)})
	}
	sort.Slice(items, func(i, j int) bool { return idx.better(items[i].d, items[j].d) })
	if m > len(items) {
		m = len(items)
	}
	picked := make([]int, 0, m)
	for _, it := range items {
		if len(picked) >= m {
			break
		}
		ok := true
		for _, p := range picked {
			// Skip if this candidate is closer to an already-picked
			// neighbor than to v — it doesn't add a new direction.
			if idx.better(idx.dist(idx.nodes[it.i].v, idx.nodes[p].v), it.d) {
				ok = false
				break
			}
		}
		if ok {
			picked = append(picked, it.i)
		}
	}
	// Fill remaining slots with closest leftovers so degree stays high.
	if len(picked) < m {
		in := map[int]struct{}{}
		for _, p := range picked {
			in[p] = struct{}{}
		}
		for _, it := range items {
			if len(picked) >= m {
				break
			}
			if _, exists := in[it.i]; exists {
				continue
			}
			picked = append(picked, it.i)
		}
	}
	return picked
}

func growLevels(neigh [][]int, layer int) [][]int {
	for len(neigh) <= layer {
		neigh = append(neigh, nil)
	}
	return neigh
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
