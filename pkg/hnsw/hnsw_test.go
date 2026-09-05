package hnsw

import (
	"math/rand"
	"path/filepath"
	"testing"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

func TestHNSWNeighborQuality(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	exact := vector.NewStore(vector.CosineDistance)
	idx := New(8, vector.CosineDistance, WithM(4), WithEfConstruction(20), WithSeed(3))
	for i := 0; i < 30; i++ {
		v := make(vector.Vector, 8)
		for j := range v {
			v[j] = rng.Float32()
		}
		v = vector.Normalize(v)
		exact.Add(itoa(i), v)
		idx.Add(itoa(i), v)
	}
	want := exact.SearchIDs(exact.Get("0"), 5)
	var gotIDs []string
	for _, ni := range idx.nodes[0].neigh[0] {
		gotIDs = append(gotIDs, idx.nodes[ni].id)
	}
	t.Logf("exact near 0: %v", want)
	t.Logf("graph neigh 0: %v", gotIDs)
	overlap := 0
	set := map[string]struct{}{}
	for _, id := range gotIDs {
		set[id] = struct{}{}
	}
	for _, h := range want {
		if h.ID == "0" {
			continue
		}
		if _, ok := set[h.ID]; ok {
			overlap++
		}
	}
	if overlap < 2 {
		t.Fatalf("overlap %d", overlap)
	}
}

func TestHNSWMatchesExactOnTinySet(t *testing.T) {
	exact := vector.NewStore(vector.CosineDistance)
	idx := New(3, vector.CosineDistance, WithM(8), WithEfConstruction(32), WithEfSearch(32))
	items := []struct {
		id string
		v  vector.Vector
	}{
		{"a", vector.Vector{1, 0, 0}},
		{"b", vector.Vector{0.9, 0.1, 0}},
		{"c", vector.Vector{0, 1, 0}},
		{"d", vector.Vector{0, 0, 1}},
	}
	for _, it := range items {
		exact.Add(it.id, it.v)
		idx.Add(it.id, it.v)
	}
	q := vector.Vector{1, 0.05, 0}
	got := idx.Search(q, 2)
	want := exact.Search(q, 2)
	if len(got) != 2 || got[0].ID != want[0].ID {
		t.Fatalf("got %+v want %+v", got, want)
	}
}

func TestHNSWExactOnMedium(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	exact := vector.NewStore(vector.CosineDistance)
	idx := New(8, vector.CosineDistance, WithM(8), WithEfConstruction(40), WithEfSearch(40), WithSeed(2))
	for i := 0; i < 50; i++ {
		v := make(vector.Vector, 8)
		for j := range v {
			v[j] = rng.Float32()
		}
		v = vector.Normalize(v)
		exact.Add(itoa(i), v)
		idx.Add(itoa(i), v)
	}
	var rec float64
	for i := 0; i < 50; i++ {
		q := exact.Get(itoa(i))
		if idx.Search(q, 1)[0].ID != itoa(i) {
			t.Fatalf("self search failed for %d", i)
		}
		rec += idx.RecallAgainst(exact, []vector.Vector{q}, 5)
	}
	rec /= 50
	t.Logf("medium recall@5 %v", rec)
	if rec < 0.95 {
		t.Fatalf("recall %v", rec)
	}
}

func TestHNSWRecall(t *testing.T) {
	const n, d, k = 400, 32, 10
	rng := rand.New(rand.NewSource(1))
	exact := vector.NewStore(vector.CosineDistance)
	idx := New(d, vector.CosineDistance, WithM(12), WithEfConstruction(64), WithEfSearch(64), WithSeed(1))
	for i := 0; i < n; i++ {
		v := make(vector.Vector, d)
		for j := range v {
			v[j] = rng.Float32()*2 - 1
		}
		v = vector.Normalize(v)
		id := itoa(i)
		exact.Add(id, v)
		idx.Add(id, v)
	}
	var queries []vector.Vector
	for i := 0; i < 20; i++ {
		queries = append(queries, exact.Get(itoa(i*7)))
	}
	rec := idx.RecallAgainst(exact, queries, k)
	if rec < 0.9 {
		t.Fatalf("recall = %v, want >= 0.9", rec)
	}
}

func TestHNSWSaveLoad(t *testing.T) {
	idx := New(2, vector.EuclideanDistance)
	idx.Add("p", vector.Vector{0, 0})
	idx.Add("q", vector.Vector{1, 0})
	path := filepath.Join(t.TempDir(), "h.gob")
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}
	idx2 := New(2, vector.CosineDistance)
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}
	if idx2.Search(vector.Vector{0, 0}, 1)[0].ID != "p" {
		t.Fatal("load search")
	}
}

func TestHNSWIgnoresEmptyAndDup(t *testing.T) {
	idx := New(2, vector.CosineDistance)
	idx.Add("a", nil)
	idx.Add("a", vector.Vector{1, 0})
	idx.Add("a", vector.Vector{0, 1})
	if idx.Len() != 1 {
		t.Fatalf("len=%d", idx.Len())
	}
	if idx.Search(vector.Vector{1, 0}, 0) != nil {
		t.Fatal("k=0")
	}
}

func itoa(i int) string {
	if i == 0 {
		return "0"
	}
	var b [12]byte
	n := len(b)
	neg := i < 0
	if neg {
		i = -i
	}
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	if neg {
		n--
		b[n] = '-'
	}
	return string(b[n:])
}
