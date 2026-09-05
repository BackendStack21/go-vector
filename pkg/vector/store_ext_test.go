package vector

import (
	"bytes"
	"path/filepath"
	"sync"
	"testing"
)

func TestStorePackedGetStable(t *testing.T) {
	s := NewStore(CosineDistance)
	first := make(Vector, 64)
	for j := range first {
		first[j] = float32(j + 1)
	}
	s.Add("first", first)
	for i := 0; i < 500; i++ {
		v := make(Vector, 64)
		for j := range v {
			v[j] = float32(i + j)
		}
		s.Add("x", v)
	}
	got := s.Get("first")
	if !Equal(got, first) {
		t.Fatalf("Get after packed growth mutated first vector: %v vs %v (packedOK=%v)", got[:4], first[:4], s.packedOK)
	}
	if !Equal(s.vectors[0], first) {
		t.Fatalf("internal row 0 stale: %v", s.vectors[0][:4])
	}
}

func TestStoreSearchMatchesDistance(t *testing.T) {
	s := NewStore(EuclideanDistance)
	for i := 0; i < 80; i++ {
		v := make(Vector, 32)
		for j := range v {
			v[j] = float32(i) + float32(j)/32
		}
		s.Add(itoaExt(i), v)
	}
	q := Clone(s.vectors[7])
	got := s.Search(q, 5)
	if got[0].ID != "7" {
		t.Fatalf("self id %s packed=%v dist=%v", got[0].ID, s.packedOK, got[0].Distance)
	}
	if !approxEqual(got[0].Distance, 0, 1e-6) {
		t.Fatalf("self dist %v", got[0].Distance)
	}
}

func itoaExt(i int) string {
	if i == 0 {
		return "0"
	}
	var b [12]byte
	n := len(b)
	for i > 0 {
		n--
		b[n] = byte('0' + i%10)
		i /= 10
	}
	return string(b[n:])
}

func TestStoreIntrospection(t *testing.T) {
	s := NewStore(EuclideanDistance)
	if s.Metric() != EuclideanDistance || s.Has("x") || s.Dims() != 0 {
		t.Fatal("empty store introspection")
	}
	s.Add("a", Vector{1, 0})
	s.Add("b", Vector{0, 1})
	if !s.Has("a") || s.Has("z") {
		t.Fatal("Has")
	}
	if s.Dims() != 2 {
		t.Fatalf("Dims() = %d", s.Dims())
	}
	ids := s.IDs()
	if len(ids) != 2 || ids[0] != "a" || ids[1] != "b" {
		t.Fatalf("IDs = %v", ids)
	}
	ids[0] = "mutated"
	if s.IDs()[0] != "a" {
		t.Fatal("IDs must clone")
	}
	s.Clear()
	if s.Len() != 0 || s.Metric() != EuclideanDistance {
		t.Fatal("Clear must drop data and keep metric")
	}
}

func TestStoreUpsertAndUnique(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0})
	s.Add("a", Vector{0, 1}) // duplicate
	if s.Len() != 2 {
		t.Fatal("Add still appends duplicates")
	}
	if !Equal(s.Get("a"), Vector{1, 0}) {
		t.Fatal("Get first match")
	}
	if s.AddUnique("a", Vector{1, 1}) {
		t.Fatal("AddUnique must fail on existing")
	}
	if !s.AddUnique("b", Vector{0, 1}) || s.Len() != 3 {
		t.Fatal("AddUnique new id")
	}
	s.Upsert("a", Vector{2, 0})
	if !Equal(s.Get("a"), Vector{2, 0}) {
		t.Fatalf("Upsert first match: %v", s.Get("a"))
	}
	if s.Len() != 3 {
		t.Fatal("Upsert must not grow when id exists")
	}
	s.Upsert("c", Vector{3, 0})
	if s.Len() != 4 || !s.Has("c") {
		t.Fatal("Upsert inserts missing id")
	}
}

func TestStoreMetadata(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("en", Vector{1, 0})
	s.Add("es", Vector{0.9, 0.1})
	s.Add("en2", Vector{0.8, 0.2})
	if s.SetMeta("missing", map[string]string{"lang": "x"}) {
		t.Fatal("SetMeta missing")
	}
	if !s.SetMeta("en", map[string]string{"lang": "en"}) || !s.SetMeta("en2", map[string]string{"lang": "en"}) {
		t.Fatal("SetMeta")
	}
	if !s.SetMeta("es", map[string]string{"lang": "es"}) {
		t.Fatal("SetMeta es")
	}
	m := s.Meta("en")
	m["lang"] = "mut"
	if s.Meta("en")["lang"] != "en" {
		t.Fatal("Meta must clone")
	}
	hits := s.SearchOpts(Vector{1, 0}, 5, WithMetaEqual("lang", "en"))
	if len(hits) != 2 {
		t.Fatalf("meta filter: %d hits", len(hits))
	}
	for _, h := range hits {
		if h.ID == "es" {
			t.Fatal("es leaked through lang=en filter")
		}
	}
}

func TestStoreSearchIDsAndWithoutVectors(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0})
	s.Add("b", Vector{0, 1})
	full := s.Search(Vector{1, 0}, 2)
	ids := s.SearchIDs(Vector{1, 0}, 2)
	if len(full) != len(ids) {
		t.Fatal("len")
	}
	for i := range full {
		if full[i].ID != ids[i].ID || !approxEqual(full[i].Distance, ids[i].Distance, 1e-6) {
			t.Fatalf("mismatch %v vs %v", full[i], ids[i])
		}
		if ids[i].ID == "" {
			t.Fatal("empty id")
		}
	}
	slim := s.SearchOpts(Vector{1, 0}, 2, WithoutVectors())
	if slim[0].Vector != nil {
		t.Fatal("WithoutVectors")
	}
}

func TestStorePredicateAndMaxDistance(t *testing.T) {
	s := NewStore(EuclideanDistance)
	s.Add("near", Vector{0, 0})
	s.Add("mid", Vector{3, 4})
	s.Add("far", Vector{10, 0})
	hits := s.SearchOpts(Vector{0, 0}, 10, WithPredicate(func(id string) bool { return id != "mid" }))
	if len(hits) != 2 || hits[0].ID != "near" {
		t.Fatalf("pred: %+v", hits)
	}
	near := s.SearchOpts(Vector{0, 0}, 10, WithMaxDistance(1))
	if len(near) != 1 || near[0].ID != "near" {
		t.Fatalf("maxDist: %+v", near)
	}
	rad := s.SearchRadius(Vector{0, 0}, 5)
	if len(rad) != 2 {
		t.Fatalf("radius: %+v", rad)
	}
}

func TestStoreSearchBatch(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0})
	s.Add("b", Vector{0, 1})
	batch := s.SearchBatch([]Vector{{1, 0}, {0, 1}}, 1)
	if len(batch) != 2 || batch[0][0].ID != "a" || batch[1][0].ID != "b" {
		t.Fatalf("%+v", batch)
	}
	if s.SearchBatch(nil, 1) != nil {
		t.Fatal("nil queries")
	}
}

func TestStoreDuplicateIDIndex(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0, 0})
	s.Add("b", Vector{0, 1, 0})
	s.Add("a", Vector{0, 0, 1})
	if !Equal(s.Get("a"), Vector{1, 0, 0}) {
		t.Fatal("first a")
	}
	if !s.Remove("a") {
		t.Fatal("remove first a")
	}
	if !Equal(s.Get("a"), Vector{0, 0, 1}) {
		t.Fatalf("second a now first: %v", s.Get("a"))
	}
	if s.Len() != 2 || !s.Has("b") {
		t.Fatal("b survived")
	}
	if !s.Remove("a") || s.Has("a") {
		t.Fatal("remove remaining a")
	}
}

func TestStoreMixedLengthFallback(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("short", Vector{1, 0})
	s.Add("long", Vector{1, 0, 0, 0})
	if s.packedOK {
		t.Fatal("mixed lengths must unpack")
	}
	hits := s.Search(Vector{1, 0}, 2)
	if len(hits) != 2 {
		t.Fatal(hits)
	}
	// mismatched cosine distance is 1; the matching short vector is closer
	if hits[0].ID != "short" {
		t.Fatalf("want short first, got %+v", hits)
	}
}

func TestStorePackedSearchMatchesJagged(t *testing.T) {
	packed := NewStore(CosineDistance)
	for i, v := range []Vector{{1, 0, 0}, {0, 1, 0}, {0.7, 0.7, 0}} {
		packed.Add(string(rune('a'+i)), v)
	}
	if !packed.packedOK {
		t.Fatal("expected packed layout")
	}
	q := Vector{1, 0.1, 0}
	a := packed.Search(q, 3)
	b := packed.SearchOpts(q, 3, WithParallel(1)) // minN=1 forces parallel
	if len(a) != len(b) {
		t.Fatal("parallel len")
	}
	for i := range a {
		if a[i].ID != b[i].ID || !approxEqual(a[i].Distance, b[i].Distance, 1e-5) {
			t.Fatalf("serial %v vs parallel %v", a[i], b[i])
		}
	}
}

func TestStorePersistRoundtripNewAPIs(t *testing.T) {
	s := NewStore(DotProductSimilarity)
	s.Add("x", Vector{1, 0, 0})
	s.SetMeta("x", map[string]string{"k": "v"})
	var buf bytes.Buffer
	if _, err := s.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	s2 := NewStore(CosineDistance)
	if _, err := s2.ReadFrom(&buf); err != nil {
		t.Fatal(err)
	}
	if s2.Metric() != DotProductSimilarity || s2.Meta("x")["k"] != "v" {
		t.Fatalf("restore meta/metric: %s %v", s2.Metric(), s2.Meta("x"))
	}
	dir := t.TempDir()
	if err := s.SaveAtomic(filepath.Join(dir, "s.gob")); err != nil {
		t.Fatal(err)
	}
	s3 := NewStore(CosineDistance)
	if err := s3.Load(filepath.Join(dir, "s.gob")); err != nil {
		t.Fatal(err)
	}
	if !Equal(s3.Get("x"), Vector{1, 0, 0}) {
		t.Fatal("atomic load")
	}
}

func TestLockedStoreConcurrent(t *testing.T) {
	ls := NewLockedStore(CosineDistance)
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			id := string(rune('a' + i%10))
			ls.Upsert(id, Vector{float32(i), 1})
			_ = ls.Search(Vector{1, 1}, 3)
			_ = ls.Get(id)
		}(i)
	}
	wg.Wait()
	if ls.Len() != 10 {
		t.Fatalf("Len = %d, want 10 unique ids", ls.Len())
	}
}

func TestRerank(t *testing.T) {
	hits := []SearchResult{
		{ID: "far", Vector: Vector{0, 1}, Distance: 0},
		{ID: "near", Vector: Vector{1, 0}, Distance: 9},
		{ID: "novec", Distance: 0},
	}
	got := Rerank(Vector{1, 0}, hits, CosineDistance, 2)
	if len(got) != 2 || got[0].ID != "near" {
		t.Fatalf("%+v", got)
	}
	if Rerank(Vector{1}, hits, CosineDistance, 0) != nil {
		t.Fatal("k<=0")
	}
}
