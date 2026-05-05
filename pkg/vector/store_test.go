package vector

import (
	"sort"
	"testing"
)

func TestStoreAddAndLen(t *testing.T) {
	s := NewStore(CosineDistance)
	if s.Len() != 0 {
		t.Error("new store should be empty")
	}

	s.Add("a", Vector{1, 0})
	s.Add("b", Vector{0, 1})
	if s.Len() != 2 {
		t.Errorf("Len() = %d, want 2", s.Len())
	}
}

func TestStoreSearchCosine(t *testing.T) {
	s := NewStore(CosineDistance)

	s.Add("cat", Vector{1.0, 0.8, 0.1})
	s.Add("dog", Vector{0.9, 0.7, 0.1})
	s.Add("car", Vector{0.1, 0.0, 0.9})
	s.Add("truck", Vector{0.0, 0.1, 1.0})

	query := Vector{1.0, 0.9, 0.1}
	results := s.Search(query, 3)

	if len(results) != 3 {
		t.Fatalf("Search() returned %d results, want 3", len(results))
	}

	// cat and dog should be closest
	ids := []string{results[0].ID, results[1].ID}
	sort.Strings(ids)
	if ids[0] != "cat" || ids[1] != "dog" {
		t.Errorf("top 2 should be cat and dog, got %v", ids)
	}

	// Distances should be non-decreasing (ascending for CosineDistance)
	for i := 1; i < len(results); i++ {
		if results[i].Distance < results[i-1].Distance {
			t.Errorf("distances not sorted: [%d]=%v > [%d]=%v",
				i-1, results[i-1].Distance, i, results[i].Distance)
		}
	}
}

func TestStoreSearchEuclidean(t *testing.T) {
	s := NewStore(EuclideanDistance)

	s.Add("a", Vector{0, 0})
	s.Add("b", Vector{1, 0})
	s.Add("c", Vector{0, 10})

	query := Vector{0, 0}
	results := s.Search(query, 3)

	if len(results) != 3 {
		t.Fatalf("Search() returned %d results, want 3", len(results))
	}

	if results[0].ID != "a" {
		t.Errorf("closest should be 'a', got %s", results[0].ID)
	}
	if !approxEqual(results[0].Distance, 0, 1e-6) {
		t.Errorf("distance to self = %v, want 0", results[0].Distance)
	}
}

func TestStoreSearchDotProduct(t *testing.T) {
	s := NewStore(DotProductSimilarity)

	// Normalized vectors so dot product ≈ cosine similarity
	s.Add("same", Vector{1, 0, 0})
	s.Add("near", Vector{0.9, 0.1, 0})
	s.Add("far", Vector{0, 1, 0})

	query := Vector{1, 0, 0}
	results := s.Search(query, 3)

	if results[0].ID != "same" {
		t.Errorf("closest should be 'same', got %s", results[0].ID)
	}

	// For DotProductSimilarity, higher = better, so descending
	if results[0].Distance < results[1].Distance {
		t.Error("DotProductSimilarity should sort descending")
	}
}

func TestStoreSearchEmpty(t *testing.T) {
	s := NewStore(CosineDistance)
	results := s.Search(Vector{1, 0}, 5)
	if results != nil {
		t.Errorf("Search(empty) = %v, want nil", results)
	}
}

func TestStoreSearchKZero(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0})
	results := s.Search(Vector{1, 0}, 0)
	if results != nil {
		t.Errorf("Search(k=0) = %v, want nil", results)
	}
}

func TestStoreSearchKExceeds(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0})
	s.Add("b", Vector{0, 1})

	results := s.Search(Vector{1, 0}, 10)
	if len(results) != 2 {
		t.Errorf("Search(k=10) returned %d results, want 2", len(results))
	}
}

func TestStoreGet(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("foo", Vector{1, 2, 3})

	v := s.Get("foo")
	if v == nil {
		t.Fatal("Get('foo') returned nil")
	}
	if !Equal(v, Vector{1, 2, 3}) {
		t.Errorf("Get('foo') = %v, want {1,2,3}", v)
	}

	if v := s.Get("nonexistent"); v != nil {
		t.Errorf("Get('nonexistent') = %v, want nil", v)
	}
}

func TestStoreRemove(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0})
	s.Add("b", Vector{0, 1})
	s.Add("c", Vector{0, 0})

	if !s.Remove("b") {
		t.Error("Remove('b') should return true")
	}
	if s.Len() != 2 {
		t.Errorf("Len() = %d, want 2", s.Len())
	}
	if s.Get("b") != nil {
		t.Error("Get('b') should return nil after removal")
	}

	// Removing nonexistent
	if s.Remove("b") {
		t.Error("Remove('b') second time should return false")
	}

	// Removing last
	if !s.Remove("c") {
		t.Error("Remove('c') should return true")
	}
	if s.Len() != 1 {
		t.Errorf("Len() = %d, want 1", s.Len())
	}

	// Removing only
	if !s.Remove("a") {
		t.Error("Remove('a') should return true")
	}
	if s.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s.Len())
	}
}

func TestStoreSearchSingleVector(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("only", Vector{1, 2, 3})

	results := s.Search(Vector{1, 2, 3}, 5)
	if len(results) != 1 {
		t.Fatalf("Search() returned %d results, want 1", len(results))
	}
	if results[0].ID != "only" {
		t.Errorf("result ID = %s, want 'only'", results[0].ID)
	}
	// CosineDist of identical vectors should be ~0
	if !approxEqual(results[0].Distance, 0, 1e-6) {
		t.Errorf("distance to self = %v, want ~0", results[0].Distance)
	}
}

func TestStoreGetReturnsClone(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("x", Vector{1, 2})

	v := s.Get("x")
	v[0] = 99

	// Original should be unchanged
	v2 := s.Get("x")
	if v2[0] != 1 {
		t.Error("Get() should return a clone, not the internal vector")
	}
}

func TestSearchResultClones(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("x", Vector{1, 2})

	results := s.Search(Vector{1, 2}, 1)
	results[0].Vector[0] = 99

	// Original should be unchanged
	v := s.Get("x")
	if v[0] != 1 {
		t.Error("SearchResult.Vector should be a clone, not the internal vector")
	}
}
