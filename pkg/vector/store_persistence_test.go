package vector

import (
	"os"
	"path/filepath"
	"testing"
)

func TestStoreSaveLoad(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 2, 3})
	s.Add("b", Vector{4, 5, 6})
	s.Add("c", Vector{7, 8, 9})

	dir := t.TempDir()
	path := filepath.Join(dir, "store.gob")

	if err := s.Save(path); err != nil {
		t.Fatalf("Save() error: %v", err)
	}

	// Load into a new store
	s2 := NewStore(EuclideanDistance) // different metric
	if err := s2.Load(path); err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if s2.Len() != 3 {
		t.Errorf("Len() = %d, want 3", s2.Len())
	}

	// Verify data
	for _, id := range []string{"a", "b", "c"} {
		v := s2.Get(id)
		if v == nil {
			t.Errorf("Get(%q) returned nil", id)
		}
	}

	// Metric should be restored (CosineDistance, not EuclideanDistance)
	result := s2.Search(Vector{1, 2, 3}, 1)
	if result[0].ID != "a" {
		t.Errorf("closest to {1,2,3} should be 'a', got %s", result[0].ID)
	}
}

func TestStoreSaveLoadRoundtrip(t *testing.T) {
	original := NewStore(ManhattanDistance)
	for i := 0; i < 100; i++ {
		original.Add("vec", Vector{float32(i), float32(i + 1), float32(i + 2)})
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "store.gob")

	if err := original.Save(path); err != nil {
		t.Fatalf("Save() error: %v", err)
	}

	restored := NewStore(CosineDistance) // different metric
	if err := restored.Load(path); err != nil {
		t.Fatalf("Load() error: %v", err)
	}

	if restored.Len() != original.Len() {
		t.Errorf("Len() = %d, want %d", restored.Len(), original.Len())
	}

	// Verify all vectors match
	for i := 0; i < original.Len(); i++ {
		if restored.ids[i] != original.ids[i] {
			t.Errorf("ids[%d] = %q, want %q", i, restored.ids[i], original.ids[i])
		}
		if !Equal(restored.vectors[i], original.vectors[i]) {
			t.Errorf("vectors[%d] differ", i)
		}
	}

	// Search results should match
	query := Vector{50, 51, 52}
	origResults := original.Search(query, 5)
	restResults := restored.Search(query, 5)

	for i := range origResults {
		if origResults[i].ID != restResults[i].ID {
			t.Errorf("result[%d].ID = %s, want %s", i, restResults[i].ID, origResults[i].ID)
		}
		if !approxEqual(origResults[i].Distance, restResults[i].Distance, 1e-5) {
			t.Errorf("result[%d].Distance = %v, want %v", i, restResults[i].Distance, origResults[i].Distance)
		}
	}
}

func TestStoreSaveLoadEmpty(t *testing.T) {
	s := NewStore(CosineDistance)
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.gob")

	if err := s.Save(path); err != nil {
		t.Fatalf("Save(empty) error: %v", err)
	}

	s2 := NewStore(EuclideanDistance)
	if err := s2.Load(path); err != nil {
		t.Fatalf("Load(empty) error: %v", err)
	}

	if s2.Len() != 0 {
		t.Errorf("Len() = %d, want 0", s2.Len())
	}
}

func TestStoreLoadMissingFile(t *testing.T) {
	s := NewStore(CosineDistance)
	err := s.Load("/nonexistent/path/store.gob")
	if err == nil {
		t.Error("Load(missing) should return error")
	}
}

func TestStoreSaveJSONLoadJSON(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("x", Vector{0.1, 0.2, 0.3})
	s.Add("y", Vector{0.4, 0.5, 0.6})

	dir := t.TempDir()
	path := filepath.Join(dir, "store.json")

	if err := s.SaveJSON(path); err != nil {
		t.Fatalf("SaveJSON() error: %v", err)
	}

	// Verify file is readable JSON
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("ReadFile() error: %v", err)
	}
	if len(data) == 0 {
		t.Error("JSON file is empty")
	}

	s2 := NewStore(EuclideanDistance)
	if err := s2.LoadJSON(path); err != nil {
		t.Fatalf("LoadJSON() error: %v", err)
	}

	if s2.Len() != 2 {
		t.Errorf("Len() = %d, want 2", s2.Len())
	}

	v := s2.Get("x")
	if v == nil {
		t.Fatal("Get(x) returned nil")
	}
	if !Equal(v, Vector{0.1, 0.2, 0.3}) {
		t.Errorf("Get(x) = %v, want {0.1, 0.2, 0.3}", v)
	}
}

func TestStoreSaveJSONRoundtrip(t *testing.T) {
	original := NewStore(DotProductSimilarity)
	original.Add("a", Vector{1, 0, 0})
	original.Add("b", Vector{0, 1, 0})

	dir := t.TempDir()
	path := filepath.Join(dir, "store.json")

	if err := original.SaveJSON(path); err != nil {
		t.Fatalf("SaveJSON() error: %v", err)
	}

	restored := NewStore(CosineDistance)
	if err := restored.LoadJSON(path); err != nil {
		t.Fatalf("LoadJSON() error: %v", err)
	}

	// Verify DotProductSimilarity metric was restored (descending sort)
	results := restored.Search(Vector{1, 0, 0}, 2)
	if results[0].ID != "a" {
		t.Errorf("DotProduct: closest to {1,0,0} should be 'a', got %s", results[0].ID)
	}
	if results[1].ID != "b" {
		t.Errorf("DotProduct: second should be 'b', got %s", results[1].ID)
	}
}
