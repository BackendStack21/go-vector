package vector

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestCompatStoreV1Gob(t *testing.T) {
	s := NewStore(EuclideanDistance)
	if err := s.Load("testdata/compat/store_v1.gob"); err != nil {
		t.Fatalf("Load v1 gob: %v", err)
	}
	if s.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", s.Len())
	}
	if s.Metric() != CosineDistance {
		t.Fatalf("metric = %s, want cosine", s.Metric())
	}
	if !Equal(s.Get("alpha"), Vector{1, 2, 3}) {
		t.Errorf("alpha = %v", s.Get("alpha"))
	}

	want := loadExpected(t)
	got := s.Search(Vector{1, 2, 3}, 3)
	if len(got) != 3 {
		t.Fatalf("search len %d", len(got))
	}
	for i, h := range want.Search {
		if got[i].ID != h.ID {
			t.Errorf("search[%d].ID = %s, want %s", i, got[i].ID, h.ID)
		}
		if !approxEqual(got[i].Distance, h.Distance, 1e-5) {
			t.Errorf("search[%d].Distance = %v, want %v", i, got[i].Distance, h.Distance)
		}
	}
}

func TestCompatStoreV1JSON(t *testing.T) {
	s := NewStore(ManhattanDistance)
	if err := s.LoadJSON("testdata/compat/store_v1.json"); err != nil {
		t.Fatalf("LoadJSON: %v", err)
	}
	if s.Len() != 3 || s.Get("beta") == nil {
		t.Fatalf("json restore incomplete: len=%d", s.Len())
	}
}

func TestCompatEmbedderV1(t *testing.T) {
	rp, err := LoadEmbedder("testdata/compat/embedder_v1.gob")
	if err != nil {
		t.Fatalf("LoadEmbedder: %v", err)
	}
	want := loadExpected(t)
	if rp.VocabSize() != want.VocabSize || rp.Dims() != want.Dims {
		t.Fatalf("vocab=%d dims=%d, want %d %d", rp.VocabSize(), rp.Dims(), want.VocabSize, want.Dims)
	}
	got, err := rp.Embed("hello world")
	if err != nil {
		t.Fatal(err)
	}
	if !EqualEps(got, Vector(want.Embed), 1e-5) {
		t.Errorf("embed mismatch\n got %v\nwant %v", got, want.Embed)
	}
}

type expectedV1 struct {
	Search []struct {
		ID       string
		Distance float32
	}
	Embed     []float32
	VocabSize int `json:"vocab_size"`
	Dims      int
}

func loadExpected(t *testing.T) expectedV1 {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("testdata", "compat", "expected_v1.json"))
	if err != nil {
		t.Fatal(err)
	}
	var e expectedV1
	if err := json.Unmarshal(b, &e); err != nil {
		t.Fatal(err)
	}
	return e
}
