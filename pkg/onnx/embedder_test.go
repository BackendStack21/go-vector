package onnx

import (
	"math"
	"os"
	"testing"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

var _ vector.Embedder = (*Embedder)(nil)

const (
	testModel = "testdata/model.onnx"
	testVocab = "testdata/vocab.txt"
)

// newTestEmbedder loads all-MiniLM-L6-v2 from testdata, skipping the test
// when the model files are absent. Fetch them with `make model`.
func newTestEmbedder(t *testing.T) *Embedder {
	t.Helper()
	if _, err := os.Stat(testModel); err != nil {
		t.Skipf("model not present (run `make model` to download all-MiniLM-L6-v2): %v", err)
	}
	e, err := New(testModel, testVocab)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { e.Close() })
	return e
}

func TestEmbedderDims(t *testing.T) {
	e := newTestEmbedder(t)
	if e.Dims() != 384 {
		t.Errorf("Dims() = %d, want 384 for MiniLM-L6", e.Dims())
	}
}

func TestEmbedderNormalizedAndDeterministic(t *testing.T) {
	e := newTestEmbedder(t)
	v1, err := e.Embed("the quick brown fox")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if len(v1) != 384 {
		t.Fatalf("len = %d, want 384", len(v1))
	}
	if math.Abs(float64(vector.Norm(v1))-1) > 1e-5 {
		t.Errorf("norm = %v, want 1", vector.Norm(v1))
	}
	v2, _ := e.Embed("the quick brown fox")
	if !vector.EqualEps(v1, v2, 1e-6) {
		t.Error("same text produced different vectors")
	}
}

func TestEmbedderSemanticSimilarity(t *testing.T) {
	e := newTestEmbedder(t)
	cat, _ := e.Embed("a cat is sleeping on the couch")
	dog, _ := e.Embed("a dog is napping on the sofa")
	fin, _ := e.Embed("the stock market closed higher today")

	if vector.Cosine(cat, dog) <= vector.Cosine(cat, fin) {
		t.Errorf("semantics inverted: sim(cat,dog)=%v <= sim(cat,finance)=%v",
			vector.Cosine(cat, dog), vector.Cosine(cat, fin))
	}
}

func TestEmbedderBatchMatchesSingle(t *testing.T) {
	e := newTestEmbedder(t)
	texts := []string{"short", "a considerably longer sentence about machine learning and embeddings"}
	batch, err := e.EmbedBatch(texts)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	for i, text := range texts {
		single, _ := e.Embed(text)
		// Padding+masking means batch results must match per-text results.
		if !vector.EqualEps(batch[i], single, 1e-4) {
			t.Errorf("batch[%d] diverges from single embed of %q (cos=%v)",
				i, text, vector.Cosine(batch[i], single))
		}
	}
}

func TestEmbedderEmptyBatch(t *testing.T) {
	e := newTestEmbedder(t)
	vecs, err := e.EmbedBatch(nil)
	if vecs != nil || err != nil {
		t.Errorf("EmbedBatch(nil) = %v, %v; want nil, nil", vecs, err)
	}
}

func TestEmbedderEmptyText(t *testing.T) {
	e := newTestEmbedder(t)
	v, err := e.Embed("")
	if err != nil {
		t.Fatalf("Embed(\"\"): %v", err)
	}
	if len(v) != 384 {
		t.Errorf("len = %d, want 384", len(v))
	}
}

func TestEmbedderStoreIntegration(t *testing.T) {
	e := newTestEmbedder(t)
	store := vector.NewStore(vector.CosineDistance)

	docs := []string{
		"cats are wonderful pets",
		"dogs are loyal companions",
		"the federal reserve raised interest rates",
	}
	vecs, err := e.EmbedBatch(docs)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	for i, doc := range docs {
		store.Add(doc, vecs[i])
	}

	q, _ := e.Embed("animals that people keep at home")
	results := store.Search(q, 2)
	for _, r := range results {
		if r.ID == docs[2] {
			t.Errorf("finance doc ranked in top 2 for a pets query: %v", results)
		}
	}
}

func TestEmbedderBadPaths(t *testing.T) {
	if _, err := os.Stat(testVocab); err != nil {
		t.Skip("vocab not present (run `make model`)")
	}
	if _, err := New("testdata/missing.onnx", testVocab); err == nil {
		t.Error("want error for missing model, got nil")
	}
	if _, err := New(testModel, "testdata/missing-vocab.txt"); err == nil {
		t.Error("want error for missing vocab, got nil")
	}
}
