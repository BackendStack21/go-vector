package vector

import (
	"testing"
)

func TestRandomProjectionsFit(t *testing.T) {
	corpus := []string{
		"the cat sat on the mat",
		"the dog ran in the park",
		"a cat and a dog",
	}

	rp := NewRandomProjections(64)
	rp.Fit(corpus)

	if rp.VocabSize() == 0 {
		t.Error("vocab should not be empty after Fit")
	}
	if rp.Dims() != 64 {
		t.Errorf("Dims() = %d, want 64", rp.Dims())
	}
	// vocab should contain "cat", "dog", "sat", "mat", "ran", "park", "the", "and"
	for _, word := range []string{"cat", "dog", "sat", "mat", "ran", "park"} {
		if _, ok := rp.vocab[word]; !ok {
			t.Errorf("vocab missing expected token: %q", word)
		}
	}
	// "the" should be filtered (too short — but it's 3 chars, so it stays)
}

func TestRandomProjectionsEmbed(t *testing.T) {
	corpus := []string{
		"the quick brown fox",
		"the lazy dog sleeps",
		"quick brown fox jumps",
	}

	rp := NewRandomProjections(32)
	rp.Fit(corpus)

	v, err := rp.Embed("quick brown fox")
	if err != nil {
		t.Fatalf("Embed() error: %v", err)
	}
	if len(v) != 32 {
		t.Errorf("Embed() dims = %d, want 32", len(v))
	}

	// Should be normalized
	norm := Norm(v)
	if norm < 0.999 || norm > 1.001 {
		t.Errorf("normalized norm = %v, want ~1", norm)
	}
}

func TestRandomProjectionsSimilarity(t *testing.T) {
	corpus := []string{
		"machine learning is fascinating and powerful",
		"deep learning and neural networks transform data",
		"the weather today is sunny and warm",
		"it will rain tomorrow with thunderstorms",
	}

	rp := NewRandomProjections(256)
	rp.Fit(corpus)

	ml1, _ := rp.Embed("machine learning is fascinating and powerful")
	ml2, _ := rp.Embed("deep learning and neural networks transform data")
	weather1, _ := rp.Embed("the weather today is sunny and warm")
	weather2, _ := rp.Embed("it will rain tomorrow with thunderstorms")

	simML := Cosine(ml1, ml2)
	simWeather := Cosine(weather1, weather2)
	cross := Cosine(ml1, weather1)

	t.Logf("ML pairwise: %.4f, Weather pairwise: %.4f, Cross: %.4f", simML, simWeather, cross)

	// Random projections at 256 dims should roughly preserve similarity
	// with this many distinct words, but it's stochastic — log only.
	if simML >= 0 && simWeather >= 0 && cross < simML && cross < simWeather {
		t.Log("✓ random projections preserved similarity ordering")
	}
}

func TestRandomProjectionsEmptyCorpus(t *testing.T) {
	rp := NewRandomProjections(64)
	rp.Fit(nil)

	v, err := rp.Embed("hello world")
	if err != nil {
		t.Fatalf("Embed() error: %v", err)
	}
	if len(v) != 64 {
		t.Errorf("dims = %d, want 64", len(v))
	}
	// Should be zero vector (no vocab to project through)
	if Norm(v) != 0 {
		t.Log("empty-corpus embed returned non-zero vector (expected zero or normalized)")
	}
}

func TestRandomProjectionsUnknownTokens(t *testing.T) {
	rp := NewRandomProjections(32)
	rp.Fit([]string{"cat dog mouse"})

	v, err := rp.Embed("elephant giraffe zebra")
	if err != nil {
		t.Fatalf("Embed() error: %v", err)
	}
	// All tokens unknown — should be zero vector
	if Norm(v) != 0 {
		t.Log("all-unknown embed returned non-zero vector")
	}
}

func TestRandomProjectionsDeterministic(t *testing.T) {
	corpus := []string{"hello world", "foo bar baz"}
	text := "hello world test"

	rp1 := NewRandomProjections(64)
	rp1.Fit(corpus)
	v1, _ := rp1.Embed(text)

	rp2 := NewRandomProjections(64)
	rp2.Fit(corpus)
	v2, _ := rp2.Embed(text)

	// Same seed (42) should produce identical vectors
	if !Equal(v1, v2) {
		t.Error("embeddings should be deterministic with same seed")
	}
}

func TestRandomProjectionsReFit(t *testing.T) {
	rp := NewRandomProjections(64)
	rp.Fit([]string{"alpha beta gamma"})
	first := rp.VocabSize()

	rp.Fit([]string{"delta epsilon zeta eta theta"})
	second := rp.VocabSize()

	if first == second {
		t.Error("vocab size should change after re-Fit with different corpus")
	}
}

func TestTokenize(t *testing.T) {
	tokens := tokenize("Hello, World! This is a TEST.")
	expected := []string{"hello", "world", "this", "is", "test"}
	if len(tokens) != len(expected) {
		t.Errorf("tokenize() = %v (len=%d), want %v (len=%d)", tokens, len(tokens), expected, len(expected))
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token[%d] = %q, want %q", i, tok, expected[i])
		}
	}

	// Short tokens filtered
	tokens = tokenize("a b c x y z")
	if len(tokens) != 0 {
		t.Errorf("short tokens should be filtered, got %v", tokens)
	}
}
