package vector

import (
	"bytes"
	"testing"
)

func TestRandomProjectionsBatchAndMust(t *testing.T) {
	rp := NewRandomProjections(16)
	rp.Fit([]string{"alpha beta", "gamma delta"})
	vecs, err := rp.EmbedBatch([]string{"alpha", "zzz unknown"})
	if err != nil || len(vecs) != 2 {
		t.Fatalf("%v %d", err, len(vecs))
	}
	if _, err := rp.EmbedBatch(nil); err != nil {
		t.Fatal(err)
	}
}

func TestRandomProjectionsFitMorePreservesOldRows(t *testing.T) {
	rp := NewRandomProjections(32)
	rp.Fit([]string{"hello world"})
	before, _ := rp.Embed("hello world")
	rp.FitMore([]string{"newtoken extra"})
	after, _ := rp.Embed("hello world")
	if !Equal(before, after) {
		t.Fatal("FitMore changed existing embedding")
	}
	if rp.VocabSize() <= 2 {
		t.Fatal("FitMore should add tokens")
	}
}

func TestRandomProjectionsHashingSeesOOV(t *testing.T) {
	plain := NewRandomProjections(32)
	plain.Fit([]string{"cat dog"})
	z, _ := plain.Embed("elephant")
	if Norm(z) != 0 {
		t.Fatal("closed vocab should ignore OOV")
	}
	hash := NewRandomProjectionsOpts(32, WithHashingTrick(8))
	hash.Fit([]string{"cat dog"})
	v, _ := hash.Embed("elephant")
	if Norm(v) == 0 {
		t.Fatal("hashing trick should embed OOV")
	}
}

func TestRandomProjectionsCustomSeed(t *testing.T) {
	a := NewRandomProjectionsOpts(16, WithSeed(1))
	b := NewRandomProjectionsOpts(16, WithSeed(2))
	a.Fit([]string{"hello world"})
	b.Fit([]string{"hello world"})
	va, _ := a.Embed("hello")
	vb, _ := b.Embed("hello")
	if Equal(va, vb) {
		t.Fatal("different seeds should differ")
	}
}

func TestRandomProjectionsWriteRead(t *testing.T) {
	rp := NewRandomProjectionsOpts(16, WithNGrams(2), WithTFIDF())
	rp.Fit([]string{"the cat sat", "the dog sat"})
	var buf bytes.Buffer
	if _, err := rp.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	got, err := ReadEmbedder(&buf)
	if err != nil {
		t.Fatal(err)
	}
	w1, _ := rp.Embed("the cat")
	w2, _ := got.Embed("the cat")
	if !Equal(w1, w2) {
		t.Fatal("roundtrip embed")
	}
}
