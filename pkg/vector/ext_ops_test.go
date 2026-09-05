package vector

import "testing"

func TestInPlaceOps(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{4, 5, 6}
	dst := make(Vector, 3)
	got := AddIn(dst, a, b)
	if !Equal(got, Vector{5, 7, 9}) || &got[0] != &dst[0] {
		t.Fatalf("AddIn %v", got)
	}
	if AddIn(nil, Vector{1}, Vector{1, 2}) != nil {
		t.Fatal("AddIn mismatch")
	}
	got = SubIn(dst, b, a)
	if !Equal(got, Vector{3, 3, 3}) {
		t.Fatalf("SubIn %v", got)
	}
	got = ScaleIn(dst, a, 2)
	if !Equal(got, Vector{2, 4, 6}) {
		t.Fatalf("ScaleIn %v", got)
	}
	got = NormalizeIn(dst, Vector{3, 4})
	if !EqualEps(got, Vector{0.6, 0.8}, 1e-6) {
		t.Fatalf("NormalizeIn %v", got)
	}
	if NormalizeIn(dst, Vector{0, 0}) != nil {
		t.Fatal("NormalizeIn zero")
	}
}

func TestChebyshevHamming(t *testing.T) {
	a := Vector{1, 5, 3}
	b := Vector{1, 1, 7}
	if !approxEqual(Chebyshev(a, b), 4, 1e-6) {
		t.Fatalf("Chebyshev = %v", Chebyshev(a, b))
	}
	if Chebyshev(a, Vector{1}) != 0 {
		t.Fatal("Chebyshev mismatch")
	}
	if !approxEqual(Hamming(Vector{1, 2, 3}, Vector{1, 0, 3}), 1, 1e-6) {
		t.Fatal("Hamming")
	}
	if Hamming(a, Vector{1}) != 0 {
		t.Fatal("Hamming mismatch")
	}
	if Distance(a, b, ChebyshevDistance) != Chebyshev(a, b) {
		t.Fatal("Distance chebyshev")
	}
	if Distance(Vector{1, 1}, Vector{0, 1}, HammingDistance) != 1 {
		t.Fatal("Distance hamming")
	}
	if !ChebyshevDistance.Ascending() || !HammingDistance.Ascending() {
		t.Fatal("ascending")
	}
	s := NewStore(ChebyshevDistance)
	s.Add("close", Vector{1, 1})
	s.Add("far", Vector{9, 9})
	if s.Search(Vector{1, 1.5}, 1)[0].ID != "close" {
		t.Fatal("chebyshev search")
	}
}

func TestMetricString(t *testing.T) {
	if CosineDistance.String() != "cosine" || Metric(99).String() != "unknown" {
		t.Fatal(CosineDistance.String(), Metric(99).String())
	}
	if EuclideanDistance.String() != "euclidean" || DotProductSimilarity.String() != "dot" {
		t.Fatal("names")
	}
}

func TestHybrid(t *testing.T) {
	if !approxEqual(Hybrid(1, 0, 0.5), 0.5, 1e-6) {
		t.Fatal(Hybrid(1, 0, 0.5))
	}
	if !approxEqual(Hybrid(1, 0, 2), 1, 1e-6) || !approxEqual(Hybrid(1, 0, -1), 0, 1e-6) {
		t.Fatal("clamp")
	}
}

func TestMustEmbed(t *testing.T) {
	rp := NewRandomProjections(8)
	rp.Fit([]string{"hello world"})
	v := rp.MustEmbed("hello")
	if len(v) != 8 {
		t.Fatal(len(v))
	}
	if len(MustEmbed(rp, "hello")) != 8 {
		t.Fatal("package MustEmbed")
	}
	defer func() {
		if recover() == nil {
			t.Fatal("want panic")
		}
	}()
	MustEmbed(failEmbedder{}, "x")
}

type failEmbedder struct{}

func (failEmbedder) Embed(string) (Vector, error) { return nil, errFail }
func (failEmbedder) Dims() int                    { return 0 }

var errFail = errString("boom")

type errString string

func (e errString) Error() string { return string(e) }

var (
	_ Embedder      = (*RandomProjections)(nil)
	_ BatchEmbedder = (*RandomProjections)(nil)
	_ BatchEmbedder = (*HTTPEmbedder)(nil)
)
