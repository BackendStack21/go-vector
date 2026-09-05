package vector

import (
	"bytes"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestUnrolledKernels(t *testing.T) {
	a := make(Vector, 20)
	b := make(Vector, 20)
	for i := range a {
		a[i] = float32(i + 1)
		b[i] = float32(20 - i)
	}
	// Sequential reference
	var dot float32
	for i := range a {
		dot += a[i] * b[i]
	}
	if !approxEqual(Dot(a, b), dot, 1e-3) {
		t.Fatalf("Dot unroll %v vs %v", Dot(a, b), dot)
	}
	if Cosine(a, a) < 0.999 {
		t.Fatal("Cosine unroll self")
	}
	if Euclidean(a, a) > 1e-5 {
		t.Fatal("Euclidean unroll self")
	}
	if Manhattan(a, a) > 1e-5 {
		t.Fatal("Manhattan unroll self")
	}
	if Norm(a) <= 0 || dotSelf(a) <= 0 {
		t.Fatal("norm unroll")
	}
}

func TestLockedStoreSurface(t *testing.T) {
	ls := NewLockedStore(ManhattanDistance)
	ls.Add("a", Vector{1, 2})
	ls.SetMeta("a", map[string]string{"k": "v"})
	if !ls.Has("a") || ls.Len() != 1 || ls.Dims() != 2 || ls.Metric() != ManhattanDistance {
		t.Fatal("meta")
	}
	if ls.Meta("a")["k"] != "v" || ls.IDs()[0] != "a" {
		t.Fatal("getters")
	}
	if !ls.AddUnique("b", Vector{3, 4}) || ls.AddUnique("a", Vector{0, 0}) {
		t.Fatal("unique")
	}
	ls.Upsert("b", Vector{5, 6})
	if ls.Search(Vector{5, 6}, 1)[0].ID != "b" {
		t.Fatal("search")
	}
	if ls.SearchIDs(Vector{1, 2}, 1)[0].ID != "a" {
		t.Fatal("ids")
	}
	if len(ls.SearchOpts(Vector{1, 2}, 1, WithoutVectors())) != 1 {
		t.Fatal("opts")
	}
	if len(ls.SearchBatch([]Vector{{1, 2}}, 1)[0]) != 1 {
		t.Fatal("batch")
	}
	if len(ls.SearchRadius(Vector{1, 2}, 100)) < 1 {
		t.Fatal("radius")
	}
	var buf bytes.Buffer
	if _, err := ls.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	ls2 := NewLockedStore(CosineDistance)
	if _, err := ls2.ReadFrom(&buf); err != nil {
		t.Fatal(err)
	}
	var js bytes.Buffer
	if err := ls.WriteJSONTo(&js); err != nil {
		t.Fatal(err)
	}
	ls3 := NewLockedStore(CosineDistance)
	if err := ls3.ReadJSONFrom(&js); err != nil {
		t.Fatal(err)
	}
	_ = ls.Store()
	if !ls.Remove("b") {
		t.Fatal("remove")
	}
	ls.Clear()
	if ls.Len() != 0 {
		t.Fatal("clear")
	}
}

func TestStoreNilByIDFallback(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("z", Vector{1})
	s.byID = nil
	if !s.Has("z") || s.Get("z") == nil || !s.Remove("z") {
		t.Fatal("nil map fallback")
	}
}

func TestHTTPEmbedderMaxBytesAndUA(t *testing.T) {
	var ua string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ua = r.Header.Get("User-Agent")
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1,2]}]}`)
	}))
	defer srv.Close()
	e := NewHTTPEmbedder(srv.URL, "m", 2, WithUserAgent("ua-test"), WithMaxResponseBytes(1024), WithRetry(0))
	if _, err := e.Embed("x"); err != nil {
		t.Fatal(err)
	}
	if ua != "ua-test" {
		t.Fatalf("ua %q", ua)
	}
}

func TestRPMinTokenAndNGrams(t *testing.T) {
	rp := NewRandomProjectionsOpts(8, WithMinTokenLen(4), WithNGrams(2))
	rp.Fit([]string{"this token filter test"})
	if rp.VocabSize() == 0 {
		t.Fatal("empty vocab")
	}
	_, _ = rp.Embed("this token")
}

func TestMetricUnknownDistance(t *testing.T) {
	if Distance(Vector{1}, Vector{1}, Metric(99)) != 0 {
		t.Fatal("unknown metric")
	}
	if ChebyshevDistance.String() != "chebyshev" || HammingDistance.String() != "hamming" {
		t.Fatal("names")
	}
}

func TestSearchRadiusEmpty(t *testing.T) {
	s := NewStore(CosineDistance)
	if s.SearchRadius(Vector{1}, 1) != nil {
		t.Fatal("empty")
	}
	if s.SearchIDs(Vector{1}, 1) != nil {
		t.Fatal("empty ids")
	}
}

func TestHammingStore(t *testing.T) {
	s := NewStore(HammingDistance)
	s.Add("same", Vector{1, 0, 1})
	s.Add("diff", Vector{0, 1, 0})
	if s.Search(Vector{1, 0, 1}, 1)[0].ID != "same" {
		t.Fatal("hamming search")
	}
}

func TestDotSelfUnroll(t *testing.T) {
	v := make(Vector, 18)
	for i := range v {
		v[i] = 1
	}
	if !approxEqual(dotSelf(v), 18, 1e-4) {
		t.Fatalf("%v", dotSelf(v))
	}
}

func TestLockedStoreFilesAndRPSave(t *testing.T) {
	dir := t.TempDir()
	ls := NewLockedStore(CosineDistance)
	ls.Add("a", Vector{1, 0})
	p := dir + "/s.gob"
	if err := ls.Save(p); err != nil {
		t.Fatal(err)
	}
	if err := ls.SaveAtomic(dir + "/s2.gob"); err != nil {
		t.Fatal(err)
	}
	if err := ls.SaveJSON(dir + "/s.json"); err != nil {
		t.Fatal(err)
	}
	ls2 := NewLockedStore(EuclideanDistance)
	if err := ls2.Load(p); err != nil {
		t.Fatal(err)
	}
	if ls2.Metric() != CosineDistance {
		t.Fatal(ls2.Metric())
	}
	ls3 := NewLockedStore(EuclideanDistance)
	if err := ls3.LoadJSON(dir + "/s.json"); err != nil {
		t.Fatal(err)
	}
	rp := NewRandomProjections(8)
	rp.Fit([]string{"hello world"})
	if err := rp.SaveEmbedder(dir + "/e.gob"); err != nil {
		t.Fatal(err)
	}
}

func TestUpsertMixedLength(t *testing.T) {
	s := NewStore(CosineDistance)
	s.Add("a", Vector{1, 0, 0})
	s.Upsert("a", Vector{1, 0}) // forces unpack + replace
	if s.packedOK {
		t.Fatal("expected unpack")
	}
	if len(s.Get("a")) != 2 {
		t.Fatal(s.Get("a"))
	}
}

func TestMoreEdges(t *testing.T) {
	rp := NewRandomProjectionsOpts(8, WithNGrams(0), WithMinTokenLen(0), nil)
	rp.FitMore([]string{"hello world"})
	if rp.VocabSize() == 0 {
		t.Fatal("FitMore on empty")
	}
	s := NewStore(CosineDistance)
	if s.Dims() != 0 || s.Meta("no") != nil {
		t.Fatal("empty dims/meta")
	}
	s.Add("a", Vector{})
	if s.Dims() != 0 {
		t.Fatal("zero-dim")
	}
	s.Add("b", Vector{1, 2})
	s.SetMeta("b", map[string]string{"x": "y"})
	s.SetMeta("b", nil)
	if s.Meta("b") != nil {
		t.Fatal("cleared meta")
	}
	// Manhattan/Cosine/Euclidean tails (n=17)
	u := make(Vector, 17)
	w := make(Vector, 17)
	for i := range u {
		u[i], w[i] = float32(i), float32(-i)
	}
	_ = Manhattan(u, w)
	_ = Cosine(u, w)
	_ = Euclidean(u, w)
}

func TestHTTPEmbedderEndpointNoSlash(t *testing.T) {
	var path string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path = r.URL.Path
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1]}]}`)
	}))
	defer srv.Close()
	e := NewHTTPEmbedder(srv.URL, "m", 1, WithEndpointPath("embeddings"))
	if _, err := e.Embed("x"); err != nil {
		t.Fatal(err)
	}
	if !strings.HasSuffix(path, "/embeddings") {
		t.Fatalf("path %s", path)
	}
}
