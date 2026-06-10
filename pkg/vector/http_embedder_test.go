package vector

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var _ Embedder = (*HTTPEmbedder)(nil)

// newEmbedServer returns a test server that responds with the given vectors,
// echoing back one vector per input in request order.
func newEmbedServer(t *testing.T, vectors map[string]Vector) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("decode request: %v", err)
		}
		type item struct {
			Index     int    `json:"index"`
			Embedding Vector `json:"embedding"`
		}
		resp := struct {
			Data []item `json:"data"`
		}{}
		for i, text := range req.Input {
			v, ok := vectors[text]
			if !ok {
				t.Errorf("unexpected input %q", text)
			}
			resp.Data = append(resp.Data, item{Index: i, Embedding: v})
		}
		json.NewEncoder(w).Encode(resp)
	}))
}

func TestHTTPEmbedderEmbed(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{"hello": {1, 2, 3}})
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "test-model", 3)
	v, err := e.Embed("hello")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if !Equal(v, Vector{1, 2, 3}) {
		t.Errorf("got %v, want [1 2 3]", v)
	}
	if e.Dims() != 3 {
		t.Errorf("Dims() = %d, want 3", e.Dims())
	}
}

func TestHTTPEmbedderRequestShape(t *testing.T) {
	var gotAuth, gotCustom, gotContentType, gotModel string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotCustom = r.Header.Get("X-Org")
		gotContentType = r.Header.Get("Content-Type")
		var req struct {
			Model string   `json:"model"`
			Input []string `json:"input"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		gotModel = req.Model
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1]}]}`)
	}))
	defer srv.Close()

	// Trailing slash on baseURL must not produce a double slash.
	e := NewHTTPEmbedder(srv.URL+"/", "text-embedding-3-small", 1,
		WithAPIKey("sk-test"), WithHeader("X-Org", "acme"))
	if _, err := e.Embed("x"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if gotAuth != "Bearer sk-test" {
		t.Errorf("Authorization = %q", gotAuth)
	}
	if gotCustom != "acme" {
		t.Errorf("X-Org = %q", gotCustom)
	}
	if gotContentType != "application/json" {
		t.Errorf("Content-Type = %q", gotContentType)
	}
	if gotModel != "text-embedding-3-small" {
		t.Errorf("model = %q", gotModel)
	}
}

func TestHTTPEmbedderBatchOrdering(t *testing.T) {
	// Server returns items in reverse order; the index field must win.
	// Also asserts the whole batch costs exactly one HTTP request.
	calls := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls++
		fmt.Fprint(w, `{"data":[{"index":1,"embedding":[2,2]},{"index":0,"embedding":[1,1]}]}`)
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	vecs, err := e.EmbedBatch([]string{"first", "second"})
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	if !Equal(vecs[0], Vector{1, 1}) || !Equal(vecs[1], Vector{2, 2}) {
		t.Errorf("got %v, want [[1 1] [2 2]]", vecs)
	}
	if calls != 1 {
		t.Errorf("EmbedBatch made %d requests, want 1", calls)
	}
}

func TestHTTPEmbedderInvalidIndices(t *testing.T) {
	// Indices that are not the permutation 0..n-1 (duplicates, negatives,
	// out of range) must error, never silently mis-assign vectors.
	for _, data := range []string{
		`{"data":[{"index":7,"embedding":[9,9]},{"index":-3,"embedding":[1,1]}]}`,
		`{"data":[{"index":0,"embedding":[1,1]},{"index":0,"embedding":[2,2]}]}`,
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, data)
		}))
		e := NewHTTPEmbedder(srv.URL, "m", 2)
		if _, err := e.EmbedBatch([]string{"a", "b"}); err == nil ||
			!strings.Contains(err.Error(), "permutation") {
			t.Errorf("response %s: want permutation error, got %v", data, err)
		}
		srv.Close()
	}
}

func TestHTTPEmbedderEmptyEmbedding(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[]}]}`)
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 0)
	if _, err := e.Embed("x"); err == nil || !strings.Contains(err.Error(), "empty embedding") {
		t.Errorf("want empty-embedding error, got %v", err)
	}
	if e.Dims() != 0 {
		t.Errorf("empty embedding must not lock dims inference, Dims() = %d", e.Dims())
	}
}

func TestHTTPEmbedderFailedBatchDoesNotLockDims(t *testing.T) {
	// First response is internally inconsistent and rejected; it must not
	// poison dims inference for the following valid response.
	bad := true
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if bad {
			fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1,2,3]},{"index":1,"embedding":[1,2]}]}`)
			return
		}
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1,2,3,4]},{"index":1,"embedding":[5,6,7,8]}]}`)
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 0)
	if _, err := e.EmbedBatch([]string{"a", "b"}); err == nil ||
		!strings.Contains(err.Error(), "inconsistent dims") {
		t.Fatalf("want inconsistent-dims error, got %v", err)
	}
	if e.Dims() != 0 {
		t.Fatalf("rejected batch locked dims to %d", e.Dims())
	}
	bad = false
	if _, err := e.EmbedBatch([]string{"a", "b"}); err != nil {
		t.Fatalf("valid batch after failure: %v", err)
	}
	if e.Dims() != 4 {
		t.Errorf("Dims() = %d, want 4", e.Dims())
	}
}

func TestHTTPEmbedderEmptyBatch(t *testing.T) {
	e := NewHTTPEmbedder("http://unused.invalid", "m", 2)
	vecs, err := e.EmbedBatch(nil)
	if vecs != nil || err != nil {
		t.Errorf("EmbedBatch(nil) = %v, %v; want nil, nil", vecs, err)
	}
}

func TestHTTPEmbedderAPIError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		fmt.Fprint(w, `{"error":{"message":"invalid api key"}}`)
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	_, err := e.Embed("x")
	if err == nil || !strings.Contains(err.Error(), "invalid api key") {
		t.Errorf("want API error message, got %v", err)
	}
}

func TestHTTPEmbedderHTTPError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadGateway)
		fmt.Fprint(w, "upstream down")
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	_, err := e.Embed("x")
	if err == nil || !strings.Contains(err.Error(), "502") {
		t.Errorf("want status error, got %v", err)
	}
}

func TestHTTPEmbedderMalformedJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, "not json")
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	if _, err := e.Embed("x"); err == nil {
		t.Error("want decode error, got nil")
	}
}

func TestHTTPEmbedderNetworkError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	srv.Close() // closed before use → connection refused

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	if _, err := e.Embed("x"); err == nil {
		t.Error("want network error, got nil")
	}
}

func TestHTTPEmbedderDimsMismatch(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{"x": {1, 2, 3}})
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 5)
	_, err := e.Embed("x")
	if err == nil || !strings.Contains(err.Error(), "3 dims, expected 5") {
		t.Errorf("want dims mismatch error, got %v", err)
	}
}

func TestHTTPEmbedderDimsInference(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{"x": {1, 2, 3, 4}})
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 0)
	if e.Dims() != 0 {
		t.Errorf("Dims() before first call = %d, want 0", e.Dims())
	}
	if _, err := e.Embed("x"); err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if e.Dims() != 4 {
		t.Errorf("Dims() after inference = %d, want 4", e.Dims())
	}
}

func TestHTTPEmbedderCountMismatch(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, `{"data":[{"index":0,"embedding":[1,1]}]}`)
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	_, err := e.EmbedBatch([]string{"a", "b"})
	if err == nil || !strings.Contains(err.Error(), "1 vectors for 2 inputs") {
		t.Errorf("want count mismatch error, got %v", err)
	}
}

func TestHTTPEmbedderNormalize(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{"x": {3, 4}})
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2, WithNormalize())
	v, err := e.Embed("x")
	if err != nil {
		t.Fatalf("Embed: %v", err)
	}
	if math.Abs(float64(Norm(v))-1) > 1e-6 {
		t.Errorf("norm = %v, want 1", Norm(v))
	}
	if !EqualEps(v, Vector{0.6, 0.8}, 1e-6) {
		t.Errorf("got %v, want [0.6 0.8]", v)
	}
}

func TestHTTPEmbedderOptions(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{"x": {1, 2}})
	defer srv.Close()

	custom := &http.Client{}
	e := NewHTTPEmbedder(srv.URL, "m", -7, WithHTTPClient(custom), WithHTTPClient(nil))
	if e.client != custom {
		t.Error("WithHTTPClient(nil) must not clear a previously set client")
	}
	if e.Dims() != 0 {
		t.Errorf("negative dims should clamp to 0, got %d", e.Dims())
	}
	if _, err := e.Embed("x"); err != nil {
		t.Fatalf("Embed with custom client: %v", err)
	}
}

func TestHTTPEmbedderBodyReadError(t *testing.T) {
	// Declare a longer body than is written: the client's read fails
	// with unexpected EOF mid-body.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Length", "1000")
		w.Write([]byte(`{"data"`))
	}))
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 2)
	if _, err := e.Embed("x"); err == nil {
		t.Error("want body read error, got nil")
	}
}

func TestHTTPEmbedderBadURL(t *testing.T) {
	e := NewHTTPEmbedder("http://bad url with spaces", "m", 2)
	if _, err := e.Embed("x"); err == nil {
		t.Error("want request-build error for invalid URL, got nil")
	}
}

func TestHTTPEmbedderContextCancel(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{"x": {1, 2}})
	defer srv.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	e := NewHTTPEmbedder(srv.URL, "m", 2)
	if _, err := e.EmbedContext(ctx, "x"); err == nil {
		t.Error("want context error, got nil")
	}
}

func TestHTTPEmbedderStoreIntegration(t *testing.T) {
	srv := newEmbedServer(t, map[string]Vector{
		"cats are great":  {1, 0.1, 0},
		"dogs are loyal":  {0.9, 0.2, 0},
		"stocks went up":  {0, 0.1, 1},
		"pets are lovely": {0.95, 0.15, 0},
	})
	defer srv.Close()

	e := NewHTTPEmbedder(srv.URL, "m", 3)
	store := NewStore(CosineDistance)

	docs := []string{"cats are great", "dogs are loyal", "stocks went up"}
	vecs, err := e.EmbedBatch(docs)
	if err != nil {
		t.Fatalf("EmbedBatch: %v", err)
	}
	for i, doc := range docs {
		store.Add(doc, vecs[i])
	}

	q, err := e.Embed("pets are lovely")
	if err != nil {
		t.Fatalf("Embed query: %v", err)
	}
	results := store.Search(q, 2)
	if len(results) != 2 {
		t.Fatalf("got %d results, want 2", len(results))
	}
	if results[0].ID == "stocks went up" || results[1].ID == "stocks went up" {
		t.Errorf("unrelated doc ranked in top 2: %v", results)
	}
}
