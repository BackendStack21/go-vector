# AGENTS.md — go-vector

Vector similarity library. Core package is zero-dependency pure Go; optional `pkg/onnx` runs local neural models.

## Project Structure

```
pkg/vector/                  ← core library (zero-dependency, pure Go)
  vector.go                  Vector type, Dot, Norm, Normalize, Add, Sub, Scale, Equal, EqualEps, Clone, Dims
  similarity.go              Metric enum, Cosine, CosineDist, Euclidean, Manhattan, Distance
  store.go                   Store: NN search + Gob/JSON persistence (Save/Load/SaveJSON/LoadJSON)
  embedder.go                Embedder interface
  http_embedder.go           HTTPEmbedder: OpenAI-compatible embeddings API adapter (stdlib net/http)
  random_projections.go      RandomProjections: sparse JL projection + tokenizer
pkg/onnx/                    ← local neural embeddings (depends on onnxruntime_go + x/text, CGo)
  embedder.go                Embedder: ONNX session, mean pooling, L2 normalization
  tokenizer.go               Pure-Go BERT WordPiece tokenizer (vocab.txt)
  testdata/                  Model files for tests (gitignored; fetch with `make model`)
cmd/go-vector/               ← minimal CLI demo
docs/                        ← GitHub Pages landing page
  index.html                 Dark-themed single-page site
  .nojekyll                  GitHub Pages raw HTML flag
```

## Conventions

- **`pkg/vector` stays zero-dependency** — it must never import anything beyond stdlib: `math`, `sort`, `encoding/gob`, `encoding/json`, `os`, `strings`, `unicode`, `math/rand`, `net/http` (HTTPEmbedder). Heavyweight integrations (CGo, third-party) live in sibling packages like `pkg/onnx` so users who don't import them pay nothing.
- **`pkg/onnx` carries the only third-party deps** — `github.com/yalue/onnxruntime_go` (CGo binding) and `golang.org/x/text` (NFD for accent stripping). It also needs the ONNX Runtime shared library at runtime (`brew install onnxruntime`).
- **Vector = []float32** — no struct, no interface, just a slice.
- **Mismatched lengths → zero** — return zero/nil rather than panicking.
- **Clone on output** — Get() and Search() return copies. Store.Add() clones on insert.
- **Zero-alloc distance** — Cosine and Euclidean compute in a single pass.
- **Gob persistence** — `storeData` internal struct bridges unexported fields to encoder.
- **Deterministic embeddings** — RandomProjections uses fixed seed 42.
- **Tests in `_test.go` files** — package `vector`.

## Random Projections

Sparse random projection (Achlioptas 2003):
- Entries: {-1, 0, +1} × sqrt(3/D) with probabilities {1/6, 2/3, 1/6}
- Vocabulary built from corpus via `Fit()`
- Tokenizer: split on non-letter/digit, lowercase, min 2 chars
- Output always L2-normalized

## Persistence

```go
// storeData bridges unexported Store fields to encoder
type storeData struct {
    Vectors []Vector
    IDs     []string
    Metric  Metric
}
```

`init()` registers `gob.Register(Vector{})` so `[]float32` serializes correctly.

## Build & Test

```bash
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && make ci'
# Benchmark
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && go test ./pkg/vector/ -bench=. -benchmem'
```

Target: >95% coverage. 40 tests, 96.8% currently.

## Adding a New Embedder

1. Implement `Embedder` interface (`Embed(text) (Vector, error)`, `Dims() int`)
2. Add constructor function
3. Add test file with Fit/Embed/determinism/similarity tests
4. Document in README under Text Embedding section

## Adding a New Metric

1. Add constant to `Metric` enum in `similarity.go`
2. Add case to `Distance()` switch, update `Ascending()` if needed
3. Add direct function — zero-alloc, single-pass
4. Add test + benchmark

## Security

- No panics. Return zero/nil for invalid input.
- No unsafe. Pure Go, no CGo, no syscalls.
- Clone hygiene. Internal state never exposed.
- Persistence: `Load` replaces all data. Caller handles atomicity.
