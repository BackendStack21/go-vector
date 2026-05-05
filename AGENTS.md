# AGENTS.md — go-vector

Zero-dependency vector similarity library. Pure Go.

## Project Structure

```
pkg/vector/                  ← library code
  vector.go                  Vector type, Dot, Norm, Normalize, Add, Sub, Scale, Equal, EqualEps, Clone, Dims
  similarity.go              Metric enum, Cosine, CosineDist, Euclidean, Manhattan, Distance
  store.go                   Store: NN search + Gob/JSON persistence (Save/Load/SaveJSON/LoadJSON)
  embedder.go                Embedder interface
  random_projections.go      RandomProjections: sparse JL projection + tokenizer
cmd/go-vector/               ← minimal CLI demo
docs/                        ← GitHub Pages landing page
  index.html                 Dark-themed single-page site
  .nojekyll                  GitHub Pages raw HTML flag
```

## Conventions

- **Zero dependencies** — never add to go.mod. stdlib only: `math`, `sort`, `encoding/gob`, `encoding/json`, `os`, `strings`, `unicode`, `math/rand`.
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
