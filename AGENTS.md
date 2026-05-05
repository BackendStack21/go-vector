# AGENTS.md — go-vector

Zero-dependency vector similarity library. Pure Go.

## Project Structure

```
pkg/vector/         ← library code (the thing people import)
  vector.go         Vector type, Dot, Norm, Normalize, Add, Sub, Scale, Equal, EqualEps, Clone, Dims
  similarity.go     Metric enum, Cosine, CosineDist, Euclidean, Manhattan, Distance
  store.go          Store: in-memory brute-force NN search (Add, Get, Remove, Search, Len)
cmd/go-vector/      ← minimal CLI demo
docs/               ← GitHub Pages landing page
  index.html        Dark-themed single-page site
  .nojekyll         GitHub Pages raw HTML flag
```

## Conventions

- **Zero dependencies** — never add to go.mod. stdlib only (`math`, `sort`).
- **Vector = []float32** — no struct, no interface, just a slice.
- **Mismatched lengths → zero** — Dot/Norm/Cosine/Euclidean/Manhattan all return 0 for mismatched-length inputs rather than panicking. Add/Sub return nil.
- **Clone on output** — Get() and Search() return copies. Store.Add() clones on insert. No internal backing arrays ever leak.
- **Zero-alloc distance** — Cosine and Euclidean compute in a single pass without intermediate allocations. All distance functions have 0 allocs in benchmarks.
- **Tests in `_test.go` files** — package `vector`, no separate test package.
- **Benchmarks** — `bench_test.go` covers all operations at 768/1536 dims. Run with `-benchmem`.

## Build & Test (in Docker container)

```bash
# Full CI
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && make ci'

# Tests with coverage
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && make test-cover'

# Benchmarks
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && go test ./pkg/vector/ -bench=. -benchmem'
```

Target: >95% coverage. All code paths exercised.

## Adding a New Metric

1. Add constant to `Metric` enum in `similarity.go`
2. Add case to `Distance()` switch
3. Update `Ascending()` if needed
4. Add direct function (e.g., `Chebyshev()`) — zero-alloc, single-pass
5. Add test case in `TestMetricAscending` and `TestDistance`
6. Add store integration test and benchmark

## Security Principles

- **No panics.** Return zero/nil for invalid input.
- **No unsafe.** Pure Go, no CGo, no syscalls, no I/O.
- **Clone hygiene.** Internal state is never exposed through return values.
- **Float32 awareness.** Document overflow limits. `MaxSafeDims` constant defines safe dimensionality.
- **Thread safety docs.** Store is safe for concurrent reads but not writes — documented, not enforced.

## Performance Rules

- Distance functions must be zero-allocation (verified by `-benchmem`)
- Single-pass computation where possible (Cosine, Euclidean already are)
- Brute-force search is O(n·d) — acceptable for the zero-dep target
- Benchmark before and after any metric or store change

## Landing Page

The `docs/` directory is deployed via GitHub Pages (Settings → Pages → Source: main / /docs). Uses the standard BackendStack21 dark theme: Outfit for body, Monaspace Neon for code. No build step — just raw HTML + CSS.
