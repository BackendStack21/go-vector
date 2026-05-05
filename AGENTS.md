# AGENTS.md — go-vector

Zero-dependency vector similarity library. Pure Go.

## Project Structure

```
pkg/vector/         ← library code (the thing people import)
  vector.go         Vector type, Dot, Norm, Normalize, Add, Sub, Scale, Equal, Clone
  similarity.go     Metric enum, Cosine, Euclidean, Manhattan, Distance
  store.go          Store: in-memory brute-force NN search
cmd/go-vector/      ← minimal CLI demo
```

## Conventions

- **Zero dependencies** — never add to go.mod. stdlib only.
- **Vector = []float32** — no struct, no interface, just a slice.
- **Mismatched lengths → zero** — Dot/Norm/Cosine/Euclidean/Manhattan all return 0 for mismatched-length inputs rather than panicking. Add/Sub return nil.
- **Clone on output** — Get() and Search() return copies. Store.Add() clones on insert.
- **Tests in `_test.go` files** — no separate test package. Use package `vector`.

## Build & Test (in Docker container)

```bash
# Build
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && go build ./...'

# Test
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && go test ./pkg/vector/ -v -count=1 -coverprofile=coverage.out && go tool cover -func=coverage.out'

# Vet
docker exec -i projects-dev bash -c 'export PATH=$PATH:/usr/local/go/bin && cd /workspace/go-vector && go vet ./...'
```

Target: >95% coverage. All code paths exercised.

## Adding a New Metric

1. Add constant to `Metric` enum in `similarity.go`
2. Add case to `Distance()` switch
3. Update `Ascending()` if it differs from default
4. Add test case in `TestMetricAscending` and `TestDistance`
5. Add store integration test

## Principles

- No panics. Return zero/nil for invalid input.
- No allocations on read paths (Search allocates distance array — unavoidable).
- SIMD/assembly would be nice but keep it pure Go for portability.
