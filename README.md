# go-vector

Zero-dependency vector similarity library for Go. Pure Go `[]float32` vectors, four distance metrics, brute-force nearest-neighbor search. No CGo, no BLAS, no third-party imports — just `math` and `sort`.

## Install

```bash
go get github.com/BackendStack21/go-vector
```

## Quick Start

```go
package main

import (
    "fmt"
    "github.com/BackendStack21/go-vector/pkg/vector"
)

func main() {
    store := vector.NewStore(vector.CosineDistance)

    store.Add("cat", vector.Vector{1.0, 0.8, 0.1})
    store.Add("dog", vector.Vector{0.9, 0.7, 0.1})
    store.Add("car", vector.Vector{0.1, 0.0, 0.9})

    query := vector.Vector{1.0, 0.9, 0.1}
    results := store.Search(query, 2)

    for _, r := range results {
        fmt.Printf("%s: %.4f\n", r.ID, r.Distance)
    }
    // cat: 0.0005
    // dog: 0.0026
}
```

## API

### Vector Type

`Vector` is `[]float32` — no struct wrapper, fully compatible with any `[]float32` data.

### Core Operations

| Function | Returns | Description |
|----------|---------|-------------|
| `Dims(v Vector) int` | `int` | Dimensionality |
| `Dot(a, b Vector) float32` | `float32` | Dot product (0 if lengths differ) |
| `Norm(v Vector) float32` | `float32` | L2 / Euclidean norm |
| `Normalize(v Vector) Vector` | `Vector` | Unit vector (nil for zero vector) |
| `Add(a, b Vector) Vector` | `Vector` | Element-wise sum (nil if lengths differ) |
| `Sub(a, b Vector) Vector` | `Vector` | Element-wise difference (nil if lengths differ) |
| `Scale(v Vector, s float32) Vector` | `Vector` | Scalar multiplication |
| `Equal(a, b Vector) bool` | `bool` | Approximate equality (ε = 1e-6) |
| `EqualEps(a, b Vector, eps float32) bool` | `bool` | Custom epsilon equality |
| `Clone(v Vector) Vector` | `Vector` | Deep copy |

### Distance Metrics

```go
vector.CosineDistance       // 1 − cos(θ)  → [0, 2],   lower = more similar
vector.EuclideanDistance    // L2 distance  → [0, ∞),  lower = more similar
vector.ManhattanDistance    // L1 distance  → [0, ∞),  lower = more similar
vector.DotProductSimilarity // dot product  → (−∞, ∞), higher = more similar
```

Direct functions available:
- `Cosine(a, b)` — cosine similarity [−1, 1]
- `CosineDist(a, b)` — 1 − cosine [0, 2]
- `Euclidean(a, b)` — L2 distance (zero-alloc)
- `Manhattan(a, b)` — L1 distance
- `Distance(a, b, metric)` — metric-dispatch version

### Vector Store

```go
store := vector.NewStore(vector.CosineDistance)

store.Add(id string, v Vector)           // insert (clones input)
store.Search(query Vector, k int)        // top-k nearest neighbors
store.Get(id string) Vector              // lookup by id (clone)
store.Remove(id string) bool             // remove by id
store.Len() int                          // count
```

`Search()` returns `[]SearchResult` sorted by distance:
- Distance metrics (Cosine, Euclidean, Manhattan): ascending order
- DotProductSimilarity: descending order

Results include **cloned** vectors — mutations won't corrupt store state.

## Performance

All benchmarks at 1536 dimensions (typical embedding size) on AMD EPYC.

```
BenchmarkDot-6                  7.5 µs     0 allocs
BenchmarkCosine-6               9.4 µs     0 allocs
BenchmarkEuclidean-6            8.7 µs     0 allocs
BenchmarkManhattan-6            9.4 µs     0 allocs
BenchmarkStoreSearch100-6       3.2 ms    63 KB
BenchmarkStoreSearch1000-6     28.5 ms    74 KB
BenchmarkStoreSearch10000-6   315  ms   185 KB
```

Distance functions are **zero-allocation**. Cosine and Euclidean compute in a single pass over the data.

Search is brute-force O(n·d). At 1536d, expect ~3ms per 100 vectors. Suitable for datasets up to ~100K vectors.

## Security

| Property | Status |
|----------|--------|
| Attack surface | 🟢 Minimal — pure float32 math, no CGo, no syscalls, no I/O |
| Panics | 🟢 None — all edge cases return zero/nil |
| Memory safety | 🟢 All outputs cloned, no shared backing arrays |
| Float overflow | 🟡 Documented — define `MaxSafeDims = 1M`; normalize large-magnitude vectors |
| Thread safety | 🟡 Store is read-safe but not write-safe — guard with `sync.Mutex` |

`SearchResult.Vector` is a `[]float32` — it's a clone from the store, but if your application mutates float32 slices returned by Search, clone them again. The store's internal state is never exposed.

### Float32 Precision

Dot products on high-dimensional vectors with large magnitudes (>1e19) can overflow float32 (±3.4e38). For typical embedding use (normalized vectors, dims < 10K), this is not a concern. If your vectors have large unnormalized magnitudes, normalize before insertion.

## Design

- **Zero dependencies** — `go.mod` has no `require` block. `math` + `sort` only.
- **Type alias** — `Vector` is `[]float32`, interoperable with any `[]float32` data
- **Brute-force search** — O(n·d) per query; pair with an approximate index for n > 100K
- **Clone safety** — `Get()`, `Search()`, and `Add()` all clone — no accidental mutation
- **Graceful degradation** — mismatched lengths return zero/nil, never panic
- **Single-pass** — Cosine and Euclidean compute in one pass without intermediate allocations

## License

MIT
