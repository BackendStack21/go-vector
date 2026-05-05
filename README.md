# go-vector

Zero-dependency vector similarity library for Go. Pure Go `[]float32` vectors, four distance metrics, and a brute-force in-memory vector store with top-K nearest-neighbor search. No CGo, no BLAS, no third-party imports.

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
    // cat: 0.0000
    // dog: 0.0005
}
```

## API

### Vector Operations

| Function | Description |
|----------|-------------|
| `Dot(a, b Vector) float32` | Dot product |
| `Norm(v Vector) float32` | L2 norm |
| `Normalize(v Vector) Vector` | Unit vector (nil for zero vector) |
| `Add(a, b Vector) Vector` | Element-wise sum |
| `Sub(a, b Vector) Vector` | Element-wise difference |
| `Scale(v Vector, s float32) Vector` | Scalar multiplication |
| `Equal(a, b Vector) bool` | Approximate equality (ε = 1e-6) |
| `EqualEps(a, b Vector, eps float32) bool` | Custom epsilon equality |
| `Clone(v Vector) Vector` | Deep copy |

### Distance Metrics

```go
vector.CosineDistance      // 1 - cos(θ)  → [0, 2], lower = similar
vector.EuclideanDistance   // L2 distance  → [0, ∞), lower = similar
vector.ManhattanDistance   // L1 distance  → [0, ∞), lower = similar
vector.DotProductSimilarity // dot product → (-∞, ∞), higher = similar
```

Direct functions: `Cosine()`, `CosineDist()`, `Euclidean()`, `Manhattan()`, `Distance()`.

### Vector Store

```go
store := vector.NewStore(vector.CosineDistance)

store.Add(id, vec)              // insert a vector
store.Search(query, k)          // top-k nearest neighbors
store.Get(id)                   // lookup by id (clone)
store.Remove(id)                // remove by id
store.Len()                     // count
```

`Search()` returns `[]SearchResult` sorted by distance — ascending for distance metrics, descending for `DotProductSimilarity`. Results include cloned vectors so mutations don't corrupt the store.

## Design

- **Zero dependencies** — `go.mod` has no `require` block
- **Type alias** — `Vector` is `[]float32`, no struct wrapper
- **Brute-force search** — O(n·d) per query; fine for up to ~100K vectors
- **Clone safety** — `Get()` and `Search()` return copies, never internal slices
- **Mismatched lengths** — return zero/nil rather than panicking

## License

MIT
