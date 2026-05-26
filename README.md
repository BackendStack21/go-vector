# go-vector

Zero-dependency vector similarity library for Go. Pure Go `[]float32` vectors, four distance metrics, text embedding via random projections, and disk-backed persistence. No CGo, no BLAS, no third-party imports.

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

## Text Embedding

```go
// Fit a random projection embedder on your corpus
rp := vector.NewRandomProjections(256)
rp.Fit([]string{
    "machine learning is fascinating",
    "deep neural networks transform AI",
    "the weather today is sunny",
})

// Embed text into a 256-dim vector
v, _ := rp.Embed("learning about machine intelligence")
// v is a normalized Vector suitable for cosine similarity search

// Use with the store
store := vector.NewStore(vector.CosineDistance)
store.Add("doc1", v)
store.Search(rp.MustEmbed("AI and learning"), 5)
```

The `Embedder` interface lets you swap backends: bring your own OpenAI, Ollama, or sentence-transformers adapter. The built-in `RandomProjections` is zero-dependency and deterministic.

## Persistence

### Store Persistence

```go
// Save to disk (gob — compact binary)
store.Save("/data/vectors.db")
store.SaveJSON("/data/vectors.json") // human-readable alternative

// Restore later
restored := vector.NewStore(vector.CosineDistance)
restored.Load("/data/vectors.db")

// Full roundtrip — metric and all data preserved
```

Gob-encoded stores are compact (~4 bytes per float32 + overhead). For a 10K × 1536d store, expect ~60 MB on disk and ~200ms save/load times.

### Embedder Persistence

```go
// Fit an embedder on your corpus
rp := vector.NewRandomProjections(256)
rp.Fit([]string{
    "machine learning is fascinating",
    "deep neural networks transform AI",
    "the weather today is sunny",
})

// Save the embedder state — vocab, projection matrix, dimensions
if err := rp.SaveEmbedder("/data/embedder.gob"); err != nil {
    log.Fatal(err)
}

// Later — load without refitting
restored, err := vector.LoadEmbedder("/data/embedder.gob")
if err != nil {
    log.Fatal(err)
}

// Embeddings are deterministic — same text, same vector
v, _ := restored.Embed("machine learning")
```

`SaveEmbedder` preserves the vocabulary, projection matrix, output dimension, and scaling factor. `LoadEmbedder` reconstructs a ready-to-use `*RandomProjections` that produces identical vectors to the original.

## API

### Vector Type

`Vector` is `[]float32` — no struct wrapper, fully compatible with any `[]float32` data.

### Core Operations

- `Dims(v Vector) int` — dimensionality
- `Dot(a, b Vector) float32` — dot product (0 if lengths differ)
- `Norm(v Vector) float32` — L2 norm
- `Normalize(v Vector) Vector` — unit vector (nil for zero vector)
- `Add(a, b Vector) Vector` — element-wise sum (nil if lengths differ)
- `Sub(a, b Vector) Vector` — element-wise difference (nil if lengths differ)
- `Scale(v Vector, s float32) Vector` — scalar multiplication
- `Equal(a, b Vector) bool` — approximate equality (ε = 1e-6)
- `EqualEps(a, b Vector, eps float32) bool` — custom epsilon
- `Clone(v Vector) Vector` — deep copy

### Distance Metrics

```go
vector.CosineDistance       // 1 − cos(θ)  → [0, 2],   lower = more similar
vector.EuclideanDistance    // L2 distance  → [0, ∞),  lower = more similar
vector.ManhattanDistance    // L1 distance  → [0, ∞),  lower = more similar
vector.DotProductSimilarity // dot product  → (−∞, ∞), higher = more similar
```

Direct functions: `Cosine`, `CosineDist`, `Euclidean`, `Manhattan`, `Distance`.

### Vector Store

```go
store := vector.NewStore(vector.CosineDistance)

store.Add(id, v)           // insert (clones input)
store.Search(query, k)     // top-k nearest neighbors
store.Get(id)              // lookup by id (clone)
store.Remove(id)           // remove by id
store.Len()                // count
store.Save(path)           // gob-encode to file
store.Load(path)           // restore from gob file
store.SaveJSON(path)       // JSON export
store.LoadJSON(path)       // JSON import
```

### Text Embedding

```go
type Embedder interface {
    Embed(text string) (Vector, error)
    Dims() int
}
```

**Built-in: `RandomProjections`**

Johnson-Lindenstrauss sparse random projection (Achlioptas 2003). Projects tokenized text into a fixed-size normalized vector. Deterministic (fixed seed), zero dependencies, ~10µs per embed.

- `NewRandomProjections(outputDim int)` — create embedder
- `Fit(corpus []string)` — build vocabulary and projection matrix
- `Embed(text string) (Vector, error)` — embed text (L2-normalized output)
- `VocabSize() int` — number of unique tokens in vocabulary
- `Dims() int` — output dimensionality
- `SaveEmbedder(path string) error` — persist embedder state to gob file
- `LoadEmbedder(path string) (*RandomProjections, error)` — restore embedder from gob file

## Performance

All benchmarks at 1536 dimensions on AMD EPYC.

```
BenchmarkDot-6                  7.5 µs     0 allocs
BenchmarkCosine-6               9.4 µs     0 allocs
BenchmarkEuclidean-6            8.7 µs     0 allocs
BenchmarkManhattan-6            9.4 µs     0 allocs
BenchmarkStoreSearch100-6       3.2 ms    63 KB
BenchmarkStoreSearch1000-6     28.5 ms    74 KB
BenchmarkStoreSearch10000-6   315  ms   185 KB
```

Distance functions are **zero-allocation**. Cosine and Euclidean compute in a single pass.

## Security

| Property | Status |
|----------|--------|
| Attack surface | 🟢 Minimal — pure float32 math, no CGo, no syscalls, no I/O |
| Panics | 🟢 None — all edge cases return zero/nil |
| Memory safety | 🟢 All outputs cloned, no shared backing arrays |
| Persistence safety | 🟡 `Load` replaces all data; atomicity is caller's responsibility |
| Float overflow | 🟡 Documented — `MaxSafeDims = 1M`; normalize large-magnitude vectors |
| Thread safety | 🟡 Store is read-safe but not write-safe — guard with `sync.Mutex` |

## Design

- **Zero dependencies** — `go.mod` has no `require` block
- **Type alias** — `Vector` is `[]float32`, interoperable with any `[]float32` data
- **Brute-force search** — O(n·d) per query; pair with an approximate index for n > 100K
- **Clone safety** — `Get()`, `Search()`, and `Add()` all clone
- **Graceful degradation** — mismatched lengths return zero/nil, never panic
- **Deterministic embeddings** — fixed seed (42) for reproducible results

## License

MIT
