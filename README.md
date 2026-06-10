# go-vector

Vector similarity library for Go. Pure Go `[]float32` vectors, four distance metrics, text embedding (random projections, OpenAI-compatible APIs, or local ONNX models), and disk-backed persistence. The core `pkg/vector` package is zero-dependency — no CGo, no BLAS, no third-party imports; the optional `pkg/onnx` package adds local neural embeddings via ONNX Runtime.

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

The `Embedder` interface lets you swap backends. The built-in `RandomProjections` is zero-dependency and deterministic.

### Real Embeddings (OpenAI, Ollama, and friends)

`HTTPEmbedder` connects to any service speaking the OpenAI-compatible embeddings protocol — OpenAI, Ollama, LM Studio, Voyage AI, llama.cpp server, vLLM — using only `net/http`, so the library stays dependency-free.

```go
// OpenAI
e := vector.NewHTTPEmbedder("https://api.openai.com/v1", "text-embedding-3-small", 1536,
    vector.WithAPIKey(os.Getenv("OPENAI_API_KEY")))

// Ollama (local, free) — pass 0 to infer dims from the first response
e := vector.NewHTTPEmbedder("http://localhost:11434/v1", "nomic-embed-text", 0)

// Index a corpus in one round-trip, then search semantically
docs := []string{"cats are great pets", "the stock market rallied", "dogs are loyal companions"}
vecs, err := e.EmbedBatch(docs)
if err != nil { /* handle network/API errors */ }

store := vector.NewStore(vector.CosineDistance)
for i, doc := range docs {
    store.Add(doc, vecs[i])
}

q, _ := e.Embed("animals that live with people")
results := store.Search(q, 2) // → the cat and dog docs
```

Options: `WithAPIKey` (Bearer auth), `WithHeader` (e.g. Azure's `api-key`), `WithHTTPClient` (custom timeout/proxy; default 30s), `WithNormalize` (L2-normalize responses — useful with `DotProductSimilarity` on backends that don't normalize, such as Ollama). Context-aware variants `EmbedContext` / `EmbedBatchContext` support cancellation and deadlines.

### Local Neural Embeddings (ONNX)

The `pkg/onnx` package runs transformer embedding models fully in-process via ONNX Runtime — no server, no API key, deterministic output. It lives in a separate package so the core `pkg/vector` stays pure Go: importing `pkg/onnx` is what pulls in the ONNX Runtime binding (CGo).

Setup: install the ONNX Runtime shared library (`brew install onnxruntime` on macOS, or download from the [onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)), then download a model — e.g. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2): `onnx/model.onnx` and `vocab.txt`.

```go
import "github.com/BackendStack21/go-vector/pkg/onnx"

e, err := onnx.New("model.onnx", "vocab.txt")
if err != nil { ... }
defer e.Close()

vecs, _ := e.EmbedBatch([]string{
    "cats are wonderful pets",
    "the federal reserve raised interest rates",
}) // 384-dim, L2-normalized, real semantics

store := vector.NewStore(vector.CosineDistance)
store.Add("doc0", vecs[0])
store.Add("doc1", vecs[1])

q, _ := e.Embed("animals that people keep at home")
store.Search(q, 1) // → doc0
```

Any BERT-style export works (inputs `input_ids`/`attention_mask`/`token_type_ids`; output `last_hidden_state` mean-pooled automatically, or a pre-pooled `sentence_embedding`). Tokenization is a pure-Go BERT WordPiece implementation — no Python, no Rust tokenizer. Options: `WithLibraryPath` (ONNX Runtime location; also honors `ONNXRUNTIME_SHARED_LIBRARY_PATH`), `WithMaxLength` (default 256), `WithCasedVocab`.

Try it end to end — downloads the model, embeds a corpus, and answers semantic queries (see `cmd/onnx-demo/`):

```bash
make model && make demo-onnx
```

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

**Built-in: `HTTPEmbedder`**

Adapter for any OpenAI-compatible embeddings API (OpenAI, Ollama, LM Studio, Voyage AI, vLLM). stdlib `net/http` only — no SDK dependency.

- `NewHTTPEmbedder(baseURL, model string, dims int, opts ...HTTPEmbedderOption)` — create embedder; `dims = 0` infers from the first response
- `Embed(text string) (Vector, error)` / `EmbedContext(ctx, text)` — embed one text
- `EmbedBatch(texts []string) ([]Vector, error)` / `EmbedBatchContext(ctx, texts)` — embed many texts in one API call
- `Dims() int` — declared or inferred dimensionality (0 until known)
- Options: `WithAPIKey(key)`, `WithHeader(k, v)`, `WithHTTPClient(c)`, `WithNormalize()`

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
