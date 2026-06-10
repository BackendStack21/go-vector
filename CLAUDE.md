# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

`AGENTS.md` is the canonical agent guide and stays authoritative for project structure and step-by-step recipes (adding a metric, adding an embedder). This file captures the architecture invariants and the commands that actually work in this environment. Note that `AGENTS.md`'s "Build & Test" section assumes a `docker exec projects-dev` container — ignore that here and use the Makefile directly.

## Commands

```bash
make test                              # run tests (-count=1, no cache)
make test-verbose                      # verbose
make test-cover                        # coverage profile + per-func report (target >95%)
make vet                               # go vet ./...
make ci                                # fmt + vet + test + build (run before pushing)
make demo                              # go run ./cmd/go-vector/ demo

go test ./pkg/vector/ -run TestCosine          # single test by name
go test ./pkg/vector/ -bench=. -benchmem       # all benchmarks
go test ./pkg/vector/ -run X -bench BenchmarkDot -benchmem  # one bench, no tests
```

Core code lives in `pkg/vector/` (tests are package `vector`, white-box); `pkg/onnx/` is the optional local-neural-embeddings package (its model-dependent tests skip unless `make model` has downloaded all-MiniLM-L6-v2 into `pkg/onnx/testdata/`, and need `brew install onnxruntime`). Note: because `make test`/`make ci` cover `pkg/onnx`, repo development requires CGo (a C toolchain); consumers importing only `pkg/vector` still build with `CGO_ENABLED=0`. `cmd/go-vector/` is a demo binary; `docs/` is the GitHub Pages site (static HTML, not built).

## Architecture

The core library is one flat package built on one type: `Vector = []float32` (a raw slice alias, no struct wrapper). Everything composes from that. Source files split by concern:

- **`vector.go`** — element-wise ops (`Dot`, `Norm`, `Normalize`, `Add`, `Sub`, `Scale`, `Clone`, `Equal`/`EqualEps`, `Dims`).
- **`similarity.go`** — the `Metric` enum and distance functions. `Distance(a, b, metric)` dispatches; `Metric.Ascending()` reports whether lower-is-better (true for all distances, false only for `DotProductSimilarity`). The sort direction in `Store.Search` keys off `Ascending()`.
- **`store.go`** — `Store`, a brute-force in-memory index (parallel `ids []string` / `vectors []Vector` slices) plus gob and JSON persistence.
- **`embedder.go`** — the `Embedder` interface (`Embed(text) (Vector, error)`, `Dims() int`) — the seam for swapping in external embedding backends.
- **`http_embedder.go`** — `HTTPEmbedder`, an adapter for OpenAI-compatible embeddings APIs (OpenAI, Ollama, LM Studio, …) built on stdlib `net/http` only. Tests use `httptest` servers — no network.
- **`random_projections.go`** + **`rp_persistence.go`** — the built-in `RandomProjections` embedder and its gob save/load.

`pkg/onnx/` (separate package, separate deps) runs BERT-family ONNX models in-process: `tokenizer.go` is a pure-Go BERT WordPiece tokenizer, `embedder.go` wraps an ONNX Runtime session (mean pooling over `last_hidden_state`, or a pre-pooled `sentence_embedding` output, then L2 normalization). It satisfies `vector.Embedder`.

### Invariants that pervade the codebase — preserve these

These rules are why edits don't break callers; every function in the package already obeys them.

- **No panics on bad input.** Mismatched-length vectors, zero vectors, and `k <= 0` return zero / `nil` rather than panicking. New functions must follow suit.
- **Clone on every output boundary.** `Store.Add` clones on insert; `Get` and `Search` return clones. Internal backing arrays are never handed out — callers can never mutate store state through a returned slice.
- **Zero-allocation, single-pass distances.** `Dot`, `Cosine`, `Euclidean`, `Manhattan` accumulate in one loop with no allocation (verified by `-benchmem` showing `0 allocs`). Don't introduce intermediate slices in these hot paths.
- **`pkg/vector` imports stdlib only** (`math`, `sort`, `encoding/gob`, `encoding/json`, `os`, `strings`, `unicode`, `math/rand`, `net/http`). Third-party/CGo integrations are quarantined in sibling packages — currently `pkg/onnx` (`onnxruntime_go`, `golang.org/x/text`) — so the core stays importable with no CGo and no BLAS. Never add an import to `pkg/vector` beyond stdlib.
- **Concurrency: read-safe, not write-safe.** `Store` supports concurrent reads but concurrent read/write needs an external `sync.Mutex` — there is no internal locking by design.

### Persistence detail

`Store`'s fields are unexported, so gob/JSON encoding goes through an internal `storeData` bridge struct (exported fields: `Vectors`, `IDs`, `Metric`). `init()` calls `gob.Register(Vector{})` so the `[]float32` alias serializes. `Load` **replaces** all store data wholesale — atomicity (e.g. write-to-temp-then-rename) is the caller's responsibility. The same bridge pattern applies to `RandomProjections` persistence in `rp_persistence.go`.

### RandomProjections specifics

Sparse Johnson-Lindenstrauss projection (Achlioptas 2003): matrix entries are `{-1, 0, +1} × sqrt(3/D)` with probabilities `{1/6, 2/3, 1/6}`, seeded with a **fixed seed (42)** so embeddings are fully deterministic — same text always yields the same vector, and tests assert this. `Fit(corpus)` builds the vocabulary; the tokenizer splits on non-letter/digit, lowercases, and drops tokens under 2 chars. Output is always L2-normalized (suited to cosine search).

## Adding metrics or embedders

See `AGENTS.md` for the exact checklists. In short: a new metric needs a `Metric` constant, a `Distance()` switch case (and `Ascending()` if it's a similarity), a zero-alloc direct function, and a test + benchmark. A new embedder implements `Embedder`, adds a constructor, and ships Fit/Embed/determinism/similarity tests.
