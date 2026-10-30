# Threat model — go-vector

## What this project does and where untrusted input enters
- go-vector is a Go vector-similarity library: `[]float32` vectors, distance metrics, an in-memory `Store`, embedders (random projections, OpenAI-compatible HTTP APIs, local ONNX models), HNSW approximate search and int8 quantization, with gob/JSON persistence.
- Untrusted: text passed to embedders and tokenizers; vectors and ids passed to the store (lengths, NaN/Inf, empty, mismatched dimensions); persisted files loaded with `Store.Load`/`LoadJSON`/`ReadFrom`/`ReadJSONFrom`, `hnsw.Index.Load`, and random-projections load (applications may load indexes from shared or downloaded locations); responses from remote embedding APIs (`HTTPEmbedder`); ONNX vocab files.
- Trusted: the application calling the library and its configuration (endpoint URLs, API keys, model paths).

## Components that matter most / least
- Highest priority: persistence decoding in `pkg/vector/store_persist.go`, `pkg/vector/rp_persistence.go`, `pkg/hnsw` (`Save`/`Load`) — malformed or hostile files; `pkg/vector/http_embedder.go` (response size caps, JSON parsing, retries, API key handling); `pkg/onnx/tokenizer.go` (WordPiece tokenizer on arbitrary text).
- Medium: `pkg/hnsw` graph search on adversarial data, `pkg/quantize`, `pkg/vector` math (`Store.Search`, distances).
- Out of scope: `cmd/onnx-demo`, `docs/`; `cmd/go-vector` is a demo CLI (lower priority).

## How to exercise it
- `make test` (runs `pkg/vector`, `pkg/onnx`, `pkg/hnsw`, `pkg/quantize`, `cmd/go-vector`). ONNX model-backed tests skip because the model is not downloaded; the tokenizer has a fuzz test (`pkg/onnx/tokenizer_fuzz_test.go`).
- `/usr/local/bin/go-vector` is a demo CLI built in the image.

## How you rate severity
- Critical: code execution from a loaded file or embedding API response.
- High: a crafted persisted file or API response causing a panic, unbounded allocation (memory exhaustion from declared lengths), or an infinite loop during load/search; leaking the API key (e.g. sent to a redirect target or logged).
- Medium: panics on bad caller input (the library promises not to panic on mismatched or zero vectors, `k <= 0`), returned slices aliasing internal store state, tokenizer CPU blowups on long inputs.
- Low: numeric precision issues, hardening suggestions.

## Anything to leave alone
- Concurrent writes to `Store` without external locking (documented as not write-safe).
- Search quality/accuracy of approximate indexes.
