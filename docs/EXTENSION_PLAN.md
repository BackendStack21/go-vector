# Extension Plan — go-vector

A backward-compatible roadmap for performance and feature work. This document is the contract for how the library grows without breaking existing callers.

Current baseline: **v1.3.x** (`pkg/vector` stdlib-only, optional `pkg/onnx`).

**Status:** Phases 0–3 are implemented on this branch. Existing exported signatures and v1.3 gob/JSON fixtures remain the compatibility suite. `Store.Search` now uses a correct worst-at-root top-k heap (the v1.3 heap only looked right when `k >= n`).

---

## 1. Compatibility contract

Every change in this plan must satisfy all of the following. If a proposal cannot, it is out of scope until a v2.

### API surface

- Do not change the signature, return type, or documented semantics of any existing exported function, method, type, or constant.
- New behavior is added as **new functions, methods, option types, or sibling packages**. Existing names keep their current meaning.
- `Embedder` cannot gain methods (that would break existing implementations). Batch support is a **new interface** that embeds `Embedder`.
- `Metric` is an `iota` enum. New metrics are **appended after** `DotProductSimilarity`. Never insert, reorder, or renumber.
- `Vector` stays `[]float32`. No struct wrapper, no interface.

### Behavioral invariants (do not weaken)

These are the rules existing tests and callers already rely on:

| Invariant | Current meaning |
|---|---|
| No panics on bad input | Mismatched lengths, zero vectors, `k <= 0` → `0` / `nil` |
| Clone on output | `Add` clones on insert; `Get` and `Search` return clones |
| Zero-alloc distances | `Dot`, `Cosine`, `Euclidean`, `Manhattan` allocate nothing |
| Duplicate IDs | `Add` appends even when `id` already exists; `Get`/`Remove` act on the **first** match |
| Search sort | Distances ascending; `DotProductSimilarity` descending |
| Cosine edge cases | Zero or mismatched vector → `Cosine` `0`, `CosineDist` `1` |
| RP determinism | Seed `42`; same corpus + same `outputDim` → identical embeddings |
| RP tokenizer | Split on non-letter/digit, lowercase, drop tokens shorter than 2 runes |
| Persistence replace | `Load` / `LoadJSON` replace all store data, including metric |
| Concurrency | `Store` is read-safe, not write-safe; no internal lock |
| Core imports | `pkg/vector` imports stdlib only |

### On-disk format

- Gob and JSON `storeData` / `rpPersistData` may gain **optional fields** with zero-value defaults.
- Old files must load on new code. New files should load on old code when the extra fields are unused (gob ignores unknown types only if the decoder does not require them; keep new fields append-only and optional).
- Check in golden fixtures from today's format (`testdata/compat/store_v1.gob`, `embedder_v1.gob`) and load them in CI forever.

### What "compatible" does **not** require

- Bit-identical timings or allocation counts on internal paths.
- Identical undocumented field layout of unexported structs.
- `Search` remaining the only way to query (new methods may return IDs without cloned vectors).
- The README snippet `rp.MustEmbed(...)` — that method is documented but **does not exist**. Adding it is a compatibility *fix*, not a break.

---

## 2. Current-state audit

### What is already strong

- Clean type model: `Vector = []float32`, four metrics, `Store`, `Embedder`.
- Distance functions are single-pass and zero-alloc.
- `Store.Search` already uses a bounded top-k heap (`O(n log k)` selection, `O(k)` scratch) and reuses the query self-dot for cosine (PR #2).
- Three embedders covering the real spectrum: lexical RP, remote HTTP, local ONNX.
- Persistence via gob/JSON for both store and RP state.
- High test coverage, no-panic policy, clone hygiene.

### Structural limits (evidence from the code)

1. **Search is memory-bandwidth bound.** Vectors live in separately heap-allocated slices. PR #2 measured ~10–15% from the heap rewrite and called out a contiguous backing array as the largest remaining win.
2. **Cosine still recomputes `‖v‖²` for every stored vector.** Query norm is cached; stored norms are not.
3. **`Get` / `Remove` are `O(n)` linear scans.** No ID index.
4. **`Search` always clones `k` vectors.** Callers who only need IDs + scores pay `k · d · 4` bytes and a copy.
5. **`Add` allows silent duplicates.** Fine as a contract, but there is no `Upsert` / unique-ID mode.
6. **No store introspection.** `metric` is unexported; there is no `Has`, `IDs`, `Clear`, or `Dims`.
7. **`Embedder` has no batch method.** `HTTPEmbedder` and `onnx.Embedder` implement `EmbedBatch`; `RandomProjections` does not; the interface cannot express it.
8. **README documents `MustEmbed`**; it is not implemented.
9. **HTTP embedder** has no retry, no automatic chunking of large batches, no 429 handling.
10. **ONNX embedder** has no `context.Context`, no execution-provider options, no session-option knobs.
11. **RP is closed-vocab.** Tokens unseen at `Fit` are dropped. Re-`Fit` rebuilds everything and changes embeddings.
12. **Persistence is path-only** (`os.Create` / `os.Open`). No `io.Reader` / `io.Writer`, no atomic replace.
13. **Brute-force only.** README already tells users to pair with FAISS/Annoy above ~100K vectors. There is no in-tree ANN.
14. **No metadata, filter, radius, or threshold search.** RAG-style "top-k where `lang=en`" is not expressible.
15. **Vector algebra is allocating.** `Add` / `Sub` / `Scale` / `Normalize` always allocate; there are no in-place variants.
16. **Scalar kernels.** Distance loops are one-element-at-a-time. Go can auto-vectorize some forms; the current shape does not invite it.

These are the gaps the phases below close, in dependency order.

---

## 3. Design principles for extensions

1. **Sibling packages for weight.** Anything that needs third-party code, CGo, or a large algorithm (HNSW, quantization tables, GPU) lives beside `pkg/vector`, the way `pkg/onnx` already does.
2. **Internal layout can change; the API cannot.** Packing vectors, adding an ID map, or caching norms is invisible if `Add`/`Get`/`Search`/`Load` keep today's results.
3. **Options structs / functional options for new knobs.** Do not add parameters to existing functions.
4. **Prefer methods on existing types** over new packages when the work is still stdlib-only and conceptually part of the type.
5. **Measure before merging performance work.** Every perf PR updates `bench_test.go` and records before/after in the PR body, same as PR #2.

---

## 4. Phase 0 — Completeness and production hardening

*Scope: new APIs and missing documented behavior. No search-layout rewrite. Low risk.*

### 0.1 Documented API that is missing

```go
// MustEmbed is Embed, panicking only on a non-nil error.
// RandomProjections.Embed currently never errors; this exists so the
// README example compiles and so HTTP/ONNX callers have a one-liner
// for trusted inputs.
func MustEmbed(e Embedder, text string) Vector

func (rp *RandomProjections) MustEmbed(text string) Vector
```

`MustEmbed` as a package function works for every `Embedder`. The method on `RandomProjections` matches the README.

### 0.2 Introspection (read-only, no semantic change)

```go
func (m Metric) String() string          // "cosine", "euclidean", "manhattan", "dot"
func (s *Store) Metric() Metric
func (s *Store) Has(id string) bool      // first-match, same as Get != nil
func (s *Store) IDs() []string           // clone of the id list (including duplicates)
func (s *Store) Clear()                  // Len()==0; metric unchanged
func (s *Store) Dims() int               // 0 if empty; else len(vectors[0])
```

`IDs` returns a copy so callers cannot mutate store state. `Dims` is informational — the store already accepts mixed lengths (mismatched queries score as zero / cosine-1).

### 0.3 Batch embedder interface (additive)

```go
type BatchEmbedder interface {
    Embedder
    EmbedBatch(texts []string) ([]Vector, error)
}
```

Implement `EmbedBatch` on `RandomProjections` (loop + shared output allocation). Assert `HTTPEmbedder` and `onnx.Embedder` satisfy it (they already have the method).

Do **not** add `EmbedBatch` to `Embedder`.

### 0.4 Streaming persistence

```go
func (s *Store) WriteTo(w io.Writer) (int64, error)
func (s *Store) ReadFrom(r io.Reader) (int64, error)
func (s *Store) WriteJSONTo(w io.Writer) error
func (s *Store) ReadJSONFrom(r io.Reader) error

func (rp *RandomProjections) WriteTo(w io.Writer) (int64, error)
func ReadEmbedder(r io.Reader) (*RandomProjections, error)
```

Path helpers (`Save` / `Load` / …) become thin wrappers. This is the hook for atomic write (write temp, `Rename`) **as an opt-in helper**, not a change to `Save`:

```go
func (s *Store) SaveAtomic(path string) error // write path+".tmp", fsync, rename
```

Existing `Save` stays "overwrite in place" so callers who depend on that (or on the file appearing mid-write) are unchanged.

### 0.5 HTTP embedder production knobs

New options only:

```go
func WithRetry(max int) HTTPEmbedderOption          // retry 429 / 5xx with backoff
func WithMaxBatch(n int) HTTPEmbedderOption         // chunk EmbedBatch into n-sized requests
func WithUserAgent(ua string) HTTPEmbedderOption
func WithMaxResponseBytes(n int64) HTTPEmbedderOption // default stays 64 MiB
```

Default retry count remains **0** (today's behavior).

### 0.6 ONNX: context + session options

```go
func (e *Embedder) EmbedContext(ctx context.Context, text string) (vector.Vector, error)
func (e *Embedder) EmbedBatchContext(ctx context.Context, texts []string) ([]vector.Vector, error)

func WithIntraOpThreads(n int) Option
func WithInterOpThreads(n int) Option
```

`Embed` / `EmbedBatch` stay as `context.Background()` wrappers.

### 0.7 Compatibility fixtures

Add `pkg/vector/testdata/compat/` with a gob/JSON store and RP embedder produced by v1.3. Tests assert `Load` / `LoadEmbedder` succeed and search/embed results match recorded values.

**Exit criteria:** all existing tests pass unchanged; README `MustEmbed` example compiles; coverage stays ≥ 95%.

---

## 5. Phase 1 — Search and kernel performance

*Scope: internal layout + new query methods. `Search` results stay byte-identical for the same inputs.*

### 1.1 Packed backing store (largest expected win)

Today each `Add` clones into its own heap slice. A 10K × 1536d store is 10K separate allocations (~60 MB of floats plus header/span overhead) and Search streams randomly.

Internal change:

```
ids      []string
data     []float32   // row-major, row i is data[i*dims : (i+1)*dims]
dims     int
norms2   []float32   // cached Σv², maintained on Add/Remove/Load
byID     map[string]int // first index of each id; see 1.3
```

Constraints that keep this compatible:

- Mixed-length vectors are still accepted (`Add` of a different length is allowed today). Packing therefore needs either (a) a single `dims` once the first vector arrives and **reject / isolate** later mismatches, or (b) a fallback to the current slice-of-slices path when lengths diverge.
- **Recommended:** pack when all vectors share a length (the 99% case); fall back to `[]Vector` if a mismatched `Add` arrives. `Search` results must match today's scores, including the mismatched → 0 / cosine-1 cases.
- `Load` of an existing gob file rebuilds the packed layout. On-disk format can stay `[]Vector`; packing is a load-time transform. Optional later: a packed gob field for faster load (additive).

Expected effect (from PR #2's diagnosis): sequential scan of one array, better prefetch, fewer GC pointers. Target: **≥ 1.5×** `BenchmarkStoreSearch10000` on the same hardware, same answers.

### 1.2 Cached stored norms

On `Add` / `Load`, store `Σvᵢ²`. Cosine then becomes:

```
1 - dot / sqrt(q2 * v2)
```

with `v2` a load, not a `d`-wide accumulation. Euclidean can use

```
‖a−b‖² = q2 + v2 − 2·dot
```

when the caller wants distance (still one pass for `dot`, no extra `d` subtractions). Keep the public `Euclidean` / `Cosine` functions unchanged (they have no cache). Only `Store.scorer` uses the cache.

Invalidate / recompute on `Remove` (swap-with-last already; swap the norm too).

### 1.3 O(1) ID index without changing duplicate semantics

```
byID map[string]int  // id → first index
```

- `Add`: if `id` is new, record index; if duplicate, leave the map pointing at the **first** entry.
- `Get` / `Has` / `Remove`: use the map, then (for `Remove`) repair the map for the swapped-in last element and, if duplicates of the removed id remain, scan forward once to the next occurrence.
- Worst case for `Remove` of a heavily duplicated id is still linear in remaining duplicates — same as today for that id — but the common unique-id case becomes O(1).

Do not change "first match wins."

### 1.4 Query APIs that avoid clone cost

```go
// SearchIDs is Search without cloning vectors. Distance/sort/k semantics
// are identical to Search.
func (s *Store) SearchIDs(query Vector, k int) []SearchHit

type SearchHit struct {
    ID       string
    Distance float32
}

// SearchOpts adds filters and result-shape controls. Search() is
// equivalent to SearchOpts(query, k).
func (s *Store) SearchOpts(query Vector, k int, opts ...SearchOption) []SearchResult

func WithoutVectors() SearchOption          // Vector field left nil
func WithPredicate(fn func(id string) bool) SearchOption
func WithMaxDistance(d float32) SearchOption // keep only scores that beat d
                                             // (≤ d if Ascending, ≥ d otherwise)
```

`Search` remains the one-liner and still clones. No signature change.

### 1.5 Kernel unrolling (pure Go, no `unsafe`)

Rewrite `Dot`, `Cosine`, `Euclidean`, `Manhattan` with 4- or 8-way accumulation and a tail loop. This stays stdlib-only and zero-alloc. Do **not** add `golang.org/x/sys` or assembly in `pkg/vector`.

Acceptable outcome: 1.2–2× on 1536-d `BenchmarkDot` / `BenchmarkCosine` depending on arch; 0 allocs preserved.

Also add in-place algebra (new functions):

```go
func AddIn(dst, a, b Vector) Vector     // nil if lengths differ; dst may be a or new
func SubIn(dst, a, b Vector) Vector
func ScaleIn(dst, v Vector, s float32) Vector
func NormalizeIn(dst, v Vector) Vector
```

Existing `Add` / `Sub` / `Scale` / `Normalize` keep allocating. The `In` variants let embedders and search helpers reuse buffers.

### 1.6 Optional parallel search

```go
func WithParallel(minN int) SearchOption // default: off
```

Split the stored rows across `GOMAXPROCS` workers, each with a local top-k heap, then merge. Off by default so a 100-vector store does not grow goroutine overhead and so results stay deterministic (merge must be stable: worse-score evicts; ties break by lower index, which matches today's left-to-right heap fill).

**Exit criteria:** existing `TestStore*` pass without edits; packed path covered by new tests (mixed-length fallback, duplicate IDs, load-from-old-gob); benches show the packed+norm win; `-benchmem` on distance functions still `0 allocs`.

---

## 6. Phase 2 — Store and embedder features

*Scope: capabilities users currently bolt on themselves. All additive.*

### 2.1 Upsert and unique-ID helpers

```go
func (s *Store) Upsert(id string, v Vector) // replace first match, or Add
func (s *Store) AddUnique(id string, v Vector) bool // false if id exists
```

`Add` still appends duplicates.

### 2.2 Batch search and range search

```go
func (s *Store) SearchBatch(queries []Vector, k int) [][]SearchResult
func (s *Store) SearchRadius(query Vector, radius float32) []SearchResult
```

`SearchBatch` is a convenience loop at first; after 1.1 it can share packed-row traversal. `SearchRadius` returns all hits within `radius` (ascending metrics: `score <= radius`; dot-product: `score >= radius`), sorted the same way as `Search`. Unbounded result — document that callers should prefer `SearchOpts(..., WithMaxDistance(r))` with a `k` cap when `n` is large.

### 2.3 Optional per-id metadata

Additive persistence field:

```go
type storeData struct {
    Vectors  []Vector
    IDs      []string
    Metric   Metric
    Metadata []map[string]string // optional; nil = no metadata
}
```

```go
func (s *Store) SetMeta(id string, meta map[string]string) bool
func (s *Store) Meta(id string) map[string]string // clone or nil
func WithMetaEqual(key, value string) SearchOption
```

Old gob files have `Metadata == nil`. Search without meta options ignores it. `Remove` / swap-with-last keeps the parallel slice aligned.

### 2.4 Concurrent wrapper

```go
type LockedStore struct { /* embeds or holds *Store + sync.RWMutex */ }

func NewLockedStore(metric Metric) *LockedStore
func (s *LockedStore) Store() *Store // snapshot? no — document that
                                    // the inner pointer is not for
                                    // unlocked use
```

Methods mirror `Store` and take `RLock` on reads, `Lock` on writes. This is a new type so the original `Store` keeps "no internal locking" as promised.

### 2.5 Additional metrics (appended)

```go
ChebyshevDistance Metric = iota // max |a_i - b_i|
HammingDistance                 // count of differing positions (exact float32 compare)
```

Plus direct functions `Chebyshev`, `Hamming` (zero-alloc, mismatch → 0). Update `Distance` and `Ascending` (both are distances). Tests + benches required by AGENTS.md.

Do not add Jaccard/sparse metrics until there is a sparse vector type — that would be a bigger API. Out of scope for v1.

### 2.6 Random projections: quality without breaking seed 42

New constructors / options; default `NewRandomProjections` + `Fit` stay bit-identical.

```go
func NewRandomProjectionsOpts(outputDim int, opts ...RPOption) *RandomProjections

func WithSeed(seed int64) RPOption          // default 42
func WithNGrams(n int) RPOption             // default 1 (today's unigrams)
func WithHashingTrick(buckets int) RPOption // OOV tokens hash into buckets
func WithTFIDF() RPOption                   // weight Fit counts; freeze idf at Fit
func WithMinTokenLen(n int) RPOption        // default 2
```

`FitMore(corpus)` **adds** tokens and extends the projection matrix with new rows, using the same RNG sequence continued from the last Fit. Existing token rows stay unchanged so previously embedded documents remain comparable **if** no TF-IDF reweight runs. Document that `WithTFIDF` + `FitMore` requires a full `Fit`.

Tokenizer changes (n-grams, min length, hashing) are option-gated so persisted embedders loaded via `LoadEmbedder` keep today's tokenize path unless the gob file records the new options (additive fields on `rpPersistData`).

### 2.7 HTTP / ONNX extras

- `HTTPEmbedder`: `WithEndpointPath(path)` for Azure-style `/openai/deployments/.../embeddings`.
- `onnx.Embedder`: `WithOutputName(name)` override; `WithMeanPoolExcludeSpecials()` (skip `[CLS]`/`[SEP]` in the mean — **opt-in**, default keeps today's include-all mean).

**Exit criteria:** new tests for upsert/meta/radius/locked store; gob fixtures from Phase 0 still load; RP default path still matches seed-42 golden vectors.

---

## 7. Phase 3 — Scale beyond brute force

*Scope: new packages. `pkg/vector.Store` remains the exact, small-n index.*

### 3.1 `pkg/hnsw` — approximate nearest neighbor

A sibling package (stdlib + the core `vector` types only, or a single small dep if a well-tested HNSW exists — prefer a from-scratch pure-Go port to keep the module honest).

```go
type Index struct { /* HNSW graph */ }

func New(dims int, metric vector.Metric, opts ...Option) *Index
func (idx *Index) Add(id string, v vector.Vector)
func (idx *Index) Search(query vector.Vector, k int) []vector.SearchResult
func (idx *Index) RecallAgainst(exact *vector.Store, queries []vector.Vector, k int) float64
```

`Store` stays brute-force exact. Documentation: use `Store` up to ~100K, `hnsw.Index` beyond, optionally re-rank top-`k'` with `Store`-style exact scores on the candidates (a `Rerank(exact []Vector, query, k)` helper in `pkg/vector` is fine and stdlib-only).

Persistence: own gob format, not mixed into `storeData`.

### 3.2 `pkg/quantize` — memory reduction

```go
func ToInt8(v vector.Vector) (q []int8, scale float32)
func FromInt8(q []int8, scale float32) vector.Vector
func DotInt8(a, b []int8, scaleA, scaleB float32) float32
```

An `Int8Store` can wrap the packed layout from 1.1. This is a new type, not a mode flag on `Store`, so exact float32 search does not quietly lose precision.

### 3.3 Hybrid lexical + vector

A small helper, not a new index:

```go
func Hybrid(vectorScore, lexicalScore float32, alpha float32) float32
```

plus a `SearchOpts` predicate that callers can combine with their own BM25. Full BM25 in-tree is a non-goal (see §9).

### 3.4 ONNX execution providers

```go
func WithCUDA() Option
func WithCoreML() Option
```

Behind `pkg/onnx` only. Default remains CPU. Document the extra runtime libraries.

### 3.5 CLI that is actually useful

`cmd/go-vector` today is a demo. Additive subcommands, old ones kept:

```
go-vector demo | embed | persist          # unchanged
go-vector index  --in docs.jsonl --out store.gob --embedder rp|http|onnx
go-vector query  --store store.gob --text "..." --k 10
go-vector bench  --n 10000 --d 1536
```

**Exit criteria:** HNSW recall ≥ 0.95 at typical `M=16, ef=64` vs exact `Store` on a 10K/768d synthetic set; int8 cosine ranking Spearman vs float32 documented; CLI commands covered by tests using temp files.

---

## 8. Suggested implementation order

The phases are also PR-sized slices. Prefer merging in this order so each PR is independently useful and revertible:

| PR | Phase | Contents | Risk to callers |
|---|---|---|---|
| A | 0.1–0.3 | `MustEmbed`, `BatchEmbedder`, `Metric.String`, `Store.Metric/Has/IDs/Clear/Dims` | None (new API) |
| B | 0.4, 0.7 | `WriteTo`/`ReadFrom`, `SaveAtomic`, golden gob fixtures | None |
| C | 0.5–0.6 | HTTP retry/chunk; ONNX context | None (defaults unchanged) |
| D | 1.5 | Unrolled kernels + `AddIn`/`ScaleIn`/… | None if answers match |
| E | 1.1–1.3 | Packed store + norm cache + ID map | **Internal only**; needs the most tests |
| F | 1.4, 1.6 | `SearchIDs`, `SearchOpts`, optional parallel | None |
| G | 2.1–2.4 | Upsert, radius, batch search, metadata, `LockedStore` | None |
| H | 2.5–2.7 | New metrics, RP options, embedder extras | None if defaults hold |
| I | 3.1 | `pkg/hnsw` | None (`Store` untouched) |
| J | 3.2–3.5 | Quantize, hybrid helper, ONNX EPs, CLI | None |

PR **E** is the one that must not ship without: mixed-length fallback tests, duplicate-ID tests, load-old-gob tests, and a before/after bench table.

---

## 9. Non-goals (v1)

These would either break the contract or pull the library off its identity:

- Adding any import to `pkg/vector` beyond the Go standard library.
- CGo, `unsafe`, or assembly in `pkg/vector`.
- Changing `Vector` to a struct, or introducing a `Vector64`.
- Making `Store` internally mutexed (use `LockedStore`).
- Changing `Add` to upsert-by-default, or rejecting duplicate IDs.
- Changing RP seed, tokenizer, or `Fit` rebuild semantics.
- Replacing brute-force `Store.Search` with ANN (ANN is a sibling).
- A network service / HTTP server in this repo (the library is embeddable; hosting is the caller's job).
- Full sparse-vector / BM25 / graph-RAG stack.
- Breaking gob compatibility to "pack on disk" without a parallel additive field.
- Raising the Go module version to v2 for any item in this plan.

---

## 10. Testing and release rules

- **Existing tests are the compatibility suite.** Do not "fix" them to match new internals. If a change fails an old test, the change is wrong.
- New functions follow AGENTS.md: constructor + tests (+ bench for metrics and kernels).
- Distance kernels: `go test ./pkg/vector/ -bench=BenchmarkDot -benchmem` must stay `0 allocs/op`.
- Search: add `BenchmarkStoreSearch10000Packed` once packing lands; keep the old names so historical numbers remain comparable.
- Coverage target remains **> 95%**.
- Releases stay on **v1.x** (module path unchanged). Group user-visible additions as minor (`v1.4.0`, `v1.5.0`); internal perf work can ride along or ship as a patch if there is no API change.
- Each minor updates README API lists and the landing page (`docs/index.html`) in the same PR as the code.

---

## 11. Success picture

After this plan, a caller on today's API still compiles and gets the same answers. New callers can:

1. Index with `Upsert` + metadata, persist atomically, and query with predicates.
2. Search 100K×1536d in-process at exact quality, faster, with less GC.
3. Switch to `pkg/hnsw` when `n` outgrows brute force, without changing embedder or `Vector` types.
4. Batch-embed through one interface whether the backend is RP, HTTP, or ONNX.
5. Run ONNX with a deadline, and HTTP with retries, without writing that glue themselves.

The library remains what it is advertised as: a zero-dependency core, optional heavy backends in siblings, and no surprises for existing importers.
