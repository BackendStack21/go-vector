// Package onnx provides a local neural text embedder that runs
// transformer models (e.g. sentence-transformers/all-MiniLM-L6-v2) fully
// in-process via ONNX Runtime — no server, no API key, deterministic.
//
// Unlike the core pkg/vector package, this package carries dependencies:
// the github.com/yalue/onnxruntime_go binding (CGo) and the ONNX Runtime
// shared library at runtime (e.g. `brew install onnxruntime`). Import it
// only if you want local neural embeddings; pkg/vector stays pure Go.
package onnx

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"sync"

	"github.com/BackendStack21/go-vector/pkg/vector"
	ort "github.com/yalue/onnxruntime_go"
)

// Embedder runs a BERT-family ONNX embedding model locally. It satisfies
// vector.Embedder, so it plugs straight into vector.Store search code.
//
// The model must take the standard BERT inputs ("input_ids",
// "attention_mask", and optionally "token_type_ids") and produce either
// token embeddings (rank-3 "last_hidden_state", mean-pooled here) or a
// pooled rank-2 "sentence_embedding". Both layouts are detected from the
// model's declared outputs. Output vectors are L2-normalized.
//
// Concurrency: safe for concurrent Embed/EmbedBatch calls (ONNX Runtime
// sessions are thread-safe).
type Embedder struct {
	session         *ort.DynamicAdvancedSession
	tok             *wordPieceTokenizer
	inputNames      []string // model-declared order, subset of the BERT trio
	pooled          bool     // true when the model outputs rank-2 sentence embeddings
	maxLen          int
	excludeSpecials bool

	mu   sync.Mutex
	dims int
}

// Option configures an Embedder.
type Option func(*config)

type config struct {
	libraryPath     string
	maxLen          int
	cased           bool
	intraOp         int
	interOp         int
	outputName      string
	excludeSpecials bool
	cuda            bool
	coreml          bool
}

// WithLibraryPath sets the path to the ONNX Runtime shared library
// (libonnxruntime.dylib / .so / onnxruntime.dll). By default the
// ONNXRUNTIME_SHARED_LIBRARY_PATH environment variable and common install
// locations (Homebrew, /usr/local/lib, /usr/lib) are tried. The library is
// initialized once per process: the first Embedder's path wins.
func WithLibraryPath(path string) Option {
	return func(c *config) { c.libraryPath = path }
}

// WithMaxLength caps tokenized sequence length (default 256). Longer
// inputs are truncated. Raise toward the model's limit (typically 512)
// for long documents; lower it for faster embedding.
func WithMaxLength(n int) Option {
	return func(c *config) {
		if n > 2 {
			c.maxLen = n
		}
	}
}

// WithCasedVocab disables lowercasing/accent-stripping, for models with a
// cased vocabulary. Default is uncased (correct for all-MiniLM-L6-v2 and
// most sentence-transformers models).
func WithCasedVocab() Option {
	return func(c *config) { c.cased = true }
}

// WithIntraOpThreads sets ONNX Runtime intra-op threads. 0 leaves the default.
func WithIntraOpThreads(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.intraOp = n
		}
	}
}

// WithInterOpThreads sets ONNX Runtime inter-op threads. 0 leaves the default.
func WithInterOpThreads(n int) Option {
	return func(c *config) {
		if n > 0 {
			c.interOp = n
		}
	}
}

// WithOutputName forces a specific model output name instead of auto-detect.
func WithOutputName(name string) Option {
	return func(c *config) { c.outputName = name }
}

// WithMeanPoolExcludeSpecials skips [CLS] and [SEP] when mean-pooling
// last_hidden_state. Default includes every unmasked token (today's behavior).
func WithMeanPoolExcludeSpecials() Option {
	return func(c *config) { c.excludeSpecials = true }
}

// WithCUDA appends the CUDA execution provider. New fails if CUDA is unavailable.
func WithCUDA() Option {
	return func(c *config) { c.cuda = true }
}

// WithCoreML appends the CoreML execution provider. New fails if unavailable.
func WithCoreML() Option {
	return func(c *config) { c.coreml = true }
}

var (
	initMu   sync.Mutex
	initDone bool
)

// initRuntime initializes the global ONNX Runtime environment. The first
// successful initialization wins; a failed attempt does not poison later
// ones, so New can be retried with a corrected library path.
func initRuntime(explicit string) error {
	initMu.Lock()
	defer initMu.Unlock()
	if initDone {
		return nil
	}
	if path := resolveLibrary(explicit); path != "" {
		ort.SetSharedLibraryPath(path)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		return err
	}
	initDone = true
	return nil
}

// resolveLibrary picks the ONNX Runtime shared library: explicit option,
// then env var, then common install locations. Empty means "let the
// binding use its platform default name" (system loader search path).
func resolveLibrary(explicit string) string {
	if explicit != "" {
		return explicit
	}
	if env := os.Getenv("ONNXRUNTIME_SHARED_LIBRARY_PATH"); env != "" {
		return env
	}
	var candidates []string
	switch runtime.GOOS {
	case "darwin":
		candidates = []string{
			"/opt/homebrew/lib/libonnxruntime.dylib",
			"/usr/local/lib/libonnxruntime.dylib",
		}
	case "linux":
		candidates = []string{
			"/usr/local/lib/libonnxruntime.so",
			"/usr/lib/libonnxruntime.so",
			"/usr/lib/x86_64-linux-gnu/libonnxruntime.so",
			"/usr/lib/aarch64-linux-gnu/libonnxruntime.so",
		}
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// New loads an ONNX embedding model and its BERT vocab.txt. For
// sentence-transformers/all-MiniLM-L6-v2, download onnx/model.onnx and
// vocab.txt from the model's Hugging Face repository.
func New(modelPath, vocabPath string, opts ...Option) (*Embedder, error) {
	cfg := config{maxLen: 256}
	for _, opt := range opts {
		opt(&cfg)
	}

	if err := initRuntime(cfg.libraryPath); err != nil {
		return nil, fmt.Errorf("onnx: initialize runtime (is the ONNX Runtime shared library installed? try `brew install onnxruntime`): %w", err)
	}

	tok, err := loadVocab(vocabPath, !cfg.cased)
	if err != nil {
		return nil, err
	}

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return nil, fmt.Errorf("onnx: inspect model: %w", err)
	}

	e := &Embedder{tok: tok, maxLen: cfg.maxLen, excludeSpecials: cfg.excludeSpecials}
	for _, in := range inputs {
		switch in.Name {
		case "input_ids", "attention_mask", "token_type_ids":
			e.inputNames = append(e.inputNames, in.Name)
		default:
			return nil, fmt.Errorf("onnx: unsupported model input %q (expected BERT-style input_ids/attention_mask/token_type_ids)", in.Name)
		}
	}
	if len(e.inputNames) == 0 {
		return nil, fmt.Errorf("onnx: model declares no recognized inputs")
	}

	if len(outputs) == 0 {
		return nil, fmt.Errorf("onnx: model declares no outputs")
	}
	// Prefer a pooled sentence embedding when exported; otherwise
	// mean-pool token embeddings ourselves.
	out := outputs[0]
	if cfg.outputName != "" {
		found := false
		for _, o := range outputs {
			if o.Name == cfg.outputName {
				out = o
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("onnx: model has no output named %q", cfg.outputName)
		}
	} else {
		for _, o := range outputs {
			if o.Name == "sentence_embedding" {
				out = o
				break
			}
			if o.Name == "last_hidden_state" {
				out = o
			}
		}
	}
	switch len(out.Dimensions) {
	case 2:
		e.pooled = true
	case 3:
		e.pooled = false
	default:
		return nil, fmt.Errorf("onnx: output %q has rank %d, expected 2 or 3", out.Name, len(out.Dimensions))
	}
	if d := out.Dimensions[len(out.Dimensions)-1]; d > 0 {
		e.dims = int(d)
	}

	var sessionOpts *ort.SessionOptions
	if cfg.intraOp > 0 || cfg.interOp > 0 || cfg.cuda || cfg.coreml {
		sessionOpts, err = ort.NewSessionOptions()
		if err != nil {
			return nil, fmt.Errorf("onnx: session options: %w", err)
		}
		defer sessionOpts.Destroy()
		if cfg.intraOp > 0 {
			if err := sessionOpts.SetIntraOpNumThreads(cfg.intraOp); err != nil {
				return nil, fmt.Errorf("onnx: intra-op threads: %w", err)
			}
		}
		if cfg.interOp > 0 {
			if err := sessionOpts.SetInterOpNumThreads(cfg.interOp); err != nil {
				return nil, fmt.Errorf("onnx: inter-op threads: %w", err)
			}
		}
		if cfg.cuda {
			cudaOpts, err := ort.NewCUDAProviderOptions()
			if err != nil {
				return nil, fmt.Errorf("onnx: CUDA provider: %w", err)
			}
			if err := sessionOpts.AppendExecutionProviderCUDA(cudaOpts); err != nil {
				cudaOpts.Destroy()
				return nil, fmt.Errorf("onnx: CUDA execution provider: %w", err)
			}
			cudaOpts.Destroy()
		}
		if cfg.coreml {
			if err := sessionOpts.AppendExecutionProviderCoreML(0); err != nil {
				return nil, fmt.Errorf("onnx: CoreML execution provider: %w", err)
			}
		}
	}

	session, err := ort.NewDynamicAdvancedSession(modelPath, e.inputNames, []string{out.Name}, sessionOpts)
	if err != nil {
		return nil, fmt.Errorf("onnx: create session: %w", err)
	}
	e.session = session
	return e, nil
}

// Close releases the underlying ONNX session. The Embedder must not be
// used afterwards.
func (e *Embedder) Close() error {
	return e.session.Destroy()
}

// Dims returns the embedding dimensionality (e.g. 384 for MiniLM-L6).
// Returns 0 only if the model declares a dynamic hidden size and nothing
// has been embedded yet.
func (e *Embedder) Dims() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.dims
}

// Embed returns the L2-normalized embedding for text.
func (e *Embedder) Embed(text string) (vector.Vector, error) {
	return e.EmbedContext(context.Background(), text)
}

// EmbedContext is Embed with cancellation. The ONNX run itself is
// blocking; ctx is checked before invocation.
func (e *Embedder) EmbedContext(ctx context.Context, text string) (vector.Vector, error) {
	vecs, err := e.EmbedBatchContext(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch embeds multiple texts in one model invocation, returning
// vectors in input order. Shorter texts are padded and masked, so results
// match per-text Embed calls. Returns nil for an empty input.
func (e *Embedder) EmbedBatch(texts []string) ([]vector.Vector, error) {
	return e.EmbedBatchContext(context.Background(), texts)
}

// EmbedBatchContext is EmbedBatch with cancellation checked before the run.
func (e *Embedder) EmbedBatchContext(ctx context.Context, texts []string) ([]vector.Vector, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(texts) == 0 {
		return nil, nil
	}

	batch := int64(len(texts))
	encoded := make([][]int64, len(texts))
	seqLen := 0
	for i, t := range texts {
		encoded[i] = e.tok.encode(t, e.maxLen)
		if len(encoded[i]) > seqLen {
			seqLen = len(encoded[i])
		}
	}

	ids := make([]int64, len(texts)*seqLen)
	mask := make([]int64, len(texts)*seqLen)
	types := make([]int64, len(texts)*seqLen) // all zeros: single-segment input
	for i, enc := range encoded {
		row := i * seqLen
		for j, id := range enc {
			ids[row+j] = id
			mask[row+j] = 1
		}
		for j := len(enc); j < seqLen; j++ {
			ids[row+j] = e.tok.padID
		}
	}

	shape := ort.NewShape(batch, int64(seqLen))
	byName := map[string][]int64{"input_ids": ids, "attention_mask": mask, "token_type_ids": types}
	inputs := make([]ort.Value, len(e.inputNames))
	for i, name := range e.inputNames {
		t, err := ort.NewTensor(shape, byName[name])
		if err != nil {
			return nil, fmt.Errorf("onnx: create %s tensor: %w", name, err)
		}
		defer t.Destroy()
		inputs[i] = t
	}

	outputs := []ort.Value{nil}
	if err := e.session.Run(inputs, outputs); err != nil {
		return nil, fmt.Errorf("onnx: run model: %w", err)
	}
	out, ok := outputs[0].(*ort.Tensor[float32])
	if !ok {
		outputs[0].Destroy()
		return nil, fmt.Errorf("onnx: model output is not a float32 tensor")
	}
	defer out.Destroy()

	data := out.GetData()
	outShape := out.GetShape()
	hidden := int(outShape[len(outShape)-1])
	// Validate the full shape before indexing into data — a model whose
	// output disagrees with (batch, seqLen, hidden) must error, not panic.
	wantRank := 3
	if e.pooled {
		wantRank = 2
	}
	if hidden <= 0 || len(outShape) != wantRank || int(outShape[0]) != len(texts) ||
		(!e.pooled && int(outShape[1]) != seqLen) {
		return nil, fmt.Errorf("onnx: unexpected output shape %v for batch=%d seq=%d", outShape, len(texts), seqLen)
	}
	e.setDims(hidden)

	result := make([]vector.Vector, len(texts))
	for i := range texts {
		v := make(vector.Vector, hidden)
		if e.pooled {
			copy(v, data[i*hidden:(i+1)*hidden])
		} else {
			// Mean over real (unmasked) token positions.
			n := len(encoded[i])
			start, end := 0, n
			if e.excludeSpecials && n > 2 {
				start, end = 1, n-1
			}
			count := end - start
			row := i * seqLen * hidden
			for j := start; j < end; j++ {
				tok := data[row+j*hidden : row+(j+1)*hidden]
				for d, x := range tok {
					v[d] += x
				}
			}
			for d := range v {
				v[d] /= float32(count)
			}
		}
		if norm := vector.Norm(v); norm > 0 {
			for d := range v {
				v[d] /= norm
			}
		}
		result[i] = v
	}
	return result, nil
}

// setDims records the hidden size observed at inference time.
func (e *Embedder) setDims(d int) {
	e.mu.Lock()
	e.dims = d
	e.mu.Unlock()
}
