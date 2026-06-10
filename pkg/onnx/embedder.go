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
	session    *ort.DynamicAdvancedSession
	tok        *wordPieceTokenizer
	inputNames []string // model-declared order, subset of the BERT trio
	pooled     bool     // true when the model outputs rank-2 sentence embeddings
	maxLen     int

	mu   sync.Mutex
	dims int
}

// Option configures an Embedder.
type Option func(*config)

type config struct {
	libraryPath string
	maxLen      int
	cased       bool
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

var (
	initOnce sync.Once
	initErr  error
)

// initRuntime initializes the global ONNX Runtime environment exactly once.
func initRuntime(explicit string) error {
	initOnce.Do(func() {
		if path := resolveLibrary(explicit); path != "" {
			ort.SetSharedLibraryPath(path)
		}
		initErr = ort.InitializeEnvironment()
	})
	return initErr
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

	e := &Embedder{tok: tok, maxLen: cfg.maxLen}
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

	// Prefer a pooled sentence embedding when exported; otherwise
	// mean-pool token embeddings ourselves.
	out := outputs[0]
	for _, o := range outputs {
		if o.Name == "sentence_embedding" {
			out = o
			break
		}
		if o.Name == "last_hidden_state" {
			out = o
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

	session, err := ort.NewDynamicAdvancedSession(modelPath, e.inputNames, []string{out.Name}, nil)
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
	vecs, err := e.EmbedBatch([]string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch embeds multiple texts in one model invocation, returning
// vectors in input order. Shorter texts are padded and masked, so results
// match per-text Embed calls. Returns nil for an empty input.
func (e *Embedder) EmbedBatch(texts []string) ([]vector.Vector, error) {
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
	e.setDims(hidden)

	result := make([]vector.Vector, len(texts))
	for i := range texts {
		v := make(vector.Vector, hidden)
		if e.pooled {
			copy(v, data[i*hidden:(i+1)*hidden])
		} else {
			// Mean over real (unmasked) token positions.
			n := len(encoded[i])
			row := i * seqLen * hidden
			for j := 0; j < n; j++ {
				tok := data[row+j*hidden : row+(j+1)*hidden]
				for d, x := range tok {
					v[d] += x
				}
			}
			for d := range v {
				v[d] /= float32(n)
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
