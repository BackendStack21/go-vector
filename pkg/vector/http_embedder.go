package vector

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"
)

// HTTPEmbedder is an Embedder backed by any service that speaks the
// OpenAI-compatible embeddings protocol (POST {baseURL}/embeddings).
// That covers OpenAI, Ollama, LM Studio, Voyage AI, llama.cpp server,
// vLLM, and most self-hosted embedding gateways.
//
// It uses only net/http and encoding/json — the library stays
// zero-dependency. Network failures and API errors are returned as
// errors, never panics.
//
// Concurrency: safe for concurrent Embed/EmbedBatch calls.
type HTTPEmbedder struct {
	baseURL   string
	model     string
	apiKey    string
	headers   map[string]string
	client    *http.Client
	normalize bool

	mu   sync.Mutex
	dims int
}

// HTTPEmbedderOption configures an HTTPEmbedder.
type HTTPEmbedderOption func(*HTTPEmbedder)

// WithAPIKey sets the bearer token sent as "Authorization: Bearer <key>".
func WithAPIKey(key string) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) { e.apiKey = key }
}

// WithHTTPClient replaces the default HTTP client (30s timeout).
// Use this to set custom timeouts, proxies, or transports.
func WithHTTPClient(c *http.Client) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) {
		if c != nil {
			e.client = c
		}
	}
}

// WithHeader adds a custom header to every request (e.g. "api-key" for
// Azure OpenAI, or organization/project headers).
func WithHeader(key, value string) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) { e.headers[key] = value }
}

// WithNormalize L2-normalizes every returned vector. Useful when the
// backend does not normalize (e.g. Ollama) and you search with
// DotProductSimilarity; cosine search is unaffected either way.
func WithNormalize() HTTPEmbedderOption {
	return func(e *HTTPEmbedder) { e.normalize = true }
}

// NewHTTPEmbedder creates an embedder for an OpenAI-compatible embeddings
// endpoint. baseURL is the API root, e.g. "https://api.openai.com/v1" or
// "http://localhost:11434/v1" (Ollama); "/embeddings" is appended. model
// names the embedding model, e.g. "text-embedding-3-small" or
// "nomic-embed-text". dims declares the expected dimensionality — responses
// of a different length are rejected. Pass 0 to infer dims from the first
// successful response.
func NewHTTPEmbedder(baseURL, model string, dims int, opts ...HTTPEmbedderOption) *HTTPEmbedder {
	if dims < 0 {
		dims = 0
	}
	e := &HTTPEmbedder{
		baseURL: strings.TrimRight(baseURL, "/"),
		model:   model,
		dims:    dims,
		headers: make(map[string]string),
		client:  &http.Client{Timeout: 30 * time.Second},
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// Dims returns the embedder's dimensionality. Returns 0 until known —
// either declared at construction or inferred from the first response.
func (e *HTTPEmbedder) Dims() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.dims
}

// Embed returns the embedding for text via a single API call.
func (e *HTTPEmbedder) Embed(text string) (Vector, error) {
	return e.EmbedContext(context.Background(), text)
}

// EmbedContext is Embed with request cancellation/deadline control.
func (e *HTTPEmbedder) EmbedContext(ctx context.Context, text string) (Vector, error) {
	vecs, err := e.EmbedBatchContext(ctx, []string{text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedBatch embeds multiple texts in one API call, returning vectors in
// input order. Returns nil for an empty input. Batching is dramatically
// cheaper than per-text calls when indexing a corpus.
func (e *HTTPEmbedder) EmbedBatch(texts []string) ([]Vector, error) {
	return e.EmbedBatchContext(context.Background(), texts)
}

// embedRequest / embedResponse mirror the OpenAI embeddings wire format.
type embedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type embedResponse struct {
	Data []struct {
		Index     int    `json:"index"`
		Embedding Vector `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

// EmbedBatchContext is EmbedBatch with request cancellation/deadline control.
func (e *HTTPEmbedder) EmbedBatchContext(ctx context.Context, texts []string) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	body, err := json.Marshal(embedRequest{Model: e.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("vector: encode embeddings request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.baseURL+"/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("vector: build embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.apiKey)
	}
	for k, v := range e.headers {
		req.Header.Set(k, v)
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("vector: embeddings request failed: %w", err)
	}
	defer resp.Body.Close()

	raw, err := io.ReadAll(io.LimitReader(resp.Body, 64<<20))
	if err != nil {
		return nil, fmt.Errorf("vector: read embeddings response: %w", err)
	}

	var parsed embedResponse
	if jsonErr := json.Unmarshal(raw, &parsed); jsonErr != nil {
		if resp.StatusCode != http.StatusOK {
			return nil, fmt.Errorf("vector: embeddings API returned status %d", resp.StatusCode)
		}
		return nil, fmt.Errorf("vector: decode embeddings response: %w", jsonErr)
	}
	if parsed.Error != nil && parsed.Error.Message != "" {
		return nil, fmt.Errorf("vector: embeddings API error (status %d): %s", resp.StatusCode, parsed.Error.Message)
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("vector: embeddings API returned status %d", resp.StatusCode)
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("vector: embeddings API returned %d vectors for %d inputs", len(parsed.Data), len(texts))
	}

	// The index field, not array position, is authoritative for ordering.
	sort.Slice(parsed.Data, func(i, j int) bool { return parsed.Data[i].Index < parsed.Data[j].Index })

	out := make([]Vector, len(texts))
	for i, d := range parsed.Data {
		if err := e.checkDims(len(d.Embedding)); err != nil {
			return nil, err
		}
		v := d.Embedding
		if e.normalize {
			if n := Norm(v); n > 0 {
				for j := range v {
					v[j] /= n
				}
			}
		}
		out[i] = v
	}
	return out, nil
}

// checkDims validates a response vector's length against the declared
// dimensionality, inferring it from the first response when unset.
func (e *HTTPEmbedder) checkDims(got int) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.dims == 0 {
		e.dims = got
		return nil
	}
	if got != e.dims {
		return fmt.Errorf("vector: embeddings API returned %d dims, expected %d", got, e.dims)
	}
	return nil
}
