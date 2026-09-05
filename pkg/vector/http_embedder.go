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
	retries   int
	maxBatch  int
	userAgent string
	maxResp   int64
	embedPath string

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

// WithRetry retries 429 and 5xx responses up to max times (exponential
// backoff starting at 50ms). Default 0 — today's no-retry behavior.
func WithRetry(max int) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) {
		if max > 0 {
			e.retries = max
		}
	}
}

// WithMaxBatch splits EmbedBatch into chunks of at most n texts.
// n <= 0 means no splitting (one request for the whole batch).
func WithMaxBatch(n int) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) {
		if n > 0 {
			e.maxBatch = n
		}
	}
}

// WithUserAgent sets the User-Agent header on every request.
func WithUserAgent(ua string) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) { e.userAgent = ua }
}

// WithMaxResponseBytes caps the response body. Default 64 MiB.
func WithMaxResponseBytes(n int64) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) {
		if n > 0 {
			e.maxResp = n
		}
	}
}

// WithEndpointPath replaces the default "/embeddings" suffix, e.g. Azure
// "/openai/deployments/<name>/embeddings".
func WithEndpointPath(path string) HTTPEmbedderOption {
	return func(e *HTTPEmbedder) { e.embedPath = path }
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
		maxResp: 64 << 20,
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

func (e *HTTPEmbedder) endpoint() string {
	path := e.embedPath
	if path == "" {
		path = "/embeddings"
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return e.baseURL + path
}

func retryableStatus(code int) bool {
	return code == http.StatusTooManyRequests || code >= 500
}

// EmbedBatchContext is EmbedBatch with request cancellation/deadline control.
func (e *HTTPEmbedder) EmbedBatchContext(ctx context.Context, texts []string) ([]Vector, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	chunk := e.maxBatch
	if chunk <= 0 || chunk >= len(texts) {
		return e.embedOnce(ctx, texts)
	}
	out := make([]Vector, 0, len(texts))
	for i := 0; i < len(texts); i += chunk {
		j := i + chunk
		if j > len(texts) {
			j = len(texts)
		}
		part, err := e.embedOnce(ctx, texts[i:j])
		if err != nil {
			return nil, err
		}
		out = append(out, part...)
	}
	return out, nil
}

func (e *HTTPEmbedder) embedOnce(ctx context.Context, texts []string) ([]Vector, error) {
	body, err := json.Marshal(embedRequest{Model: e.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("vector: encode embeddings request: %w", err)
	}

	limit := e.maxResp
	if limit <= 0 {
		limit = 64 << 20
	}

	var lastErr error
	attempts := e.retries + 1
	for attempt := 0; attempt < attempts; attempt++ {
		if attempt > 0 {
			delay := 50 * time.Millisecond * time.Duration(1<<uint(attempt-1))
			timer := time.NewTimer(delay)
			select {
			case <-ctx.Done():
				timer.Stop()
				return nil, ctx.Err()
			case <-timer.C:
			}
		}

		req, err := http.NewRequestWithContext(ctx, http.MethodPost, e.endpoint(), bytes.NewReader(body))
		if err != nil {
			return nil, fmt.Errorf("vector: build embeddings request: %w", err)
		}
		req.Header.Set("Content-Type", "application/json")
		if e.userAgent != "" {
			req.Header.Set("User-Agent", e.userAgent)
		}
		if e.apiKey != "" {
			req.Header.Set("Authorization", "Bearer "+e.apiKey)
		}
		for k, v := range e.headers {
			req.Header.Set(k, v)
		}

		resp, err := e.client.Do(req)
		if err != nil {
			lastErr = fmt.Errorf("vector: embeddings request failed: %w", err)
			if attempt+1 < attempts {
				continue
			}
			return nil, lastErr
		}

		raw, readErr := io.ReadAll(io.LimitReader(resp.Body, limit))
		resp.Body.Close()
		if readErr != nil {
			return nil, fmt.Errorf("vector: read embeddings response: %w", readErr)
		}

		if retryableStatus(resp.StatusCode) && attempt+1 < attempts {
			lastErr = fmt.Errorf("vector: embeddings API returned status %d", resp.StatusCode)
			continue
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

		// The index field, not array position, is authoritative for ordering —
		// and must form the exact permutation 0..n-1, or texts and vectors
		// would be silently mismatched.
		sort.Slice(parsed.Data, func(i, j int) bool { return parsed.Data[i].Index < parsed.Data[j].Index })
		got := len(parsed.Data[0].Embedding)
		for i, d := range parsed.Data {
			if d.Index != i {
				return nil, fmt.Errorf("vector: embeddings API returned indices that are not a permutation of 0..%d", len(texts)-1)
			}
			if len(d.Embedding) != got {
				return nil, fmt.Errorf("vector: embeddings API returned inconsistent dims (%d and %d) in one response", got, len(d.Embedding))
			}
		}
		// Validate (and possibly lock in) dims only after the whole batch is
		// known-consistent, so a rejected response can never poison inference.
		if err := e.checkDims(got); err != nil {
			return nil, err
		}

		out := make([]Vector, len(texts))
		for i, d := range parsed.Data {
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
	return nil, lastErr
}

// checkDims validates a response vector's length against the declared
// dimensionality, inferring it from the first valid response when unset.
func (e *HTTPEmbedder) checkDims(got int) error {
	if got == 0 {
		return fmt.Errorf("vector: embeddings API returned an empty embedding")
	}
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
