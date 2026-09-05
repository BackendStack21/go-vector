package vector

// Embedder converts text into a Vector embedding.
//
// Implementations may be pure-Go (RandomProjections) or external adapters
// (OpenAI, Ollama, sentence-transformers via subprocess). The interface is
// the contract — users can swap backends without changing search code.
//
// This interface is closed: new methods are not added so existing
// implementations keep compiling. See BatchEmbedder for batched embedding.
type Embedder interface {
	// Embed returns a vector representation of text.
	// Returns an error if embedding fails (e.g., external API error).
	Embed(text string) (Vector, error)

	// Dims returns the dimensionality of vectors produced by this embedder.
	Dims() int
}

// BatchEmbedder is an Embedder that can embed many texts in one call.
// HTTPEmbedder, RandomProjections, and onnx.Embedder all satisfy it.
type BatchEmbedder interface {
	Embedder
	EmbedBatch(texts []string) ([]Vector, error)
}

// MustEmbed calls e.Embed and panics if it returns a non-nil error.
// Intended for trusted inputs and for the README one-liner; prefer Embed
// when the backend can fail (HTTP, ONNX).
func MustEmbed(e Embedder, text string) Vector {
	v, err := e.Embed(text)
	if err != nil {
		panic(err)
	}
	return v
}
