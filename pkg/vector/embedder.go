package vector

// Embedder converts text into a Vector embedding.
//
// Implementations may be pure-Go (RandomProjections) or external adapters
// (OpenAI, Ollama, sentence-transformers via subprocess). The interface is
// the contract — users can swap backends without changing search code.
type Embedder interface {
	// Embed returns a vector representation of text.
	// Returns an error if embedding fails (e.g., external API error).
	Embed(text string) (Vector, error)

	// Dims returns the dimensionality of vectors produced by this embedder.
	Dims() int
}
