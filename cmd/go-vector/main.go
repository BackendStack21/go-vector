// Command go-vector is a minimal CLI demonstrating the vector package.
package main

import (
	"fmt"
	"os"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: go-vector <command>")
		fmt.Fprintln(os.Stderr, "  demo     — vector store search")
		fmt.Fprintln(os.Stderr, "  embed    — text embedding similarity")
		fmt.Fprintln(os.Stderr, "  persist  — save/load roundtrip")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "demo":
		demo()
	case "embed":
		embed()
	case "persist":
		persist()
	default:
		fmt.Fprintf(os.Stderr, "unknown command: %s\n", os.Args[1])
		os.Exit(1)
	}
}

func demo() {
	store := vector.NewStore(vector.CosineDistance)

	store.Add("cat", vector.Vector{1.0, 0.8, 0.2})
	store.Add("dog", vector.Vector{0.9, 0.7, 0.1})
	store.Add("car", vector.Vector{0.1, 0.0, 0.9})
	store.Add("truck", vector.Vector{0.0, 0.1, 1.0})

	query := vector.Vector{1.0, 0.9, 0.1}
	results := store.Search(query, 3)

	fmt.Printf("Query: %v\n\n", query)
	fmt.Println("Top 3 nearest neighbors (CosineDistance):")
	for i, r := range results {
		fmt.Printf("  %d. %s (distance: %.4f)\n", i+1, r.ID, r.Distance)
	}
}

func embed() {
	corpus := []string{
		"machine learning is fascinating and powerful",
		"deep learning and neural networks transform data",
		"the weather today is sunny and warm",
		"it will rain tomorrow with thunderstorms",
		"artificial intelligence drives the future",
	}

	rp := vector.NewRandomProjections(128)
	rp.Fit(corpus)

	store := vector.NewStore(vector.CosineDistance)
	for _, doc := range corpus {
		v, _ := rp.Embed(doc)
		store.Add(doc, v)
	}

	query := "learning about machine intelligence and AI"
	qv, _ := rp.Embed(query)
	results := store.Search(qv, 3)

	fmt.Printf("Corpus: %d documents, Vocab: %d tokens, Dims: %d\n\n",
		len(corpus), rp.VocabSize(), rp.Dims())
	fmt.Printf("Query: %q\n\n", query)
	fmt.Println("Top 3 matches (CosineDistance):")
	for i, r := range results {
		fmt.Printf("  %d. %q (distance: %.4f)\n", i+1, r.ID, r.Distance)
	}
}

func persist() {
	store := vector.NewStore(vector.CosineDistance)
	store.Add("alpha", vector.Vector{1, 0, 0})
	store.Add("beta", vector.Vector{0, 1, 0})
	store.Add("gamma", vector.Vector{0, 0, 1})

	tmp := os.TempDir()
	path := tmp + "/go-vector-demo.gob"

	if err := store.Save(path); err != nil {
		fmt.Fprintf(os.Stderr, "Save: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Saved %d vectors to %s\n", store.Len(), path)

	restored := vector.NewStore(vector.EuclideanDistance) // different metric
	if err := restored.Load(path); err != nil {
		fmt.Fprintf(os.Stderr, "Load: %v\n", err)
		os.Exit(1)
	}

	results := restored.Search(vector.Vector{1, 0, 0}, 3)
	fmt.Println("\nRestored and searched — closest to {1,0,0}:")
	for i, r := range results {
		fmt.Printf("  %d. %s (distance: %.4f)\n", i+1, r.ID, r.Distance)
	}
}
