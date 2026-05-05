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
		fmt.Fprintln(os.Stderr, "  demo  — run a quick demonstration")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "demo":
		demo()
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
