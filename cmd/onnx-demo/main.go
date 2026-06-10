// Command onnx-demo runs an end-to-end semantic search demo using a local
// ONNX transformer model (all-MiniLM-L6-v2): embed a corpus in one batch,
// index it in a vector.Store, and answer natural-language queries.
//
// Fetch the model first with `make model`, then run `make demo-onnx`.
// It lives apart from cmd/go-vector so the main demo stays CGo-free.
package main

import (
	"flag"
	"fmt"
	"os"
	"time"

	"github.com/BackendStack21/go-vector/pkg/onnx"
	"github.com/BackendStack21/go-vector/pkg/vector"
)

func main() {
	modelPath := flag.String("model", "pkg/onnx/testdata/model.onnx", "path to the ONNX model")
	vocabPath := flag.String("vocab", "pkg/onnx/testdata/vocab.txt", "path to the BERT vocab.txt")
	flag.Parse()

	if _, err := os.Stat(*modelPath); err != nil {
		fmt.Fprintf(os.Stderr, "model not found at %s — run `make model` first\n", *modelPath)
		os.Exit(1)
	}

	start := time.Now()
	embedder, err := onnx.New(*modelPath, *vocabPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer embedder.Close()
	fmt.Printf("Loaded all-MiniLM-L6-v2 (%d dims) in %v\n\n", embedder.Dims(), time.Since(start).Round(time.Millisecond))

	corpus := []string{
		"The cat curled up on the warm windowsill and fell asleep.",
		"Golden retrievers are friendly dogs that love to play fetch.",
		"The Federal Reserve raised interest rates by a quarter point.",
		"Quarterly earnings beat analyst expectations, lifting the stock.",
		"Preheat the oven to 200°C and roast the vegetables for 30 minutes.",
		"Whisk the eggs with sugar until the mixture turns pale and fluffy.",
		"The spacecraft entered orbit around Jupiter after a six-year journey.",
		"Astronomers detected water vapor in the atmosphere of a distant exoplanet.",
	}

	start = time.Now()
	vecs, err := embedder.EmbedBatch(corpus)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("Embedded %d documents in one batch (%v)\n", len(corpus), time.Since(start).Round(time.Millisecond))

	store := vector.NewStore(vector.CosineDistance)
	for i, doc := range corpus {
		store.Add(doc, vecs[i])
	}

	queries := []string{
		"pets and animals at home",
		"central bank monetary policy",
		"baking a dessert",
		"space exploration",
	}
	for _, q := range queries {
		start = time.Now()
		qv, err := embedder.Embed(q)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		results := store.Search(qv, 2)
		fmt.Printf("\nQuery: %q (%v)\n", q, time.Since(start).Round(time.Millisecond))
		for i, r := range results {
			fmt.Printf("  %d. [%.4f] %s\n", i+1, r.Distance, r.ID)
		}
	}
}
