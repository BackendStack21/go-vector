// Command go-vector is a CLI for the vector package.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: go-vector <command>")
		fmt.Fprintln(os.Stderr, "  demo     — vector store search")
		fmt.Fprintln(os.Stderr, "  embed    — text embedding similarity")
		fmt.Fprintln(os.Stderr, "  persist  — save/load roundtrip")
		fmt.Fprintln(os.Stderr, "  index    — build a store from JSONL")
		fmt.Fprintln(os.Stderr, "  query    — search a saved store")
		fmt.Fprintln(os.Stderr, "  bench    — brute-force search microbench")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "demo":
		demo()
	case "embed":
		embed()
	case "persist":
		persist()
	case "index":
		indexCmd(os.Args[2:])
	case "query":
		queryCmd(os.Args[2:])
	case "bench":
		benchCmd(os.Args[2:])
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

type jsonlDoc struct {
	ID     string    `json:"id"`
	Text   string    `json:"text"`
	Vector []float32 `json:"vector"`
}

func indexCmd(args []string) {
	fs := flag.NewFlagSet("index", flag.ExitOnError)
	in := fs.String("in", "", "JSONL input ({id,text} or {id,vector})")
	out := fs.String("out", "store.gob", "output gob path")
	dim := fs.Int("dim", 64, "RP output dims when embedding text")
	_ = fs.Parse(args)
	if *in == "" {
		fmt.Fprintln(os.Stderr, "index: -in is required")
		os.Exit(1)
	}
	f, err := os.Open(*in)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer f.Close()

	var docs []jsonlDoc
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		var d jsonlDoc
		if err := json.Unmarshal([]byte(line), &d); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		docs = append(docs, d)
	}

	store := vector.NewStore(vector.CosineDistance)
	var rp *vector.RandomProjections
	needEmbed := false
	for _, d := range docs {
		if len(d.Vector) == 0 {
			needEmbed = true
			break
		}
	}
	if needEmbed {
		rp = vector.NewRandomProjections(*dim)
		corpus := make([]string, 0, len(docs))
		for _, d := range docs {
			if d.Text != "" {
				corpus = append(corpus, d.Text)
			}
		}
		rp.Fit(corpus)
	}
	for i, d := range docs {
		id := d.ID
		if id == "" {
			id = strconv.Itoa(i)
		}
		v := vector.Vector(d.Vector)
		if len(v) == 0 {
			v = rp.MustEmbed(d.Text)
		}
		store.Add(id, v)
	}
	if err := store.SaveAtomic(*out); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("indexed %d vectors → %s\n", store.Len(), *out)
}

func queryCmd(args []string) {
	fs := flag.NewFlagSet("query", flag.ExitOnError)
	storePath := fs.String("store", "", "gob store path")
	text := fs.String("text", "", "query text (RP-embedded; needs -corpus or use -vector)")
	vec := fs.String("vector", "", "comma-separated float32 query")
	k := fs.Int("k", 5, "top-k")
	_ = fs.Parse(args)
	if *storePath == "" {
		fmt.Fprintln(os.Stderr, "query: -store is required")
		os.Exit(1)
	}
	s := vector.NewStore(vector.CosineDistance)
	if err := s.Load(*storePath); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var q vector.Vector
	if *vec != "" {
		for _, p := range strings.Split(*vec, ",") {
			p = strings.TrimSpace(p)
			f, err := strconv.ParseFloat(p, 32)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(1)
			}
			q = append(q, float32(f))
		}
	} else if *text != "" {
		rp := vector.NewRandomProjections(s.Dims())
		if rp.Dims() == 0 {
			rp = vector.NewRandomProjections(64)
		}
		rp.Fit([]string{*text})
		q = rp.MustEmbed(*text)
		fmt.Fprintln(os.Stderr, "warning: query RP was fit on the query alone; use precomputed -vector for stored embeddings")
	} else {
		fmt.Fprintln(os.Stderr, "query: provide -text or -vector")
		os.Exit(1)
	}
	for i, r := range s.Search(q, *k) {
		fmt.Printf("%d\t%s\t%.6f\n", i+1, r.ID, r.Distance)
	}
}

func benchCmd(args []string) {
	fs := flag.NewFlagSet("bench", flag.ExitOnError)
	n := fs.Int("n", 1000, "number of vectors")
	d := fs.Int("d", 384, "dimensions")
	k := fs.Int("k", 10, "top-k")
	_ = fs.Parse(args)
	s := vector.NewStore(vector.CosineDistance)
	for i := 0; i < *n; i++ {
		v := make(vector.Vector, *d)
		for j := range v {
			v[j] = float32((i+j)%97) / 97
		}
		s.Add(strconv.Itoa(i), v)
	}
	q := make(vector.Vector, *d)
	for j := range q {
		q[j] = float32(j%97) / 97
	}
	start := time.Now()
	hits := s.Search(q, *k)
	elapsed := time.Since(start)
	fmt.Printf("n=%d d=%d k=%d search=%s hits=%d\n", *n, *d, *k, elapsed, len(hits))
}
