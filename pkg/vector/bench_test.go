package vector

import (
	"math/rand"
	"testing"
)

// --- Helpers ---

func randVector(dims int) Vector {
	v := make(Vector, dims)
	for i := range v {
		v[i] = rand.Float32()
	}
	return v
}

func randStore(n, dims int) *Store {
	s := NewStore(CosineDistance)
	for i := 0; i < n; i++ {
		s.Add("vec", randVector(dims))
	}
	return s
}

// --- Vector Ops ---

func BenchmarkDot(b *testing.B) {
	a, c := randVector(1536), randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Dot(a, c)
	}
}

func BenchmarkNorm(b *testing.B) {
	v := randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Norm(v)
	}
}

func BenchmarkNormalize(b *testing.B) {
	v := randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Normalize(v)
	}
}

func BenchmarkAdd(b *testing.B) {
	a, c := randVector(1536), randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Add(a, c)
	}
}

// --- Similarity ---

func BenchmarkCosine(b *testing.B) {
	a, c := randVector(1536), randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Cosine(a, c)
	}
}

func BenchmarkEuclidean(b *testing.B) {
	a, c := randVector(1536), randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Euclidean(a, c)
	}
}

func BenchmarkManhattan(b *testing.B) {
	a, c := randVector(1536), randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		Manhattan(a, c)
	}
}

// --- Store ---

func BenchmarkStoreAdd(b *testing.B) {
	v := randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s := NewStore(CosineDistance)
		s.Add("x", v)
	}
}

func BenchmarkStoreSearch100(b *testing.B) {
	s := randStore(100, 1536)
	query := randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(query, 10)
	}
}

func BenchmarkStoreSearch1000(b *testing.B) {
	s := randStore(1000, 1536)
	query := randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(query, 10)
	}
}

func BenchmarkStoreSearch10000(b *testing.B) {
	s := randStore(10000, 1536)
	query := randVector(1536)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(query, 10)
	}
}

func BenchmarkStoreGet(b *testing.B) {
	s := NewStore(CosineDistance)
	s.Add("a", randVector(1536))
	s.Add("b", randVector(1536))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Get("a")
	}
}

// --- Metric-specific search ---

func BenchmarkSearchCosine(b *testing.B) {
	s := randStore(1000, 768)
	query := randVector(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(query, 10)
	}
}

func BenchmarkSearchEuclidean(b *testing.B) {
	s := NewStore(EuclideanDistance)
	for i := 0; i < 1000; i++ {
		s.Add("vec", randVector(768))
	}
	query := randVector(768)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Search(query, 10)
	}
}
