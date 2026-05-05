package vector

import (
	"math"
	"testing"
)

func TestCosine(t *testing.T) {
	tests := []struct {
		name   string
		a, b   Vector
		expect float32
	}{
		{"identical", Vector{1, 0, 0}, Vector{1, 0, 0}, 1},
		{"orthogonal", Vector{1, 0, 0}, Vector{0, 1, 0}, 0},
		{"opposite", Vector{1, 0, 0}, Vector{-1, 0, 0}, -1},
		{"45deg-ish", Vector{1, 1}, Vector{1, 0}, float32(1.0 / math.Sqrt2)},
		{"zero vector", Vector{0, 0}, Vector{1, 2}, 0},
		{"mismatched", Vector{1, 2}, Vector{1}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Cosine(tt.a, tt.b)
			if !approxEqual(got, tt.expect, 1e-5) {
				t.Errorf("Cosine() = %v, want %v", got, tt.expect)
			}
		})
	}
}

func TestCosineDist(t *testing.T) {
	// CosineDist of identical vectors should be 0
	dist := CosineDist(Vector{1, 2, 3}, Vector{1, 2, 3})
	if !approxEqual(dist, 0, 1e-6) {
		t.Errorf("CosineDist(identical) = %v, want 0", dist)
	}

	// CosineDist of orthogonal vectors should be 1
	dist = CosineDist(Vector{1, 0}, Vector{0, 1})
	if !approxEqual(dist, 1, 1e-6) {
		t.Errorf("CosineDist(orthogonal) = %v, want 1", dist)
	}
}

func TestEuclidean(t *testing.T) {
	tests := []struct {
		name   string
		a, b   Vector
		expect float32
	}{
		{"same point", Vector{1, 2}, Vector{1, 2}, 0},
		{"345", Vector{0, 0}, Vector{3, 4}, 5},
		{"negative", Vector{-1, -2}, Vector{2, 2}, 5},
		{"mismatched", Vector{1, 2}, Vector{1}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Euclidean(tt.a, tt.b)
			if !approxEqual(got, tt.expect, 1e-5) {
				t.Errorf("Euclidean() = %v, want %v", got, tt.expect)
			}
		})
	}
}

func TestManhattan(t *testing.T) {
	tests := []struct {
		name   string
		a, b   Vector
		expect float32
	}{
		{"same point", Vector{1, 2}, Vector{1, 2}, 0},
		{"grid", Vector{0, 0}, Vector{3, 4}, 7},
		{"negative", Vector{-1, 2}, Vector{3, -1}, 7},
		{"mismatched", Vector{1, 2}, Vector{1}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Manhattan(tt.a, tt.b)
			if !approxEqual(got, tt.expect, 1e-5) {
				t.Errorf("Manhattan() = %v, want %v", got, tt.expect)
			}
		})
	}
}

func TestDistance(t *testing.T) {
	a := Vector{1, 0}
	b := Vector{0, 1}

	// CosineDistance: 1 - 0 = 1
	if got := Distance(a, b, CosineDistance); !approxEqual(got, 1, 1e-5) {
		t.Errorf("Distance(CosineDistance) = %v, want 1", got)
	}

	// Euclidean: sqrt(2)
	if got := Distance(a, b, EuclideanDistance); !approxEqual(got, 1.4142, 1e-4) {
		t.Errorf("Distance(EuclideanDistance) = %v, want ~1.4142", got)
	}

	// Manhattan: 2
	if got := Distance(a, b, ManhattanDistance); !approxEqual(got, 2, 1e-5) {
		t.Errorf("Distance(ManhattanDistance) = %v, want 2", got)
	}

	// DotProduct: 0
	if got := Distance(a, b, DotProductSimilarity); !approxEqual(got, 0, 1e-5) {
		t.Errorf("Distance(DotProductSimilarity) = %v, want 0", got)
	}
}

func TestMetricAscending(t *testing.T) {
	tests := []struct {
		metric    Metric
		ascending bool
	}{
		{CosineDistance, true},
		{EuclideanDistance, true},
		{ManhattanDistance, true},
		{DotProductSimilarity, false},
	}

	for _, tt := range tests {
		if got := tt.metric.Ascending(); got != tt.ascending {
			t.Errorf("Metric(%d).Ascending() = %v, want %v", tt.metric, got, tt.ascending)
		}
	}
}
