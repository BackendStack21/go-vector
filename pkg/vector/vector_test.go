package vector

import (
	"math"
	"testing"
)

func TestDims(t *testing.T) {
	if got := Dims(Vector{1, 2, 3}); got != 3 {
		t.Errorf("Dims([1,2,3]) = %d, want 3", got)
	}
	if got := Dims(Vector{}); got != 0 {
		t.Errorf("Dims([]) = %d, want 0", got)
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		name   string
		a, b   Vector
		expect float32
	}{
		{"simple", Vector{1, 2, 3}, Vector{4, 5, 6}, 32},
		{"zeros", Vector{0, 0}, Vector{1, 2}, 0},
		{"negative", Vector{-1, 2}, Vector{3, -4}, -11},
		{"mismatched", Vector{1, 2}, Vector{1, 2, 3}, 0},
		{"empty", Vector{}, Vector{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			if !approxEqual(got, tt.expect, 1e-6) {
				t.Errorf("Dot() = %v, want %v", got, tt.expect)
			}
		})
	}
}

func TestNorm(t *testing.T) {
	tests := []struct {
		name   string
		v      Vector
		expect float32
	}{
		{"unit x", Vector{1, 0, 0}, 1},
		{"345 triangle", Vector{3, 4}, 5},
		{"zeros", Vector{0, 0, 0}, 0},
		{"empty", Vector{}, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Norm(tt.v)
			if !approxEqual(got, tt.expect, 1e-6) {
				t.Errorf("Norm() = %v, want %v", got, tt.expect)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	v := Vector{3, 4}
	got := Normalize(v)
	expected := Vector{0.6, 0.8}
	if !Equal(got, expected) {
		t.Errorf("Normalize({3,4}) = %v, want %v", got, expected)
	}
	if !approxEqual(Norm(got), 1, 1e-6) {
		t.Errorf("Normalized vector norm = %v, want 1", Norm(got))
	}

	// Zero vector returns nil
	if got := Normalize(Vector{0, 0}); got != nil {
		t.Errorf("Normalize({0,0}) = %v, want nil", got)
	}
}

func TestAdd(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{4, 5, 6}
	got := Add(a, b)
	expected := Vector{5, 7, 9}
	if !Equal(got, expected) {
		t.Errorf("Add() = %v, want %v", got, expected)
	}

	// Mismatched lengths
	if got := Add(Vector{1}, Vector{1, 2}); got != nil {
		t.Errorf("Add(mismatched) = %v, want nil", got)
	}
}

func TestSub(t *testing.T) {
	a := Vector{5, 7, 9}
	b := Vector{4, 5, 6}
	got := Sub(a, b)
	expected := Vector{1, 2, 3}
	if !Equal(got, expected) {
		t.Errorf("Sub() = %v, want %v", got, expected)
	}

	// Mismatched lengths
	if got := Sub(Vector{1}, Vector{1, 2}); got != nil {
		t.Errorf("Sub(mismatched) = %v, want nil", got)
	}
}

func TestScale(t *testing.T) {
	v := Vector{1, 2, 3}
	got := Scale(v, 2)
	expected := Vector{2, 4, 6}
	if !Equal(got, expected) {
		t.Errorf("Scale(x2) = %v, want %v", got, expected)
	}

	got = Scale(v, 0)
	expected = Vector{0, 0, 0}
	if !Equal(got, expected) {
		t.Errorf("Scale(x0) = %v, want %v", got, expected)
	}
}

func TestEqual(t *testing.T) {
	a := Vector{1, 2, 3}
	b := Vector{1, 2, 3}
	if !Equal(a, b) {
		t.Error("Equal() should be true")
	}

	c := Vector{1, 2, 3.1}
	if Equal(a, c) {
		t.Error("Equal() should be false for different vectors")
	}

	d := Vector{1, 2}
	if Equal(a, d) {
		t.Error("Equal() should be false for different lengths")
	}
}

func TestClone(t *testing.T) {
	v := Vector{1, 2, 3}
	c := Clone(v)
	c[0] = 99
	if v[0] == 99 {
		t.Error("Clone() should not share backing array")
	}
	if !Equal(c, Vector{99, 2, 3}) {
		t.Error("Clone() should produce correct copy")
	}
}

func TestEqualEps(t *testing.T) {
	a := Vector{1.0, 2.0}
	b := Vector{1.0001, 1.9999}
	if !EqualEps(a, b, 1e-3) {
		t.Error("should be equal within 1e-3")
	}
	if EqualEps(a, b, 1e-6) {
		t.Error("should not be equal within 1e-6")
	}
}

func approxEqual(a, b, eps float32) bool {
	return math.Abs(float64(a-b)) <= float64(eps)
}
