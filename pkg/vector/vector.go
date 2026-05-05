// Package vector provides zero-dependency vector operations, similarity
// metrics, and in-memory nearest-neighbor search in pure Go.
//
// The Vector type is a []float32 — no struct wrapping, no alloc overhead.
// All functions operate on equal-length vectors; mismatched lengths return
// zero/empty results rather than panicking.
package vector

import "math"

// Vector is a sequence of float32 values.
type Vector []float32

// Dot returns the dot product of a and b. Returns 0 if lengths differ.
func Dot(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm returns the L2 (Euclidean) norm of v.
func Norm(v Vector) float32 {
	return float32(math.Sqrt(float64(Dot(v, v))))
}

// Normalize returns a unit vector in the direction of v.
// Returns nil for the zero vector.
func Normalize(v Vector) Vector {
	n := Norm(v)
	if n == 0 {
		return nil
	}
	return Scale(v, 1/n)
}

// Add returns element-wise sum a + b. Returns nil if lengths differ.
func Add(a, b Vector) Vector {
	if len(a) != len(b) {
		return nil
	}
	out := make(Vector, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

// Sub returns element-wise difference a - b. Returns nil if lengths differ.
func Sub(a, b Vector) Vector {
	if len(a) != len(b) {
		return nil
	}
	out := make(Vector, len(a))
	for i := range a {
		out[i] = a[i] - b[i]
	}
	return out
}

// Scale returns v multiplied by scalar s.
func Scale(v Vector, s float32) Vector {
	out := make(Vector, len(v))
	for i := range v {
		out[i] = v[i] * s
	}
	return out
}

// Equal reports whether a and b are approximately equal within epsilon
// (default 1e-6). Vectors of different lengths are never equal.
func Equal(a, b Vector) bool {
	return EqualEps(a, b, 1e-6)
}

// EqualEps reports whether a and b are approximately equal within eps.
func EqualEps(a, b Vector, eps float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Abs(float64(a[i]-b[i])) > float64(eps) {
			return false
		}
	}
	return true
}

// Clone returns a copy of v.
func Clone(v Vector) Vector {
	out := make(Vector, len(v))
	copy(out, v)
	return out
}
