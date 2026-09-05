// Package vector provides zero-dependency vector operations, similarity
// metrics, and in-memory nearest-neighbor search in pure Go.
//
// The Vector type is a []float32 — no struct wrapping, no alloc overhead.
// All functions operate on equal-length vectors; mismatched lengths return
// zero/empty results rather than panicking.
//
// # Security & Precision
//
// All operations use float32. Dot products can overflow for vectors with
// large magnitudes (>1e19) or high dimensions (>10⁵ with moderate values).
// Consider normalizing vectors before storage if magnitude safety is required.
// No CGo, no syscalls, no I/O — the attack surface is the Go float32 runtime.
//
// # Performance
//
// Distance computation is O(d) per vector pair where d = dimensionality.
// Store.Search is brute-force O(n·d) for n vectors. Suitable for up to ~100K
// vectors at typical embedding dimensions (384–1536). For larger datasets,
// pair with an approximate index (pkg/hnsw, or FAISS/Annoy) and use this
// for exact re-ranking.
package vector

import "math"

// Vector is a sequence of float32 values.
type Vector []float32

// MaxSafeDims is the maximum recommended dimensionality for float32 dot
// products without overflow risk, assuming normalized or small values (<10.0).
// For unnormalized vectors with large magnitudes, reduce proportionally.
const MaxSafeDims = 1_000_000

// unrollMin is the length at which kernels switch to 4-way accumulation.
// Shorter vectors keep the original sequential loop so existing unit-test
// inputs stay bit-identical.
const unrollMin = 16

// Dims returns the dimensionality of v.
func Dims(v Vector) int { return len(v) }

// Dot returns the dot product of a and b. Returns 0 if lengths differ.
func Dot(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	n := len(a)
	if n < unrollMin {
		var sum float32
		for i := range a {
			sum += a[i] * b[i]
		}
		return sum
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += a[i] * b[i]
		s1 += a[i+1] * b[i+1]
		s2 += a[i+2] * b[i+2]
		s3 += a[i+3] * b[i+3]
	}
	sum := s0 + s1 + s2 + s3
	for ; i < n; i++ {
		sum += a[i] * b[i]
	}
	return sum
}

// Norm returns the L2 (Euclidean) norm of v.
func Norm(v Vector) float32 {
	return float32(math.Sqrt(float64(dotSelf(v))))
}

// dotSelf is Σvᵢ² without a second slice.
func dotSelf(v Vector) float32 {
	n := len(v)
	if n < unrollMin {
		var sum float32
		for _, x := range v {
			sum += x * x
		}
		return sum
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		s0 += v[i] * v[i]
		s1 += v[i+1] * v[i+1]
		s2 += v[i+2] * v[i+2]
		s3 += v[i+3] * v[i+3]
	}
	sum := s0 + s1 + s2 + s3
	for ; i < n; i++ {
		sum += v[i] * v[i]
	}
	return sum
}

// Normalize returns a unit vector in the direction of v.
// Returns nil for the zero vector.
func Normalize(v Vector) Vector {
	return NormalizeIn(nil, v)
}

// NormalizeIn is Normalize writing into dst. dst may be nil or too short, in
// which case a new slice is allocated. dst may alias v.
func NormalizeIn(dst, v Vector) Vector {
	n := Norm(v)
	if n == 0 {
		return nil
	}
	return ScaleIn(dst, v, 1/n)
}

// Add returns element-wise sum a + b. Returns nil if lengths differ.
func Add(a, b Vector) Vector {
	return AddIn(nil, a, b)
}

// AddIn is Add writing into dst. Returns nil if lengths differ.
func AddIn(dst, a, b Vector) Vector {
	if len(a) != len(b) {
		return nil
	}
	dst = resize(dst, len(a))
	for i := range a {
		dst[i] = a[i] + b[i]
	}
	return dst
}

// Sub returns element-wise difference a - b. Returns nil if lengths differ.
func Sub(a, b Vector) Vector {
	return SubIn(nil, a, b)
}

// SubIn is Sub writing into dst. Returns nil if lengths differ.
func SubIn(dst, a, b Vector) Vector {
	if len(a) != len(b) {
		return nil
	}
	dst = resize(dst, len(a))
	for i := range a {
		dst[i] = a[i] - b[i]
	}
	return dst
}

// Scale returns v multiplied by scalar s.
func Scale(v Vector, s float32) Vector {
	return ScaleIn(nil, v, s)
}

// ScaleIn is Scale writing into dst. dst may alias v.
func ScaleIn(dst, v Vector, s float32) Vector {
	dst = resize(dst, len(v))
	for i := range v {
		dst[i] = v[i] * s
	}
	return dst
}

func resize(dst Vector, n int) Vector {
	if cap(dst) < n {
		return make(Vector, n)
	}
	return dst[:n]
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
