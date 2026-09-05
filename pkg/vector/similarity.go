package vector

import "math"

// Metric identifies a distance or similarity function for nearest-neighbor search.
type Metric int

const (
	// CosineDistance is 1 - cosine similarity. Range [0, 2]. Lower = more similar.
	CosineDistance Metric = iota

	// EuclideanDistance is the straight-line distance. Range [0, ∞). Lower = more similar.
	EuclideanDistance

	// ManhattanDistance is the L1 / city-block distance. Range [0, ∞). Lower = more similar.
	ManhattanDistance

	// DotProductSimilarity is the raw dot product. Range (−∞, ∞). Higher = more similar.
	// Best used with normalized vectors.
	DotProductSimilarity

	// ChebyshevDistance is the L∞ / max-norm distance. Range [0, ∞). Lower = more similar.
	ChebyshevDistance

	// HammingDistance counts differing positions (exact float32 compare).
	// Range [0, d]. Lower = more similar.
	HammingDistance
)

// String returns a stable lowercase name for m.
func (m Metric) String() string {
	switch m {
	case CosineDistance:
		return "cosine"
	case EuclideanDistance:
		return "euclidean"
	case ManhattanDistance:
		return "manhattan"
	case DotProductSimilarity:
		return "dot"
	case ChebyshevDistance:
		return "chebyshev"
	case HammingDistance:
		return "hamming"
	default:
		return "unknown"
	}
}

// Ascending reports whether this metric is "lower is better" (true for distances,
// false for similarities like dot product).
func (m Metric) Ascending() bool {
	return m != DotProductSimilarity
}

// Cosine returns the cosine similarity of a and b (range [-1, 1]).
// Returns 0 if either vector is zero-length or lengths differ.
func Cosine(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	n := len(a)
	var dot, na, nb float32
	if n < unrollMin {
		for i := range a {
			dot += a[i] * b[i]
			na += a[i] * a[i]
			nb += b[i] * b[i]
		}
	} else {
		var d0, d1, d2, d3, a0, a1, a2, a3, b0, b1, b2, b3 float32
		i := 0
		for ; i+4 <= n; i += 4 {
			d0 += a[i] * b[i]
			d1 += a[i+1] * b[i+1]
			d2 += a[i+2] * b[i+2]
			d3 += a[i+3] * b[i+3]
			a0 += a[i] * a[i]
			a1 += a[i+1] * a[i+1]
			a2 += a[i+2] * a[i+2]
			a3 += a[i+3] * a[i+3]
			b0 += b[i] * b[i]
			b1 += b[i+1] * b[i+1]
			b2 += b[i+2] * b[i+2]
			b3 += b[i+3] * b[i+3]
		}
		dot, na, nb = d0+d1+d2+d3, a0+a1+a2+a3, b0+b1+b2+b3
		for ; i < n; i++ {
			dot += a[i] * b[i]
			na += a[i] * a[i]
			nb += b[i] * b[i]
		}
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / float32(math.Sqrt(float64(na)*float64(nb)))
}

// CosineDist returns 1 - Cosine(a, b). Range [0, 2]. Lower = more similar.
func CosineDist(a, b Vector) float32 {
	return 1 - Cosine(a, b)
}

// Euclidean returns the Euclidean (L2) distance between a and b.
// Zero-allocation: computes directly without intermediate vectors. Returns 0 if lengths differ.
func Euclidean(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	n := len(a)
	var sum float64
	if n < unrollMin {
		for i := range a {
			d := float64(a[i] - b[i])
			sum += d * d
		}
	} else {
		var s0, s1, s2, s3 float64
		i := 0
		for ; i+4 <= n; i += 4 {
			d0 := float64(a[i] - b[i])
			d1 := float64(a[i+1] - b[i+1])
			d2 := float64(a[i+2] - b[i+2])
			d3 := float64(a[i+3] - b[i+3])
			s0 += d0 * d0
			s1 += d1 * d1
			s2 += d2 * d2
			s3 += d3 * d3
		}
		sum = s0 + s1 + s2 + s3
		for ; i < n; i++ {
			d := float64(a[i] - b[i])
			sum += d * d
		}
	}
	return float32(math.Sqrt(sum))
}

// Manhattan returns the Manhattan (L1) distance between a and b.
// Returns 0 if lengths differ.
func Manhattan(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	n := len(a)
	var sum float32
	if n < unrollMin {
		for i := range a {
			d := a[i] - b[i]
			if d < 0 {
				d = -d
			}
			sum += d
		}
		return sum
	}
	var s0, s1, s2, s3 float32
	i := 0
	for ; i+4 <= n; i += 4 {
		d0 := a[i] - b[i]
		d1 := a[i+1] - b[i+1]
		d2 := a[i+2] - b[i+2]
		d3 := a[i+3] - b[i+3]
		if d0 < 0 {
			d0 = -d0
		}
		if d1 < 0 {
			d1 = -d1
		}
		if d2 < 0 {
			d2 = -d2
		}
		if d3 < 0 {
			d3 = -d3
		}
		s0 += d0
		s1 += d1
		s2 += d2
		s3 += d3
	}
	sum = s0 + s1 + s2 + s3
	for ; i < n; i++ {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return sum
}

// Chebyshev returns the L∞ distance (max absolute component difference).
// Returns 0 if lengths differ.
func Chebyshev(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	var max float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		if d > max {
			max = d
		}
	}
	return max
}

// Hamming returns the number of positions where a and b differ (exact
// float32 compare). Returns 0 if lengths differ.
func Hamming(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	var n float32
	for i := range a {
		if a[i] != b[i] {
			n++
		}
	}
	return n
}

// Distance computes the distance/similarity between a and b using the given metric.
// For distance metrics, lower is more similar.
// For DotProductSimilarity, higher is more similar.
func Distance(a, b Vector, m Metric) float32 {
	switch m {
	case CosineDistance:
		return CosineDist(a, b)
	case EuclideanDistance:
		return Euclidean(a, b)
	case ManhattanDistance:
		return Manhattan(a, b)
	case DotProductSimilarity:
		return Dot(a, b)
	case ChebyshevDistance:
		return Chebyshev(a, b)
	case HammingDistance:
		return Hamming(a, b)
	default:
		return 0
	}
}

// Hybrid blends a vector score with a lexical (or other) score.
// alpha is the weight on vectorScore and is clamped to [0, 1]:
//
//	alpha*vectorScore + (1-alpha)*lexicalScore
func Hybrid(vectorScore, lexicalScore, alpha float32) float32 {
	if alpha < 0 {
		alpha = 0
	}
	if alpha > 1 {
		alpha = 1
	}
	return alpha*vectorScore + (1-alpha)*lexicalScore
}
