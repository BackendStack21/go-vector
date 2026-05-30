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

	// DotProductSimilarity is the raw dot product. Range (-∞, ∞). Higher = more similar.
	// Best used with normalized vectors.
	DotProductSimilarity
)

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
	var dot, na, nb float32
	for i := range a {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
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
	var sum float64
	for i := range a {
		d := float64(a[i] - b[i])
		sum += d * d
	}
	return float32(math.Sqrt(sum))
}

// Manhattan returns the Manhattan (L1) distance between a and b.
// Returns 0 if lengths differ.
func Manhattan(a, b Vector) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		if d < 0 {
			d = -d
		}
		sum += d
	}
	return sum
}

// Distance computes the distance/similarity between a and b using the given metric.
// For CosineDistance/EuclideanDistance/ManhattanDistance, lower is more similar.
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
	default:
		return 0
	}
}
