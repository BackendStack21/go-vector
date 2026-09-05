// Package quantize reduces vector memory with int8 scaling.
// It is a sibling of pkg/vector so the core store stays exact float32.
package quantize

import (
	"math"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

// ToInt8 quantizes v to int8 with a single scale so that
// v[i] ≈ q[i] * scale. scale is max(|v|)/127, or 0 for a zero vector.
func ToInt8(v vector.Vector) (q []int8, scale float32) {
	if len(v) == 0 {
		return []int8{}, 0
	}
	var max float32
	for _, x := range v {
		a := x
		if a < 0 {
			a = -a
		}
		if a > max {
			max = a
		}
	}
	if max == 0 {
		return make([]int8, len(v)), 0
	}
	scale = max / 127
	q = make([]int8, len(v))
	inv := float32(1)
	if scale > 0 {
		inv = 1 / scale
	}
	for i, x := range v {
		r := x * inv
		if r > 127 {
			r = 127
		}
		if r < -127 {
			r = -127
		}
		q[i] = int8(math.Round(float64(r)))
	}
	return q, scale
}

// FromInt8 reconstructs an approximate float32 vector.
func FromInt8(q []int8, scale float32) vector.Vector {
	out := make(vector.Vector, len(q))
	for i, x := range q {
		out[i] = float32(x) * scale
	}
	return out
}

// DotInt8 is the dequantized dot product of two int8 vectors.
// Returns 0 if lengths differ.
func DotInt8(a, b []int8, scaleA, scaleB float32) float32 {
	if len(a) != len(b) {
		return 0
	}
	var sum int32
	for i := range a {
		sum += int32(a[i]) * int32(b[i])
	}
	return float32(sum) * scaleA * scaleB
}

// Int8Store is a packed int8 index with brute-force search using
// dequantized dot products / reconstructed distances.
type Int8Store struct {
	ids    []string
	data   []int8
	scales []float32
	dims   int
	metric vector.Metric
}

// NewInt8Store creates an empty quantized store.
func NewInt8Store(metric vector.Metric) *Int8Store {
	return &Int8Store{metric: metric}
}

// Add quantizes v and appends it. The first vector sets dims; later
// mismatched lengths are ignored.
func (s *Int8Store) Add(id string, v vector.Vector) {
	q, scale := ToInt8(v)
	if s.dims == 0 {
		s.dims = len(q)
	}
	if len(q) != s.dims {
		return
	}
	s.ids = append(s.ids, id)
	s.data = append(s.data, q...)
	s.scales = append(s.scales, scale)
}

// Len returns the number of stored vectors.
func (s *Int8Store) Len() int { return len(s.ids) }

// Search reconstructs each row and scores with vector.Distance.
func (s *Int8Store) Search(query vector.Vector, k int) []vector.SearchResult {
	n := len(s.ids)
	if k <= 0 || n == 0 || s.dims == 0 {
		return nil
	}
	tmp := vector.NewStore(s.metric)
	for i := 0; i < n; i++ {
		row := s.data[i*s.dims : (i+1)*s.dims]
		tmp.Add(s.ids[i], FromInt8(row, s.scales[i]))
	}
	return tmp.Search(query, k)
}
