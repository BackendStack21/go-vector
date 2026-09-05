package quantize

import (
	"math"
	"testing"

	"github.com/BackendStack21/go-vector/pkg/vector"
)

func TestInt8Roundtrip(t *testing.T) {
	v := vector.Vector{0.5, -0.25, 1, 0}
	q, scale := ToInt8(v)
	back := FromInt8(q, scale)
	if len(back) != len(v) {
		t.Fatal(len(back))
	}
	for i := range v {
		if math.Abs(float64(back[i]-v[i])) > 0.02 {
			t.Fatalf("dim %d: %v vs %v", i, back[i], v[i])
		}
	}
	if DotInt8(q, q, scale, scale) <= 0 {
		t.Fatal("self dot")
	}
	if DotInt8(q, q[:1], scale, scale) != 0 {
		t.Fatal("mismatch")
	}
	z, sc := ToInt8(vector.Vector{0, 0})
	if sc != 0 || z[0] != 0 {
		t.Fatal("zero")
	}
}

func TestInt8StoreRanking(t *testing.T) {
	s := NewInt8Store(vector.CosineDistance)
	s.Add("near", vector.Vector{1, 0, 0})
	s.Add("mid", vector.Vector{0.7, 0.7, 0})
	s.Add("far", vector.Vector{0, 1, 0})
	s.Add("bad", vector.Vector{1}) // mismatched, ignored
	if s.Len() != 3 {
		t.Fatalf("len=%d", s.Len())
	}
	hits := s.Search(vector.Vector{1, 0.05, 0}, 2)
	if hits[0].ID != "near" {
		t.Fatalf("%+v", hits)
	}
	if s.Search(vector.Vector{1}, 0) != nil {
		t.Fatal("k<=0")
	}
}
