package onnx

import (
	"os"
	"path/filepath"
	"testing"
)

// FuzzTokenizerEncode asserts structural invariants of encode on arbitrary
// input: never panics, output bounded by maxLen, always wrapped in
// [CLS]...[SEP], and every ID within the vocabulary range.
func FuzzTokenizerEncode(f *testing.F) {
	path := filepath.Join(f.TempDir(), "vocab.txt")
	vocab := "[PAD]\n[UNK]\n[CLS]\n[SEP]\nthe\ncat\nsat\nun\n##able\n##s\n.\n你\n"
	if err := os.WriteFile(path, []byte(vocab), 0o644); err != nil {
		f.Fatal(err)
	}
	tok, err := loadVocab(path, true)
	if err != nil {
		f.Fatal(err)
	}
	vocabSize := int64(len(tok.vocab))

	f.Add("the cat sat", 16)
	f.Add("thé Ca­ts 你好!!", 8)
	f.Add("", 3)
	f.Add("\x00‍�", 512)
	f.Fuzz(func(t *testing.T, text string, maxLen int) {
		if maxLen < 3 || maxLen > 4096 {
			t.Skip()
		}
		ids := tok.encode(text, maxLen)
		if len(ids) < 2 || len(ids) > maxLen {
			t.Fatalf("len(ids) = %d, want 2..%d", len(ids), maxLen)
		}
		if ids[0] != tok.clsID || ids[len(ids)-1] != tok.sepID {
			t.Fatalf("not wrapped in [CLS]/[SEP]: %v", ids)
		}
		for _, id := range ids {
			if id < 0 || id >= vocabSize {
				t.Fatalf("id %d out of vocab range [0,%d)", id, vocabSize)
			}
		}
	})
}
