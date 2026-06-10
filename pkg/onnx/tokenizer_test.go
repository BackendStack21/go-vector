package onnx

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

// writeVocab writes a vocab file and returns a tokenizer loaded from it.
func writeVocab(t *testing.T, tokens []string) *wordPieceTokenizer {
	t.Helper()
	path := filepath.Join(t.TempDir(), "vocab.txt")
	content := strings.Join(tokens, "\n") + "\n"
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
	tok, err := loadVocab(path, true)
	if err != nil {
		t.Fatalf("loadVocab: %v", err)
	}
	return tok
}

// testVocab: [PAD]=0 [UNK]=1 [CLS]=2 [SEP]=3 the=4 cat=5 sat=6 un=7
// ##able=8 ##s=9 .=10
var testVocabTokens = []string{"[PAD]", "[UNK]", "[CLS]", "[SEP]", "the", "cat", "sat", "un", "##able", "##s", "."}

func TestTokenizerEncode(t *testing.T) {
	tok := writeVocab(t, testVocabTokens)

	cases := []struct {
		text string
		want []int64
	}{
		{"the cat sat", []int64{2, 4, 5, 6, 3}},
		{"The CAT", []int64{2, 4, 5, 3}},      // lowercased
		{"thé cat", []int64{2, 4, 5, 3}},      // accent stripped
		{"cats", []int64{2, 5, 9, 3}},         // WordPiece: cat + ##s
		{"unable", []int64{2, 7, 8, 3}},       // un + ##able
		{"the cat.", []int64{2, 4, 5, 10, 3}}, // punctuation split
		{"zebra", []int64{2, 1, 3}},           // unknown → [UNK]
		{"  the\tcat\n", []int64{2, 4, 5, 3}}, // whitespace handling
		{"", []int64{2, 3}},                   // empty → just specials
		{"the unable. cats", []int64{2, 4, 7, 8, 10, 5, 9, 3}},
	}
	for _, c := range cases {
		if got := tok.encode(c.text, 128); !reflect.DeepEqual(got, c.want) {
			t.Errorf("encode(%q) = %v, want %v", c.text, got, c.want)
		}
	}
}

func TestTokenizerTruncation(t *testing.T) {
	tok := writeVocab(t, testVocabTokens)
	got := tok.encode("the cat sat the cat sat", 5)
	// [CLS] + 3 tokens + [SEP]
	want := []int64{2, 4, 5, 6, 3}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("truncated encode = %v, want %v", got, want)
	}
}

func TestTokenizerLongWordIsUnk(t *testing.T) {
	tok := writeVocab(t, testVocabTokens)
	long := strings.Repeat("a", maxWordPieceChars+1)
	if got := tok.encode(long, 16); !reflect.DeepEqual(got, []int64{2, 1, 3}) {
		t.Errorf("overlong word = %v, want [CLS] [UNK] [SEP]", got)
	}
}

func TestTokenizerPartialMatchIsUnk(t *testing.T) {
	tok := writeVocab(t, testVocabTokens)
	// "catx": "cat" matches but "##x" doesn't → whole word becomes [UNK].
	if got := tok.encode("catx", 16); !reflect.DeepEqual(got, []int64{2, 1, 3}) {
		t.Errorf("partial match = %v, want [CLS] [UNK] [SEP]", got)
	}
}

func TestTokenizerFormatCharsStripped(t *testing.T) {
	tok := writeVocab(t, testVocabTokens)
	// Soft hyphen (U+00AD) and zero-width joiner (U+200D) are Unicode
	// format (Cf) characters; HF's BERT tokenizer removes them entirely,
	// joining the surrounding letters.
	if got := tok.encode("ca­ts", 16); !reflect.DeepEqual(got, []int64{2, 5, 9, 3}) {
		t.Errorf("soft hyphen: got %v, want cat+##s [2 5 9 3]", got)
	}
	if got := tok.encode("cat‍s", 16); !reflect.DeepEqual(got, []int64{2, 5, 9, 3}) {
		t.Errorf("ZWJ: got %v, want cat+##s [2 5 9 3]", got)
	}
}

func TestTokenizerCJKSplit(t *testing.T) {
	tok := writeVocab(t, append(testVocabTokens, "你", "好"))
	if got := tok.encode("你好", 16); len(got) != 4 {
		t.Errorf("CJK should split per character: got %d ids %v, want 4", len(got), got)
	}
}

func TestTokenizerCasedMode(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vocab.txt")
	os.WriteFile(path, []byte("[PAD]\n[UNK]\n[CLS]\n[SEP]\nCat\n"), 0o644)
	tok, err := loadVocab(path, false)
	if err != nil {
		t.Fatal(err)
	}
	if got := tok.encode("Cat", 16); !reflect.DeepEqual(got, []int64{2, 4, 3}) {
		t.Errorf("cased encode = %v, want [2 4 3]", got)
	}
	if got := tok.encode("cat", 16); !reflect.DeepEqual(got, []int64{2, 1, 3}) {
		t.Errorf("cased encode of lowercase = %v, want [UNK]", got)
	}
}

func TestTokenizerMissingSpecial(t *testing.T) {
	path := filepath.Join(t.TempDir(), "vocab.txt")
	os.WriteFile(path, []byte("the\ncat\n"), 0o644)
	if _, err := loadVocab(path, true); err == nil {
		t.Error("want error for vocab missing special tokens, got nil")
	}
}

func TestTokenizerMissingFile(t *testing.T) {
	if _, err := loadVocab(filepath.Join(t.TempDir(), "nope.txt"), true); err == nil {
		t.Error("want error for missing vocab file, got nil")
	}
}
