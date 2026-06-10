package onnx

import (
	"bufio"
	"fmt"
	"os"
	"strings"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// wordPieceTokenizer implements the BERT uncased tokenization scheme:
// basic tokenization (clean, lowercase, strip accents, split punctuation
// and CJK) followed by greedy longest-match-first WordPiece against a
// vocab.txt vocabulary. This is the tokenizer used by BERT-family
// sentence-transformers models such as all-MiniLM-L6-v2.
type wordPieceTokenizer struct {
	vocab map[string]int64
	unkID int64
	clsID int64
	sepID int64
	padID int64
	lower bool
}

// maxWordPieceChars mirrors BERT's max_input_chars_per_word (100): longer
// words map straight to [UNK] instead of being split.
const maxWordPieceChars = 100

// loadVocab reads a BERT vocab.txt (one token per line, ID = line index).
func loadVocab(path string, lower bool) (*wordPieceTokenizer, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("onnx: open vocab: %w", err)
	}
	defer f.Close()

	t := &wordPieceTokenizer{vocab: make(map[string]int64), lower: lower}
	scanner := bufio.NewScanner(f)
	var id int64
	for scanner.Scan() {
		tok := strings.TrimRight(scanner.Text(), "\r\n")
		if _, dup := t.vocab[tok]; !dup {
			t.vocab[tok] = id
		}
		id++
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("onnx: read vocab: %w", err)
	}

	for _, special := range []struct {
		name string
		dst  *int64
	}{
		{"[UNK]", &t.unkID},
		{"[CLS]", &t.clsID},
		{"[SEP]", &t.sepID},
		{"[PAD]", &t.padID},
	} {
		v, ok := t.vocab[special.name]
		if !ok {
			return nil, fmt.Errorf("onnx: vocab is missing required token %s", special.name)
		}
		*special.dst = v
	}
	return t, nil
}

// encode converts text to token IDs wrapped in [CLS]...[SEP], truncated so
// the result never exceeds maxLen IDs.
func (t *wordPieceTokenizer) encode(text string, maxLen int) []int64 {
	words := t.basicTokenize(text)

	ids := make([]int64, 0, len(words)+2)
	ids = append(ids, t.clsID)
	budget := maxLen - 2 // room for [CLS] and [SEP]
	for _, w := range words {
		pieces := t.wordPiece(w)
		if len(ids)-1+len(pieces) > budget {
			remaining := budget - (len(ids) - 1)
			if remaining <= 0 {
				break
			}
			pieces = pieces[:remaining]
		}
		ids = append(ids, pieces...)
	}
	return append(ids, t.sepID)
}

// basicTokenize cleans, normalizes, and splits text into words, with
// punctuation and CJK characters isolated as single-character words.
func (t *wordPieceTokenizer) basicTokenize(text string) []string {
	if t.lower {
		// BERT uncased: lowercase, then strip combining marks (accents)
		// from the NFD decomposition.
		text = strings.ToLower(text)
		text = strings.Map(func(r rune) rune {
			if unicode.Is(unicode.Mn, r) {
				return -1
			}
			return r
		}, norm.NFD.String(text))
	}

	var words []string
	var cur strings.Builder
	flush := func() {
		if cur.Len() > 0 {
			words = append(words, cur.String())
			cur.Reset()
		}
	}
	for _, r := range text {
		switch {
		case r == 0 || r == 0xFFFD || unicode.IsControl(r) || unicode.Is(unicode.Cf, r):
			// drop invalid, control, and format characters (ZWJ, soft
			// hyphen, BOM, …) like HF's _clean_text; whitespace-class
			// controls (tab/newline) still split words
			if unicode.IsSpace(r) {
				flush()
			}
		case unicode.IsSpace(r):
			flush()
		case isBertPunct(r) || isCJK(r):
			flush()
			words = append(words, string(r))
		default:
			cur.WriteRune(r)
		}
	}
	flush()
	return words
}

// wordPiece splits one word into subword IDs by greedy longest-match-first,
// prefixing continuations with "##". Unknown words become [UNK].
func (t *wordPieceTokenizer) wordPiece(word string) []int64 {
	runes := []rune(word)
	if len(runes) > maxWordPieceChars {
		return []int64{t.unkID}
	}

	var ids []int64
	start := 0
	for start < len(runes) {
		end := len(runes)
		var match int64 = -1
		for end > start {
			sub := string(runes[start:end])
			if start > 0 {
				sub = "##" + sub
			}
			if id, ok := t.vocab[sub]; ok {
				match = id
				break
			}
			end--
		}
		if match < 0 {
			return []int64{t.unkID}
		}
		ids = append(ids, match)
		start = end
	}
	return ids
}

// isBertPunct reports whether r is punctuation per BERT's definition:
// the four ASCII symbol ranges plus the Unicode P category.
func isBertPunct(r rune) bool {
	if (r >= 33 && r <= 47) || (r >= 58 && r <= 64) || (r >= 91 && r <= 96) || (r >= 123 && r <= 126) {
		return true
	}
	return unicode.IsPunct(r)
}

// isCJK reports whether r falls in the CJK ideograph blocks that BERT
// tokenizes character-by-character.
func isCJK(r rune) bool {
	return (r >= 0x4E00 && r <= 0x9FFF) ||
		(r >= 0x3400 && r <= 0x4DBF) ||
		(r >= 0x20000 && r <= 0x2A6DF) ||
		(r >= 0x2A700 && r <= 0x2B73F) ||
		(r >= 0x2B740 && r <= 0x2B81F) ||
		(r >= 0x2B820 && r <= 0x2CEAF) ||
		(r >= 0xF900 && r <= 0xFAFF) ||
		(r >= 0x2F800 && r <= 0x2FA1F)
}
