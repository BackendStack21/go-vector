package main

import (
	"io"
	"strings"
	"testing"
)

func TestReadJSONL(t *testing.T) {
	docs, err := readJSONL(strings.NewReader("{\"id\":\"a\",\"text\":\"hello\"}\n\n{\"id\":\"b\",\"vector\":[1,2]}\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 2 || docs[0].ID != "a" || docs[1].ID != "b" {
		t.Fatalf("docs = %+v", docs)
	}
}

func TestReadJSONLRejectsBadJSON(t *testing.T) {
	if _, err := readJSONL(strings.NewReader("{not json}\n")); err == nil {
		t.Fatal("expected JSON error")
	}
}

type errReader struct{}

func (errReader) Read([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestReadJSONLReportsScannerError(t *testing.T) {
	if _, err := readJSONL(errReader{}); err == nil {
		t.Fatal("expected scanner I/O error")
	}
}

func TestReadJSONLLongLine(t *testing.T) {
	text := strings.Repeat("a", 100000)
	docs, err := readJSONL(strings.NewReader(`{"id":"big","text":"` + text + `"}` + "\n"))
	if err != nil {
		t.Fatal(err)
	}
	if len(docs) != 1 || len(docs[0].Text) != 100000 {
		t.Fatalf("got %+v", docs)
	}
}
