.PHONY: all build test test-verbose test-cover vet fmt tidy clean demo demo-onnx model

MODEL_REPO := https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main

all: vet test build

build:
	go build ./...

test:
	go test ./pkg/vector/ ./pkg/onnx/ -count=1

test-verbose:
	go test ./pkg/vector/ ./pkg/onnx/ -v -count=1

test-cover:
	go test ./pkg/vector/ ./pkg/onnx/ -coverprofile=coverage.out
	go tool cover -func=coverage.out

# Download all-MiniLM-L6-v2 for pkg/onnx tests (skipped when absent)
model:
	mkdir -p pkg/onnx/testdata
	curl -fL -o pkg/onnx/testdata/model.onnx $(MODEL_REPO)/onnx/model.onnx
	curl -fL -o pkg/onnx/testdata/vocab.txt $(MODEL_REPO)/vocab.txt

vet:
	go vet ./...

fmt:
	go fmt ./...

tidy:
	go mod tidy

demo:
	go run ./cmd/go-vector/ demo

# Semantic search demo with a local ONNX model (run `make model` first)
demo-onnx:
	go run ./cmd/onnx-demo/

clean:
	rm -f coverage.out

ci: fmt vet test build
