.PHONY: all build test test-verbose test-cover vet fmt tidy clean demo demo-onnx model

# Pinned revision of sentence-transformers/all-MiniLM-L6-v2; downloads are
# verified against pkg/onnx/model.sha256
MODEL_REV  := 1110a243fdf4706b3f48f1d95db1a4f5529b4d41
MODEL_REPO := https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/$(MODEL_REV)

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
	cd pkg/onnx/testdata && shasum -a 256 -c ../model.sha256

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
