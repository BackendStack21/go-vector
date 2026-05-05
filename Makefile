.PHONY: all build test test-verbose test-cover vet fmt tidy clean demo

all: vet test build

build:
	go build ./...

test:
	go test ./pkg/vector/ -count=1

test-verbose:
	go test ./pkg/vector/ -v -count=1

test-cover:
	go test ./pkg/vector/ -coverprofile=coverage.out
	go tool cover -func=coverage.out

vet:
	go vet ./...

fmt:
	go fmt ./...

tidy:
	go mod tidy

demo:
	go run ./cmd/go-vector/ demo

clean:
	rm -f coverage.out

ci: fmt vet test build
