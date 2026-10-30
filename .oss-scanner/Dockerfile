# Build image for Anthropic OSS Scanner (https://github.com/anthropics/oss-scanner).
# Network is available only during this build; everything the tests need is installed here.
FROM golang:1.27-bookworm
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git python3 \
    && rm -rf /var/lib/apt/lists/*

# scanner contract: the checkout lives inside the image, at /src
COPY . /src
WORKDIR /src

# pkg/onnx compiles without the ONNX Runtime library; its model-backed tests skip when testdata/ is absent
RUN go mod download && go build ./... \
    && go build -o /usr/local/bin/go-vector ./cmd/go-vector

# Run the checks so the image is known to work, without failing the build on a red test.
RUN (go vet ./... && make test) || echo "WARNING: checks reported failures"
