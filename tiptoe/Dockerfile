FROM --platform=linux/amd64 rust:1.76-slim-bullseye as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies in venv
RUN pip3 install requests

# Set up the workspace structure
WORKDIR /workspace
COPY . .

# Build the project for x86_64 Linux
ENV RUSTFLAGS="-C target-feature=+crt-static"
RUN rustup target add x86_64-unknown-linux-gnu
RUN cargo build --release --target x86_64-unknown-linux-gnu

# Runtime stage
FROM --platform=linux/amd64 debian:bullseye-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libssl-dev \
    python3 \
    python3-venv \
    netcat \
    && rm -rf /var/lib/apt/lists/*

# Set up Python virtual environment and install dependencies
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip3 install requests

WORKDIR /app

# Copy the built binary and necessary files
COPY --from=builder /workspace/target/x86_64-unknown-linux-gnu/release/encoding_server /app/encoding_server
COPY --from=builder /workspace/target/x86_64-unknown-linux-gnu/release/embedding_server /app/embedding_server
COPY --from=builder /workspace/tiptoe/src/python /app/src/python

# Set environment variables
ENV CARGO_MANIFEST_DIR=/app
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Default command (will be overridden by CDK)
CMD ["./encoding_server"] 