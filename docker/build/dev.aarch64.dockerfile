#FROM ubuntu:18.04
FROM arm64v8/ubuntu:18.04

# Be sure to install any runtime dependencies
RUN apt-get clean && apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    apt-utils \
    pkg-config \
    zip \
    g++ \
    zlib1g-dev \
    unzip \
    python \
    wget

COPY installers /tmp/installers
RUN bash /tmp/installers/build_bazel_aarch64.sh
