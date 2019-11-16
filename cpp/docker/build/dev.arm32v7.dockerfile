#FROM arm64v8/ubuntu:18.04
FROM ubuntu:18.04

# Be sure to install any runtime dependencies
RUN apt-get clean && apt-get update && apt-get install -y \
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

RUN bash /tmp/installers/install_bazel_x86.sh
