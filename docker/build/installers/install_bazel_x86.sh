set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

wget https://github.com/bazelbuild/bazel/releases/download/0.23.2/bazel-0.23.2-installer-linux-x86_64.sh &&
    bash bazel-0.23.2-installer-linux-x86_64.sh

