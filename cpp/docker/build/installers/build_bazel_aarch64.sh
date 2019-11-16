#!/usr/bin/env bash

###############################################################################
# Copyright 2018 The Apollo Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################

# Fail on first error.
set -e

# Install Java8.
add-apt-repository -y ppa:webupd8team/java
echo "oracle-java8-installer shared/accepted-oracle-license-v1-1 select true" | debconf-set-selections
# echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | tee /etc/apt/sources.list.d/bazel.list
# curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -

apt-get update -y
apt-get install -y oracle-java8-installer

# Install Bazel.
cd "$(dirname "${BASH_SOURCE[0]}")"

wget https://github.com/bazelbuild/bazel/releases/download/0.5.3/bazel-0.5.3-dist.zip
mkdir bazel-0.5.3
unzip bazel-0.5.3-dist.zip -d bazel-0.5.3
cd bazel-0.5.3

cat <<'EOF' >>arm64.patch
diff --git a/scripts/bootstrap/buildenv.sh b/scripts/bootstrap/buildenv.sh
index 502f2c1..a2ab4dc 100755
--- a/scripts/bootstrap/buildenv.sh
+++ b/scripts/bootstrap/buildenv.sh
@@ -40,7 +40,7 @@ PLATFORM="$(uname -s | tr 'A-Z' 'a-z')"
 
 MACHINE_TYPE="$(uname -m)"
 MACHINE_IS_64BIT='no'
-if [ "${MACHINE_TYPE}" = 'amd64' -o "${MACHINE_TYPE}" = 'x86_64' -o "${MACHINE_TYPE}" = 's390x' ]; then
+if [ "${MACHINE_TYPE}" = 'amd64' -o "${MACHINE_TYPE}" = 'x86_64' -o "${MACHINE_TYPE}" = 's390x'  -o "${MACHINE_TYPE}" = 'aarch64' ]; then
   MACHINE_IS_64BIT='yes'
 fi
   
diff --git a/third_party/BUILD b/third_party/BUILD
index 9cd2fac..f1cd14c 100755
--- a/third_party/BUILD
+++ b/third_party/BUILD
@@ -583,6 +583,11 @@ config_setting(
 )
 
 config_setting(
+    name = "aarch64",
+    values = {"host_cpu": "aarch64"},
+)
+
+config_setting(
     name = "freebsd",
     values = {"host_cpu": "freebsd"},
 )
EOF

wget https://github.com/bazelbuild/bazel/commit/cc8e7166e29fee39d44e578cf98a06486084a6bd.patch/ -O errorprone.patch

patch -p1 < arm64.patch
patch -p1 < errorprone.patch
./compile.sh
cp output/bazel /usr/local/bin/bazel

# wget https://github.com/bazelbuild/bazel/releases/download/0.5.3/bazel-0.5.3-installer-linux-x86_64.sh
# bash bazel-0.5.3-installer-linux-x86_64.sh

cd "$(dirname "${BASH_SOURCE[0]}")"
# Clean up.
apt-get clean && rm -rf /var/lib/apt/lists/*
rm -fr bazel-0.5.3*
