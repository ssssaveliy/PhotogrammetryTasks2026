#!/bin/bash
set -euo pipefail

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xzf eigen-3.4.0.tar.gz
cd eigen-3.4.0
cmake -B build -DCMAKE_INSTALL_PREFIX=/opt/eigen3
sudo cmake --build build --target install
