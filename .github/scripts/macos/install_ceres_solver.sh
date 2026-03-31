#!/bin/bash
set -euo pipefail

brew install eigen glog gflags suite-sparse

wget https://github.com/ceres-solver/ceres-solver/archive/2.2.0.zip
unzip 2.2.0.zip
cd ceres-solver-2.2.0
rm -rf build
mkdir build && cd build
cmake -DCMAKE_INSTALL_PREFIX=/opt/ceres-solver220 -DUSE_CUDA=OFF ..
make -j$(sysctl -n hw.ncpu)
sudo make install
