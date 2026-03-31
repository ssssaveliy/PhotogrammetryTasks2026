$ErrorActionPreference = "Stop"
$INSTALL_PREFIX = "C:\eigen-3.4.0"

Invoke-WebRequest -Uri "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip" -OutFile "eigen-3.4.0.zip"
Expand-Archive -Path "eigen-3.4.0.zip" -DestinationPath "."
cd eigen-3.4.0
cmake -S . -B _build -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --install _build
if ($LASTEXITCODE -ne 0) { throw "cmake install failed" }
