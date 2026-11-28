# Clean and serve documentation with auto-build
set -e
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
make clean
make serve
