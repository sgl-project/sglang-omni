# Clean and serve documentation with auto-build
set -e

cd -- "$(dirname -- "$0")"
make clean
make serve
