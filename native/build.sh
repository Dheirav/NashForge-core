#!/usr/bin/env bash

# The module is installed by rename, not by copying over the old file. A copy
# rewrites the .so in place and a process that has it mapped dies with SIGBUS
# (on 13 September that killed a raise-cap-2 solver 80 minutes into its run);
# a rename gives the new file a new inode and a running trainer keeps the old
# one until it exits. So a rebuild under a running training is safe now, and
# the refusal that used to guard it is gone.
# Build the native module into native/build and copy it beside the Python.
#
# The interpreter is passed explicitly: this repo has a venv, and CMake's own
# search will happily find the system Python and build a module that imports and
# then crashes on a subtly different ABI.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
python="${here}/../venv/bin/python"
cmake -S "$here" -B "$here/build" -G Ninja \
      -DPython_EXECUTABLE="$python" -DCMAKE_BUILD_TYPE=Release "$@" >/dev/null
cmake --build "$here/build" --parallel
for so in "$here"/build/pokerbot_native*.so; do
  cp "$so" "$here/../$(basename "$so").tmp" && mv -f "$here/../$(basename "$so").tmp" "$here/../$(basename "$so")"
done
echo "built $(ls "$here"/build/pokerbot_native*.so)"
