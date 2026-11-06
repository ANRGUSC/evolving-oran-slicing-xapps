#!/usr/bin/env bash
# Runs the five reproduction scripts. Standard library only; no network.
# Usage: bash reproduce_all.sh        (set PY=/path/to/python to pick an interpreter)
set -u
PY="${PY:-python3}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
rc=0
for d in two_slice editability hardware four_slice source_repair; do
  echo
  echo "================================================================"
  echo "  $d"
  echo "================================================================"
  ( cd "$ROOT/$d" && "$PY" reproduce.py ) || { echo ">>> $d: reproduce.py failed"; rc=1; }
done
echo
if [ $rc -eq 0 ]; then echo "All five reproductions passed."; else echo "At least one reproduction failed (see above)."; fi
exit $rc
