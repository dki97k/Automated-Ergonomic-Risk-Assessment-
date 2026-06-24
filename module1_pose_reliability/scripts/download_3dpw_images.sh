#!/usr/bin/env bash

set -u

url="https://virtualhumans.mpi-inf.mpg.de/3DPW/imageFiles.zip"
destination="${1:-../data/3dpw/downloads/imageFiles.zip}"
expected_size=4610672154

mkdir -p "$(dirname "$destination")"

while true; do
  current_size=0
  if [[ -f "$destination" ]]; then
    current_size="$(stat -f%z "$destination")"
  fi

  if (( current_size == expected_size )); then
    echo "Download complete: $destination ($current_size bytes)"
    break
  fi

  if (( current_size > expected_size )); then
    echo "Refusing to continue: partial file exceeds expected size." >&2
    exit 1
  fi

  echo "Resuming at byte $current_size of $expected_size"
  curl --fail --location --continue-at - --output "$destination" "$url" || true
  sleep 3
done

