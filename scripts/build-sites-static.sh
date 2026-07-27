#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
site_dir="$project_dir/_site"
dist_dir="$project_dir/dist"
worker_source="$project_dir/sites/worker.js"

test -f "$site_dir/index.html" || {
  echo "Сначала соберите Quarto-сайт: make render" >&2
  exit 2
}
test -f "$worker_source"
test "$dist_dir" = "$project_dir/dist"

stage="$(mktemp -d)"
trap 'rm -rf "$stage"' EXIT

mkdir -p "$stage/dist/client" "$stage/dist/server"
cp -R "$site_dir"/. "$stage/dist/client"/
cp "$worker_source" "$stage/dist/server/index.js"

rm -rf "$dist_dir"
mv "$stage/dist" "$dist_dir"
