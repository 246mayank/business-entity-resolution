#!/usr/bin/env bash
# Build <team_name>_submission.zip in the layout the organisers' README asks for:
#   output/{matching_results,candidate_pairs}.tsv, code/business_entity_resolution/, Documentation_template.md
# Validates the two TSVs against dataset/test first and refuses to package a failing submission.
#
#   ./make_submission.sh <team_name>
set -euo pipefail

TEAM="${1:?usage: ./make_submission.sh <team_name>}"
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

MATCH=output/matching_results.tsv
CAND=output/candidate_pairs.tsv
CODE=code/business_entity_resolution
DOC=Documentation_template.md
ZIP="$ROOT/${TEAM}_submission.zip"

for f in "$MATCH" "$CAND" "$DOC" "$CODE/README.md" "$CODE/requirements.txt"; do
  [[ -f "$f" ]] || { echo "missing: $f" >&2; exit 1; }
done
[[ -d "$CODE/src" ]] || { echo "missing: $CODE/src" >&2; exit 1; }

if grep -q "TBD" "$DOC"; then
  echo "WARNING: $DOC still contains TBD placeholders:" >&2
  grep -n "TBD" "$DOC" >&2
fi

if [[ -d dataset/test ]]; then
  python3 utils/validate_submission.py --matching "$MATCH" --candidate "$CAND" --test-dir dataset/test
else
  echo "dataset/test not found: cannot validate, refusing to package" >&2
  exit 1
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/output" "$STAGE/code"
cp "$MATCH" "$CAND" "$STAGE/output/"
cp "$DOC" "$STAGE/"
# src, tests, README, requirements; no venv, caches or bytecode
rsync -a --exclude '.venv' --exclude '__pycache__' --exclude '*.pyc' --exclude '.pytest_cache' \
  "$CODE" "$STAGE/code/"

rm -f "$ZIP"
(cd "$STAGE" && zip -qr "$ZIP" output code "$DOC")

echo
echo "wrote $ZIP ($(du -h "$ZIP" | cut -f1))"
unzip -l "$ZIP" | awk 'NR>3 {print $4}' | grep -E '^(output/|code/business_entity_resolution/[^/]+/?$|Documentation)' | sort
