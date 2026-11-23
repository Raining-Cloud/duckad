#!/usr/bin/env bash
#
# Download the ImageNet-1k validation images (~6.7GB) from Kaggle into imagenet-val/ as *.JPEG,
# for data.background_aug and optim.imagenet_stop_prob.
#
# Credentials: KAGGLE_API_TOKEN (kaggle.json contents or bare key + KAGGLE_USERNAME),
# KAGGLE_USERNAME + KAGGLE_KEY, ~/.kaggle/kaggle.json, or `kaggle auth login`.
#
# Usage: [KAGGLE_SLUG=<user>/<dataset>] ./scripts/setup/fetch_imagenet_kaggle.sh [dest]
set -euo pipefail

DEST="${1:-imagenet-val}"
SLUG="${KAGGLE_SLUG:-titericz/imagenet1k-val}"
MIN_IMAGES="${MIN_IMAGES:-2000}"

# The loader's glob is case-sensitive.
count_jpeg() { find "$DEST" -name '*.JPEG' 2>/dev/null | wc -l; }

mkdir -p "$DEST"
have="$(count_jpeg)"
if [ "$have" -ge "$MIN_IMAGES" ]; then
  echo ">>> $DEST already has $have *.JPEG files - nothing to do."
  exit 0
fi

# --- preflight ----------------------------------------------------------------------------------
if ! command -v kaggle >/dev/null 2>&1; then
  echo ">>> installing kaggle CLI ..."
  uv tool install --quiet kaggle
  export PATH="$(uv tool dir --bin):$PATH"
fi

# The kaggle CLI does not read KAGGLE_API_TOKEN.
if [ -n "${KAGGLE_API_TOKEN:-}" ]; then
  if [ "${KAGGLE_API_TOKEN#\{}" != "$KAGGLE_API_TOKEN" ]; then   # kaggle.json blob
    mkdir -p "${HOME}/.kaggle"
    printf '%s' "$KAGGLE_API_TOKEN" > "${HOME}/.kaggle/kaggle.json"
    chmod 600 "${HOME}/.kaggle/kaggle.json"
    echo ">>> materialised \$KAGGLE_API_TOKEN -> ~/.kaggle/kaggle.json"
  else
    export KAGGLE_KEY="$KAGGLE_API_TOKEN"
    if [ -z "${KAGGLE_USERNAME:-}" ]; then
      echo "ERROR: \$KAGGLE_API_TOKEN looks like a bare key, so also: export KAGGLE_USERNAME=..."
      exit 1
    fi
    echo ">>> using \$KAGGLE_API_TOKEN as KAGGLE_KEY (user: $KAGGLE_USERNAME)"
  fi
fi

if [ -z "${KAGGLE_USERNAME:-}" ] && [ ! -f "${HOME}/.kaggle/kaggle.json" ] \
    && [ ! -f "${HOME}/.kaggle/credentials.json" ]; then
  echo "ERROR: no Kaggle credentials found."
  echo "  export KAGGLE_API_TOKEN='<contents of kaggle.json>'   (or KAGGLE_USERNAME + KAGGLE_KEY)"
  echo "  or: kaggle auth login"
  exit 1
fi

# --- download -----------------------------------------------------------------------------------
echo ">>> downloading Kaggle dataset '$SLUG' into $DEST ... (~6.7GB)"
if ! kaggle datasets download -d "$SLUG" -p "$DEST" --unzip; then
  echo
  echo "ERROR: could not download '$SLUG'."
  echo "  - if it 404s, the slug has moved: search kaggle.com for 'imagenet 1k validation'"
  echo "    and re-run with:  KAGGLE_SLUG=<user>/<dataset> $0"
  echo "  - if it 403s, open the dataset page once in a browser and accept its terms."
  exit 1
fi

# --- normalise the extension --------------------------------------------------------------------
# Some mirrors use .jpg/.jpeg, which the loader would skip.
renamed=0
while IFS= read -r -d '' f; do
  mv "$f" "${f%.*}.JPEG"
  renamed=$(( renamed + 1 ))
done < <(find "$DEST" \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.JPG' \) -print0)
[ "$renamed" -gt 0 ] && echo ">>> renamed $renamed lowercase-extension files to .JPEG"

total="$(count_jpeg)"
echo "=============================================================="
echo " $DEST now has $total *.JPEG"
if [ "$total" -lt "$MIN_IMAGES" ]; then
  echo " WARNING: under $MIN_IMAGES images - background-aug/stop-aug will lack diversity"
  exit 1
fi
echo " background pool ready"
echo "=============================================================="
