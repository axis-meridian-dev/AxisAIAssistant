#!/bin/bash
# ============================================================
# CT Practice Book Downloader — AXMH Legal Research
# Downloads the official CT Practice Book from jud.ct.gov
# Run: chmod +x download_ct_practice_book.sh && ./download_ct_practice_book.sh
# ============================================================

DEST="$HOME/LegalResearch"
mkdir -p "$DEST"

echo "=================================================="
echo "  CT Practice Book Downloader — AXMH Legal Tools"
echo "=================================================="
echo ""

# URLs — both 2025 and 2026 (2026 is latest as of April 2026)
URLS=(
  "https://www.jud.ct.gov/Publications/PracticeBook/PB2025.pdf|CT_Practice_Book_2025.pdf"
  "https://www.jud.ct.gov/publications/PracticeBook/PB.pdf|CT_Practice_Book_2026_latest.pdf"
  "https://www.jud.ct.gov/Publications/PracticeBook/PB_Mobile_2024.pdf|CT_Practice_Book_2024_mobile.pdf"
)

download_file() {
  local URL="$1"
  local FILENAME="$2"
  local OUTPATH="$DEST/$FILENAME"

  echo "→ Downloading: $FILENAME"
  echo "  Source: $URL"

  # Try wget first (better with government sites)
  if command -v wget &>/dev/null; then
    wget -q --show-progress \
      --user-agent="Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0" \
      --referer="https://www.jud.ct.gov/pb.htm" \
      -O "$OUTPATH" "$URL"
  else
    # Fallback to curl
    curl -L --silent --show-error --progress-bar \
      -A "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0" \
      -e "https://www.jud.ct.gov/pb.htm" \
      -o "$OUTPATH" "$URL"
  fi

  # Verify it's actually a PDF (not a 503 page)
  if file "$OUTPATH" 2>/dev/null | grep -q "PDF"; then
    SIZE=$(du -h "$OUTPATH" | cut -f1)
    echo "  ✓ Success — $SIZE saved to $OUTPATH"
    return 0
  else
    echo "  ✗ Failed (server blocked or file invalid)"
    rm -f "$OUTPATH"
    return 1
  fi
}

SUCCESS=0
for ENTRY in "${URLS[@]}"; do
  URL="${ENTRY%%|*}"
  FILE="${ENTRY##*|}"
  download_file "$URL" "$FILE" && SUCCESS=$((SUCCESS + 1))
  echo ""
done

echo "=================================================="
echo "  Done. $SUCCESS file(s) downloaded to:"
echo "  $DEST"
echo ""
echo "  If downloads failed, open these URLs directly"
echo "  in Firefox and use File > Save As:"
echo ""
echo "  2026 (latest):"
echo "  https://www.jud.ct.gov/publications/PracticeBook/PB.pdf"
echo ""
echo "  2025:"
echo "  https://www.jud.ct.gov/Publications/PracticeBook/PB2025.pdf"
echo ""
echo "  2024 (mobile/compact):"
echo "  https://www.jud.ct.gov/Publications/PracticeBook/PB_Mobile_2024.pdf"
echo "=================================================="
