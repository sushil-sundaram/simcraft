#!/usr/bin/env bash
# Install the SimCraft launcher entry and icons for the current user (Linux / freedesktop).
# Usage: scripts/install-desktop.sh            (after `cargo build --release`)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/target/release/simcraft"
[ -x "$BIN" ] || { echo "Build first: cargo build --release" >&2; exit 1; }

DATA="${XDG_DATA_HOME:-$HOME/.local/share}"
ICONS="$DATA/icons/hicolor"
# Docks/launchers (e.g. COSMIC) cache icons by name, so the icon name carries a content hash:
# a new logo gets a new name and shows up immediately.
HASH="$(sha1sum "$ROOT/assets/simcraft.svg" | cut -c1-8)"
ICON="simcraft-$HASH"
find "$ICONS" -path '*/apps/simcraft*' \( -name '*.svg' -o -name '*.png' \) -delete 2>/dev/null || true
mkdir -p "$ICONS/scalable/apps" "$DATA/applications"
install -m644 "$ROOT/assets/simcraft.svg" "$ICONS/scalable/apps/$ICON.svg"
if command -v magick >/dev/null 2>&1; then
  for s in 16 24 32 48 64 128 256 512; do
    mkdir -p "$ICONS/${s}x${s}/apps"
    magick -background none -density 384 "$ROOT/assets/simcraft.svg" -resize "${s}x${s}" "$ICONS/${s}x${s}/apps/$ICON.png"
  done
else
  mkdir -p "$ICONS/256x256/apps"
  install -m644 "$ROOT/assets/simcraft-256.png" "$ICONS/256x256/apps/$ICON.png"
fi

cat > "$DATA/applications/simcraft.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Version=1.0
Name=SimCraft
GenericName=FEA Simulation
Comment=Structural, thermal and modal finite-element simulation
Exec="$BIN" %f
Icon=$ICON
StartupWMClass=simcraft
Terminal=false
Categories=Science;Engineering;
Keywords=FEA;FEM;simulation;stress;thermal;modal;CAE;
DESKTOP

command -v update-desktop-database >/dev/null && update-desktop-database "$DATA/applications" 2>/dev/null || true
command -v gtk-update-icon-cache >/dev/null && gtk-update-icon-cache -q -t "$ICONS" 2>/dev/null || true
echo "Installed SimCraft launcher: $DATA/applications/simcraft.desktop"
