#!/bin/bash
# Linux Desktop Entry Installer for FITS Preview

set -e

APP_NAME="py-fits-preview.desktop"
DESKTOP_DIR="$HOME/.local/share/applications"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing $APP_NAME to $DESKTOP_DIR..."

mkdir -p "$DESKTOP_DIR"

cat > "$DESKTOP_DIR/$APP_NAME" << EOF
[Desktop Entry]
Version=1.0
Name=FITS Preview
Comment=Extremely fast preloading FITS image viewer
Exec=bash -c "cd '$REPO_DIR' && uv run python3 main.py %f"
Icon=image-x-generic
Terminal=false
Type=Application
Categories=Science;Astronomy;Graphics;Viewer;
MimeType=image/fits;
EOF

chmod +x "$DESKTOP_DIR/$APP_NAME"

echo "Updating desktop database..."
update-desktop-database "$DESKTOP_DIR" 2>/dev/null || true

echo "Setting as default handler for image/fits..."
if command -v xdg-mime >/dev/null 2>&1; then
    xdg-mime default "$APP_NAME" image/fits
    echo "Done! Py-FITS-Preview is now registered in GNOME/KDE."
else
    echo "xdg-mime not found, skipping default handler registration."
fi
