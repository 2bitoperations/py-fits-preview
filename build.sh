#!/usr/bin/env bash
set -e

echo "Building FITS Preview.app..."
uv run pyinstaller fits-preview.spec --noconfirm

echo ""
echo "Done. App bundle at: dist/FITS Preview.app"
echo ""
echo "To register with Finder, run:"
echo "  /System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f \"dist/FITS Preview.app\""
