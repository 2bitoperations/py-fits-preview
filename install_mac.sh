#!/bin/bash
# MacOS App Wrapper Installer for FITS Preview

set -e

APP_NAME="FITS Preview.app"
APP_DIR="$HOME/Applications/$APP_NAME"
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing $APP_NAME to ~/Applications..."

# Generate the AppleScript wrapper using osacompile
osacompile -o "$APP_DIR" -e 'on open theFiles' \
    -e "do shell script \"cd '$REPO_DIR' && uv run python3 main.py \\\"\" & POSIX path of item 1 of theFiles & \"\\\" >/dev/null 2>&1 &\"" \
    -e 'end open' \
    -e 'on run' \
    -e "do shell script \"cd '$REPO_DIR' && uv run python3 main.py >/dev/null 2>&1 &\"" \
    -e 'end run'

# Add CFBundleDocumentTypes to Info.plist to register as a handler for .fit and .fits
PLIST="$APP_DIR/Contents/Info.plist"

# Use PlistBuddy to inject the DocumentTypes dictionary
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes array" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0 dict" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeRole string Viewer" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions array" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:0 string fits" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:1 string fit" "$PLIST"
/usr/libexec/PlistBuddy -c "Add :CFBundleDocumentTypes:0:CFBundleTypeExtensions:2 string fts" "$PLIST"

echo "Rebuilding LaunchServices database..."
/System/Library/Frameworks/CoreServices.framework/Versions/A/Frameworks/LaunchServices.framework/Versions/A/Support/lsregister -f "$APP_DIR"

echo "Done! You can now Right-Click any .fits file -> Get Info -> Open With -> FITS Preview -> Change All."
