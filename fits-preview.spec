# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        "astropy",
        "astropy.io.fits",
        "numpy",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="fits-preview",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    # argv_emulation converts Apple Events → argv as a fallback;
    # our QFileOpenEvent handler is the primary mechanism.
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    name="fits-preview",
)

app = BUNDLE(
    coll,
    name="FITS Preview.app",
    icon=None,  # replace with "assets/icon.icns" when ready
    bundle_identifier="io.fitspreview.app",
    info_plist={
        # High-DPI support
        "NSHighResolutionCapable": True,
        "NSPrincipalClass": "NSApplication",
        # Minimum macOS version
        "LSMinimumSystemVersion": "11.0",
        # Register the app to handle FITS files
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName": "FITS Image",
                "CFBundleTypeRole": "Viewer",
                "LSHandlerRank": "Default",
                "CFBundleTypeExtensions": ["fits", "fit", "fts"],
                "LSItemContentTypes": ["io.fitspreview.fits-image"],
            }
        ],
        # Export a custom UTI for FITS so Finder can identify the file type
        "UTExportedTypeDeclarations": [
            {
                "UTTypeIdentifier": "io.fitspreview.fits-image",
                "UTTypeDescription": "FITS Image",
                "UTTypeConformsTo": ["public.data", "public.image"],
                "UTTypeTagSpecification": {
                    "public.filename-extension": ["fits", "fit", "fts"],
                    "public.mime-type": ["image/fits", "application/fits"],
                },
            }
        ],
    },
)
