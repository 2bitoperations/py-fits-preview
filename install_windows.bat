@echo off
setlocal
echo Installing py-fits-preview registry entries...
set "APP_DIR=%~dp0"
:: Remove trailing backslash if present
if "%APP_DIR:~-1%"=="\" set "APP_DIR=%APP_DIR:~0,-1%"

reg add "HKCU\Software\Classes\.fit" /ve /d "py_fits_preview_file" /f >nul
reg add "HKCU\Software\Classes\.fits" /ve /d "py_fits_preview_file" /f >nul
reg add "HKCU\Software\Classes\.fts" /ve /d "py_fits_preview_file" /f >nul
reg add "HKCU\Software\Classes\py_fits_preview_file" /ve /d "FITS Image" /f >nul
reg add "HKCU\Software\Classes\py_fits_preview_file\shell\open\command" /ve /d "cmd.exe /c \"cd /d \"%APP_DIR%\" && uv run pythonw main.py \"%%1\"\"" /f >nul

echo Done! py-fits-preview is now registered as the default handler for .fits files.
pause
