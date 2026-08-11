@echo off
setlocal
title Itinerary Repair Copilot
cd /d "%~dp0"

where python >nul 2>nul
if not errorlevel 1 (
  python scripts\run_product_app.py --open %*
  goto :done
)

where py >nul 2>nul
if not errorlevel 1 (
  py -3 scripts\run_product_app.py --open %*
  goto :done
)

echo.
echo Python was not found on PATH.
echo Open a project terminal and run:
echo python scripts\run_product_app.py --open
pause
exit /b 1

:done
set "PRODUCT_EXIT_CODE=%errorlevel%"
if not "%PRODUCT_EXIT_CODE%"=="0" pause
endlocal & exit /b %PRODUCT_EXIT_CODE%
