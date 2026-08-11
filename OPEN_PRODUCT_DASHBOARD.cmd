@echo off
rem Compatibility wrapper for the stable Itinerary Repair Copilot entrypoint.
call "%~dp0OPEN_ITINERARY_COPILOT.cmd" %*
exit /b %errorlevel%
