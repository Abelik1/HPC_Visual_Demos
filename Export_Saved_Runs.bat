@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Export saved runs

rem Packs favourites, showcase picks and Discoverer/Leonardo runs into one zip
rem under exports\. Put that zip in runs\_import\ on the other machine and start
rem the viewer; it unpacks the runs automatically.
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

"%PYTHON%" tools\export_runs.py --list
if errorlevel 1 ( pause & exit /b 1 )
echo.
choice /c YN /n /m "Write these runs to a zip? [Y/N] "
if errorlevel 2 exit /b 0

"%PYTHON%" tools\export_runs.py
if errorlevel 1 ( pause & exit /b 1 )
start "" "%~dp0exports"
pause
