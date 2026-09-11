@echo off
setlocal EnableExtensions
cd /d "%~dp0\.."
title Leonardo Visual Demos

set "PYTHON=.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
  echo Creating the local Python environment. This only happens once.
  where py >nul 2>nul
  if not errorlevel 1 (
    py -3 -m venv .venv
  ) else (
    python -m venv .venv
  )
)

if not exist "%PYTHON%" (
  echo.
  echo A Windows Python environment could not be created.
  echo Install the official 64-bit Python 3.10 or newer, then run this file again.
  pause
  exit /b 1
)

"%PYTHON%" -c "import numpy, PIL, fastapi, uvicorn, pydantic" >nul 2>nul
if errorlevel 1 (
  echo Installing the demo requirements. This only happens once.
  "%PYTHON%" -m pip install -r requirements.txt
  if errorlevel 1 goto :setup_failed
)

:menu
cls
echo ======================================
echo        LEONARDO VISUAL DEMOS
echo ======================================
echo.
echo  1. Black-hole lensing
echo  2. Primordial black-hole threshold
echo  3. Virtual wind tunnel
echo  4. Cosmic-web formation
echo  5. Galaxy collision
echo  6. Living mathematics
echo  7. Crystal growth
echo  8. Neural-network wall
echo  9. Star in a Bottle - fusion plasma
echo  A. Storm Factory - weather ensemble
echo  B. Molecular Machine - molecular dynamics
echo.
echo  V. Open the interactive web viewer
echo  Q. Quit
echo.
choice /c 123456789ABVQ /n /m "Choose a demo"

if errorlevel 13 goto :done
if errorlevel 12 goto :viewer
if errorlevel 11 goto :molecular_dynamics
if errorlevel 10 goto :weather_ensemble
if errorlevel 9 goto :fusion_plasma
if errorlevel 8 goto :neural_wall
if errorlevel 7 goto :crystal
if errorlevel 6 goto :reaction_diffusion
if errorlevel 5 goto :galaxy_collision
if errorlevel 4 goto :cosmic_web
if errorlevel 3 goto :fluid
if errorlevel 2 goto :pbh
if errorlevel 1 goto :black_hole

:black_hole
set "DEMO=black_hole"
goto :profile
:pbh
set "DEMO=pbh"
goto :profile
:fluid
set "DEMO=fluid"
goto :profile
:cosmic_web
set "DEMO=cosmic_web"
goto :profile
:galaxy_collision
set "DEMO=galaxy_collision"
goto :profile
:reaction_diffusion
set "DEMO=reaction_diffusion"
goto :profile
:crystal
set "DEMO=crystal"
goto :profile
:neural_wall
set "DEMO=neural_wall"
goto :profile
:fusion_plasma
set "DEMO=fusion_plasma"
goto :profile
:weather_ensemble
set "DEMO=weather_ensemble"
goto :profile
:molecular_dynamics
set "DEMO=molecular_dynamics"
goto :profile

:profile
cls
echo %DEMO%
echo.
echo  1. Local     - recommended; low-resolution and fast
echo  2. Desktop   - larger simulation; use when local looks good
echo  3. HPC       - supercomputer-sized settings; use on a suitable machine or cluster
echo.
choice /c 123 /n /m "Choose a profile"
if errorlevel 3 goto :hpc_profile
if errorlevel 2 goto :desktop_profile
if errorlevel 1 goto :local_profile

:local_profile
set "PROFILE=local"
goto :frame_count
:desktop_profile
set "PROFILE=desktop"
goto :frame_count
:hpc_profile
set "PROFILE=hpc"

:frame_count
set "FRAMES="
set /p "FRAMES=Number of frames [70]: "
if not defined FRAMES set "FRAMES=70"

echo.
echo Running %DEMO% with the %PROFILE% profile (%FRAMES% frames)...
echo.
set "RUN_DIR="
for /f "usebackq delims=" %%R in (`"%PYTHON%" run_demo.py %DEMO% --profile %PROFILE% --frames %FRAMES%`) do set "RUN_DIR=%%R"

if not defined RUN_DIR (
  echo.
  echo The demo did not complete. Check the message above and try again.
  pause
  goto :menu
)

echo.
echo Finished: %RUN_DIR%
echo Opening the reveal image and the frame folder...
start "" "%RUN_DIR%\reveal.jpg"
start "" "%RUN_DIR%\frames"
echo.
pause
goto :menu

:viewer
cls
echo Starting the viewer at http://127.0.0.1:8000
echo Press Ctrl+C to return to this menu.
echo.
"%PYTHON%" app.py
pause
goto :menu

:setup_failed
echo.
echo Setup failed. Check your internet connection and Python installation, then run this file again.
pause
exit /b 1

:done
endlocal
