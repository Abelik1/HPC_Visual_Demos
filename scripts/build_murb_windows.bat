@echo off
rem Build the NBody-EuroHPC "murb" executable for the MUrB N-body dashboard demo.
rem
rem Expects the repository next to this one:  ..\NBody-EuroHPC
rem   git clone https://github.com/albtad01/NBody-EuroHPC ..\NBody-EuroHPC
rem Needs CMake, MSYS2 MinGW g++ (C:\msys64\mingw64\bin) and Ninja (bundled with
rem Visual Studio 2022). MSVC cannot compile MUrB's GCC-style code, so this uses
rem g++, CPU/OpenMP only, statically linked so murb.exe runs without MSYS2 on PATH.
setlocal
set "HERE=%~dp0"
set "REPO=%HERE%..\..\NBody-EuroHPC"
if not "%MURB_REPO%"=="" set "REPO=%MURB_REPO%"
if not exist "%REPO%\CMakeLists.txt" (
  echo NBody-EuroHPC not found at %REPO%
  echo Clone it there or set MURB_REPO.
  exit /b 1
)
set "PATH=C:\msys64\mingw64\bin;%PATH%"
set "NINJA="
for %%E in (Community Professional Enterprise BuildTools) do (
  for %%R in ("%ProgramFiles%" "%ProgramFiles(x86)%") do (
    if exist "%%~R\Microsoft Visual Studio\2022\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "NINJA=%%~R\Microsoft Visual Studio\2022\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
  )
)
where ninja >nul 2>nul && set "NINJA=ninja"
if "%NINJA%"=="" (
  echo Ninja not found. Install Visual Studio 2022 C++ tools or "pip install ninja".
  exit /b 1
)
cmake -S "%REPO%" -B "%REPO%\build-windows" -G Ninja "-DCMAKE_MAKE_PROGRAM=%NINJA%" ^
  -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release ^
  -DENABLE_VISU=OFF -DENABLE_TEST=OFF -DENABLE_MURB_OMP=ON ^
  -DCMAKE_EXE_LINKER_FLAGS=-static || exit /b 1
cmake --build "%REPO%\build-windows" -j 8 || exit /b 1
echo.
echo Built %REPO%\build-windows\bin\murb.exe
echo Restart the viewer so the dashboard picks it up.
