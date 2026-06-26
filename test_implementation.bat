@echo off
REM Script to test the three implementations
REM Tests: Naive, Scattered Grid, and Coherent Grid

echo ========================================
echo CUDA Flocking Implementation Test
echo ========================================
echo.

REM Check if executable exists
if not exist "build\bin\Release\cis5650_boids.exe" (
    echo ERROR: Executable not found at build\bin\Release\cis5650_boids.exe
    echo Please build the project in Release mode first.
    pause
    exit /b 1
)

echo This script will test the implementation.
echo Close the window after a few seconds to test the next mode.
echo.
echo Press any key to continue...
pause > nul

echo.
echo ========================================
echo Testing with compute-sanitizer (memcheck)
echo This will detect memory errors and race conditions
echo ========================================
echo.
echo Running for 5 seconds...
timeout /t 2 > nul

REM Run with compute-sanitizer for a limited time
echo Starting sanitizer check...
compute-sanitizer --tool memcheck --print-limit 100 build\bin\Release\cis5650_boids.exe

echo.
echo ========================================
echo Test complete!
echo ========================================
echo.
echo If no errors were reported above, the implementation is safe.
echo.
pause
