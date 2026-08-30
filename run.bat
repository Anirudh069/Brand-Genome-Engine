@echo off
echo Starting Brand Genome Engine (backend + frontend)...

REM Try docker compose first
docker compose version >nul 2>&1
if %errorlevel% equ 0 (
    docker compose up --build
    exit /b
)

REM Fallback to docker-compose
docker-compose version >nul 2>&1
if %errorlevel% equ 0 (
    docker-compose up --build
    exit /b
)

echo.
echo Error: Docker is not installed, or neither "docker compose" nor "docker-compose" was found.
echo Please refer to the README.md for instructions on how to run the project manually.
pause
