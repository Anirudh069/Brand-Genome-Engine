#!/bin/bash
echo "Starting Brand Genome Engine (backend + frontend)..."

if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    docker compose up --build
elif command -v docker-compose >/dev/null 2>&1; then
    docker-compose up --build
else
    echo ""
    echo "Error: Docker is not installed, or neither 'docker compose' nor 'docker-compose' was found."
    echo "Please refer to the README.md for instructions on how to run the project manually."
    exit 1
fi
