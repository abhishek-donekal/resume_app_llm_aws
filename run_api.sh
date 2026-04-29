#!/bin/bash

# Start FastAPI server for resume customizer

set -e

echo "🚀 Starting FastAPI server..."
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🏥 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Load environment
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | xargs)
fi

# Start server
python -m uvicorn inference_api:app \
    --host 0.0.0.0 \
    --port 8000 \
    --reload \
    --log-level info
