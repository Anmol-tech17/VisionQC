"""
VisionQC — Application Launcher
---------------------------------
Run this file from the VisionQC/ directory to start the API server:

    python main.py

The server will be accessible at:
    http://localhost:8000         → Browser UI
    http://localhost:8000/health  → Health check
    http://localhost:8000/predict → POST image upload
    http://localhost:8000/docs    → Interactive API docs (Swagger UI)
"""

import sys
import os
from pathlib import Path

# Ensure the project root is on the Python path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("  VisionQC — Automated PCB Defect Inspection")
    print("=" * 60)
    print(f"  UI:      http://localhost:8000")
    print(f"  API:     http://localhost:8000/health")
    print(f"  Swagger: http://localhost:8000/docs")
    print("=" * 60)
    print()

    uvicorn.run(
        "src.visionqc.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
