#!/usr/bin/env python3
"""
Startup script for the Student Score Prediction Server.
This script initializes the ML pipeline and starts the FastAPI server.
"""

import argparse
import sys
from pathlib import Path

from src.pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from src.pipelines.training_pipeline import run_training_pipeline
from src.utils.logger import setup_logger


def initialize_system():
    """Initialize the ML system by running ingest, preprocess, and train"""
    logger = setup_logger()
    logger.info("Initializing Student Score Prediction System...")

    try:
        # Step 1: Ingest data
        logger.info("Step 1: Ingesting raw data...")
        from src.data.ingest import run_ingestion
        run_ingestion()

        # Step 2: Preprocess data
        logger.info("Step 2: Preprocessing data...")
        run_preprocessing_pipeline()

        # Step 3: Train models
        logger.info("Step 3: Training models...")
        run_training_pipeline()

        logger.info("✅ System initialization completed successfully!")
        return True

    except Exception as e:
        logger.error(f"❌ Initialization failed: {str(e)}")
        return False


def start_server(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the FastAPI server"""
    logger = setup_logger()
    logger.info(f"Starting server on {host}:{port}...")

    try:
        import uvicorn
        from server import app

        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            reload=reload,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Student Score Prediction Server")
    parser.add_argument("--init-only", action="store_true",
                       help="Only initialize the system, don't start server")
    parser.add_argument("--no-init", action="store_true",
                       help="Skip initialization, just start server")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000,
                       help="Server port (default: 8000)")
    parser.add_argument("--reload", action="store_true",
                       help="Enable auto-reload for development")

    args = parser.parse_args()

    # Initialize system unless --no-init is specified
    if not args.no_init:
        if not initialize_system():
            sys.exit(1)

    # Start server unless --init-only is specified
    if not args.init_only:
        start_server(host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()