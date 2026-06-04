#!/usr/bin/env python3
"""
Serve the Predictive Maintenance Platform API via uvicorn.

Usage:
    python scripts/serve_api.py [--host 0.0.0.0] [--port 8000] [--workers 4]

The host, port, and worker count default to the values defined in
``config.settings`` (which are themselves driven by ``PMP_*`` environment
variables).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import uvicorn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("serve_api")


def main() -> None:
    # Import settings *after* adding src/ to sys.path
    from config.settings import settings

    parser = argparse.ArgumentParser(
        description="Start the Predictive Maintenance API server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=settings.api_host,
        help="Bind address (default: %(default)s)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help="Bind port (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=settings.api_workers,
        help="Number of uvicorn workers (default: %(default)s)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=settings.api_reload,
        help="Enable hot-reload (development only)",
    )
    args = parser.parse_args()

    logger.info(
        "Starting API server on %s:%d with %d worker(s) (reload=%s)",
        args.host,
        args.port,
        args.workers,
        args.reload,
    )

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
