"""Entry point for the research-agent."""
from __future__ import annotations

import argparse
import asyncio
import logging
import os

from orchestrator_client import OrchestratorClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Research Agent")
    parser.add_argument(
        "--orchestrator-url",
        default=os.environ.get("ORCHESTRATOR_URL", "http://localhost:8000"),
        help="Base URL of the agent orchestrator (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    client = OrchestratorClient(orchestrator_url=args.orchestrator_url)
    await client.start()


if __name__ == "__main__":
    asyncio.run(main())
