"""CLI inference for a trained Z-Jev tiny checkpoint.

Reads a JSON request on stdin and prints a Jev-compatible JSON response.
This mirrors what the FastAPI server does for a single request, but in a
scriptable shell form useful for smoke tests and demos.
"""

from __future__ import annotations

import argparse
import json
import sys

from z_jev.model import ZJevModel
from z_jev.protocol import DecisionsRequest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a single Z-Jev evaluation.")
    parser.add_argument("--checkpoint", required=True, help="Path to model.pt")
    parser.add_argument(
        "--request",
        required=False,
        help="Path to a JSON request file (defaults to stdin).",
    )
    args = parser.parse_args(argv)

    model = ZJevModel.load_checkpoint(args.checkpoint, hf_cfg_only=True)

    raw = sys.stdin.read() if not args.request else open(args.request, encoding="utf-8").read()
    data = json.loads(raw)
    req = DecisionsRequest.from_dict(data)
    response = model.evaluate(req)
    json.dump(response.to_dict(), sys.stdout, indent=2, ensure_ascii=False)
    sys.stdout.write("\n")


if __name__ == "__main__":  # pragma: no cover
    main()
