"""inferlab command line interface.

    inferlab bench     -- measure throughput/latency/VRAM of a running server
    inferlab quality   -- measure what a quantization config cost in quality
    inferlab report    -- regenerate docs/RESULTS.md from results/*.json
"""
from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inferlab",
        description="Reproducible LLM inference benchmarking & optimization lab.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    from . import harness as _bench
    from . import quality as _quality
    from . import report as _report

    sub.add_parser("bench", help="benchmark a running OpenAI-compatible server")
    sub.add_parser("quality", help="measure the quality cost of a config")
    sub.add_parser("report", help="regenerate docs/RESULTS.md")

    return parser, _bench, _quality, _report


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser, bench, quality, report = build_parser()

    # Parse only the first token to pick a subcommand, then hand the rest to the module's own
    # parser. This keeps each subcommand's --help self-contained.
    if not argv:
        parser.print_help()
        return 2

    command, rest = argv[0], argv[1:]
    if command in ("-h", "--help"):
        parser.print_help()
        return 0

    if command == "bench":
        return bench.main(rest)
    if command == "quality":
        return quality.main(rest)
    if command == "report":
        return report.main(rest)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
