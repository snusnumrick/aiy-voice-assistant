#!/usr/bin/env python3
"""Print tool usage statistics collected by optimize_tool_usage_tracking_enabled."""

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Show tool usage stats")
    parser.add_argument(
        "--file",
        default="tool_usage_stats.json",
        help="Path to stats JSON (default: tool_usage_stats.json)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"No stats file found: {path}")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools", {})
    total = int(data.get("total_calls", 0))

    print(f"file: {path}")
    print(f"total_calls: {total}")
    print(f"updated_at: {data.get('updated_at')}")
    print("")
    print("tool\tcount\tlast_used")
    for name, row in sorted(
        tools.items(),
        key=lambda kv: int((kv[1] or {}).get("count", 0)),
        reverse=True,
    ):
        print(f"{name}\t{int(row.get('count', 0))}\t{row.get('last_used')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
