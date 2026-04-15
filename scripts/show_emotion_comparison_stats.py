#!/usr/bin/env python3
"""
Summarize emotion-engine comparison JSONL data.
"""

import argparse
import json
from collections import Counter
from statistics import mean


def _safe_mean(values):
    return round(mean(values), 2) if values else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Show summary stats for emotion comparison JSONL data."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default="logs/emotion_comparison.jsonl",
        help="Path to the comparison JSONL file",
    )
    args = parser.parse_args()

    rows = []
    with open(args.path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        print("No comparison rows found.")
        return 0

    top_1_matches = 0
    primary_latencies = []
    shadow_latencies = []
    extra_latencies = []
    disagreement_pairs = Counter()

    for row in rows:
        agreement = row.get("agreement", {})
        timing = row.get("timing", {})
        primary_name = agreement.get("primary_top_1")
        shadow_name = agreement.get("shadow_top_1")

        if agreement.get("top_1_match"):
            top_1_matches += 1
        elif primary_name or shadow_name:
            disagreement_pairs[(primary_name, shadow_name)] += 1

        if isinstance(timing.get("primary_return_latency_ms"), (int, float)):
            primary_latencies.append(timing["primary_return_latency_ms"])
        if isinstance(timing.get("shadow_latency_ms"), (int, float)):
            shadow_latencies.append(timing["shadow_latency_ms"])
        if isinstance(timing.get("shadow_extra_latency_ms"), (int, float)):
            extra_latencies.append(timing["shadow_extra_latency_ms"])

    print(f"rows={len(rows)}")
    print(f"top1_match_rate={round((top_1_matches / len(rows)) * 100, 2)}%")
    print(f"avg_primary_latency_ms={_safe_mean(primary_latencies)}")
    print(f"avg_shadow_latency_ms={_safe_mean(shadow_latencies)}")
    print(f"avg_shadow_extra_latency_ms={_safe_mean(extra_latencies)}")

    if disagreement_pairs:
        print("top_disagreements:")
        for (primary_name, shadow_name), count in disagreement_pairs.most_common(5):
            print(f"  {primary_name} -> {shadow_name}: {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
