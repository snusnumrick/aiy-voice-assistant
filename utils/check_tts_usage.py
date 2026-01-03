#!/usr/bin/env python3
"""
Quick script to check TTS usage from logs
"""

import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

def check_usage(days: int = 30):
    """Check TTS usage for the last N days"""

    log_file = 'logs/tts_usage.log'

    if not os.path.exists(log_file):
        print(f"❌ No usage log found at {log_file}")
        print("   Run the application first to generate usage logs")
        return

    print("=" * 70)
    print(f"TTS USAGE REPORT - Last {days} days")
    print("=" * 70)
    print()

    # Read all logs
    records = []
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except:
                pass

    if not records:
        print("No usage records found")
        return

    # Filter by date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    filtered = []
    for record in records:
        try:
            ts = datetime.fromisoformat(record['timestamp'])
            if start_date <= ts <= end_date:
                filtered.append(record)
        except:
            pass

    if not filtered:
        print(f"No usage in the last {days} days")
        return

    # Calculate totals
    total_cost = sum(r['cost_usd'] for r in filtered)
    total_chars = sum(r['text_length'] for r in filtered)
    total_requests = len(filtered)

    # Group by date
    daily = defaultdict(lambda: {'requests': 0, 'chars': 0, 'cost': 0})
    for record in filtered:
        date = record['timestamp'][:10]
        daily[date]['requests'] += 1
        daily[date]['chars'] += record['text_length']
        daily[date]['cost'] += record['cost_usd']

    # Show summary
    print(f"Total Requests: {total_requests:,}")
    print(f"Total Characters: {total_chars:,}")
    print(f"Total Cost: ${total_cost:.6f} USD")
    print(f"Average per request: {total_chars / total_requests:.1f} chars")
    print(f"Average daily cost: ${total_cost / days:.6f} USD")
    print()

    # Show daily breakdown
    print("Daily Breakdown:")
    print("-" * 70)
    for date in sorted(daily.keys()):
        data = daily[date]
        print(f"{date}: {data['requests']:4d} req | {data['chars']:7d} chars | ${data['cost']:8.6f}")
    print("-" * 70)
    print()

    # Show engines used
    engines = defaultdict(int)
    for record in filtered:
        engines[record['engine']] += 1

    print("By Engine:")
    for engine, count in sorted(engines.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_requests) * 100
        print(f"  {engine}: {count} requests ({percentage:.1f}%)")
    print()

    # Show top 10 longest requests
    top_longest = sorted(filtered, key=lambda x: x['text_length'], reverse=True)[:10]
    print("Top 10 Longest Requests:")
    print("-" * 70)
    for i, record in enumerate(top_longest, 1):
        date = record['timestamp'][:10]
        time = record['timestamp'][11:19]
        print(f"{i:2d}. {date} {time} | {record['text_length']:5d} chars | ${record['cost_usd']:.6f} | {record['engine']}")
        if record['text_preview']:
            print(f"     \"{record['text_preview']}\"")
    print()

    # Check for anomalies
    print("Anomaly Check:")
    print("-" * 70)
    costs = [data['cost'] for data in daily.values()]
    if costs:
        avg_cost = sum(costs) / len(costs)
        max_cost = max(costs)
        max_day = max(daily.keys(), key=lambda d: daily[d]['cost'])

        print(f"Average daily cost: ${avg_cost:.6f}")
        print(f"Maximum daily cost: ${max_cost:.6f} on {max_day}")

        if max_cost > avg_cost * 5:
            print(f"⚠️  WARNING: {max_day} cost is {max_cost/avg_cost:.1f}x average!")
            print("   Possible reasons:")
            print("   - API key compromise")
            print("   - Unusual usage pattern")
            print("   - Long text synthesized")
        else:
            print("✓ No major anomalies detected")

    print()

if __name__ == '__main__':
    check_usage(days=30)
