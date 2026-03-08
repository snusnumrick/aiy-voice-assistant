#!/usr/bin/env python3
"""
Enhanced TTS Engine with comprehensive usage logging
Add this to your YandexTTSEngine to track every request with cost
"""

import logging
import json
import os
from datetime import datetime
from typing import Optional

# Configure the usage logger
usage_logger = logging.getLogger('tts_usage')
usage_logger.setLevel(logging.INFO)

# Create file handler for usage logs
usage_log_file = 'logs/tts_usage.log'
os.makedirs('logs', exist_ok=True)
handler = logging.FileHandler(usage_log_file)
handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(message)s')
handler.setFormatter(formatter)
usage_logger.addHandler(handler)

class TTSUsageLogger:
    """Tracks TTS usage and costs"""

    V3_RATE_PER_UNIT = 0.001333  # USD per 250-char billing unit

    def __init__(self, log_file: Optional[str] = None):
        self.log_file = log_file or 'logs/tts_usage.jsonl'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def log_request(self, text: str, engine_name: str, lang: str, tone: str, filename: str):
        """Log a TTS request with cost calculation"""

        char_count = len(text)
        billing_units = (char_count + 249) // 250
        estimated_cost_usd = billing_units * self.V3_RATE_PER_UNIT

        # Create usage record
        usage_record = {
            'timestamp': datetime.now().isoformat(),
            'engine': engine_name,
            'text_length': char_count,
            'billing_units': billing_units,
            'estimated_cost_usd': round(estimated_cost_usd, 6),
            'language': lang,
            'tone': tone,
            'filename': filename,
            'text_preview': text[:100] + '...' if len(text) > 100 else text
        }

        # Write to JSONL file (one JSON per line)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(usage_record, ensure_ascii=False) + '\n')

        # Also log to regular log file
        usage_logger.info(
            f"TTS Request | {engine_name} | {char_count} chars | "
            f"{billing_units} units | ${estimated_cost_usd:.6f} | "
            f"{lang}/{tone} | {filename}"
        )

        return usage_record

    def calculate_daily_cost(self, date: str) -> dict:
        """Calculate total cost for a specific date (YYYY-MM-DD)"""
        total_units = 0
        total_chars = 0
        request_count = 0

        if not os.path.exists(self.log_file):
            return {'date': date, 'requests': 0, 'chars': 0, 'units': 0, 'cost_usd': 0}

        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                if record['timestamp'].startswith(date):
                    request_count += 1
                    total_units += record['billing_units']
                    total_chars += record['text_length']

        return {
            'date': date,
            'requests': request_count,
            'chars': total_chars,
            'units': total_units,
            'cost_usd': round(total_units * self.V3_RATE_PER_UNIT, 6)
        }

    def calculate_monthly_cost(self, year: int, month: int) -> dict:
        """Calculate total cost for a month"""
        total_units = 0
        total_chars = 0
        request_count = 0
        daily_breakdown = {}

        if not os.path.exists(self.log_file):
            return {
                'year': year,
                'month': month,
                'requests': 0,
                'chars': 0,
                'units': 0,
                'cost_usd': 0,
                'daily': {}
            }

        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                date_str = record['timestamp'][:10]  # YYYY-MM-DD
                year_month = date_str[:7]  # YYYY-MM

                if year_month == f"{year:04d}-{month:02d}":
                    request_count += 1
                    total_units += record['billing_units']
                    total_chars += record['text_length']

                    day = date_str[-2:]
                    if day not in daily_breakdown:
                        daily_breakdown[day] = {'chars': 0, 'units': 0, 'requests': 0}
                    daily_breakdown[day]['chars'] += record['text_length']
                    daily_breakdown[day]['units'] += record['billing_units']
                    daily_breakdown[day]['requests'] += 1

        return {
            'year': year,
            'month': month,
            'requests': request_count,
            'chars': total_chars,
            'units': total_units,
            'cost_usd': round(total_units * self.V3_RATE_PER_UNIT, 6),
            'daily': daily_breakdown
        }

    def generate_report(self, days: int = 30) -> str:
        """Generate usage report for last N days"""
        from datetime import datetime, timedelta

        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        total_cost = 0
        total_requests = 0
        total_chars = 0

        if not os.path.exists(self.log_file):
            return f"No usage data found in {self.log_file}"

        with open(self.log_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]

        # Filter by date range
        filtered_records = [
            r for r in records
            if start_date <= datetime.fromisoformat(r['timestamp']) <= end_date
        ]

        for record in filtered_records:
            total_cost += record['estimated_cost_usd']
            total_requests += 1
            total_chars += record['text_length']

        report = f"""
TTS Usage Report ({days} days)
{'=' * 60}
Period: {start_date.date()} to {end_date.date()}

Total Requests: {total_requests:,}
Total Characters: {total_chars:,}
Total Cost: ${total_cost:.6f} USD

Average per request: {total_chars / total_requests if total_requests > 0 else 0:.1f} chars
Average daily cost: ${total_cost / days:.6f} USD

Top 10 longest requests:
"""

        # Sort by length and show top 10
        top_longest = sorted(filtered_records, key=lambda x: x['text_length'], reverse=True)[:10]
        for i, record in enumerate(top_longest, 1):
            report += f"\n{i:2d}. {record['text_length']:5d} chars | ${record['estimated_cost_usd']:.6f} | {record['timestamp'][:19]}"

        return report


# Example usage in your TTS engine:
# Add this to your YandexTTSEngine.synthesize() method:
"""
from tts_usage_logger import TTSUsageLogger

# At the top of your file, create a logger instance
_usage_logger = TTSUsageLogger()

def synthesize(self, text: str, filename: str, tone: Tone = Tone.PLAIN, lang=Language.RUSSIAN) -> None:
    if not text:
        logger.warning("Empty text to synthesize, skipping")
        return

    # Log the request BEFORE synthesizing
    _usage_logger.log_request(
        text=text,
        engine_name="YandexTTSEngine",
        lang=lang.name,
        tone=tone.name,
        filename=filename
    )

    model = self.voice_model(tone=tone, lang=lang)
    try:
        logger.debug(f"Synthesizing text: {text[:50]}...")
        result = model.synthesize(text, raw_format=False)
        result.export(filename, "wav")
        logger.debug(f"Audio content written to file {filename}")
    except Exception as e:
        logger.error(f"Error during speech synthesis: {str(e)}")
        raise
"""

if __name__ == '__main__':
    # Demo
    logger = TTSUsageLogger()

    # Log a test request
    logger.log_request(
        text="Hello, this is a test of the TTS usage logging system",
        engine_name="YandexTTSEngine",
        lang="RUSSIAN",
        tone="PLAIN",
        filename="test.wav"
    )

    # Generate report
    print(logger.generate_report(1))
