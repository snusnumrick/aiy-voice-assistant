#!/usr/bin/env python3
"""
Run conversation.txt assistant responses through clean_response() and print results.
Usage: python scripts/check_clean_response.py [conversation.txt]
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.tools import clean_response

path = sys.argv[1] if len(sys.argv) > 1 else "conversation.txt"

with open(path, encoding="utf-8") as f:
    lines = f.readlines()

in_assistant = False
block = []

def flush(block):
    raw = " ".join(block).strip()
    cleaned = clean_response(raw)
    if raw != cleaned:
        print("BEFORE:", raw)
        print("AFTER: ", cleaned)
        print()

for line in lines:
    stripped = line.strip()
    if stripped in ("A:", "assistant:"):
        if block:
            flush(block)
            block = []
        in_assistant = True
    elif stripped and not line.startswith(" ") and not line.startswith("\t"):
        if block:
            flush(block)
            block = []
        in_assistant = False
    elif in_assistant and stripped:
        block.append(stripped)

if block:
    flush(block)
