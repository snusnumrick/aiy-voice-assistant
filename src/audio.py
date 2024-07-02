"""
Text-to-Speech (TTS) module.

This module provides functions for speech synthesis.
"""

import os
import tempfile
import shutil
from typing import List
import re
from pydub import AudioSegment
from src.config import Config


def split_text(text: str, max_length: int) -> List[str]:
    """Split text into chunks of maximum length."""
    sentences = re.split('(?<=[.!?]) +', text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def combine_audio_files(file_list: List[str], output_filename: str) -> None:
    """Combine multiple audio files into a single file."""
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    combined.export(output_filename, format="wav")


def synthesize_speech(engine, text: str, filename: str, config: Config) -> None:
    """Synthesize speech from text, handling long texts by splitting and combining audio chunks."""
    max_size_tts = config.get('max_size_tts', 4096)
    chunks = split_text(text, max_length=max_size_tts)

    temp_dir = tempfile.mkdtemp()
    try:
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(temp_dir, f"chunk_{i}.wav")
            engine.synthesize(chunk, chunk_file)
            chunk_files.append(chunk_file)

        if len(chunk_files) > 1:
            combine_audio_files(chunk_files, filename)
        else:
            shutil.move(chunk_files[0], filename)
    finally:
        shutil.rmtree(temp_dir)
