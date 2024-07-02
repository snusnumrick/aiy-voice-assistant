"""
Text-to-Speech (TTS) module.

This module provides functions for speech synthesis.
"""

import os
import tempfile
import shutil
from typing import List, Any
import re
from pydub import AudioSegment
from src.config import Config

def split_text(text: str, max_length: int) -> List[str]:
    """
    Split text into chunks of maximum length.

    Args:
        text (str): The input text to be split.
        max_length (int): The maximum length of each chunk.

    Returns:
        List[str]: A list of text chunks.
    """
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
    """
    Combine multiple audio files into a single file.

    Args:
        file_list (List[str]): List of audio file paths to combine.
        output_filename (str): Path to save the combined audio file.
    """
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    combined.export(output_filename, format="wav")

def synthesize_speech(engine: Any, text: str, filename: str, config: Config) -> None:
    """
    Synthesize speech from text, handling long texts by splitting and combining audio chunks.

    Args:
        engine (Any): The TTS engine to use for synthesis.
        text (str): The text to synthesize into speech.
        filename (str): The path to save the synthesized audio file.
        config (Config): The application configuration object.
    """
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

# Additional TTS engine classes can be added here if needed

class YandexTTSEngine:
    """Yandex SpeechKit TTS engine."""

    def __init__(self, config: Config):
        """
        Initialize the Yandex TTS engine.

        Args:
            config (Config): Configuration object.
        """
        from speechkit import configure_credentials, creds, model_repository

        self.api_key = os.environ.get('YANDEX_API_KEY') or config.get('yandex_api_key')
        if not self.api_key:
            raise ValueError("Yandex API key is not provided in environment variables or configuration")

        configure_credentials(yandex_credentials=creds.YandexCredentials(api_key=self.api_key))
        self.model = model_repository.synthesis_model()
        self.model.voice = config.get('yandex_tts_voice', 'ermil')
        self.model.role = config.get('yandex_tts_role', 'neutral')
        self.model.language = config.get('yandex_tts_language', 'ru-RU')
        self.model.speed = config.get('yandex_tts_speed', 1.0)

    def synthesize(self, text: str, filename: str) -> None:
        """
        Synthesize speech from text and save it to a file.

        Args:
            text (str): The text to synthesize.
            filename (str): The path to save the synthesized audio file.
        """
        result = self.model.synthesize(text, raw_format=False)
        result.export(filename, 'wav')

def create_tts_engine(config: Config) -> Any:
    """
    Factory function to create TTS engine based on configuration.

    Args:
        config (Config): Configuration object.

    Returns:
        Any: Initialized TTS engine.

    Raises:
        ValueError: If an unsupported TTS engine is specified.
    """
    engine_name = config.get('tts_engine', 'yandex').lower()
    if engine_name == 'yandex':
        return YandexTTSEngine(config)
    else:
        raise ValueError(f"Unsupported TTS engine: {engine_name}")