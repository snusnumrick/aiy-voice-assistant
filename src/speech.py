"""
Synthesize speech (TTS)
"""
import logging
import os
import re
from abc import ABC, abstractmethod
from typing import List
from pydub import AudioSegment
import tempfile
import shutil

if __name__ == '__main__':
    from config import Config
else:
    from .config import Config

logger = logging.getLogger(__name__)

MAX_SIZE_TTS = 4096


class TTSEngine(ABC):
    @abstractmethod
    def synthesize(self, text: str, filename: str) -> None:
        pass


class OpenAITTSEngine(TTSEngine):
    def __init__(self, model="tts-1-hd", voice="onyx"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model
        self.voice = voice

    def synthesize(self, text: str, filename: str) -> None:
        self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            response_format="wav",
            input=text
        ).stream_to_file(filename)


class GoogleTTSEngine(TTSEngine):
    def __init__(self, config: Config, language_code="en-US"):
        from google.oauth2 import service_account
        from google.cloud import texttospeech

        self.config = config
        service_accout_file = self.config.get('service_account_file', '~/gcloud.json')
        service_accout_file = os.path.expanduser(service_accout_file)
        credentials = service_account.Credentials.from_service_account_file(service_accout_file)
        self.client = texttospeech.TextToSpeechClient(credentials=credentials)
        self.language_code = language_code

    def synthesize(self, text: str, filename: str) -> None:
        from google.cloud import texttospeech
        synthesis_input = texttospeech.SynthesisInput(text=text)
        voice = texttospeech.VoiceSelectionParams(
            language_code=self.language_code,
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16
        )
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=voice, audio_config=audio_config
        )
        with open(filename, "wb") as out:
            out.write(response.audio_content)


def split_text(text: str, max_length: int) -> List[str]:
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
    logger.debug(f"Combining {len(file_list)} audio files into {output_filename}")
    combined = AudioSegment.empty()
    for file in file_list:
        audio = AudioSegment.from_wav(file)
        combined += audio
    logger.debug(f"Exporting combined audio to {output_filename}")
    combined.export(output_filename, format="wav")
    logger.debug(f"Exported combined audio to {output_filename}")


def synthesize_speech(engine: TTSEngine, text: str, filename: str, config: Config) -> None:
    logger.debug('Synthesizing speech for: %s', text)
    max_size_tts = config.get('max_size_tts', 4096)  # Default to 4096 if not in config
    chunks = split_text(text, max_length=max_size_tts)
    temp_dir = tempfile.mkdtemp()
    try:
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(temp_dir, f"chunk_{i}.wav")
            logger.debug(f"Synthesizing chunk {i}: {chunk} into {chunk_file}")
            engine.synthesize(chunk, chunk_file)
            logger.debug(f"Saved chunk {i} to {chunk_file}")
            chunk_files.append(chunk_file)

        combine_audio_files(chunk_files, filename)
    finally:
        shutil.rmtree(temp_dir)
    logger.debug(f"saved at {filename}")


def test():
    config = Config()
    tts = GoogleTTSEngine(config, "ru-RU")
    tts.synthesize("Привет, как дела?", "test.wav")


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    test()
