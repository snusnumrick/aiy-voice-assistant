import logging
import openai
import os
from aiy.leds import (Leds, Pattern, PrivacyLed, RgbLeds, Color)
from aiy.voice.audio import AudioFormat, play_wav, record_file, Recorder
from openai import OpenAI
import re
from pydub import AudioSegment
import tempfile
import shutil

logger = logging.getLogger(__name__)
openai = OpenAI()
RUN_DIR = '/run/user/%d' % os.getuid()
AUDIO_SAMPLE_RATE_HZ = 24000
AUDIO_FORMAT = AudioFormat(sample_rate_hz=AUDIO_SAMPLE_RATE_HZ,
                           num_channels=1,
                           bytes_per_sample=2)
MAX_SIZE_TTS = 4096


def synthesize_speech(text, filename):
    def split_text(text, max_length=MAX_SIZE_TTS):
        # Split text into sentences
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

    def text_to_speech_chunk(text, chunk_index, output_dir):
        filename = os.path.join(output_dir, f"chunk_{chunk_index}.wav")
        logger.info(f"Synthesizing chunk {chunk_index}: {text} into {filename}")
        openai.audio.speech.create(
            model="tts-1-hd",
            voice="onyx",
            response_format="wav",
            input=text
        ).stream_to_file(filename)
        logger.info(f"Saved chunk {chunk_index} to {filename}")
        return filename

    def combine_audio_files(file_list, output_filename):
        combined = AudioSegment.empty()
        for file in file_list:
            audio = AudioSegment.from_wav(file)
            combined += audio
        combined.export(output_filename, format="wav")

    logger.info('Synthesizing speech for: %s', text)
    chunks = split_text(text)
    temp_dir = tempfile.mkdtemp()
    try:
        chunk_files = []
        for i, chunk in enumerate(chunks):
            chunk_file = text_to_speech_chunk(chunk, i, temp_dir)
            chunk_files.append(chunk_file)

        combine_audio_files(chunk_files, filename)
    finally:
        shutil.rmtree(temp_dir)
    logger.debug(f"saved at {filename}")


