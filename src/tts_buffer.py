"""
TTS Sentence Buffering Module

Buffers multiple sentences before sending to TTS to reduce costs.
Combines sentences based on:
1. Timeout (wait X seconds for more sentences)
2. Max length (send immediately if text gets too long)
"""

import asyncio
import logging
from typing import Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class TTSBuffer:
    """
    Buffers sentences before sending to TTS to reduce the number of requests.

    Combines sentences based on:
    - timeout: Wait this long for more sentences (default: 1.5 seconds)
    - max_length: Send immediately if buffer exceeds this length (default: 200 chars)
    """

    def __init__(
        self,
        timeout: float = 1.5,
        max_length: int = 200,
        min_sentences: int = 1
    ):
        """
        Initialize the TTS buffer.

        Args:
            timeout: Seconds to wait for more sentences before sending
            max_length: Maximum accumulated characters before sending immediately
            min_sentences: Minimum sentences to buffer before considering send
        """
        self.timeout = timeout
        self.max_length = max_length
        self.min_sentences = min_sentences

        self.buffer: List[str] = []
        self.buffer_chars = 0
        self.send_task: Optional[asyncio.Task] = None
        self.last_sentence_time: float = 0
        self.is_flushing = False

    async def add_sentence(self, sentence: str, tts_func, *tts_args):
        """
        Add a sentence to the buffer and send if conditions are met.

        Args:
            sentence: The sentence to add
            tts_func: The TTS function to call (e.g., tts_engine.synthesize_async)
            *tts_args: Arguments to pass to tts_func
        """
        if not sentence or self.is_flushing:
            return

        # Add sentence to buffer
        self.buffer.append(sentence)
        self.buffer_chars += len(sentence)
        self.last_sentence_time = asyncio.get_event_loop().time()

        logger.debug(
            f"TTS Buffer: Added sentence {len(self.buffer)} "
            f"({len(sentence)} chars, total: {self.buffer_chars})"
        )

        # Cancel existing timeout task
        if self.send_task and not self.send_task.done():
            self.send_task.cancel()

        # Check if we should send immediately
        should_send = False

        if self.buffer_chars >= self.max_length:
            logger.info(
                f"TTS Buffer: Max length reached ({self.buffer_chars}/{self.max_length} chars), "
                f"sending immediately"
            )
            should_send = True
        elif len(self.buffer) >= self.min_sentences:
            # Start timeout task for this buffer
            self.send_task = asyncio.create_task(self._timeout_sender(tts_func, *tts_args))

        # Send immediately if max length reached
        if should_send:
            await self._send_buffer(tts_func, *tts_args)

    async def _timeout_sender(self, tts_func, *tts_args):
        """Wait for timeout, then send buffer if still not empty."""
        try:
            await asyncio.sleep(self.timeout)
            if self.buffer and not self.is_flushing:
                await self._send_buffer(tts_func, *tts_args)
        except asyncio.CancelledError:
            # Task was cancelled because new sentence arrived
            pass

    async def _send_buffer(self, tts_func, *tts_args):
        """Send the buffered text to TTS."""
        if not self.buffer:
            return

        self.is_flushing = True

        try:
            # Combine all sentences
            combined_text = " ".join(self.buffer)

            # Log the buffering
            sent_count = len(self.buffer)
            saved_requests = sent_count - 1  # We saved these many requests

            logger.info(
                f"TTS Buffer: Sending {sent_count} sentences "
                f"({self.buffer_chars} chars) as 1 request "
                f"(saved {saved_requests} requests)"
            )

            # Send to TTS
            await tts_func(combined_text, *tts_args)

        finally:
            # Clear buffer
            self.buffer.clear()
            self.buffer_chars = 0
            self.is_flushing = False
            self.send_task = None

    async def flush(self, tts_func, *tts_args):
        """Force send all buffered text immediately."""
        if self.buffer and not self.is_flushing:
            logger.info(f"TTS Buffer: Flushing {len(self.buffer)} sentences ({self.buffer_chars} chars)")
            await self._send_buffer(tts_func, *tts_args)

    def get_stats(self) -> dict:
        """Get current buffer statistics."""
        return {
            'buffered_sentences': len(self.buffer),
            'buffered_chars': self.buffer_chars,
            'timeout': self.timeout,
            'max_length': self.max_length,
            'min_sentences': self.min_sentences,
        }

    def is_buffering(self) -> bool:
        """Check if currently buffering sentences."""
        return len(self.buffer) > 0 and not self.is_flushing

    def reset(self):
        """Clear the buffer without sending."""
        self.buffer.clear()
        self.buffer_chars = 0
        self.is_flushing = False
        if self.send_task and not self.send_task.done():
            self.send_task.cancel()
            self.send_task = None


class TTSBufferManager:
    """
    Manager for multiple TTS buffers (e.g., one per language or voice).
    """

    def __init__(self, config):
        """
        Initialize from config.

        Config options:
        - tts_buffer_timeout: Seconds to wait (default: 1.5)
        - tts_buffer_max_length: Max chars before immediate send (default: 200)
        - tts_buffer_min_sentences: Min sentences to buffer (default: 1)
        - tts_buffer_enabled: Enable buffering (default: true)
        """
        self.enabled = config.get("tts_buffer_enabled", True)

        if not self.enabled:
            logger.info("TTS Buffering is DISABLED")
            return

        timeout = config.get("tts_buffer_timeout", 1.5)
        max_length = config.get("tts_buffer_max_length", 200)
        min_sentences = config.get("tts_buffer_min_sentences", 1)

        self.buffers: dict = {}  # key: (lang, tone) -> TTSBuffer

        logger.info(
            f"TTS Buffering ENABLED: timeout={timeout}s, max_length={max_length} chars, "
            f"min_sentences={min_sentences}"
        )

        # Create default buffer
        self.get_buffer(timeout, max_length, min_sentences)

    def get_buffer(self, timeout: float, max_length: int, min_sentences: int):
        """Get or create the default buffer."""
        key = "default"
        if key not in self.buffers:
            self.buffers[key] = TTSBuffer(timeout, max_length, min_sentences)
            logger.debug(f"Created TTS buffer: {key}")
        return self.buffers[key]

    async def add_sentence(self, sentence: str, tts_func, *tts_args, lang=None, tone=None):
        """Add a sentence to the appropriate buffer."""
        if not self.enabled:
            # Buffering disabled, send immediately
            await tts_func(sentence, *tts_args)
            return

        # Use default buffer (could expand to per-language buffers later)
        buffer = self.get_buffer(
            1.5,  # Default timeout
            200,  # Default max length
            1     # Default min sentences
        )

        await buffer.add_sentence(sentence, tts_func, *tts_args)

    async def flush_all(self, tts_func, *tts_args):
        """Flush all buffers."""
        if not self.enabled:
            return

        for buffer in self.buffers.values():
            await buffer.flush(tts_func, *tts_args)

    def get_all_stats(self) -> dict:
        """Get statistics for all buffers."""
        if not self.enabled:
            return {'enabled': False}

        return {
            'enabled': True,
            'buffers': {key: buffer.get_stats() for key, buffer in self.buffers.items()}
        }
