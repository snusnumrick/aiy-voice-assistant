from abc import ABC, abstractmethod
import logging

if __name__ == '__main__':
    from config import Config
else:
    from .config import Config


logger = logging.getLogger(__name__)

class STTEngine(ABC):
    """
    Abstract class for Speech-to-Text Engine
    """
    @abstractmethod
    def transcribe(self, audio_file: str) -> str:
        pass


class OpenAISTTEngine(STTEngine):
    """
    Implementation of STTEngine using OpenAI API
    """
    def transcribe(self, audio_file: str) -> str:
        import openai
        with open(audio_file, 'rb') as f:
            return openai.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ru",
                response_format="text"
            )


class GoogleSTTEngine(STTEngine):
    """
    Implementation of STTEngine using Google Speech Recognition API
    """
    def transcribe(self, audio_file: str) -> str:
        import speech_recognition as sr
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio = recognizer.record(source)
        try:
            return recognizer.recognize_google(audio, language="ru")
        except sr.UnknownValueError:
            logger.error("unknown error occurred")
            return ""
        except sr.RequestError:
            logger.error("Could not request results from Google Speech Recognition service")
            return ""


def test():
    logging.basicConfig(level=logging.DEBUG)
    config = Config()
    engine = GoogleSTTEngine()
    text = engine.transcribe("/Users/antont/recording.wav")
    print(f"Google transcribed text: {text}")
    # engine = OpenAISTTEngine()
    # text = engine.transcribe("/Users/antont/recording.wav")
    # print(f"OpenAI transcribed text: {text}")


if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    test()