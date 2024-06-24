import logging
import signal
import sys
from dotenv import load_dotenv
from src.speech import transcribe_speech, synthesize_speech
from src.openai_interaction import get_openai_response
from src.led_feedback import led_on, led_off
import aiy.voicehat

# Load environment variables from .env file
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, filename='logs/aiy_voice_assistant.log')
logger = logging.getLogger(__name__)

# Initialize the Google AIY Voice Kit components
button = aiy.voicehat.get_button()
led = aiy.voicehat.get_led()


# Main function to handle interactions
def main():
    logging.basicConfig(level=logging.DEBUG)
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))


    while True:
        text = transcribe_speech(button, led)
        if text:
            ai_response = get_openai_response(text)
            logger.info('AI says: %s', ai_response)

            speech_audio_file = synthesize_speech(ai_response)
            aiy.audio.play_wave(speech_audio_file)


if __name__ == '__main__':
    main()