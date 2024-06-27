import logging
import signal
import sys
import time
from dotenv import load_dotenv
from aiy.board import Board, Button
from aiy.leds import Leds, Color

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(0))
load_dotenv()

from src.STTEngine import OpenAISTTEngine
from src.dialog import main_loop
from src.config import Config
from src.speech import OpenAITTSEngine


def main():
    from google.cloud import speech
    from aiy.cloudspeech import CloudSpeechClient, AUDIO_FORMAT, END_OF_SINGLE_UTTERANCE
    from aiy.voice.audio import Recorder

    class SpeechListener(CloudSpeechClient):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def recognize(self, language_code='en-US', hint_phrases=None):
            """
            Performs speech-to-text for a single utterance using the default ALSA soundcard driver.
            Once it detects the user is done speaking, it stops listening and delivers the top
            result as text.

            By default, this method calls :meth:`start_listening` and :meth:`stop_listening` as the
            recording begins and ends, respectively.

            Args:
                language_code:  Language expected from the user, in IETF BCP 47 syntax (default is
                    "en-US"). See the `list of Cloud's supported languages`_.
                hint_phrase: A list of strings containing words and phrases that may be expected from
                    the user. These hints help the speech recognizer identify them in the dialog and
                    improve the accuracy of your results.

            Returns:
                The text transcription of the user's dialog.
            """
            streaming_config=speech.types.StreamingRecognitionConfig(
                config=self._make_config(language_code, hint_phrases))

            with Recorder() as recorder:
                for chunk in recorder.record(AUDIO_FORMAT,
                                             chunk_duration_sec=0.1,
                                             on_start=self.start_listening,
                                             on_stop=self.stop_listening):

                    requests = [speech.types.StreamingRecognizeRequest(audio_content=chunk)]
                    responses = self._client.streaming_recognize(config=streaming_config, requests=requests)

                    for response in responses:
                        if response.speech_event_type == END_OF_SINGLE_UTTERANCE:
                            recorder.done()

                        for result in response.results:
                            print (result.alternatives[0].transcript)
                            # if #result.is_final:
                            #     return result.alternatives[0].transcript

            return None


    client = SpeechListener("/home/anton/gcloud.json")
    while True:
        logging.info('Say something.')
        text = client.recognize(language_code="ru-RU")
        if text is None:
            logging.info('You said nothing.')
            continue
        logging.info('You said: "%s"' % text)

    config = Config()
    with Board() as board, Leds() as leds:

        leds.update(Leds.rgb_on(Color.WHITE))
        time.sleep(1)
        leds.update(Leds.rgb_off())

        main_loop(board.button, leds, OpenAISTTEngine(), OpenAITTSEngine(), config)


if __name__ == '__main__':
    main()
