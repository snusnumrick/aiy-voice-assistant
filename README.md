# AIY Voice Assistant

This project uses the Google AIY Voice Kit to create an AI voice assistant that interacts with various AI models, including OpenAI's GPT and speech services.

## Features

- Speech-to-Text using Google Cloud Speech-to-Text or OpenAI Whisper
- Text-to-Speech using Google Cloud Text-to-Speech or OpenAI TTS
- Conversation management with OpenAI GPT or Anthropic Claude
- LED feedback for user interactions
- Configurable settings through config files and environment variables

## Setup

### Prerequisites

- Raspberry Pi with Raspbian installed
- Google AIY Voice Kit assembled
- Follow official instructions to flash microSD card with AIY Voice Kit image

### Installation

1. Clone the repository and navigate to the project directory.

2. Create and activate a virtual environment:

    ```bash
    python3 -m venv --system-site-packages venv
    source venv/bin/activate
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory with your API keys:

    ```plaintext
    OPENAI_API_KEY=your_openai_api_key_here
    ANTHROPIC_API_KEY=your_anthropic_api_key_here
    GOOGLE_APPLICATION_CREDENTIALS=path_to_your_google_credentials.json
    ```

5. Configure your `config.json` file with desired settings.

### Running the Project

Run the main script:

```bash
python main.py
```

Press the button on the AIY Voice Kit to start a conversation. The LED will indicate when the assistant is listening.

### Project Structure

* main.py: Entry point of the application
* src/: Contains all the source code

    * config.py: Configuration management
    * audio.py: Audio processing and transcription
    * stt_engine.py: Speech-to-Text engines
    * tts_engine.py: Text-to-Speech engines
    * ai_models.py: AI model implementations
    * conversation_manager.py: Manages conversation flow and history
    * dialog.py: Main dialog loop

### Testing
To run tests:
```bash
python -m unittest discover tests
```

### Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

### License
This project is licensed under the MIT License - see the LICENSE file for details.
