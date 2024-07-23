# AI-Powered Voice Assistant: An Adaptive Companion (Russian Language Default)

## Overview

This project implements a sophisticated AI-powered voice assistant using Python, designed to work with the Google Voice Kit V2 hardware platform. While created with children in mind, this assistant serves as an interactive, educational, and fun companion for users of all ages. It features advanced capabilities such as emotional expression, memory formation, and self-improvement mechanisms, all tailored to provide a safe and enriching experience. The assistant primarily operates in Russian but can be configured for other languages.

Key aspects of the assistant include:
- Adaptive interactions based on user age and preferences
- Educational content and engaging conversations
- Emotional intelligence and expression
- Web searches for current information
- Physical interaction through a custom cardboard cube interface

## Key Features

- **Adaptive Conversations**: Engages in dialogues appropriate to the user's age and interests, from children to adults.
- **Educational Content**: Offers learning opportunities through interactive discussions, quizzes, and informational exchanges.
- **Emotional Intelligence**: Expresses and recognizes emotions, enhancing the interactive experience.
- **Web Search Integration**: Performs web searches to answer queries and provide access to current information beyond the AI model's knowledge cutoff date, enhancing the assistant's ability to discuss recent events and up-to-date facts.
- **Adaptive Personality**: Learns and adapts to the user's communication style and interests over time.
- **Multilingual Support**: Primarily configured for Russian, with potential for other languages.
- **Physical Interaction**: Utilizes a cardboard cube interface with a button and LED for intuitive interaction.

## Hardware Requirements

- **Google Voice Kit V2**: This project is specifically designed for the Google Voice Kit V2. While this kit has been discontinued by Google, it can still be found on secondary markets like eBay.
- For details on the kit, visit: https://aiyprojects.withgoogle.com/voice/

**Note**: Setting up the Google Voice Kit V2 is a prerequisite for running this project. Please follow Google's official instructions at the above URL before proceeding with this project's setup.

## Software Requirements

- Python 3.7+
- Various Python libraries (see `requirements.txt` for a complete list)

## Setup

1. Set up the Google Voice Kit V2:
   - Follow the official setup guide at https://aiyprojects.withgoogle.com/voice/
   - Ensure your Raspberry Pi and Voice Kit hardware are properly configured

2. Clone the repository:
   ```
   git clone https://github.com/yourusername/voice-assistant.git
   cd voice-assistant
   ```

3. Set up a virtual environment:
   ```
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

4. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

5. Configure the assistant:
   Create a `config.json` file in the project root directory with the following content:

   ```json
   {
       "language_code": "ru-RU",
       "system_prompt": "Тебя зовут Кубик. Ты мой друг и помощник. Ты умеешь шутить и быть саркастичным. Отвечай естественно, как в устной речи. Говори максимально просто и понятно. Не используй списки и нумерации. Например, не говори 1. что-то; 2. что-то. говори во-первых, во-вторых или просто перечисляй. При ответе на вопрос где важно время, помни какое сегодня число. Если чего-то не знаешь, так и скажи. Я буду разговаривать с тобой через голосовой интерфейс. Будь краток, избегай банальностей и непрошенных советов.",
       "speech_recognition_service": "yandex",
       "claude_model": "claude-3-5-sonnet-20240620",
       "openrouter_model": "anthropic/claude-3.5-sonnet",
       "openrouter_model_simple": "anthropic/claude-3-haiku",
       "perplexity_model": "llama-3-sonar-large-32k-online",
       "max_tokens": 4096,
       "token_threshold": 2500,
       "min_number_of_messages_to_summarize": 20,
       "min_number_of_messages_to_keep": 10,
       "summary_prompt": "Обобщи основные моменты разговора, сосредоточившись на наиболее важных фактах и контексте. Будь лаконичен. Начни ответ с \"Ранее мы говорили о \". \nРазговор: \n-----\n",
       "audio_file_name": "speech.wav",
       "sample_rate_hertz": 16000,
       "audio_recording_chunk_duration_sec": 0.1,
       "max_number_of_chunks": 5,
       "number_of_chuncks_to_record_after_button_depressed": 3,
       "ready_breathing_period_ms": 10000,
       "ready_breathing_color": [0, 1, 0],
       "ready_breathing_duration": 60,
       "recording_color": [0, 255, 0],
       "processing_color": [0, 1, 0],
       "processing_blink_period_ms": 300,
       "yandex_tts_voice_russian": "ermil",
       "yandex_tts_voice_english": "john",
       "yandex_tts_voice_german": "lea",
       "yandex_tts_role_plain": "neutral",
       "yandex_tts_role_happy": "good",
       "yandex_tts_speed": 1.0,
       "elevenlabs_voice_id_en": "ID of voice you want elevenlabs TTS use for english",
       "elevenlabs_voice_id_de": "ID of voice you want elevenlabs TTS use for german",
       "elevenlabs_voice_id_ru": "ID of voice you want elevenlabs TTS use for russian", 
       "elevenlabs_stability":  0.5,
       "elevenlabs_similarity_boost": 0.75,
       "elevenlabs_style": 0.0,
       "elevenlabs_use_speaker_boost": false,
       "elevenlabs_max_retries": 5,
       "elevenlabs_initial_retry_delay": 1,
       "elevenlabs_retry_jitter_factor": 0.1,
       "elevenlabs_output_format": "eleven_multilingual_v2", 
       "openai_tts_model": "tts",
       "openai_tts_voice": "alloy",
       "google_service_account_file": "~/gcloud.json",
       "google_tts_language": "ru-RU",
       "google_tts_voice": "ru-RU-Wavenet-A",
       "cleaning_time_start_hour": 3,
       "cleaning_time_stop_hour": 4,
       "form_new_memories_at_night": true,
       "clean_message_history_at_night": true,
       "profanity_filter": false,
       "literature_text": true,
       "assistant_email_address": "Enter the email address you want the assistant to use for sending emails",
       "user_email_address": "Enter your email address where you want to receive emails from the assistant",
       "smtp_server": "Enter the SMTP server address for sending emails",
       "smtp_port": "Enter the SMTP port number",
       "assistant_email_username": "Enter the username for the email account the assistant will use"
   }
   ```
   
   Note:
   * listed are dafault values. If you are comfortable with them you don't have to put such entries into config file.

   Adjust the following fields in the `config.json`:
   - `assistant_email_address`: The email address the assistant will use to send emails
   - `user_email_address`: Your email address to receive emails from the assistant
   - `smtp_server`: SMTP server address for the email service
   - `smtp_port`: SMTP port number (usually 587 for TLS or 465 for SSL)
   - `assistant_email_username`: Username for the assistant's email account

6. Set up environment variables:
   Create a `.env` file in the project root and 
   add the following (replace with your actual API keys and sensitive information):
   ```
   OPENAI_API_KEY=your_openai_api_key
   GOOGLE_API_KEY=your_google_api_key
   YANDEX_API_KEY=your_yandex_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   EMAIL_PASSWORD=your_SNOT_server_password
   GOOGLE_CUSTOMSEARCH_KEY=your_google_customsearch_key
   TAVILY_API_KEY=your_tavily_api_key
   OPENROUTER_API_KEY=your_open_router_api_key
   PERPLEXITY_API_KEY=your_perplexity_api_key
   ```

   Notes
      * Make sure to keep your `.env` file secure and never commit it to version control.
      * Depending on configuration, some of these API keys may be unnecessary.

## Usage

Ensure you're in the virtual environment before running the assistant:

```
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
python main.py
```

Once the assistant is running, here's how to interact with it:

1. **Initial Introduction:**
   - When you first start interacting with the assistant, introduce yourself.
   - This helps the assistant set an appropriate tone for subsequent conversations.
   - For example, you might say:
     "Привет! Меня зовут Алиса. Мне семь лет."
     (Hello! My name is Alisa. I'm seven years old.)
   - Or for an adult user:
     "Здравствуй! Я Иван, мне 35 лет. Я интересуюсь историей."
     (Hello! I'm Ivan, I'm 35 years old. I'm interested in history.)

2. **Initiating Interaction:**
   - The assistant is housed in a cardboard cube with a button and embedded LED on top.
   - When ready for interaction, the LED will display a greenish breathing pattern.
   - Note: To conserve energy and avoid light pollution, the LED will go blank after a period of inactivity.

3. **Speaking to the Assistant:**
   - To speak to the assistant, press and hold the button on top of the cube.
   - While you're holding the button, the LED will turn solid green, indicating that it's recording your voice.
   - Speak clearly into the microphone while holding the button.
   - Release the button when you've finished speaking.

4. **Listening to the Response:**
   - After you release the button, the assistant will process your input and respond.
   - The LED may display different colors or patterns to express the assistant's emotional state during the response.

5. **Interrupting the Assistant:**
   - If you need to interrupt the assistant's response, simply press the button again.
   - This will stop the current playback and allow you to speak again.

6. **Continuous Interaction:**
   - You can have a continuous conversation by repeating steps 3-5.
   - The assistant will maintain context throughout the conversation.

Remember, the assistant is primarily configured to interact in Russian. It will adapt its conversation style and content based on the user's age and interests as indicated in the initial introduction.

While the assistant is designed to be safe and educational for children, it can also engage in more complex discussions with adult users. However, for young children, parental supervision is recommended to ensure a safe and productive experience.

## Project Structure

- `main.py`: Entry point of the application
- `src/`: Contains the core modules of the voice assistant
  - `ai_models.py`: Implementations of various AI models
  - `ai_models_with_tools.py`: AI models with support for tools/plugins
  - `audio.py`: Audio processing and speech recognition
  - `config.py`: Configuration management
  - `conversation_manager.py`: Manages conversation flow, context, and memory formation
  - `dialog.py`: Implements the main conversation loop
  - `email_tools.py`: Email functionality
  - `tts_engine.py`: Text-to-speech engines with emotion support
  - `web_search.py`: Web search functionality
  - `web_search_tool.py`: Web search tool implementation
  - `tools.py`: Utility functions and tools
  - `llm_tools.py`: Language model specific tools
  - `responce_player.py`: Manages audio playback and LED behavior for emotional expression

## Extending the Assistant

To add new capabilities:
1. Implement a new tool in `src/ai_models_with_tools.py`
2. Register the tool in the `ConversationManager` class
3. Update the AI model prompts to include information about the new tool
4. If applicable, add new emotional expressions or memory categories in the respective modules

## Troubleshooting

- Ensure the Google Voice Kit V2 is properly set up and functioning.
- If speech recognition is not working, check your microphone settings and ensure you have the necessary API keys set up.
- For TTS issues, verify that the chosen engine is properly configured in `config.json`.
- If the assistant is not responding correctly, check the console output for any error messages or unexpected behaviors.
- Ensure all required environment variables are set correctly in your `.env` file.
- If emotional expressions or memory formation seem off, review the related configuration settings and logs.

## Contributing

Contributions to this project are welcome! Please fork the repository and submit a pull request with your changes. We especially appreciate contributions that enhance the educational value and adaptability of the assistant for users of all ages.

For more details on how to contribute, please see the CONTRIBUTING.md file in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.