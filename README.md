# AI-Powered Voice Assistant: An Adaptive Companion (Russian Language Default)

## Overview

This project implements a sophisticated AI-powered voice assistant using Python, 
designed to work with the Google Voice Kit V2 hardware platform. 
While created with children in mind, this assistant serves as an interactive, educational, 
and fun companion for users of all ages. It features advanced capabilities such as emotional expression, 
memory formation, and self-improvement mechanisms, all tailored to provide a safe and enriching experience.

Key aspects of the assistant include:
- Adaptive interactions based on user age and preferences
- Educational content and engaging conversations
- Emotional intelligence and expression
- Web searches for current information
- Physical interaction through a custom cardboard cube interface
- Multi-model AI approach, supporting Claude AI, OpenAI, and Google's Gemini models
- Email functionality for sending messages
- Streaming responses for natural conversation flow

The assistant primarily operates in Russian but can be configured for other languages.

## Key Features

- **Adaptive Conversations**: Engages in dialogues appropriate to the user's age and interests, from children to adults.
- **Educational Content**: Offers learning opportunities through interactive discussions, quizzes, and informational exchanges.
- **Emotional Intelligence**: Expresses and recognizes emotions, enhancing the interactive experience through LED patterns and voice tones.
- **Web Search Integration**: Performs web searches to answer queries and provide access to current information 
beyond the AI model's knowledge cutoff date, enhancing the assistant's ability to discuss recent events and up-to-date facts.
- **Adaptive Personality**: Learns and adapts to the user's communication style and interests over time.
- **Multilingual Support**: Primarily configured for Russian, with potential for other languages.
- **Physical Interaction**: Utilizes a cardboard cube interface with a button and LED for intuitive interaction.
- **Multi-Model AI**: Supports Claude AI, OpenAI, and Google's Gemini models, allowing for flexible and powerful natural language processing.
- **Email Functionality**: Ability to send emails to the user.
- **Memory Formation**: Remembers facts and rules across conversations for improved context retention.
- **Streaming Responses**: Provides more natural conversation flow with real-time responses.

## Hardware Requirements

- **Google Voice Kit V2**: This project is specifically designed for the Google Voice Kit V2. While this kit has been discontinued by Google, it can still be found on secondary markets like eBay.
- For details on the kit, visit: https://aiyprojects.withgoogle.com/voice/

**Note**: Setting up the Google Voice Kit V2 is a prerequisite for running this project. Please follow Google's official instructions at the above URL before proceeding with this project's setup.

## Software Requirements

- Python 3.7+
- Various Python libraries (see `requirements.txt` for a complete list)
- API keys for: OpenAI, Google, **Yandex**, **Anthropic**, **ElevenLabs**, **Tavily**, OpenRouter, **Perplexity** 
(in **bold** are keys for default configuration)

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
   source venv/bin/activate  
   ```

4. Install required Python packages:
   ```
   pip install -r requirements.txt
   ```

5. Configure the assistant:
   Customize `config.json` file in the project root directory. 
   Refer to the provided json in the repository for the structure and available options.
   
   Note:
   * listed are dafault values.

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
    EMAIL_PASSWORD=your_SMTP_server_password
    GOOGLE_CUSTOMSEARCH_KEY=your_google_customsearch_key
    TAVILY_API_KEY=your_tavily_api_key
    OPENROUTER_API_KEY=your_open_router_api_key
    PERPLEXITY_API_KEY=your_perplexity_api_key
    ELEVENLABS_API_KEY=your_elevenlabs_api_key
    GEMINI_API_KEY=your_gemini_api_key
   ```

   Notes
      * Make sure to keep your `.env` file secure and never commit it to version control.
      * Depending on configuration, some of these API keys may be unnecessary.

7. Set up the systemd service:
   * Ensure you're in the project directory
   * Run the setup script:
    ```bash
    sudo ./setup_service.sh
    ```
   * This script will:
     * Make run.sh executable
     * Create a systemd service file
     * Enable the service to start on boot
     * Start the service

     After running the script, you can check the service status with:
     ```bash
     sudo systemctl status aiy.service
     ```

## Usage

To start the assistant manually (if not using the systemd service):

```bash
source venv/bin/activate  
python main.py
```

If you've set up the systemd service, the assistant will start automatically on boot. 
You can manually control the service with these commands:
* Start the service: ```sudo systemctl start aiy.service```
* Stop the service: ```sudo systemctl stop aiy.service```
* Restart the service: ```sudo systemctl restart aiy.service```
* Check service status: ```sudo systemctl status aiy.service```

To view the terminal output of the running assistant:

1. Ensure the service is running
2. Attach to the tmux session:
    ```bash
    tmux attach -t aiy
    ```
3. To detach from the session without stopping the assistant, press ```Ctrl-B``` then ```D```

This allows you to monitor the assistant's operation and view any debug information or errors in real-time.

Once the assistant is running, here's how to interact with it:

1. **Initial Introduction:**
   - When you first start interacting with the assistant, introduce yourself.
   - This helps the assistant set an appropriate tone for later conversations.
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

5. **Interrupting the Assistant:**
   - If you need to interrupt the assistant's response, simply press the button again.
   - This will stop the current playback and allow you to speak.

6. **Continuous Interaction:**
   - You can have a continuous conversation by repeating steps 3-5.
   - The assistant will maintain context throughout the conversation.

Remember, the assistant is primarily configured to interact in Russian. It will adapt its conversation style and content based on the user's age and interests as indicated in the initial introduction.

While the assistant is designed to be safe and educational for children, it can also engage in more complex discussions with adult users. However, for young children, parental supervision is recommended to ensure a safe and productive experience.

Additional Features:
- **Web Search**: Ask questions about current events or topics beyond the AI's training data.
- **Email**: Request the assistant to send you an email with specific information.
- **Emotional Expression**: Observe LED patterns and voice tone changes reflecting the assistant's emotional state.

## Customization

- Switch between AI models (Claude, OpenAI, Gemini) in the `config.json` file.
- Adjust the system prompt and other configuration options in `config.json`.
- Customize TTS voices and languages in the configuration.

## Project Structure

- `main.py`: Entry point of the application
- `src/`: Contains core modules:
  - `ai_models.py`: AI model implementations
  - `ai_models_with_tools.py`: AI models with tool support
  - `audio.py`: Audio processing and speech recognition
  - `config.py`: Configuration management
  - `conversation_manager.py`: Manages conversation flow and memory
  - `dialog.py`: Main conversation loop
  - `email_tools.py`: Email functionality
  - `tts_engine.py`: Text-to-speech engines
  - `web_search.py` & `web_search_tool.py`: Web search functionality
  - `tools.py`: Utility functions
  - `llm_tools.py`: Language model specific tools
  - `responce_player.py`: Audio playback and LED control
  - `stt_engine.py`: Speech-to-text engines

## Troubleshooting

- Ensure proper setup of the Google Voice Kit V2
- Verify all API keys are correctly set in the `.env` file
- Check console output for error messages
- For API rate limit issues, consider implementing backoff strategies
- For email configuration issues, verify SMTP settings in `config.json`


## Performance Considerations

- Response time and quality may vary between different AI models
- Be aware of potential limitations when running on Raspberry Pi hardware

## Security Considerations

- Properly secure all API keys and email credentials
- Be mindful of data privacy, especially when used with children
- Regularly update the software and dependencies to address potential vulnerabilities

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes. Areas where contributions would be particularly valuable include:
- Integration of new AI models or services
- Additional tools or capabilities
- Performance optimizations
- Improved security features

For more details on how to contribute, please see the CONTRIBUTING.md file in the repository.

## License

This project is licensed under the MIT License - see the LICENSE file for details.