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
- Multi-model AI approach, supporting Claude AI, OpenAI, Google's Gemini models, and Deepseek's advanced analysis capabilities
- Email functionality for sending messages
- Streaming responses for natural conversation flow
- Code interpretation and execution capabilities
- Deep analytical reasoning through the WizardTool feature

The assistant primarily operates in Russian but can be configured for other languages.

## Key Features

- **Adaptive Conversations**: Engages in dialogues appropriate to the user's age and interests, from children to adults.
- **Educational Content**: Offers learning opportunities through interactive discussions, quizzes, and informational exchanges.
- **Emotional Intelligence**: Expresses and recognizes emotions, enhancing the interactive experience through LED patterns and voice tones.
- **Deep Analysis Capabilities**: Utilizes the WizardTool to break down and thoroughly analyze complex questions, providing comprehensive, well-reasoned responses for sophisticated queries.
- **Web Search Integration**: Performs web searches to answer queries and provide access to current information 
beyond the AI model's knowledge cutoff date, enhancing the assistant's ability to discuss recent events and up-to-date facts.
- **Adaptive Personality**: Learns and adapts to the user's communication style and interests over time.
- **Multilingual Support**: Primarily configured for Russian, with potential for other languages.
- **Physical Interaction**: Utilizes a cardboard cube interface with a button and LED for intuitive interaction.
- **Multi-Model AI**: Supports Claude AI, OpenAI, Google's Gemini models, and Deepseek's analytical capabilities, allowing for flexible and powerful natural language processing.
- **Email Functionality**: Ability to send emails to the user.
- **Memory Formation**: Remembers facts and rules across conversations for improved context retention.
- **Streaming Responses**: Provides more natural conversation flow with real-time responses.
- **Russian Stress Marking**: Ability to add stress marks to Russian words, enhancing pronunciation guidance and language learning.
- **Code Interpretation**: Can execute Python code, allowing for complex computations and data analysis, with results conveyed verbally.
- **Volume Control**: Ability to adjust speaker volume through voice commands, enhancing user comfort and accessibility.
- **Comprehensive Weather Information**: Provides detailed weather data including:
  - Current conditions and forecasts (hourly/daily)
  - UV index and ozone levels
  - Air quality information
  - Solar data (sunrise, sunset, dawn, dusk)
  - Lunar phase and illumination
  - **Deep Analysis with WizardTool**: Ask complex questions that require thorough analysis and reasoning. For example:
  - "Какие философские последствия квантовой запутанности?" (What are the philosophical implications of quantum entanglement?)
  - "Как искусственный интеллект может повлиять на будущее образования?" (How might artificial intelligence impact the future of education?)
  - "Объясни взаимосвязь между климатическими изменениями и глобальной экономикой." (Explain the relationship between climate change and the global economy.)
  The WizardTool will break down these complex questions, analyze them from multiple perspectives, and provide comprehensive, well-reasoned responses.
- **Complex Problem Solving**: Ask the assistant to solve complex problems that may require computational assistance.
- **Automated Tailscale Management**: Automatically manages Tailscale VPN state based on time of day to optimize CPU usage, enabling remote maintenance during quiet hours while ensuring optimal performance during active use.

## Hardware Requirements

- **Google Voice Kit V2**: This project is specifically designed for the Google Voice Kit V2. While this kit has been discontinued by Google, it can still be found on secondary markets like eBay.
- For details on the kit, visit: https://aiyprojects.withgoogle.com/voice/

**Note**: Setting up the Google Voice Kit V2 is a prerequisite for running this project. Please follow Google's official instructions at the above URL before proceeding with this project's setup.

## Software Requirements

- Raspberry Pi OS
- Python 3.9 (managed via pyenv)
- Poetry for dependency management
- Rust compiler
- ZSH shell with Oh My Zsh
- API keys for: OpenAI, Google, **Yandex**, **Anthropic**, **ElevenLabs**, **Tavily**, OpenRouter, **Perplexity**,  **Tomorrow.io**, **Maps.co Geocoding**
(in **bold** are keys for default configuration)
- Additional system packages and development tools (detailed in setup instructions)

For a complete list of Python dependencies, refer to the pyproject.toml file in the project repository. Poetry will handle the installation of these dependencies during the setup process.

## Setup

Follow these steps to set up the AI Voice Assistant on your Raspberry Pi:

1. **Set up the Google Voice Kit V2:**
   - Follow the official guide at https://aiyprojects.withgoogle.com/voice/

2. **Enable easy SSH access:**
   ```
   ssh-copy-id <your-raspberry-pi-ip>
   ```

3. **Increase VM size:**
   ```bash
   sudo dphys-swapfile swapoff
   sudo nano /etc/dphys-swapfile
   # Find CONF_SWAPSIZE=100 and increase (e.g., to 1024 for 1GB)
   sudo dphys-swapfile setup
   sudo dphys-swapfile swapon
   sudo reboot
   free -h  # Verify new swap size
   ```

4. **Update the system:**
   ```bash
   sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys B53DC80D13EDEF05
   sudo apt update
   sudo apt upgrade
   ```

5. **Install and configure ZSH:**
   ```bash
   sudo apt install zsh
   chsh -s $(which zsh)
   # Logout and login again
   sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
   ```

6. **Set up remote editing:**
   ```bash
   sudo apt install ruby
   sudo gem install rmate
   ```

7. **Install Poetry:**
   ```bash
   sudo apt install pipx
   pipx install poetry
   echo 'export PATH=/home/anton/.local/bin:$PATH' >> ~/.zshrc
   mkdir $ZSH_CUSTOM/plugins/poetry
   poetry completions zsh > $ZSH_CUSTOM/plugins/poetry/_poetry
   # Add 'poetry' to your plugins array in ~/.zshrc

8. **Install Python 3.9 using pyenv:**
   ```bash
   curl https://pyenv.run | bash
   echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
   echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
   echo 'eval "$(pyenv init -)"' >> ~/.zshrc
   poetry config virtualenvs.prefer-active-python true
   # Logout and login again
   sudo apt update
   sudo apt install build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libclang-dev libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev
   pyenv install 3.9
   # Logout and login again
   ```

9. **Install Rust:**
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs > rust.sh
   bash rust.sh -y
   ```

10. **Clone the repository and set up the project:**
    ```bash
    git clone https://github.com/snusnumrick/aiy-voice-assistant.git
    cd aiy-voice-assistant
    pyenv local 3.9
    sudo apt install cmake cython
    poetry install
    pip install google-cloud-speech google-cloud-texttospeech
    ```

11. **Configure the assistant:**

    The assistant uses two configuration files:
    * config.json: Main configuration file that should be kept in version control
    * user.json: User-specific overrides that should not be committed to version control

    The configuration system follows these precedence rules:

    1. Direct arguments passed to constructor
    2. Environment variables (prefixed with APP_)
    3. Values from user.json (user-specific overrides)
    4. Values from config.json (shared configuration)

    Start by copying the example configuration file:
    ```shell
    cp user.json.example user.json
    ```
    Customize user.json according to your needs. 

12. **Set up environment variables:**
    Create a `.env` file in the project root (Environment variables will override settings from both config files) 
    and  add the following (replace with your actual API keys and sensitive information):
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
        TOMORROW_API_KEY=your_tomorrow_io_api_key        # For weather data
        GEOCODE_API_KEY=your_maps_co_geocoding_api_key   # For location lookup
        WAQI_API_KEY=your_waqi_api_key                   # For air quality data
        OPENUV_API_KEY=your_openuv_api_key 
    ```
    Notes
    1. Make sure to keep your `.env` file secure and never commit it to version control.
    2. Depending on configuration, some of these API keys may be unnecessary.


13. **Set up the systemd service:**
    * Ensure you're in the project directory
    * Run the setup script:
    ```
    chmod +x ./setup_service.sh
    sudo ./setup_service.sh
    ```
    * This script will:
        * Make run.sh executable
        * Create a systemd service file
        * Create necessary logs directory
        * Enable the service to start on boot
        * Start the service
        * Configure log rotation
    
    After running the script, you can check the service status with:
    ```
    sudo systemctl status aiy.service
    ```

14. **Logging System:**
The assistant uses a comprehensive logging system with the following features:
* Daily log rotation with 5-day retention
* Logs are stored in the logs directory within the project folder
* System service logs are directed to the systemd journal
* Log files follow the naming convention:
  * Current day: assistant.log
  * Previous days: assistant.log.YYYY-MM-DD
  
To view logs:
    * Application logs: Check the logs directory
    * Service logs: Use journalctl -u aiy.service

You can modify the logging level when running manually:
```
python main.py --log-dir logs --log-level INFO
```
Available log levels: DEBUG, INFO, WARNING, ERROR, CRITICAL

15. **Set up the systemd service and Tailscale management:**
* Ensure you're in the project directory
* Run the setup script with sudo:
```bash
sudo ./setup_service.sh
```
* This script will:
  * Make run.sh executable
  * Create a systemd service file
  * Create necessary logs directory
  * Enable the service to start on boot
  * Start the service
  * Configure log rotation
  * Install Tailscale management scripts
  * Set up scheduled Tailscale management (enabled at night, disabled during day)

After running the script, you can check:
* Service status: ```sudo systemctl status aiy.service```
* Tailscale schedules: ```sudo crontab -l```
* Tailscale operation logs: ```grep tailscale-scheduler /var/log/syslog``

After completing these steps, your AI Voice Assistant should be set up and ready to use on your Raspberry Pi.

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

7. **Volume Control:**
   - You can ask the assistant to adjust the volume using natural language commands.
   - For example:
     - To increase volume: "Говори погромче" (Speak louder)
     - To decrease volume: "Сделай потише" (Make it quieter)
     - To set a specific volume: "Установи громкость на 50 процентов" (Set the volume to 50 percent)
   - The assistant will confirm the volume change after adjusting it.

8. **Weather Information:**
   - Get comprehensive current weather: "Какая сейчас погода в Москве?" (What's the current weather in Moscow?)
   - Get hourly forecast: "Почасовой прогноз погоды для Санкт-Петербурга" (Hourly weather forecast for Saint Petersburg)
   - Get daily forecast: "Прогноз погоды на неделю в Лондоне" (Weekly weather forecast for London)
   - Use coordinates: "Текущая погода на координатах 55.7558,37.6173" (Current weather at coordinates 55.7558,37.6173)

   The enhanced weather system provides:
   - Basic weather metrics (temperature, humidity, wind, pressure)
   - UV index and ozone level information
   - Air quality data and nearest monitoring station
   - Solar information (sunrise, sunset, dawn, dusk times)
   - Lunar phase details (phase name, illumination percentage, moon age)
       
   Note: Detailed additional information (UV, air quality, solar/lunar data) is available for current weather queries. Hourly and daily forecasts provide basic weather metrics only.

Remember, the assistant is primarily configured to interact in Russian. It will adapt its conversation style and content based on the user's age and interests as indicated in the initial introduction.

While the assistant is designed to be safe and educational for children, it can also engage in more complex discussions with adult users. However, for young children, parental supervision is recommended to ensure a safe and productive experience.

Additional Features:
- **Web Search**: Ask questions about current events or topics beyond the AI's training data.
- **Email**: Request the assistant to send you an email with specific information.
- **Emotional Expression**: Observe LED patterns and voice tone changes reflecting the assistant's emotional state.
- **Complex Problem Solving**: Ask the assistant to solve complex problems that may require computational assistance. For example:
  - "Вычисли сумму всех простых чисел меньше 10000." (Calculate the sum of all prime numbers below 10000.)
  - "Если бы у нас был список из 1000000 случайных чисел от 1 до 100, какова вероятность того, что число 42 появится более 15000 раз?" (If we had a list of 1000000 random numbers from 1 to 100, what's the probability that the number 42 would appear more than 15000 times?)
  - "Сколько различных способов есть подняться на лестницу из 30 ступенек, если за один шаг можно подниматься на 1 или 2 ступеньки?" (How many different ways are there to climb a staircase of 30 steps if you can take either 1 or 2 steps at a time?)
  - "Какова сумма цифр в числе 2^1000 (2 в степени 1000)?" (What is the sum of the digits in the number 2^1000 (2 to the power of 1000)?)

- **Tailscale Management:**
  - The assistant automatically manages Tailscale VPN state:
    - Enables Tailscale during night hours (default: 10 PM - 7 AM) for remote maintenance
    - Disables Tailscale during day hours to optimize CPU usage and assistant responsiveness
  - Configuration can be customized in config.json or user.json
  - State changes are logged in the assistant's log files
  - No manual intervention required once configured

## Customization

To modify settings:

* For personal changes: Edit user.json
* For project-wide changes: Edit config.json
* For temporary changes: Use environment variables
    
*Never commit user.json to version control to protect sensitive information.*

Possible customizations:

- Change email addresses and SMTP server information
- Switch between AI models (Claude, OpenAI, Gemini) in the `config.json` file.
- Adjust the system prompt and other configuration options in `config.json`.
- Customize TTS voices and languages in the configuration.
- Adjust volume control settings (min/max volume, step size) in the `config.json` file

WizardTool Configuration:

- Adjust analysis depth levels in the configuration
- Customize thinking templates for different types of questions
- Configure maximum response length and detail level
- Set up preferred AI models for different types of analysis

**Tailscale Management:**

The assistant uses cron to manage Tailscale for optimal performance:
* Default schedule:
  * Enables Tailscale at 10 PM for remote maintenance
  * Disables Tailscale at 7 AM to optimize CPU usage
* To modify the schedule:
  ```bash
  sudo crontab -e
  ```
  Update the times in these lines:
  ```crontab
  0 22 * * * /usr/local/bin/tailscale-up.sh   # 10 PM
  0 7 * * * /usr/local/bin/tailscale-down.sh  # 7 AM
  ```
* Scripts location: `/usr/local/bin/tailscale-up.sh` and `/usr/local/bin/tailscale-down.sh`
* All operations are logged to syslog for monitoring

## Project Structure

- `main.py`: Entry point of the application
- `user.json.example`: Example user-specific configuration file
- `src/`: Contains core modules:
  - `ai_models.py`: AI model implementations
  - `ai_models_with_tools.py`: AI models with tool support
  - `aqi.py`: Air Quality Index data fetching using WAQI API
  - `audio.py`: Audio processing and speech recognition
  - `config.py`: Configuration management
  - `conversation_manager.py`: Manages conversation flow and memory
  - `dialog.py`: Main conversation loop
  - `email_tools.py`: Email functionality
  - `moon.py`: Astronomical calculations for lunar phases and information
  - `openuv.py`: UV index and ozone data fetching using OpenUV API
  - `responce_player.py`: Audio playback and LED control
  - `stt_engine.py`: Speech-to-text engines
  - `sunrise.py`: Solar calculations and data (sunrise, sunset, dawn, dusk)
  - `stress_tool.py`: Tool for adding stress marks to Russian words
  - `tools.py`: Utility functions
  - `tts_engine.py`: Text-to-speech engines
  - `weather_tool.py`: Enhanced weather tool integrating multiple data sources:
    - Basic weather conditions and forecasts
    - Moon phase information
    - UV index and air quality data
    - Solar data integration
  - `web_search.py` & `web_search_tool.py`: Web search functionality
  - `llm_tools.py`: Language model specific tools
  - `code_interpreter_tool.py`: Tool for executing Python code and returning results
  - `volume_control_tool.py`: Tool for adjusting speaker volume
  - `wizard_tool.py`: Advanced analytical reasoning tool for complex questions

## Troubleshooting

- Ensure proper setup of the Google Voice Kit V2
- Verify all API keys are correctly set in the `.env` file
- Check console output for error messages
- For API rate limit issues, consider implementing backoff strategies
- For email configuration issues, verify SMTP settings in `config.json`
- Log files are automatically rotated to prevent disk space issues. You can find recent logs in the project directory and older, compressed logs with date suffixes.
- For configuration issues:
  - Check both config.json and user.json for conflicts
  - Verify environment variables aren't overriding desired settings
  - Use --debug flag to see which configuration source is being used
- For weather-related issues:
  - Verify all weather-related API keys (Tomorrow.io, WAQI, OpenUV) are valid and have sufficient quota
  - Check geocoding API key if location queries fail
  - For coordinate-based queries, ensure format is "latitude,longitude"
  - Weather data might be temporarily unavailable due to API limits or service issues
  - Some additional data (UV, air quality) may not be available for all locations
  - Hourly and daily forecasts include only basic weather metrics

- For Tailscale management issues:
  * Check cron is running: ```sudo systemctl status cron```
  * Verify cron jobs: ```sudo crontab -l```
  * Check script permissions: ```ls -l /usr/local/bin/tailscale-*.sh```
  * View recent operations: ```grep tailscale-scheduler /var/log/syslog```
  * Test scripts manually:
    ```bash
    sudo /usr/local/bin/tailscale-up.sh
    sudo /usr/local/bin/tailscale-down.sh
    ```
  * For immediate remote access: ```sudo tailscale up```
  * To disable immediately: ```sudo tailscale down```
  * 
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