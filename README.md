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
- **Voice Emotion Detection**: Real-time detection of user emotions from voice using Hume AI, enabling emotionally-aware responses that match the user's mood.
- **Music Generation**: Generate and play music with lyrics using MiniMax API, supporting lullabies, songs, and custom musical compositions with streaming playback.
- **Reminders**: Set reminders that trigger a bell + LED pattern and optionally spoken reminders, with context injection so you can ask “what was the reminder about?”.
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
- ffmpeg (required for music generation audio conversion)
- API keys for: OpenAI, Google, **Yandex**, **Anthropic**, **ElevenLabs**, **Tavily**, **Parallel**, OpenRouter, **Perplexity**,  **Tomorrow.io**, **Maps.co Geocoding**, MiniMax (optional, for music generation)
(in **bold** are keys for default configuration)
- Additional system packages and development tools (detailed in setup instructions)

For a complete list of Python dependencies, refer to the pyproject.toml file in the project repository. Poetry will handle the installation of these dependencies during the setup process.

## Setup

If you already have a working assistant, cloning its SD card is usually the fastest way
to provision a new device. Use the first-time setup steps below when building a device
from a fresh Raspberry Pi OS image.

### Clone an existing SD card for a new device

Cloning copies the full system image, including this repository, local config,
`.env`, `user.json`, logs, SSH host keys, and Tailscale state. Only clone cards into
devices you control, and keep the source device powered down until the clone has its
own hostname and network identity.

1. **Shut down the working Raspberry Pi cleanly:**
   ```bash
   sudo shutdown -h now
   ```
   Wait until the activity LED stops blinking, then remove the source SD card.

2. **Create an image from the source card on your computer.**

   Double-check every device name before running `dd`: `if=` is the source and `of=`
   is overwritten.

   On macOS:
   ```bash
   diskutil list
   diskutil unmountDisk /dev/diskN
   sudo dd if=/dev/rdiskN of=~/aiy-voice-source.img bs=4m status=progress
   diskutil eject /dev/diskN
   ```

   On Linux:
   ```bash
   lsblk
   sudo umount /dev/sdX*
   sudo dd if=/dev/sdX of=~/aiy-voice-source.img bs=4M status=progress conv=fsync
   sync
   ```

   Replace `diskN` or `sdX` with the whole SD-card device, not a partition such as
   `diskNs1` or `sdX1`. On macOS, use the same number for `diskN` and `rdiskN`;
   `rdiskN` is the raw-device path for the same SD card.

3. **Write the image to the new SD card.**

   Use a destination card with the same or larger capacity than the source.
   Raspberry Pi Imager is the recommended way to write the clone because it gives
   you a safer target-card picker and verifies the write.
   Newer Imager versions do not offer OS customization when writing an existing
   image, so do hostname, SSH, and Tailscale cleanup after the first boot.

   In Raspberry Pi Imager:
   1. Choose **Choose Device** and select the Raspberry Pi model.
   2. Choose **Choose OS** -> **Use custom**.
   3. Select `aiy-voice-source.img`.
   4. Choose **Choose Storage** and select the new SD card.
   5. Click **Write** and wait for verification to finish.
   6. Eject the SD card when Imager completes.

   If you prefer a terminal-only restore, use `dd`.

   On macOS:
   ```bash
   diskutil list
   diskutil unmountDisk /dev/diskM
   sudo dd if=~/aiy-voice-source.img of=/dev/rdiskM bs=4m status=progress
   sync
   diskutil eject /dev/diskM
   ```

   On Linux:
   ```bash
   lsblk
   sudo umount /dev/sdY*
   sudo dd if=~/aiy-voice-source.img of=/dev/sdY bs=4M status=progress conv=fsync
   sync
   ```

   On macOS, use the same destination-card number for `diskM` and `rdiskM`.

4. **Boot the new Raspberry Pi and give it a unique identity.**

   SSH into the new device, then set a new hostname:
   ```bash
   sudo hostnamectl set-hostname aiy-voice-2
   sudo nano /etc/hosts
   sudo reboot
   ```

   In `/etc/hosts`, update the `127.0.1.1` entry to match the new hostname.

5. **Regenerate SSH host keys on the cloned device:**
   ```bash
   sudo rm /etc/ssh/ssh_host_*
   sudo dpkg-reconfigure openssh-server
   sudo systemctl restart ssh
   ```

   On your computer, remove the old host key for the cloned device if SSH warns about
   a changed fingerprint:
   ```bash
   ssh-keygen -R <new-device-hostname-or-ip>
   ```

6. **Reset Tailscale identity if the source card had Tailscale configured:**
   ```bash
   sudo tailscale logout || true
   sudo rm -rf /var/lib/tailscale
   sudo systemctl restart tailscaled
   sudo tailscale up --hostname aiy-voice-2
   ```

   This prevents the new device from appearing as the original assistant in the
   Tailscale admin console.

7. **Expand the filesystem if the destination SD card is larger:**
   ```bash
   sudo raspi-config
   ```

   Choose **Advanced Options** -> **Expand Filesystem**, reboot, then verify space:
   ```bash
   df -h /
   ```

8. **Review per-device configuration and service health:**
   ```bash
   cd ~/aiy-voice-assistant
   nano user.json
   nano .env
   sudo systemctl status aiy.service
   sudo crontab -l
   tailscale status
   ```

   Check that API keys, email settings, location, reminders, and any child/user
   profile details are correct for the new physical device.

### First-time setup

Follow these steps to set up the AI Voice Assistant on your Raspberry Pi from a
fresh OS image:

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
   ```

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
    sudo apt install cmake cython ffmpeg
    poetry install
    pip install google-cloud-speech google-cloud-texttospeech
    ```

    Note: ffmpeg is required for music generation features (MP3 to WAV conversion).

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

    Reminders:
    * Reminders are stored in `reminders.json` at the project root.
    * Nightly cleanup removes completed reminder entries from `reminders.json`.
    * Optional config keys: `reminders_enabled`, `reminders_check_interval_sec`, `reminders_file`,
      `reminder_bell_file`, `reminder_silence_file`, `reminder_speech_delay_sec`.

    LLM cost optimization flags (all optional, safe defaults are `false`):
    * `optimize_prompt_split_dynamic`: split system prompt into static + dynamic parts (for cache-friendly payloads).
    * `claude_enable_prompt_caching`: send Claude prompt-caching headers and cacheable system blocks.
    * `claude_prompt_caching_hybrid_enabled`: keep explicit system caching and also add a message-history cache breakpoint for longer conversations.
    * `claude_hybrid_cache_min_messages` / `claude_hybrid_cache_recent_uncached_messages`: tune where the history breakpoint is placed.
    * `claude_prompt_cache_window_seconds`: freeze dynamic time/location prefix for this duration to improve cache reuse.
    * `claude_prompt_cache_freeze_dynamic_context`: enable/disable time/location freezing while caching.
    * `optimize_prompt_compact`: use shorter built-in instruction text.
    * `optimize_prompt_internal_language`: `ru` or `en` for meta-instructions (responses still follow `$lang` tags).
    * `optimize_dynamic_tool_profiles`: attach only tool subsets per message intent (`chat_only`, `home_control`, `creative`, `research`, `organizer`, `memory`).
    * `optimize_volume_router_enabled`: directly handle obvious volume commands without an LLM turn.
    * `optimize_response_length_control`: use per-turn `max_tokens` caps (`optimize_response_max_tokens_default` / `optimize_response_max_tokens_detailed`).
    * `optimize_tool_usage_tracking_enabled`: persist per-tool usage stats to `optimize_tool_usage_stats_file`.
    * `optimize_tool_rules_prune_by_usage`: shorten system prompt by omitting low-usage tool rules after warmup (`optimize_tool_rules_usage_*`).
    * `cost_per_turn_logging_enabled`: logs per-turn Claude usage/cost in app logs.
    * `claude_cost_*_per_million`: pricing inputs used for cost calculation (set to your current model pricing).
    * Inspect collected stats with: `python scripts/show_tool_usage_stats.py`.

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
        PARALLEL_API_KEY=your_parallel_api_key
        OPENROUTER_API_KEY=your_open_router_api_key
        PERPLEXITY_API_KEY=your_perplexity_api_key
        ELEVENLABS_API_KEY=your_elevenlabs_api_key
        SONIOX_API_KEY=your_soniox_api_key
        GEMINI_API_KEY=your_gemini_api_key                 # For Gemini models, emotion, and speaker embeddings
        WESPEAKER_API_KEY=your_wespeaker_service_token     # Optional bearer token for the speaker server
        TOMORROW_API_KEY=your_tomorrow_io_api_key        # For weather data
        GEOCODE_API_KEY=your_maps_co_geocoding_api_key   # For location lookup
        WAQI_API_KEY=your_waqi_api_key                   # For air quality data
        OPENUV_API_KEY=your_openuv_api_key               # For UV index data
        MINIMAX_API_KEY=your_minimax_api_key             # For music generation (optional)
        HUME_API_KEY=your_hume_api_key                   # For Hume voice emotion detection (optional)
    ```
    Notes
    1. Make sure to keep your `.env` file secure and never commit it to version control.
    2. Depending on configuration, some of these API keys may be unnecessary.
    3. GOOGLE_API_KEY should support the timezone API.
    4. `web_search_providers` controls the search provider order. Parallel Search is used when `PARALLEL_API_KEY` is set; optional config keys include `parallel_search_mode`, `parallel_search_max_results`, `parallel_search_location`, and `parallel_search_after_date`. The `internet_search` tool accepts `location` for localized search, and still forwards `after_date` if a caller supplies it. Additional query variants are pooled into one markdown evidence set, capped by `web_search_max_query_variants` (default 3). Returned web search markdown is saved by default to `web_search_reports_dir` (`web_search_reports`) and, when cubie-server is reachable, to its shared documents folder for web viewing. Set `web_search_save_reports` or `web_search_save_to_documents` to `false` to disable either behavior. Nightly cleanup removes saved web search reports older than `web_search_reports_retention_days` (default 30). Use `list_web_search_reports` and `get_web_search_report` to reuse saved results later.


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

8. **Music Generation:**
   - Ask the assistant to sing songs, generate music, or play lullabies
   - Examples:
     - "Спой колыбельную про звёздочки" (Sing a lullaby about stars)
     - "Спой песню про лето" (Sing a song about summer)
     - "Сгенерируй весёлую музыку" (Generate happy music)
     - "Пой песню с такими словами: [your lyrics]" (Sing a song with these words: [your lyrics])
   - Music starts playing within 1-2 seconds of generation
   - You can interrupt playback at any time by pressing the button
   - Requires MiniMax API key (see setup instructions)

9. **Weather Information:**
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
- Configure speech recognition and STT context hints:
  - `speech_recognition_service` (google/yandex/openai/elevenlabs/soniox)
  - `stt_context_enabled`, `stt_context_prefix`, `stt_context_max_age_sec`, `stt_context_max_length`

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
  - `audio.py`: Audio processing and speech recognition
  - `config.py`: Configuration management
  - `conversation_manager.py`: Manages conversation flow and memory
  - `dialog.py`: Main conversation loop
  - `email_tools.py`: Email functionality
  - `emotion_engine.py`: Voice emotion detection using Hume AI
  - `responce_player.py`: Audio playback and LED control
  - `stt_engine.py`: Speech-to-text engines
  - `stress_tool.py`: Tool for adding stress marks to Russian words
  - `tools.py`: Utility functions
  - `tts_engine.py`: Text-to-speech engines
  - `weather/`: Weather-related package:
    - `aqi.py`: Air Quality Index data fetching using WAQI API
    - `moon.py`: Astronomical calculations for lunar phases and information
    - `openuv.py`: UV index and ozone data fetching using OpenUV API
    - `sunrise.py`: Solar calculations and data (sunrise, sunset, dawn, dusk)
    - `tool.py`: Enhanced weather tool integrating multiple data sources:
      - Basic weather conditions and forecasts
      - Moon phase information
      - UV index and air quality data
      - Solar data integration
  - `web_search.py` & `web_search_tool.py`: Web search functionality
  - `llm_tools.py`: Language model specific tools
  - `code_interpreter_tool.py`: Tool for executing Python code and returning results
  - `volume_control_tool.py`: Tool for adjusting speaker volume
  - `wizard_tool.py`: Advanced analytical reasoning tool for complex questions
  - `minimax_music_tool.py`: Music generation tool using MiniMax API for creating songs and lullabies

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

- For music generation issues:
  * Verify MINIMAX_API_KEY is set in .env file
  * Ensure ffmpeg is installed: ```ffmpeg -version```
  * Check MiniMax API key validity and quota
  * Check /tmp directory has sufficient space for temporary files
  * Music generation requires streaming support - verify network connection
  * If playback is interrupted, temporary WAV files are automatically cleaned up
  * Maximum 50 temporary music files are kept, older files are auto-deleted

- For voice emotion detection issues:
  * Select provider with `emotion_engine_provider`: `gemini` (default), `hume`, or `none`
  * For transition-mode comparison, set `emotion_comparison_enabled` to `true`; the primary provider stays on the critical path while the shadow provider runs in parallel and writes JSONL rows to `emotion_comparison_log_path` (default: `logs/emotion_comparison.jsonl`)
  * Control annotation verbosity with `emotion_annotation_omit_labels` (default: `["neutral", "calm"]`), `emotion_annotation_score_decimals` (default: `1`), and `emotion_annotation_include_scores` (default: `true`)
  * Verify HUME_API_KEY is set in .env file
  * If using Gemini emotion detection, verify GEMINI_API_KEY is set in .env file
  * Ensure `emotion_detection_enabled` is set to `true` in config.json
  * Install websockets package: ```pip install websockets```
  * Check Hume API key validity at https://platform.hume.ai
  * Gemini emotion detection uses end-of-turn audio understanding by default; Gemini Live WebSocket streaming requires a Live model, not Flash-Lite
  * Comparison rows include both providers' top emotions and per-provider latency measurements
  * Summarize collected comparison rows with: `python scripts/show_emotion_comparison_stats.py`
  * Emotion detection runs in parallel with STT - if STT works but emotions aren't detected, check Hume API connectivity
  * If emotion detection times out, adjust `emotion_detection_timeout` in config (default: 10 seconds)
  * Emotions are added as annotations to user messages, e.g., `[User emotion: excited (0.82)]`

- For passive speaker recognition:
  * `speaker_embedding_provider` selects `wespeaker`, `gemini`, or `none`. The recommended
    `wespeaker` provider sends WAV audio to the dedicated speaker embedding service in
    `speaker_server/`; set its HTTPS endpoint in `wespeaker_embedding_url`.
  * Store the optional service bearer token in `WESPEAKER_API_KEY`. The server uses the
    speaker-specific WeSpeaker ResNet34-LM ONNX model by default and returns a
    model-hash-namespaced embedding, so its profiles cannot be mixed with the old Gemini
    embedding space.
  * The Gemini provider remains available for comparison, using `gemini-embedding-2` and
    `GEMINI_API_KEY`, but real Anton/Tanya profiles showed insufficient speaker separation.
  * Natural self-introductions are interpreted by the existing conversation LLM rather than a
    fixed phrase parser. When the user explicitly identifies themself, the model adds a hidden
    `$speaker: Anton$` annotation to its raw response. The annotation is removed before history
    storage and speech synthesis, then associates the current audio embedding with that name.
  * Profiles are progressive normalized centroids stored in the untracked
    `speaker_profiles.json` file. Copy or synchronize that file to share profiles between Cubies.
  * Tune `speaker_match_threshold`, `speaker_match_margin`, and
    `speaker_profile_update_threshold` only after comparing same-speaker and different-speaker
    scores from real microphones. Existing values are conservative migration placeholders, not
    calibrated WeSpeaker thresholds.
    Run `python scripts/evaluate_speaker_embeddings.py Anton=a.wav Anton=b.wav Maria=c.wav`
    with several varied recordings per person to print score ranges and a candidate threshold.
  * Speaker embedding runs alongside STT and emotion analysis. A late result updates profiles in
    the background and never delays the response. The last recognized conversational speaker is
    reused for up to `speaker_context_max_age_sec` (default 300 seconds) while the new embedding
    is in flight, then cleared after repeated mismatches. Context-only identity remains internal
    and is not added to the user message; annotations require current-turn evidence.
  * Recognized speakers are added to model context, e.g., `[User speaker: Anton (0.82)]`.
  * Build and deploy the server using `speaker_server/Dockerfile`; detailed local commands and
    endpoint configuration are in `speaker_server/README.md`.

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
