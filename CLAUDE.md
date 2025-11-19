# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Dependency Management
```bash
# Install dependencies
poetry install

# Add a new dependency
poetry add <package>

# Remove a dependency
poetry remove <package>

# Activate virtual environment
source venv/bin/activate
# or use: poetry run <command>
```

### Running Tests
```bash
# Run all tests
python run_tests.py

# Run specific test file
pytest tests/test_dialog.py

# Run tests with verbose output
pytest tests/ -v
```

### Running the Application
```bash
# Manual run
python main.py --log-dir logs --log-level INFO

# Via poetry
poetry run python main.py

# Check service status (if installed as systemd service)
sudo systemctl status aiy.service

# View service logs
journalctl -u aiy.service -f
```

### Code Formatting
```bash
# Check code with ruff
ruff check src/

# Format code with ruff
ruff format src/
```

## Code Architecture & Structure

### High-Level Flow
The application follows this execution flow:
1. **main.py** (entry point) → Initializes hardware, config, AI models
2. **dialog.py** (`main_loop_async()`) → Button handling, recording, STT, conversation loop
3. **conversation_manager.py** → Context management, user adaptation
4. **ai_models.py** / **ai_models_with_tools.py** → AI integration (Claude, OpenAI, etc.)
5. **tools/** → Specialized capabilities (weather, web search, code execution, etc.)

### Key Directories

- **`src/`** - Core application logic (26 Python modules, ~10K lines)
- **`aiy/`** - Hardware abstraction for Google Voice Kit V2
- **`tests/`** - Unit tests (9 test files)
- **`scripts/`** - Shell deployment scripts
- **`logs/`** - Application logs (auto-rotated)

### Critical Modules

**Entry & Control:**
- `main.py:1` - Application entry point, initialization
- `src/dialog.py:1` - Main conversation loop, button/audio handling
- `src/conversation_manager.py:1` - Conversation state and memory

**AI Integration:**
- `src/ai_models.py:1` - Basic AI model implementations (Claude, OpenAI, Gemini)
- `src/ai_models_with_tools.py:1` - AI models with tool-calling support

**Speech Processing:**
- `src/audio.py:1` - Audio recording/playback
- `src/stt_engine.py:1` - Speech-to-text (Yandex, Google Cloud)
- `src/tts_engine.py:1` - Text-to-speech (Yandex, ElevenLabs)

**Tools (Specialized Capabilities):**
- `src/weather_tool.py:1` - Weather data (Tomorrow.io, WAQI, OpenUV)
- `src/web_search_tool.py:1` - Web search (DuckDuckGo, Google, Tavily)
- `src/code_interpreter_tool.py:1` - Python code execution
- `src/wizard_tool.py:1` - Advanced analytical reasoning
- `src/email_tools.py:1` - Email sending functionality
- `src/volume_control_tool.py:1` - Audio volume control
- `src/minimax_music_tool.py:1` - Music generation using MiniMax API

**Hardware:**
- `aiy/voice/` - Audio hardware interfaces
- `aiy/board.py:1` - Hardware board control
- `aiy/leds.py:1` - LED pattern control

### Configuration System

The application uses a hierarchical configuration system with this precedence:
1. Constructor arguments
2. Environment variables (`APP_*` prefix)
3. `user.json` (user-specific, not committed)
4. `config.json` (shared configuration)

**Key Configuration Files:**
- **`config.json`** - Shared settings: LLM selection, streaming, chunk duration, admin email
- **`user.json.example`** - Template for user-specific overrides (copy to `user.json`)
- **`.env`** - API keys and sensitive data (never committed)

### System Integration

**Service Management:**
- Installed via `setup_service.sh` as `aiy.service`
- Auto-starts on boot
- Runs in tmux session for monitoring

**Tailscale VPN Management:**
- Automatically enables VPN at night (10 PM - 7 AM) for remote maintenance
- Disables during day for performance
- Configurable in cron (`sudo crontab -e`)

## Tool Development Guidelines

### Overview
The application uses a **dynamic rule generation system** for tool usage guidance. Tools can optionally provide `rule_instructions` that specify when and how the AI should use them. These rules are automatically collected and included in the AI model's system prompt based on enabled tools.

### How It Works

**Rule Separation:**
- **Base Rules**: General, tool-independent instructions (timezone handling, memory patterns, formatting rules)
- **Tool-Specific Rules**: When and how to use particular tools (trigger phrases, usage patterns)

**Dynamic Generation:**
The `ConversationManager` collects `rule_instructions` from all enabled tools and combines them with base rules to create the complete system prompt.

### Adding a New Tool with Rule Instructions

1. **Define the Tool Class** with `rule_instructions`:

```python
from src.ai_models_with_tools import Tool, ToolParameter

class YourTool:
    def tool_definition(self) -> Tool:
        return Tool(
            name="your_tool_name",
            description="Brief description of what the tool does",
            iterative=True,  # or False
            parameters=[
                ToolParameter(
                    name="parameter_name",
                    type="string",
                    description="Description of the parameter"
                ),
            ],
            processor=self.your_processor_method,
            required=["parameter_name"],

            # OPTIONAL: Add rule_instructions for when to use this tool
            rule_instructions={
                "russian": (
                    "Когда пользователь просит <что делать>, "
                    "используй инструмент your_tool_name. "
                    "Триггеры: 'фраза 1', 'фраза 2', 'фраза 3'. "
                    "Обязательно <как использовать>."
                ),
                "english": (
                    "When user asks to <what to do>, "
                    "use your_tool_name tool. "
                    "Triggers: 'phrase 1', 'phrase 2', 'phrase 3'. "
                    "Always <how to use>."
                )
            }
        )
```

2. **Register the Tool in main.py**:

```python
# Create tool instance
your_tool = YourTool(config)

# Add to tools list
tools.append(your_tool.tool_definition())

# Pass to ConversationManager
conversation_manager = ConversationManager(
    config, ai_model, timezone, enabled_tools=tools
)
```

### Rule Instruction Best Practices

**Be Specific about Triggers:**
- List common phrases that should trigger the tool
- Include variations in both languages
- Think about implicit triggers, not just explicit mentions

**Example from VolumeControlTool:**
```python
"russian": (
    "Используй инструмент control_speaker_volume, когда пользователь просит "
    "изменить громкость, даже если он не упоминает слово 'громкость' явно. "
    "Триггеры: 'тише', 'громче', 'убавь звук', 'прибавь звук', 'сделай тише', "
    "'сделай громче', 'уменьши громкость', 'увеличь громкость'."
)
```

**Explain Usage:**
- What parameters to use when
- Any validation or special handling
- Expected behavior or responses

**Keep it Concise:**
- Rules should be 1-3 sentences per language
- Focus on when to use the tool, not implementation details
- Use natural language, not technical jargon

### Tool Classes That Don't Need Rules

Tools with obvious usage don't need `rule_instructions`:
- **Email tool**: "Send email" is self-explanatory
- **Web search**: "Search for..." is clear
- **Code interpreter**: "Run code" is straightforward

Only add rules if the tool has:
- Non-obvious trigger phrases
- Specific usage patterns
- Parameter combinations that need explanation

### Testing Tool Rules

To verify your tool's rules are working:

1. Check that the tool is included in the enabled tools list
2. Verify `rule_instructions` are present in the tool definition
3. Test that the rules appear in the ConversationManager's hard_rules
4. Confirm the AI model receives the rules in its system prompt

## Development Notes

- **Primary Language**: Python 3.9+
- **Package Manager**: Poetry
- **Hardware Target**: Raspberry Pi with Google Voice Kit V2
- **Primary Language**: Russian (configurable)
- **Key APIs**: Yandex SpeechKit (default STT/TTS), Claude/OpenAI/Gemini for LLM
- **Tests**: Use `python run_tests.py` or `pytest`

## Logging

Logs are stored in `logs/assistant.log` with:
- Daily rotation (5-day retention)
- Configurable log levels via `--log-level` flag
- Both file logging and systemd journal integration
