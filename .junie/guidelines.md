Project-specific development guidelines

This document collects practical, project-specific notes for advanced contributors to the AI-powered voice assistant in this repository. It focuses on configuration, build/run/test specifics, and development gotchas discovered in the current codebase and test suite.

1. Build and configuration instructions

1.1. Runtime and dependency management
- Python: 3.9.x (project is validated on 3.9; tests and dependencies assume this). pyenv is recommended to pin 3.9 locally (see README.md for detailed steps).
- Poetry: used for dependency management. Install dependencies via:
  - poetry install
  - If you prefer the system interpreter and an already-activated venv, you can also run: pip install -r generated requirements if needed, but Poetry is the source of truth.
- Protobuf/grpc constraints:
  - protobuf is pinned to 3.20.3 and grpcio to 1.67.0 in pyproject.toml. Do not upgrade lightly; several generated stubs are vendored and aligned with these versions.

1.2. Service/API keys and sensitive configuration
- The assistant relies on multiple external services (OpenAI, Anthropic, Google, Yandex, ElevenLabs, Tavily, OpenRouter, Perplexity, Tomorrow.io, Maps.co, WAQI, OpenUV, etc.).
- Configuration precedence (implemented in src/config.py and described in README.md) is:
  1) Direct arguments to Config
  2) Environment variables with APP_ prefix (e.g., APP_TEST_ENV)
  3) user.json (user overrides, not committed)
  4) config.json (shared defaults, committed)
- For local development, place sensitive values in a .env at the repo root and load them via python-dotenv (used in several scripts). Example vars from README:
  - OPENAI_API_KEY, ANTHROPIC_API_KEY, EMAIL_PASSWORD, GOOGLE_CUSTOMSEARCH_KEY, TAVILY_API_KEY, OPENROUTER_API_KEY, PERPLEXITY_API_KEY, ELEVENLABS_API_KEY, GEMINI_API_KEY, TOMORROW_API_KEY, GEOCODE_API_KEY, WAQI_API_KEY, OPENUV_API_KEY

1.3. Running the assistant locally
- The main entry point is main.py; scripts/run.sh contains an opinionated launch wrapper that exports environment and starts the assistant.
- Hardware abstractions: The code targets Google Voice Kit V2 (AIY). For non-hardware environments and in tests, AIY modules are mocked (see tests/mock_aiy.py). If you add code that imports aiy.* directly, ensure tests can inject the mocks before import.

1.4. Email sending
- src/email_tools.py supports both sync (smtplib) and async (aiosmtplib) email sending. Email password is read from EMAIL_PASSWORD env var. SMTP settings are configured through Config (with sensible defaults in absence of user.json).

2. Testing information

2.1. Test framework and discovery
- Tests are written with the standard library unittest (not pytest).
- The repository provides a runner:
  - python3 run_tests.py
- Standard unittest discovery also works:
  - python3 -m unittest discover -s tests -p 'test_*.py' -v
- To run a specific test module/class/method:
  - python3 -m unittest tests.test_tools -v
  - python3 -m unittest tests.test_dialog.TestDialogManager.test_main_loop_async -v

2.2. Environment for tests
- Tests are designed to run offline and mock external dependencies:
  - AIY hardware modules are faked by tests/mock_aiy.py and are injected into sys.modules before importing src.* in dialog/response player tests.
  - Cloud and TTS/STT dependencies (google.cloud.*, pydub, yandex, etc.) are mocked in tests where needed.
- Config-dependent tests (tests/test_config.py) create and remove temporary files (e.g., test_config.json) and rely on APP_ prefixed env vars. If you modify Config, keep backward compatibility to avoid breaking these tests.
- Async tests use unittest.IsolatedAsyncioTestCase.

2.3. Known compatibility notes when running all tests
- OpenAI SDK: Some tests patch openai.AsyncOpenAI. Ensure your installed openai package matches the newer >=1.0 API that includes AsyncOpenAI, or adapt the tests. If you cannot control your environment version, run a subset of tests or adjust your virtualenv accordingly (poetry install will pull a compatible version per pyproject.toml).
- Third-party mocks: If you add new imports from aiy.*, google.cloud.*, or audio libraries in src, expand the mocks in tests/mock_aiy.py and in the relevant test modules to keep test isolation.

2.4. How to add a new test
- Create a file under tests/ named test_<feature>.py.
- Use unittest.TestCase or unittest.IsolatedAsyncioTestCase.
- Import from src.* only after setting up any necessary sys.modules mocks (if your code imports aiy.* or vendor SDKs at module scope).
- Example skeleton:

  import unittest
  from src.tools import extract_sentences

  class TestMyFeature(unittest.TestCase):
      def test_basic(self):
          self.assertEqual(extract_sentences("Hello."), ["Hello."])

  if __name__ == '__main__':
      unittest.main()

2.5. Demonstration: creating and running a simple test
- Example file created during validation: tests/test_guidelines_smoke.py
  Content:

  import unittest
  from src.tools import extract_sentences

  class TestGuidelinesSmoke(unittest.TestCase):
      def test_extract_sentences_smoke(self):
          text = "Hello world. Привет мир!"
          self.assertEqual(extract_sentences(text), ["Hello world.", "Привет мир!"])

  if __name__ == "__main__":
      unittest.main()

- Run just this test:
  python3 -m unittest tests.test_guidelines_smoke -v

- Expected output (observed locally):
  test_extract_sentences_smoke (tests.test_guidelines_smoke.TestGuidelinesSmoke) ... ok
  ----------------------------------------------------------------------
  Ran 1 test in 0.001s
  OK

Note: Running the entire suite via run_tests.py currently depends on your installed OpenAI SDK featuring AsyncOpenAI and on behavior that may differ between environments. For rapid iteration on unrelated changes, prefer running targeted modules or methods as shown above.

3. Additional development information

3.1. Code style and typing
- The codebase is gradually typed; public APIs in src/ commonly use type hints. Maintain or improve typing when editing public modules. Pydantic v1.9.2 models appear in src/ai_models.py, so avoid introducing Pydantic v2-specific patterns unless you add compatibility shims.

3.2. Concurrency model
- Dialog flow blends asyncio (network I/O, TTS synthesis) with background threads (ResponsePlayer merge/play pipelines). When modifying ResponsePlayer (src/responce_player.py):
  - Keep queue/condition usage, thread lifecycle, and _should_play/_stopped flags consistent to avoid deadlocks.
  - Light behavior is manipulated via LEDs mock in tests; adjust_rgb_brightness must keep 0..255 bounds.
- Email sending in src/email_tools.py offers both sync and async variants; the async path uses aiosmtplib with opportunistic STARTTLS (falling back if not supported). Maintain integer port types for aiosmtplib.

3.3. External API clients
- AI models are abstracted in src/ai_models.py and src/ai_models_with_tools.py. Avoid importing heavyweight SDKs at module import time inside tests; where unavoidable, feature-detect or lazy-import within functions to simplify mocking.
- Web search (src/web_search.py, src/web_search_tool.py) depends on external APIs; preserve timeouts and backoff strategies if you make changes.

3.4. Configuration system (src/config.py)
- Config resolves values from env (APP_ prefix), user.json, and config.json in that order. Tests rely on .get('test_env') reading APP_TEST_ENV and on reading temporary test_config.json when provided to Config(path) in the constructor. If you change key normalization or env var mapping, update tests accordingly.

3.5. Running on Raspberry Pi / AIY
- For production, follow the more exhaustive steps in README.md (ZSH/Oh-My-Zsh, swap adjustments, systemd service setup). Development on non-ARM desktops is practical due to the comprehensive mocks in tests.

3.6. Scripts
- scripts/run.sh: starts the assistant (loads .env, activates venv, runs main.py).
- scripts/check_logs.sh: simple log inspection helper.
- scripts/send_email.sh: demonstrates email sending via environment-configured SMTP.
- setup_service.sh: installs a systemd service to run the assistant at boot (see README.md for details and safety considerations).

4. Quickstart commands for contributors
- Install deps: poetry install
- Run targeted tests:
  - python3 -m unittest tests.test_tools -v
  - python3 -m unittest tests.test_dialog.TestDialogManager.test_main_loop_async -v
- Run all tests (may depend on SDK versions; prefer Poetry venv):
  - poetry run python run_tests.py
- Launch the app (development):
  - poetry run python main.py

Housekeeping for this note
- During preparation of this file, a demonstration test tests/test_guidelines_smoke.py was created and executed successfully as shown above; it should be removed after reading to keep the repo clean if not needed. In automated workflows, ensure temporary example tests are not committed unless serving documentation purposes.
