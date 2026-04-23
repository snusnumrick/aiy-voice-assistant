import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union

import aiofiles
import requests

from src.ai_models import GeminiAIModel, OpenAIModel, to_reasoning_effort
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
from src.server_utils import get_server_url

logger = logging.getLogger(__name__)


def _discover_doc_folder() -> Optional[Path]:
    """Discover cubie-server and get documents folder path"""
    server_url = get_server_url()

    try:
        logger.info(f"🔍 Discovering cubie-server at {server_url}...")
        response = requests.get(f"{server_url}/api/config/folders", timeout=5)

        if response.status_code == 200:
            config = response.json()
            doc_folder = config.get("documents", "")
            if doc_folder and os.path.exists(doc_folder):
                logger.info(f"✅ Server discovered! Doc folder: {doc_folder}")
                return Path(doc_folder)
            else:
                logger.warning(
                    f"⚠️ Server returned config but doc folder not found: {doc_folder}"
                )
        else:
            logger.warning(f"❌ Server returned status {response.status_code}")

    except requests.exceptions.RequestException as e:
        logger.warning(f"❌ Cannot connect to cubie-server: {e}")
    except Exception as e:
        logger.warning(f"❌ Error discovering doc folder: {e}")

    return None

def fix_markdown_formatting(text: str) -> str:
    """
    Fix common markdown formatting issues with conservative pattern matching.

    Target patterns (in the malformed input):
    - )1. ** -> )\n\n1. **    (numbered list after closing paren)
    - ):- -> ):\n\n-           (dash list after colon)
    - ).- -> ).\n-             (dash list after period)
    - ).2. ** -> ).\n\n2. **   (numbered list after paren+period)
    - ).## -> ).\n\n##         (header after paren+period)
    - ")Capital -> ")\n\nCapital (new sentence after quote+paren)
    - word## -> word\n\n##     (header directly after word)
    """
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")

    # ============================================================
    # Pattern 1: Numbered list after closing paren (no period between)
    # ")1. **text" -> ")\n\n1. **text"
    # ============================================================
    text = re.sub(r"\)(\d+\.\s+\*\*)", r")\n\n\1", text)

    # ============================================================
    # Pattern 2: Dash list or numbered list after colon
    # ":- text" -> ":\n\n- text"
    # ============================================================
    text = re.sub(r":(\s*)(-\s)", r":\n\n\2", text)

    # ============================================================
    # Pattern 3: Dash list after period (end of sentence)
    # ".- **text" -> ".\n- **text"
    # ============================================================
    text = re.sub(r"\.(-\s+\*\*)", r".\n\1", text)
    text = re.sub(r"\.(-\s+[А-Яа-яA-Za-z])", r".\n\1", text)

    # ============================================================
    # Pattern 3b: Dash list after closing paren (header ends, list starts)
    # ")- **text" -> ")\n\n- **text"
    # ============================================================
    text = re.sub(r"\)(-\s+\*\*)", r")\n\n\1", text)
    text = re.sub(r"\)(-\s+[А-Яа-яA-Za-z])", r")\n\n\1", text)

    # ============================================================
    # Pattern 4: Numbered list after period
    # ".2. **text" -> ".\n\n2. **text"
    # ============================================================
    text = re.sub(r"\.(\d+\.\s+\*\*)", r".\n\n\1", text)

    # ============================================================
    # Pattern 5: Header after paren+period or just period
    # ").## " or ".## " -> before header
    # ============================================================
    text = re.sub(r"\)\.(#{1,6}\s)", r").\n\n\1", text)
    text = re.sub(r"([А-Яа-яA-Za-z\"\»])\.(#{1,6}\s)", r"\1.\n\n\2", text)
    text = re.sub(
        r"\.(#{1,6}\s)", r".\n\n\1", text
    )  # General: any period before header

    # ============================================================
    # Pattern 6: New sentence after closing quote+paren
    # ")Capital -> ")\n\nCapital
    # But be careful - only when followed by Cyrillic/Latin capital
    # ============================================================
    text = re.sub(r"(\"\))([А-ЯA-Z])", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 7: Header directly after word/letter (no newline)
    # "word## Header" -> "word\n\n## Header"
    # ============================================================
    text = re.sub(r"([А-Яа-яA-Za-z])(#{1,6}\s+)", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 8: New sentence after bold text ends
    # "**.Capital" -> "**.\n\nCapital" (period after bold, then new sentence)
    # ============================================================
    text = re.sub(r"(\*\*\.)([А-ЯA-Z])", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 9: Two headers concatenated (header ends, another begins)
    # "## Header1)### Header2" -> "## Header1)\n\n### Header2"
    # ============================================================
    text = re.sub(r"(\)|\"|\'|»)(#{1,6}\s+)", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 10: New sentence after closing paren (not header numbering)
    # "слово)Сознательное" -> "слово)\n\nСознательное"
    # Require letter/quote before paren (not digit, to preserve "## 1)Title")
    # ============================================================
    text = re.sub(r"([а-яa-zА-Яа-я\"\»])\)([А-Я][а-я])", r"\1)\n\n\2", text)

    # ============================================================
    # Pattern 10b: Add space after closing paren when followed by lowercase letter
    # "психопатология)заметно" -> "психопатология) заметно"
    # ============================================================
    text = re.sub(r"(?<!\d)\)([А-Яа-яA-Za-z])", r") \1", text)

    # ============================================================
    # Pattern 11: Add space after header numbering paren
    # "## 1)Разбор" -> "## 1) Разбор"
    # ============================================================
    text = re.sub(r"(#{1,6}\s+\d+\))([А-ЯA-Zа-яa-z])", r"\1 \2", text)

    # ============================================================
    # Pattern 11a: Add space after ## when followed directly by digit
    # "##1." -> "## 1."
    # ============================================================
    text = re.sub(r"(#{1,6})(\d)", r"\1 \2", text)

    # ============================================================
    # Pattern 11a2: Split inline headers after punctuation or text
    # ".## 2" -> ".\n\n## 2"
    # "text## 2" -> "text\n\n## 2"
    # ============================================================
    text = re.sub(r"([.!?…»\"\)])\s*(#{1,6}\s*\S)", r"\1\n\n\2", text)
    text = re.sub(r"([А-Яа-яA-Za-z])\s*(#{2,6}\s*\S)", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 11b: Add space after letter+paren in headers (like "### A)Text")
    # "### A)Нейронаучная" -> "### A) Нейронаучная"
    # ============================================================
    text = re.sub(r"(#{1,6}\s+[A-ZА-Я]\))([А-Яа-яA-Za-z])", r"\1 \2", text)

    # ============================================================
    # Pattern 11c: Inline numbered enumeration after colon (BEFORE colon-space pattern!)
    # ":1)**text" -> ":\n\n1) **text"
    # ============================================================
    text = re.sub(r":(\d+)\)(?=\S)", r":\n\n\1) ", text)

    # ============================================================
    # Pattern 11c2: Inline numbered enumeration with digit-period format after colon
    # ":1. Для" -> ":\n\n1. Для"
    # ============================================================
    text = re.sub(r":(\d+)\.\s*(\S)", r":\n\n\1. \2", text)

    # ============================================================
    # Pattern 11d: Inline numbered enumeration after period
    # ".2)**text" -> ".\n\n2) **text"
    # ============================================================
    text = re.sub(r"\.(\d+)\)(?=\S)", r".\n\n\1) ", text)

    # ============================================================
    # Pattern 11e: Inline numbered enumeration with period-digit-period format
    # ").2. Сознание" -> ").\n\n2. Сознание"
    # "мозга.4. Теории" -> "мозга.\n\n4. Теории"
    # BUT NOT "## 5.1." or "4.1." (section numbers)
    # Require a letter before the period
    # ============================================================
    text = re.sub(r"([а-яa-zА-Яа-я])\.(\d+)\.\s*(\S)", r"\1.\n\n\2. \3", text)

    # ============================================================
    # Pattern 11f: Inline enumeration after closing paren + period
    # ").2. Сознание" -> ").\n\n2. Сознание"
    # ============================================================
    text = re.sub(r"\)\.(\d+)\.\s*(\S)", r").\n\n\1. \2", text)

    # ============================================================
    # Pattern 12a: Split same-line horizontal rule + title/header
    # "--- # Title" -> "---\n\n# Title"
    # ============================================================
    text = re.sub(r"(?m)^(---+)\s+(#{1,6}\s*[^\n]+)$", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 12: Add space after colon when followed directly by letter or quote
    # "Перспектива A:нейронаучная" -> "Перспектива A: нейронаучная"
    # "Перспектива D:«радикальные»" -> "Перспектива D: «радикальные»"
    # ============================================================
    text = re.sub(r":([А-Яа-яA-Za-z«\"])", r": \1", text)

    # ============================================================
    # Pattern 12b: Separate horizontal rules (---) from preceding text
    # "активность.---" -> "активность.\n\n---"
    # ============================================================
    text = re.sub(r"([^\n\-])(---+)", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 12c: Ensure horizontal rules have blank line after them too
    # "---## Header" -> "---\n\n## Header"
    # ============================================================
    text = re.sub(r"(---+)([^\n\-])", r"\1\n\n\2", text)

    # ============================================================
    # Pattern 14: Bold text after closing paren (new idea/concept)
    # ")** Идея" -> ")\n\n**Идея"
    # ============================================================
    text = re.sub(r"\)(\*\*[А-ЯA-Z])", r")\n\n\1", text)

    # ============================================================
    # Pattern 15: Missing space between adjacent sentences
    # "причинность.На" -> "причинность. На"
    # "состояний.С философской" -> "состояний. С философской"
    # ============================================================
    text = re.sub(r"([а-яa-z])\.([А-ЯA-Z])", r"\1. \2", text)

    # ============================================================
    # Pattern 15b: Missing space after period following quote/paren
    # '").Это' -> '"). Это'
    # ============================================================
    text = re.sub(r"(\"|»|\)|\*)\.([А-ЯA-Z])", r"\1. \2", text)

    # ============================================================
    # Pattern 16: Add space after digit+paren when followed by ** (bold enumeration)
    # "1)**text" -> "1) **text" (cleanup any remaining)
    # ============================================================
    text = re.sub(r"(\d\))(\*\*)", r"\1 \2", text)

    # ============================================================
    # Pattern 13: Normalize list indentation - remove leading spaces before dash lists
    # "   - item" -> "- item" (when it's a top-level list item, not a sub-item)
    # ============================================================
    # Remove leading whitespace before dash when preceded by newline and comma-ending line
    lines = text.split("\n")
    normalized_lines = []
    for i, line in enumerate(lines):
        # Check if line is an indented dash list item
        match = re.match(r"^(\s+)(-\s+)", line)
        if match and i > 0:
            prev_line = normalized_lines[-1] if normalized_lines else ""
            # If previous line ends with comma and is a list item, keep same level
            # If previous line is a regular dash list item (no indent), remove indent
            prev_is_list = re.match(r"^-\s+", prev_line.strip())
            if prev_is_list and not prev_line.startswith(" "):
                # Previous is unindented list, this should be too
                line = re.sub(r"^\s+(-\s+)", r"\1", line)
        normalized_lines.append(line)
    text = "\n".join(normalized_lines)

    # ============================================================
    # LINE-BY-LINE PROCESSING for blank line insertion
    # ============================================================
    lines = text.split("\n")
    result_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()

        is_dash_list = bool(re.match(r"^[-*+]\s", stripped))
        is_num_list = bool(re.match(r"^\d+\.\s+", stripped))
        is_list_item = is_dash_list or is_num_list
        is_header = bool(re.match(r"^#{1,6}\s", stripped))

        if i > 0 and result_lines:
            prev_stripped = result_lines[-1].strip()
            prev_is_list = bool(
                re.match(r"^[-*+]\s", prev_stripped)
                or re.match(r"^\d+\.\s+", prev_stripped)
            )
            prev_is_header = bool(re.match(r"^#{1,6}\s", prev_stripped))
            prev_is_empty = prev_stripped == ""

            # Blank line before header (if not already blank)
            if is_header and not prev_is_empty:
                result_lines.append("")

            # Blank line before start of list block
            elif (
                is_list_item
                and not prev_is_list
                and not prev_is_header
                and not prev_is_empty
            ):
                result_lines.append("")

        result_lines.append(line)

    text = "\n".join(result_lines)

    # ============================================================
    # Merge continuation paragraphs back into the preceding list item
    # when a generic sentence-splitting rule over-separates a bullet.
    # ============================================================
    lines = text.split("\n")
    merged_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        is_list_item = bool(
            re.match(r"^[-*+]\s", stripped) or re.match(r"^\d+\.\s+", stripped)
        )

        if is_list_item:
            current = line.rstrip()
            j = i + 1
            while (
                j + 1 < len(lines)
                and lines[j].strip() == ""
                and lines[j + 1].strip()
                and not re.match(r"^[-*+]\s", lines[j + 1].strip())
                and not re.match(r"^\d+\.\s+", lines[j + 1].strip())
                and not re.match(r"^#{1,6}\s", lines[j + 1].strip())
                and not re.match(r"^---+$", lines[j + 1].strip())
            ):
                current = f"{current} {lines[j + 1].strip()}"
                j += 2
            merged_lines.append(current)
            i = j
            continue

        merged_lines.append(line)
        i += 1

    text = "\n".join(merged_lines)

    # Remove accidental indentation from normalized headers.
    text = re.sub(r"(?m)^[ \t]+(#{1,6}\s)", r"\1", text)

    # ============================================================
    # Ensure blank line after headers
    # ============================================================
    text = re.sub(
        r"(^#{1,6}\s+[^\n]+)\n([^\n\s#])", r"\1\n\n\2", text, flags=re.MULTILINE
    )

    # ============================================================
    # Cleanup: remove excessive blank lines, trim whitespace
    # ============================================================
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = text.split("\n")
    lines = [line.rstrip() for line in lines]
    text = "\n".join(lines)

    return text.strip()


class WizardTool:
    """
    A sophisticated tool for handling complex analytical questions that require deep thinking
    and analysis. It uses a configurable reasoning model (OpenAI or Gemini) to process
    questions thoroughly and provide well-reasoned answers.

    The tool is designed to:
    1. Break down complex questions into analyzable components
    2. Apply systematic thinking to each component
    3. Synthesize findings into comprehensive answers
    4. Handle both synchronous and asynchronous requests
    5. Provide detailed explanations of the reasoning process
    6. Save analysis reports to markdown files
    7. Manage and retrieve saved reports

    Methods:
    - tool_definitions(self) -> List[Tool]
        Returns a list of tool definitions for the WizardTool, including:
        * wise_wizard - For deep analytical reasoning
        * list_wizard_reports - For listing saved wizard reports
        * get_wizard_report - For retrieving a specific report by filename

    - __init__(self, config: Config)
        Initializes the WizardTool with configuration and sets up the configured
        reasoning model.

    - analyze_question(self, question: str) -> Dict
        Breaks down a complex question into its core components for analysis.

    - do_wizardry(self, parameters: Dict[str, any]) -> Union[str, Dict]
        Performs synchronous analysis of the question and returns a detailed response.

    - do_wizardry_async(self, parameters: Dict[str, any]) -> str
        Performs asynchronous analysis and returns the complete answer text (non-streaming; it awaits model streaming internally and returns a single string).

    - save_report_async(self, question: str, answer: str) -> str
        Asynchronously saves analysis report to a markdown file and returns the filepath.

    - save_report(self, question: str, answer: str) -> str
        Synchronously saves analysis report to a markdown file and returns the filepath.

    - list_reports(self) -> str
        Lists all saved wizard reports with metadata (filename, creation time, size) - synchronous version.

    - list_reports_async(self) -> str
        Lists all saved wizard reports with metadata (filename, creation time, size) - asynchronous version.

    - list_reports_sync(self, parameters: Dict[str, any]) -> str
        Synchronous wrapper for list_reports that returns JSON-formatted report metadata.

    - get_report(self, filename: str) -> str
        Retrieves the content of a specific report by filename (async method).

    - get_report_sync(self, parameters: Dict[str, any]) -> str
        Synchronous wrapper for get_report that handles the filename parameter.
    """

    def tool_definitions(self) -> list[Tool]:
        """
        Returns a list of tool definitions for the WizardTool.

        This includes three tools:
        1. wise_wizard - For deep analytical reasoning
        2. list_wizard_reports - For listing saved wizard reports
        3. get_wizard_report - For retrieving a specific report by filename

        Returns:
            List[Tool]: List of Tool definitions
        """
        return [
            Tool(
                name="wise_wizard",
                description="Analyzes complex questions requiring deep thinking and provides "
                            "comprehensive, well-reasoned answers with detailed explanations. ",
                iterative=True,
                programmatic_code_execution_candidate=False,
                parameters=[
                    ToolParameter(
                        name="question",
                        type="string",
                        description="The complex question or problem to analyze",
                    ),
                    ToolParameter(
                        name="context",
                        type="string",
                        description="Optional additional context or background information",
                        required=False
                    ),
                    ToolParameter(
                        name="analysis_depth",
                        type="string",
                        description="Desired depth of analysis (quick, thorough, or comprehensive)",
                        required=False
                    )
                ],
                required=["question"],
                processor=self.do_wizardry_async,
                rule_instructions={
                    "russian": (
                        "О сложных вещах ты можешь спросить мудреца. "
                        "Ответ может занять очень много времени; используйте с осторожностью. "
                        "Перед тем, как задать вопрос, проверьте существующие отчеты, "
                        "используя инструмент list_wizard_reports, возможно, он уже был задан ранее."
                    ),
                    "english": (
                        "For complex questions, you can ask the wise wizard. "
                        "Response may take a very long time; use sparingly. "
                        "Check existing reports, "
                        "using list_wizard_reports tool, before asking wizard, "
                        "maybe the question was asked before."
                    )
                },
            ),
            Tool(
                name="list_wizard_reports",
                description="Lists all saved wizard analysis reports with metadata including filename, "
                            "creation time, and file size. Returns a list of report summaries.",
                iterative=True,
                parameters=[],
                required=[],
                processor=self.list_reports_async,
            ),
            Tool(
                name="get_wizard_report",
                description="Retrieves the full content of a specific wizard report by filename. "
                            "Returns the complete markdown content of the saved analysis.",
                iterative=True,
                programmatic_code_execution_candidate=False,
                parameters=[
                    ToolParameter(
                        name="filename",
                        type="string",
                        description="The filename of the report to retrieve"
                    )
                ],
                required=["filename"],
                processor=self.get_report_async,
            ),
        ]

    def __init__(self, config: Config):
        """Initialize the WizardTool with necessary configurations."""
        # Store config reference for later use
        self.config = config

        # Use a configurable reasoning model; preserve OpenAI as the default.
        self.default_effort = None
        self.thinking_model = self._build_thinking_model(config)

        self.thinking_template = """
Analyze this question:
1. Break down the core components
2. Identify key concepts and relationships
3. Consider multiple perspectives
4. Draw upon relevant knowledge
5. Synthesize insights into a coherent answer

**Formatting requirements:**
- Use proper markdown with blank lines before and after headers (## and ###)
- Use blank lines before and after lists (both - and numbered)
- Use blank lines before and after horizontal rules (---)
- Always put a space after colons, parentheses, and list markers
- For inline enumerations like "1) item 2) item", put each on its own line
- Never concatenate sentences without proper spacing
- Each section should be visually separated

Provide clean markdown content with a concise title.
Do not ask follow-up questions.

Question: {question}
Context: {context}
Depth: {depth}
        """

    @staticmethod
    def _normalize_gemini_model_id(model_id: Union[str, object]) -> str:
        """Allow shorthand Gemini model names like '3.1 pro' in config."""
        normalized = str(model_id).strip().lower()
        normalized = re.sub(r"[\s_]+", "-", normalized)
        if not normalized.startswith("gemini-"):
            normalized = f"gemini-{normalized}"
        return normalized

    def _build_thinking_model(self, config: Config):
        """Create the reasoning model configured for wizard analyses."""
        provider = str(config.get("wizard_model_api", "") or "").strip().lower()
        explicit_gemini_model = config.get("wizard_gemini_model")
        configured_model_id = config.get("wizard_model_id")
        max_tokens = config.get("wizard_max_tokens", 8192)

        if not provider:
            if explicit_gemini_model:
                provider = "gemini"
            elif isinstance(configured_model_id, str) and configured_model_id.strip().lower().startswith("gemini"):
                provider = "gemini"
            else:
                provider = "openai"

        if provider == "gemini":
            thinking_model_id = explicit_gemini_model or configured_model_id or config.get(
                "gemini_model_id", "gemini-1.5-pro-exp-0801"
            )
            thinking_model_id = self._normalize_gemini_model_id(thinking_model_id)
            logger.info("WizardTool using Gemini model: %s", thinking_model_id)
            return GeminiAIModel(
                config,
                model_id=thinking_model_id,
                max_tokens=max_tokens,
            )

        if provider == "openai":
            thinking_model_id = configured_model_id or "gpt-5"
            logger.info("WizardTool using OpenAI model: %s", thinking_model_id)
            return OpenAIModel(
                config,
                model_id=thinking_model_id,
                max_tokens=max_tokens,
            )

        raise ValueError(
            f"Unsupported wizard_model_api: {provider}. Expected 'openai' or 'gemini'."
        )

    def analyze_question(self, question: str) -> dict[str, Any]:
        """Break down a complex question into analyzable components."""
        components: dict[str, Any] = {
            "core_concepts": [],
            "relationships": [],
            "assumptions": [],
            "required_knowledge": []
        }

        # Use the model to analyze the question structure
        json_pattern = '{"core_concepts": [], "relationships": [], "assumptions": [], "required_knowledge": []}'
        analysis_prompt = f"Analyze this question and break it down, return JSON {json_pattern}: {question}"
        response = self.thinking_model.get_response([
            {"role": "user", "content": analysis_prompt}
        ])

        try:
            # Parse the response and structure the components
            parsed = json.loads(response)
            components.update(parsed)
        except json.JSONDecodeError:
            logger.warning(f"Failed to decode JSON response: {response}")
            # Handle free-form text response
            components["analysis"] = response

        logger.info(f"analyze_question: {question} => {components}")
        return components

    def do_wizardry(self, parameters: dict[str, Any]) -> Union[str, dict[str, Any]]:
        """
        Process a complex question and provide a detailed analysis and answer.

        Args:
            parameters: Dictionary containing:
                - question: The main question to analyze
                - context: Optional additional context
                - analysis_depth: Desired depth of analysis (quick, thorough, or comprehensive)

        Returns:
            A structured response containing the analysis and answer
        """
        if "question" not in parameters:
            logger.error("Missing required parameter 'question'")
            return {"error": "Question parameter is required"}

        question = parameters["question"]
        context = parameters.get("context", "")
        # Accept multiple keys and typos; normalize into reasoning_effort
        effort = (
            parameters.get("reasoning_effort")
            or parameters.get("analysis_depth")
            or parameters.get("depth")
            or self.default_effort
        )

        # Format the thinking prompt
        prompt = self.thinking_template.format(
            question=question,
            context=context,
            depth=effort
        )

        try:
            # Get the model's analysis
            response = self.thinking_model.get_response([
                {"role": "user", "content": prompt}
            ], reasoning_effort=to_reasoning_effort(effort))

            filename = self.save_report(question, response)
            logger.info(f"do_wizardry: {response} saved in {filename}")

            # Structure the response
            return {
                "analysis": self.analyze_question(question),
                "response": response,
                "report_filename": filename,
                "meta": {
                    "depth": effort,
                    "timestamp": asyncio.get_event_loop().time()
                }
            }

        except Exception as e:
            logger.error(f"Error in wizard analysis: {str(e)}")
            raise

    async def do_wizardry_async(self, parameters: dict[str, Any]) -> str:
        """
        Asynchronously process a complex question and return the full analysis and answer.

        Args:
            parameters: Dictionary containing:
                - question: The main question to analyze
                - context: Optional additional context
                - analysis_depth: Desired depth of analysis

        Returns:
            str: The complete analysis/answer text produced asynchronously.
        """
        if "question" not in parameters:
            logger.error("Missing required parameter 'question'")
            return ""

        logger.info(f"do_wizardry_async: {parameters}")

        question = parameters["question"]
        context = parameters.get("context", "")
        # Accept multiple keys and typos; normalize into reasoning_effort
        effort = (
            parameters.get("reasoning_effort")
            or parameters.get("analysis_depth")
            or parameters.get("depth")
            or self.default_effort
        )

        # logger.info(f"wizard question: {question}")
        # logger.info(f"wizard context: {context}")
        # logger.info(f"wizard effort: {effort}")

        # Format the thinking prompt
        prompt = self.thinking_template.format(
            question=question,
            context=context,
            depth=effort
        )

        try:
            result = ""
            async for response_chunk in self.thinking_model.get_response_async([
                {"role": "user", "content": prompt}
            ], reasoning_effort=to_reasoning_effort(effort)):
                result += response_chunk

            logger.info(f"do_wizardry_async: wizard response\n{result}")
            filename = await self.save_report_async(question, result)
            logger.info(f"do_wizardry_async: {result} saved in {filename}")

            return result

        except Exception as e:
            logger.error(f"Error in async wizard analysis: {str(e)}")
            raise
    def save_report(self, question: str, answer: str) -> str:
        """Synchronously save the analysis report to a markdown file."""
        return self._save_report(question, answer)

    async def save_report_async(self, question: str, answer: str) -> str:
        """Asynchronously save the analysis report to a markdown file."""
        return self._save_report(question, answer)

    def _save_report(self, question: str, answer: str) -> str:
        """
        Save the analysis report to a markdown file.

        Args:
            question: The original question
            answer: The analysis/answer content

        Returns:
            str: The filepath where the report was saved
        """

        # fix formatting
        content = fix_markdown_formatting(answer)
        title = content.split("\n")[0]
        while title.startswith("#") or title.startswith(" "):
            title = title[1:]

        # Get reports directory from config or use default
        reports_dir = Path(self.config.get("wizard_reports_dir", "wizard_reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp and sanitized question title
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c.isspace() else "_" for c in question)
        safe_title = "_".join(safe_title.split())

        # Truncate title to prevent filename too long errors
        # Conservative limit for cross-platform compatibility (200 bytes total for filename)
        # Reserve space for timestamp (~20 bytes) + extension (4 bytes) + separators
        max_filename_bytes = 200  # Conservative limit for cross-platform compatibility
        max_title_bytes = max_filename_bytes - len(timestamp.encode('utf-8')) - 9  # -9 for ".md" and separators

        # Check byte length of title (important for UTF-8 Cyrillic characters)
        title_bytes = safe_title.encode('utf-8')

        if len(title_bytes) > max_title_bytes:
            # Truncate the title to fit within the available bytes
            # Timestamp already provides uniqueness, so no need for hash
            while len(safe_title.encode('utf-8')) > max_title_bytes:
                safe_title = safe_title[:-1]

        filename = f"{timestamp}_{safe_title}.md"
        filepath = reports_dir / filename

        # Save the markdown file synchronously
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {question}\n\n{answer}")

        logger.info(f"Report saved to: {filepath}")

        # save to doc folder
        doc_folder = _discover_doc_folder()
        if title and content:
            if doc_folder:
                # Apply same filename length limits to doc folder save
                doc_filename = f"{timestamp}_{safe_title}.md"
                # Ensure it doesn't exceed filesystem limits (typically 255 bytes)
                max_doc_filename_bytes = 250  # Conservative limit
                doc_filename_bytes = doc_filename.encode('utf-8')

                if len(doc_filename_bytes) > max_doc_filename_bytes:
                    # Truncate to fit
                    while len(doc_filename.encode('utf-8')) > max_doc_filename_bytes:
                        doc_filename = doc_filename[:-1]
                    # Ensure it still has .md extension
                    if not doc_filename.endswith('.md'):
                        doc_filename = doc_filename[:-3] + '.md'

                doc_path = doc_folder / doc_filename
                with open(doc_path, "w", encoding="utf-8") as f:
                    f.write(content)
                logger.info(f"Saved doc file: {doc_path}")
            else:
                logger.info(f"Doc folder not found, skipping save. generated content:\n{title}\n{content}")

        return str(filepath)

    def list_reports(self) -> str:
        """
        List all saved wizard reports (synchronous version).

        Returns comma-separated list of filenames
        """
        logger.info("Listing saved wizard reports...")
        reports_dir = Path(self.config.get("wizard_reports_dir", "wizard_reports"))

        if not reports_dir.exists():
            logger.warning(f"Reports directory {reports_dir} does not exist")
            return ""

        reports = [filepath.name for filepath in reports_dir.glob("*.md")]
        logger.info(f"Found {len(reports)} reports in {reports_dir}")
        return ','.join(reports)

    async def list_reports_async(self, parameters: dict[str, Any]) -> str:
        """
        List all saved wizard reports (asynchronous version).

        Returns: comma separated list of filenames
        """
        # Since file system operations are fast, we can run the sync version in a thread pool
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.list_reports)

    def list_reports_sync(self, parameters: dict[str, Any]) -> str:
        """
        Synchronous wrapper for listing wizard reports.

        Args:
            parameters: Dictionary (not used, as this method takes no parameters)

        Returns:
            str: comma-separated list of filenames
        """
        return self.list_reports()

    async def get_report_async(self, parameters: dict[str, Any]) -> str:
        """
        Retrieve the content of a specific wizard report.

        Args:
            filename: The filename of the report to retrieve

        Returns:
            str: The content of the report

        Raises:
            ValueError: If the filename is invalid or outside the reports directory
        """
        if "filename" not in parameters:
            raise ValueError("Missing required parameter 'filename'")

        filename = parameters["filename"]

        reports_dir = Path(self.config.get("wizard_reports_dir", "wizard_reports"))
        reports_dir = reports_dir.resolve()

        # Security: ensure filename is within reports_dir (prevent directory traversal)
        filepath = (reports_dir / filename).resolve()

        if not filepath.exists():
            return f"Report '{filename}' not found"

        # Read the file asynchronously
        async with aiofiles.open(filepath, encoding="utf-8") as f:
            content = await f.read()

        logger.info(f"Report '{filename}' retrieved")
        return content

    def get_report_sync(self, parameters: dict[str, Any]) -> str:
        """
        Synchronous wrapper for getting a wizard report.

        Args:
            parameters: Dictionary containing:
                - filename: The filename of the report to retrieve

        Returns:
            str: The content of the report

        Raises:
            ValueError: If the filename is invalid or missing
        """
        if "filename" not in parameters:
            raise ValueError("Missing required parameter 'filename'")

        # Run the async get_report method
        content = asyncio.run(self.get_report_async(parameters))
        return content

# Example usage in debugging/testing
async def test_wizard():
    config = Config()
    wizard = WizardTool(config)

    question = (
        "Как связаны время и сознание с точки зрения современной науки? "
        "Какие существуют теории о природе этой связи и "
        "что говорят последние исследования в нейронауке и философии сознания?"
    )
    logger.info(
        "Wizard test scaffold ready for %s with question: %s",
        wizard.__class__.__name__,
        question,
    )

if __name__ == "__main__":
    import asyncio

    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    asyncio.run(test_wizard())
