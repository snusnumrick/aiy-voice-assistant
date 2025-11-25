from typing import Dict, Union, List
from pathlib import Path
import aiofiles
from datetime import datetime

from src.ai_models import OpenAIModel, to_reasoning_effort
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
import logging
import asyncio
import json

logger = logging.getLogger(__name__)

class WizardTool:
    """
    A sophisticated tool for handling complex analytical questions that require deep thinking
    and analysis. It uses the GPT-5 model (OpenAI Responses API) to process questions thoroughly and provide
    well-reasoned answers.

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
        Initializes the WizardTool with configuration and sets up the GPT-5 model via OpenAI Responses API.

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

    - list_reports(self) -> List[Dict]
        Lists all saved wizard reports with metadata (filename, creation time, size) - synchronous version.

    - list_reports_async(self) -> List[Dict]
        Lists all saved wizard reports with metadata (filename, creation time, size) - asynchronous version.

    - list_reports_sync(self, parameters: Dict[str, any]) -> str
        Synchronous wrapper for list_reports that returns JSON-formatted report metadata.

    - get_report(self, filename: str) -> str
        Retrieves the content of a specific report by filename (async method).

    - get_report_sync(self, parameters: Dict[str, any]) -> str
        Synchronous wrapper for get_report that handles the filename parameter.
    """

    def tool_definitions(self) -> List[Tool]:
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
                        "Ответ может занять очень много времени; используйте с осторожностью. "
                        # "Перед тем, как задать вопрос, проверьте существующие отчеты, "
                        # "используя инструмент list_wizard_reports, возможно, он уже был задан ранее."
                    ),
                    "english": (
                        "Response may take a very long time; use sparingly. "
                        # "Check existing reports, "
                        # "using list_wizard_reports tool, before asking wizard, "
                        # "maybe the question was asked before."
                    )
                },
            ),
            # Tool(
            #     name="list_wizard_reports",
            #     description="Lists all saved wizard analysis reports with metadata including filename, "
            #                 "creation time, and file size. Returns a list of report summaries.",
            #     iterative=True,
            #     parameters=[],
            #     required=[],
            #     processor=self.list_reports_async,
            # ),
            # Tool(
            #     name="get_wizard_report",
            #     description="Retrieves the full content of a specific wizard report by filename. "
            #                 "Returns the complete markdown content of the saved analysis.",
            #     iterative=True,
            #     parameters=[
            #         ToolParameter(
            #             name="filename",
            #             type="string",
            #             description="The filename of the report to retrieve"
            #         )
            #     ],
            #     required=["filename"],
            #     processor=self.get_report_async,
            # ),
        ]

    def __init__(self, config: Config):
        """Initialize the WizardTool with necessary configurations."""
        # Store config reference for later use
        self.config = config

        # Use GPT-5 with reasoning via OpenAI Responses API; default to thorough effort
        self.default_effort = None
        thinking_model_id = config.get("wizard_model_id", "gpt-5")
        max_tokens = config.get("wizard_max_tokens", 8192)
        self.thinking_model = OpenAIModel(
            config,
            model_id=thinking_model_id,
            max_tokens=max_tokens
        )

        self.thinking_template = """
        Analyze this question:
        1. Break down the core components
        2. Identify key concepts and relationships
        3. Consider multiple perspectives
        4. Draw upon relevant knowledge
        5. Synthesize insights into a coherent answer in a clean markdown format

        Question: {question}
        Context: {context}
        Depth: {depth}
        """

    def analyze_question(self, question: str) -> Dict:
        """Break down a complex question into analyzable components."""
        components = {
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

    def do_wizardry(self, parameters: Dict[str, any]) -> Union[str, Dict]:
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
            logger.info(f"do_wizardry_async: {response} saved in {filename}")

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

    async def do_wizardry_async(self, parameters: Dict[str, any]) -> str:
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

        logger.info(f"wizard question: {question})")
        logger.info(f"wizard context: {context})")
        logger.info(f"wizard effort: {effort})")

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

            filename = self.save_report(question, result)
            logger.info(f"do_wizardry_async: {result} saved in {filename}")

            return result

        except Exception as e:
            logger.error(f"Error in async wizard analysis: {str(e)}")
            raise

    def save_report(self, question: str, answer: str) -> str:
        """
        Synchronously save the analysis report to a markdown file.

        Args:
            question: The original question
            answer: The analysis/answer content

        Returns:
            str: The filepath where the report was saved
        """
        # Get reports directory from config or use default
        reports_dir = Path(self.config.get("wizard_reports_dir", "wizard_reports"))
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp and sanitized question title
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() or c.isspace() else "_" for c in question)
        safe_title = "_".join(safe_title.split())
        filename = f"{timestamp}_{safe_title}.md"
        filepath = reports_dir / filename

        # Save the markdown file synchronously
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {question}\n\n{answer}")

        logger.info(f"Report saved to: {filepath}")
        return str(filepath)

    def list_reports(self) -> List[Dict[str, Union[str, int]]]:
        """
        List all saved wizard reports (synchronous version).

        Returns:
            List of dictionaries containing report metadata:
            - filename: str
            - created: str (ISO format timestamp)
            - size: int (bytes)
        """
        reports_dir = Path(self.config.get("wizard_reports_dir", "wizard_reports"))

        if not reports_dir.exists():
            return []

        reports = []
        for filepath in reports_dir.glob("*.md"):
            # stat = filepath.stat()
            reports.append(filepath.name)
            # reports.append({
            #     "filename": filepath.name,
            #     "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            #     "size": stat.st_size
            # })

        logger.info(f"Found {len(reports)} reports in {reports_dir}")
        return reports

    async def list_reports_async(self, parameters: Dict[str, any]) -> List[Dict[str, Union[str, int]]]:
        """
        List all saved wizard reports (asynchronous version).

        Returns:
            List of dictionaries containing report metadata:
            - filename: str
            - created: str (ISO format timestamp)
            - size: int (bytes)
        """
        # Since file system operations are fast, we can run the sync version in a thread pool
        loop = asyncio.get_event_loop()
        reports = await loop.run_in_executor(None, self.list_reports)
        return reports

    def list_reports_sync(self, parameters: Dict[str, any]) -> str:
        """
        Synchronous wrapper for listing wizard reports.

        Args:
            parameters: Dictionary (not used, as this method takes no parameters)

        Returns:
            str: JSON-formatted list of report metadata
        """
        reports = self.list_reports()
        return json.dumps(reports, indent=2, ensure_ascii=False)

    async def get_report_async(self, parameters: Dict[str, any]) -> str:
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
        async with aiofiles.open(filepath, "r", encoding="utf-8") as f:
            content = await f.read()

        logger.info(f"Report '{filename}' retrieved")
        return content

    def get_report_sync(self, parameters: Dict[str, any]) -> str:
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

        filename = parameters["filename"]

        # Run the async get_report method
        content = asyncio.run(self.get_report_async(filename))
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

    test_params = {
        # "question": "What are the philosophical implications of quantum entanglement?",
        "question": question,
        # "context": "Consider both scientific and metaphysical perspectives",
        "analysis_depth": "thorough"
    }

    result = await wizard.do_wizardry_async(test_params)
    print(result, end="")

if __name__ == "__main__":
    import asyncio
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    load_dotenv()

    asyncio.run(test_wizard())