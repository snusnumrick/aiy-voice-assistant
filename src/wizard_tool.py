from typing import Dict, Union
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

    Methods:
    - tool_definition(self) -> Tool
        Returns the definition of the tool including parameters and processing options.

    - __init__(self, config: Config)
        Initializes the WizardTool with configuration and sets up the GPT-5 model via OpenAI Responses API.

    - analyze_question(self, question: str) -> Dict
        Breaks down a complex question into its core components for analysis.

    - do_wizardry(self, parameters: Dict[str, any]) -> Union[str, Dict]
        Performs synchronous analysis of the question and returns a detailed response.

    - do_wizardry_async(self, parameters: Dict[str, any]) -> str
        Performs asynchronous analysis and returns the complete answer text (non-streaming; it awaits model streaming internally and returns a single string).
    """

    def tool_definition(self) -> Tool:
        return Tool(
            name="wise_wizard",
            description="Analyzes complex questions requiring deep thinking and provides "
                        "comprehensive, well-reasoned answers with detailed explanations. "
                        "Response may take a very long time; use sparingly. ",
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
        )

    def __init__(self, config: Config):
        """Initialize the WizardTool with necessary configurations."""
        # Use GPT-5 with reasoning via OpenAI Responses API; default to thorough effort
        self.default_effort = None
        self.model = OpenAIModel(
            config,
            model_id="gpt-5"
        )
        self.max_tokens = config.get("max_tokens", 4096)
        self.thinking_template = """
        Analyze this question:
        1. Break down the core components
        2. Identify key concepts and relationships
        3. Consider multiple perspectives
        4. Draw upon relevant knowledge
        5. Synthesize insights into a coherent answer

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
        response = self.model.get_response([
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
            or parameters.get("deoth")
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
            response = self.model.get_response([
                {"role": "user", "content": prompt}
            ], reasoning_effort=to_reasoning_effort(effort))

            # Structure the response
            return {
                "analysis": self.analyze_question(question),
                "response": response,
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
            or parameters.get("deoth")
            or self.default_effort
        )

        # Format the thinking prompt
        prompt = self.thinking_template.format(
            question=question,
            context=context,
            depth=effort
        )

        try:
            result = ""
            async for response_chunk in self.model.get_response_async([
                {"role": "user", "content": prompt}
            ], reasoning_effort=to_reasoning_effort(effort)):
                result += response_chunk
            logger.info(f"do_wizardry_async: {result}")
            return result

        except Exception as e:
            logger.error(f"Error in async wizard analysis: {str(e)}")
            raise

# Example usage in debugging/testing
async def test_wizard():
    config = Config()
    wizard = WizardTool(config)

    question = "Как связаны время и сознание с точки зрения современной науки? Какие существуют теории о природе этой связи и что говорят последние исследования в нейронауке и философии сознания?"

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