import asyncio
import os
from typing import Dict, List, Any, Optional
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
import logging
import aiohttp

logger = logging.getLogger(__name__)

class InterpreterTool:
    """
    This class represents an interpreter tool that can be used to execute Python code in a sandbox environment.
    It provides both synchronous and asynchronous execution methods.

    Class: InterpreterTool

    Methods:
    - tool_definition(self) -> Tool
        Returns the definition of the tool. The Tool object includes the name, description, whether it is iterative, and the parameters required for code execution.

    - __init__(self, config: Config)
        Initializes the InterpreterTool object.

    - _start_processing(self)
        (Private method) Starts the processing indicator (e.g. LED blinking).

    - _stop_processing(self)
        (Private method) Stops the processing indicator (e.g. LED off).

    - execute_code(self, parameters: Dict[str, any]) -> str
        Performs a synchronous code execution using the given parameters. It starts the processing indicator, executes the code, stops the processing indicator, and returns the result.

    - execute_code_async(self, parameters: Dict[str, any]) -> str
        Performs an asynchronous code execution using the given parameters. It starts the processing indicator, executes the code asynchronously, stops the processing indicator, and returns the result.
    """

    base_description = """Evaluates python code in a sandbox environment. 
The environment resets on every execution. 
You must send the whole script every time and print your outputs. 
Script should be pure python code that can be evaluated. 
It should be in python format NOT markdown. 
The code should NOT be wrapped in backticks. 
No python packages are available, so don't use import statements.
print() any output and results so you can capture the output.
The result is JSON dictionary with keys stdout and stderr.
"""

    def tool_definition(self) -> Tool:
        return Tool(
            name="code_interpreter",
            description=self.base_description,
            iterative=True,
            parameters=[
                ToolParameter(name='code', type='string', description='Python code to execute')
            ],
            required=["code"],
            processor=self.execute_code_async
        )

    def __init__(self, config: Config):
        self.api_key = os.environ["RAPID_API_KEY"]
        self.api_base = config.get('judge0_api_base', 'https://judge0-ce.p.rapidapi.com')
        self.headers: Dict[str, str] = {
            "x-rapidapi-host": "judge0-ce.p.rapidapi.com",
            "x-rapidapi-key": self.api_key,
            "Content-Type": "application/json"
        }
        self.python_language_id: int = 71

    async def execute_code_async(self, parameters: Dict[str, Any]) -> str:
        if 'code' not in parameters:
            logger.error("Missing 'code' parameter")
            return "Error: Missing required parameter 'code'"

        code = parameters['code']
        # code = """
# for dist in __import__('pkg_resources').working_set:
#     print (dist.project_name.replace('Python', ''))"""

        submission_data = {
            "source_code": code,
            "language_id": self.python_language_id,
            "stdin": parameters.get('stdin', ""),
        }

        try:
            submission = await self._create_submission(submission_data)
            if not submission:
                return "Error: Failed to create submission"

            result = await self._get_submission_result(submission['token'])
            if not result:
                return "Error: Failed to get submission result"

            return self._format_result(result)
        except aiohttp.ClientError as e:
            logger.error(f"Network error occurred: {e}")
            return f"Error: Network issue occurred - {e}"
        except asyncio.TimeoutError:
            logger.error("Request timed out")
            return "Error: Request timed out"
        except Exception as e:
            logger.error(f"Unexpected error occurred: {e}")
            return f"Error: An unexpected error occurred - {e}"

    async def _create_submission(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                    f"{self.api_base}/submissions?base64_encoded=false&wait=false",
                    headers=self.headers,
                    json=data,
                    timeout=10  # 10 seconds timeout
            ) as response:
                if response.status == 201:
                    return await response.json()
                else:
                    logger.error(f"Failed to create submission: {response.status}")
                    return None

    async def _get_submission_result(self, token: str) -> Optional[Dict[str, Any]]:
        max_attempts = 10
        for attempt in range(max_attempts):
            async with aiohttp.ClientSession() as session:
                async with session.get(
                        f"{self.api_base}/submissions/{token}?base64_encoded=false",
                        headers=self.headers,
                        timeout=10  # 10 seconds timeout
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        if result['status']['id'] not in [1, 2]:  # Not In Queue or Processing
                            return result
                    else:
                        logger.error(f"Failed to get submission result: {response.status}")
                        return None
            await asyncio.sleep(1)  # Wait before next attempt
        logger.error("Max attempts reached while waiting for submission result")
        return None

    def _format_result(self, result: Dict[str, Any]) -> str:
        formatted_result = f"Status: {result['status']['description']}\n"

        if result.get('compile_output'):
            formatted_result += f"Compile Output:\n{result['compile_output']}\n"
        if result.get('stdout'):
            formatted_result += f"Standard Output:\n{result['stdout']}\n"
        if result.get('stderr'):
            formatted_result += f"Standard Error:\n{result['stderr']}\n"
        if result.get('time'):
            formatted_result += f"Execution Time: {result['time']} seconds\n"
        if result.get('memory'):
            formatted_result += f"Memory Used: {result['memory']} KB\n"

        return formatted_result

    async def get_python_version(self) -> str:
        languages = await self.get_languages()
        for lang in languages:
            if lang['id'] == self.python_language_id:
                return lang['name']
        return "Python version information not found"

    async def get_languages(self) -> List[Dict[str, Any]]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                    f"{self.api_base}/languages",
                    headers=self.headers,
                    timeout=10  # 10 seconds timeout
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Failed to get languages: {response.status}")
                    return []
