import asyncio
import os
from typing import Dict
from src.ai_models_with_tools import Tool, ToolParameter
from src.config import Config
import logging
import json
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
All python packages including requests, matplotlib, scipy, numpy, pandas, \
etc are available.
Convey all results through print(), so you can capture the output.
"""

    def tool_definition(self) -> Tool:
        return Tool(
            name="code_interpreter",
            description=self.base_description,
            iterative=True,
            parameters=[
                ToolParameter(
                    name="code", type="string", description="Python code to execute"
                )
            ],
            required=["code"],
            processor=self.execute_code_async,
        )

    def __init__(self, config: Config):
        self.api_key = os.environ["OPENAI_API_KEY"]
        self.openai_api_base = config.get(
            "openai_api_base", "https://api.openai.com/v1"
        )
        self.model = config.get("code_interpreter_model", "gpt-4o")
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "OpenAI-Beta": "assistants=v2",
        }

    async def execute_code_async(self, parameters: Dict[str, any]) -> str:
        if "code" not in parameters:
            logger.error("Missing 'code' parameter")
            return "Error: Missing 'code' parameter"

        code = parameters["code"]
        logger.info(f"Executing code: \n{code}")
        message = (
            f"run interpreter for this code```\n{code}\n```\n; "
            f"return json dictionary only with keys stdout and stderr, omit explanations"
        )

        # Create an Assistant
        assistant = await self._create_assistant()

        # Create a Thread
        thread = await self._create_thread()

        # Add a Message to the Thread
        await self._add_message_to_thread(thread["id"], message)

        # Run the Assistant
        run = await self._run_assistant(thread["id"], assistant["id"])

        # Wait for the Run to complete
        run = await self._wait_for_run(thread["id"], run["id"])

        # Retrieve the results
        messages = await self._get_messages(thread["id"])

        # Extract and return the last assistant message
        for message in reversed(messages["data"]):
            if message["role"] == "assistant":
                result = message["content"][0]["text"]["value"]

                # Remove the ```json and ``` markers
                json_string = result.strip().replace("```json", "").replace("```", "")

                # Parse the JSON string
                data = json.loads(json_string)

                if data["stderr"]:
                    result = f"there was an error: {data['stderr']}"
                else:
                    result = f"the result is {data['stdout']}"

                logger.info(f"openai assistant result:\n{result}")
                return result

        return "No result found"

    async def _create_assistant(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openai_api_base}/assistants",
                headers=self.headers,
                json={"model": self.model, "tools": [{"type": "code_interpreter"}]},
            ) as response:
                return await response.json()

    async def _create_thread(self):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openai_api_base}/threads", headers=self.headers
            ) as response:
                return await response.json()

    async def _add_message_to_thread(self, thread_id: str, content: str):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openai_api_base}/threads/{thread_id}/messages",
                headers=self.headers,
                json={"role": "user", "content": content},
            ) as response:
                return await response.json()

    async def _run_assistant(self, thread_id: str, assistant_id: str):
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openai_api_base}/threads/{thread_id}/runs",
                headers=self.headers,
                json={"assistant_id": assistant_id},
            ) as response:
                return await response.json()

    async def _wait_for_run(self, thread_id: str, run_id: str):
        while True:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.openai_api_base}/threads/{thread_id}/runs/{run_id}",
                    headers=self.headers,
                ) as response:
                    run = await response.json()
                    if run["status"] in ["completed", "failed", "cancelled"]:
                        return run
                    await asyncio.sleep(1)

    async def _get_messages(self, thread_id: str):
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.openai_api_base}/threads/{thread_id}/messages",
                headers=self.headers,
            ) as response:
                return await response.json()
