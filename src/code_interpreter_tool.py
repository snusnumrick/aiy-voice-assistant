import asyncio
import os
from typing import Dict, Any, List
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
        }

    async def execute_code_async(self, parameters: Dict[str, any]) -> str:
        if "code" not in parameters:
            logger.error("Missing 'code' parameter")
            return "Error: Missing 'code' parameter"

        code = parameters["code"]
        logger.info(f"Executing code via Responses API: \n{code}")
        # Ask the model to strictly return JSON with stdout/stderr only.
        user_prompt = (
            "Run this Python code using Code Interpreter and capture printed output and errors. "
            "Return only a JSON object with keys 'stdout' and 'stderr' (both strings). "
            "Do not include any explanations or additional text. Code follows:\n\n" \
            + code
        )

        # Create a Response (non-stream) with code_interpreter tool enabled
        response = await self._create_response(user_prompt)

        # If the API returned an error object, surface it immediately
        if isinstance(response, dict) and response.get("error", None):
            err = response.get("error", {})
            msg = err.get("message") or str(err)
            logger.error(f"OpenAI Responses error: {msg}")
            return f"Error: {msg}"

        # Some long code runs may return in-progress; wait until completed
        status = response.get("status")
        response_id = response.get("id")
        if response_id and status and status not in ("completed", "failed", "cancelled"):
            response = await self._wait_for_response(response_id)

            # If after waiting we have an error object, return it
            if isinstance(response, dict) and response.get("error", None):
                err = response.get("error", {})
                msg = err.get("message") or str(err)
                logger.error(f"OpenAI Responses error after wait: {msg}")
                return f"Error: {msg}"

        # Extract assistant text result
        text = self._extract_text_from_response(response)
        if not text:
            logger.warning(f"No textual output extracted from response: {json.dumps(response)[:500]}")
            return "No result found"

        # Remove code fences if present and parse JSON
        json_string = text.strip().replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(json_string)
        except Exception as e:
            logger.error(f"Failed to parse JSON from response text: {e}; text=\n{text}")
            return "Error: Failed to parse tool output"

        stdout = data.get("stdout", "")
        stderr = data.get("stderr", "")
        if stderr:
            result = f"there was an error: {stderr}"
        else:
            result = f"the result is {stdout}"

        logger.info(f"openai responses result:\n{result}")
        return result

    async def _create_response(self, input_text: str) -> Dict[str, Any]:
        payload = {
            "model": self.model,
            "input": input_text,
            "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.openai_api_base}/responses",
                headers=self.headers,
                json=payload,
            ) as resp:
                return await resp.json()

    async def _get_response(self, response_id: str) -> Dict[str, Any]:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.openai_api_base}/responses/{response_id}",
                headers=self.headers,
            ) as resp:
                return await resp.json()

    async def _wait_for_response(self, response_id: str, poll_interval: float = 1.0) -> Dict[str, Any]:
        while True:
            resp = await self._get_response(response_id)
            status = resp.get("status")
            if status in ("completed", "failed", "cancelled"):
                return resp
            await asyncio.sleep(poll_interval)

    def _extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """
        Try to robustly extract assistant text content from Responses API schemas.
        """
        # 1) Newer schema: response["output"] is a list of items; pick assistant message text
        output = response.get("output")
        if isinstance(output, list) and output:
            texts: List[str] = []
            for item in output:
                # Message item with content array
                if isinstance(item, dict) and item.get("type") == "message":
                    content = item.get("content") or []
                    texts.extend(self._extract_texts_from_content_array(content))
                # Some variants may flatten text directly
                if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
                    texts.append(item["text"]) 
            if texts:
                return "\n".join(t for t in texts if t)
        # 2) Fallback: response has top-level "content" like a message
        content = response.get("content")
        if isinstance(content, list):
            texts = self._extract_texts_from_content_array(content)
            if texts:
                return "\n".join(t for t in texts if t)
        # 3) Fallback: response["message"]["content"]
        message = response.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            if isinstance(content, list):
                texts = self._extract_texts_from_content_array(content)
                if texts:
                    return "\n".join(t for t in texts if t)
        # 4) Last resort: look for a top-level field "output_text" or "text"
        if isinstance(response.get("output_text"), str):
            return response["output_text"]
        if isinstance(response.get("text"), str):
            return response["text"]
        return ""

    def _extract_texts_from_content_array(self, content: List[Any]) -> List[str]:
        texts: List[str] = []
        for part in content:
            if not isinstance(part, dict):
                continue
            # Common shapes: {"type":"text","text":"..."} or {"type":"output_text","text":"..."}
            t = part.get("text")
            if isinstance(t, str):
                texts.append(t)
            # Sometimes nested as {"type":"text","text":{"value":"..."}}
            if isinstance(t, dict) and isinstance(t.get("value"), str):
                texts.append(t["value"])
        return texts
