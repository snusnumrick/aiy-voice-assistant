import unittest
from unittest.mock import AsyncMock, patch

from src.ai_models_with_tools import ClaudeAIModelWithTools, Tool
from src.config import Config


class TestClaudeIncompleteTools(unittest.IsolatedAsyncioTestCase):
    def make_model(self):
        processor = AsyncMock(return_value="sent")
        tool = Tool(
            name="send_email_to_user",
            description="Send email",
            iterative=False,
            parameters=[],
            required=[],
            processor=processor,
        )
        model = ClaudeAIModelWithTools(
            Config(config_file="missing.json", user_config_file="missing-user.json"),
            tools=[tool],
        )
        return model, processor

    async def run_stream(self, ending):
        model, processor = self.make_model()

        async def events(*args, **kwargs):
            yield {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use", "name": "send_email_to_user", "id": "t1",
                },
            }
            yield {
                "type": "content_block_delta",
                "delta": {"type": "input_json_delta", "partial_json": '{"body": "unfinished'},
            }
            for event in ending:
                yield event

        model._get_response_async = events
        with patch("src.tools.asyncio.sleep", new=AsyncMock()):
            output = [item async for item in model._get_response_async_streaming([])]
        processor.assert_not_awaited()
        self.assertIn("[[TOOL_RESULT]]", output)
        self.assertTrue(any("не был выполнен" in item for item in output))
        records = model.consume_tool_provenance_messages()
        self.assertEqual(len(records), 1)
        self.assertIn("NOT executed", records[0]["content"])

    async def test_truncated_json_at_block_stop_reports_failure(self):
        await self.run_stream([
            {"type": "content_block_stop"},
            {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
            {"type": "message_stop"},
        ])

    async def test_message_stop_with_open_tool_reports_failure(self):
        await self.run_stream([{"type": "message_stop"}])

    async def test_stream_ending_with_open_tool_reports_failure(self):
        await self.run_stream([])

    async def test_valid_email_executes_and_retains_result_without_body(self):
        model, processor = self.make_model()
        output = [item async for item in model._process_tool_use_streaming(
            {"name": "send_email_to_user", "id": "t1",
             "input": '{"subject": "Draft", "body": "private draft"}'},
            [],
        )]
        processor.assert_awaited_once()
        self.assertEqual(output, ["[[TOOL_RESULT]]"])
        record = model.consume_tool_provenance_messages()[0]["content"]
        self.assertIn("Draft", record)
        self.assertIn("sent", record)
        self.assertNotIn("private draft", record)

    async def test_token_limit_retry_doubles_budget_and_sends_once(self):
        model, processor = self.make_model()
        model.set_request_options(response_max_tokens=4096)
        budgets = []

        async def events(*args, **kwargs):
            budgets.append(kwargs["response_max_tokens"])
            success = len(budgets) == 3
            yield {"type": "content_block_start", "content_block": {
                "type": "tool_use", "name": "send_email_to_user", "id": "t1"}}
            yield {"type": "content_block_delta", "delta": {
                "type": "input_json_delta",
                "partial_json": '{"body": "done"}' if success else '{"body": "cut'}}
            yield {"type": "content_block_stop"}
            yield {"type": "message_delta", "delta": {
                "stop_reason": "tool_use" if success else "max_tokens"}}
            yield {"type": "message_stop"}

        model._get_response_async = events
        with patch("src.tools.asyncio.sleep", new=AsyncMock()):
            output = [item async for item in model._get_response_async_streaming([])]
        self.assertEqual(budgets, [4096, 8192, 16384])
        processor.assert_awaited_once_with({"body": "done"})
        self.assertEqual(output.count("[[TOOL_RESULT]]"), 1)
        self.assertEqual(model._runtime_response_max_tokens, 4096)
        self.assertNotIn("NOT executed", str(model.consume_tool_provenance_messages()))

    async def test_malformed_json_retries_without_increasing_budget(self):
        model, processor = self.make_model()
        budgets = []

        async def events(*args, **kwargs):
            budgets.append(kwargs["response_max_tokens"])
            yield {"type": "content_block_start", "content_block": {
                "type": "tool_use", "name": "send_email_to_user", "id": "t1"}}
            yield {"type": "content_block_delta", "delta": {
                "type": "input_json_delta", "partial_json": 'broken'}}
            yield {"type": "content_block_stop"}
            yield {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}
            yield {"type": "message_stop"}

        model._get_response_async = events
        with patch("src.tools.asyncio.sleep", new=AsyncMock()):
            output = [item async for item in model._get_response_async_streaming([])]
        self.assertEqual(budgets, [4096, 4096, 4096])
        processor.assert_not_awaited()
        self.assertTrue(any("не был выполнен" in item for item in output))
