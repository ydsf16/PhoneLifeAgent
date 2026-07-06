import unittest
from unittest.mock import patch

from life_report.model_engine.aliyun import AliyunTextLLM


class AliyunModelEngineTest(unittest.TestCase):
    def test_text_llm_disables_thinking_explicitly(self) -> None:
        calls = {}

        class Message:
            content = "ok"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        class Completions:
            def create(self, **kwargs):
                calls.update(kwargs)
                return Response()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        with patch("life_report.model_engine.aliyun._openai_client", return_value=Client()):
            text = AliyunTextLLM(api_key="key").generate_text("system", "user")

        self.assertEqual(text, "ok\n")
        self.assertEqual(calls["extra_body"], {"enable_thinking": False})

    def test_text_llm_can_enable_thinking_explicitly(self) -> None:
        calls = {}

        class Message:
            content = "ok"

        class Choice:
            message = Message()

        class Response:
            choices = [Choice()]

        class Completions:
            def create(self, **kwargs):
                calls.update(kwargs)
                return Response()

        class Chat:
            completions = Completions()

        class Client:
            chat = Chat()

        with patch("life_report.model_engine.aliyun._openai_client", return_value=Client()):
            text = AliyunTextLLM(api_key="key", enable_thinking=True).generate_text("system", "user")

        self.assertEqual(text, "ok\n")
        self.assertEqual(calls["extra_body"], {"enable_thinking": True})


if __name__ == "__main__":
    unittest.main()
