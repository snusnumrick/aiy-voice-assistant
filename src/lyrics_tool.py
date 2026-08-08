import asyncio
import logging

from dotenv import load_dotenv

from src.ai_models import AIModel, ClaudeAIModel, GeminiAIModel, OpenAIModel, OpenRouterModel
from src.config import Config

logger = logging.getLogger(__name__)


LYRICS_SYSTEM_INSTRUCTION = """Пиши песню как цельный маленький мир, а не как набор красивых
строк. Перед написанием найди один главный образ, ситуацию или деталь, вокруг которой будет
держаться песня. Этот образ должен быть конкретным и запоминающимся, а не абстрактным. Не
перегружай текст метафорами. Лучше один сильный образ, который развивается по ходу песни, чем
много несвязанных красивых фраз. Избегай шаблонных слов и сочетаний вроде «сердце», «мечта»,
«судьба», «душа», «звёзды», «тишина», «навсегда», если они не необходимы именно этой истории. Не
используй их просто для создания поэтичности. В песне должно что-то происходить или меняться:
герой что-то замечает, делает, вспоминает, понимает, ждёт, теряет или находит. Куплеты должны
двигать эту маленькую историю вперёд, а не повторять одну мысль другими словами. Припев должен
содержать главную находку песни — фразу или образ, который хочется запомнить и повторить. Не делай
припев просто общим эмоциональным резюме. Пиши простым естественным языком. Не ставь рифму выше
смысла. Не добавляй строку только потому, что она рифмуется. Допустимы неточные рифмы и отсутствие
рифмы, если так текст звучит живее. Если задан музыкальный стиль, передавай его через длину строк,
ритм, лексику, настроение и устройство песни, но не копируй существующие песни и не набивай текст
очевидными атрибутами жанра. Учитывай возраст и замысел пользователя. Если идея смешная — песня
может быть смешной. Если бытовая — не превращай её искусственно в драму. Если нежная — не делай её
приторной. Перед окончательным ответом мысленно спроси: «Что в этой песне можно вспомнить завтра?»
Если ответа нет, найди более сильную центральную деталь и перепиши текст.

Следуй языку, теме, точке зрения, жанру, настроению, структуре, длине и другим ограничениям из
запроса пользователя. При редактировании существующего текста меняй только то, что попросил
пользователь, и сохраняй указанные им части. Пиши оригинальный текст и не воспроизводи текст
существующих песен. Верни только готовый текст песни без анализа, предисловия, кавычек и блоков
кода. Где уместно, используй короткие метки разделов: [Verse], [Chorus], [Bridge], [Outro]."""


SUPPORTED_LYRICS_PROVIDERS = {"gemini", "openrouter", "openai", "claude"}


def normalize_lyrics_provider(provider) -> str:
    normalized = str(provider or "gemini").strip().lower().replace("-", "_")
    aliases = {
        "google": "gemini",
        "open_router": "openrouter",
        "anthropic": "claude",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in SUPPORTED_LYRICS_PROVIDERS:
        expected = ", ".join(sorted(SUPPORTED_LYRICS_PROVIDERS))
        raise ValueError(f"Unsupported lyrics_provider: {provider}. Expected one of: {expected}.")
    return normalized


class LyricsTool:
    """Generate or revise song lyrics with a configurable text model provider."""

    def __init__(self, config: Config):
        self.config = config
        self.provider = normalize_lyrics_provider(config.get("lyrics_provider", "gemini"))
        self.model = str(config.get("lyrics_model", "") or "").strip()
        self._model = None

    def tool_definition(self):
        from src.ai_models_with_tools import Tool, ToolParameter

        return Tool(
            name="generate_lyrics",
            description=(
                "Write, rewrite, translate, or refine original song lyrics using the configured "
                "specialist lyrics model. "
                "Use the returned lyrics as the exact lyrics input for generate_music."
            ),
            iterative=True,
            parameters=[
                ToolParameter(
                    name="prompt",
                    type="string",
                    description=(
                        "Complete songwriting request. Include the language, subject or story, "
                        "genre, mood, point of view, desired sections or length, rhyme or meter, "
                        "and any words, ideas, or content to include or avoid."
                    ),
                ),
                ToolParameter(
                    name="existing_lyrics",
                    type="string",
                    description=(
                        "Optional existing lyrics to revise, extend, translate, or polish. Leave "
                        "empty when writing a new song."
                    ),
                ),
            ],
            required=["prompt"],
            processor=self.generate_lyrics_async,
            rule_instructions={
                "russian": (
                    "Когда пользователь просит написать, переписать, улучшить или перевести текст "
                    "песни, используй generate_lyrics. Для новой песни с текстом, если пользователь "
                    "не дал окончательный текст дословно, сначала вызови generate_lyrics, затем "
                    "передай полученный текст без изменений в параметре lyrics инструмента "
                    "generate_music. Для инструментальной музыки generate_lyrics не используй."
                ),
                "english": (
                    "When the user asks to write, rewrite, improve, or translate song lyrics, use "
                    "generate_lyrics. For a new song with lyrics, unless the user supplied final "
                    "lyrics verbatim, call generate_lyrics first and then pass its returned text "
                    "unchanged as the lyrics parameter of generate_music. Do not call "
                    "generate_lyrics for instrumental music."
                ),
            },
        )

    async def generate_lyrics_async(self, parameters: dict) -> str:
        prompt = str(parameters.get("prompt", "") or "").strip()
        if not prompt:
            return "Error: 'prompt' is required"
        if len(prompt) < 10:
            return "Error: 'prompt' should be at least 10 characters"
        if not self.model:
            return "Error: lyrics_model is not configured"

        existing_lyrics = str(parameters.get("existing_lyrics", "") or "").strip()
        timeout = self.config.get("lyrics_timeout", 120)

        request_text = f"Songwriting request:\n{prompt}"
        if existing_lyrics:
            request_text += f"\n\nExisting lyrics:\n{existing_lyrics}"

        logger.info("Generating lyrics: provider=%s model=%s", self.provider, self.model)

        try:
            model = self._get_model()
            messages = [
                {"role": "system", "content": LYRICS_SYSTEM_INSTRUCTION},
                {"role": "user", "content": request_text},
            ]
            reasoning_effort = self.config.get("lyrics_reasoning_effort")
            lyrics = await asyncio.wait_for(
                asyncio.to_thread(
                    model.get_response,
                    messages,
                    reasoning_effort=reasoning_effort,
                ),
                timeout=timeout,
            )
            lyrics = str(lyrics or "").strip()
            if not lyrics:
                logger.error("Lyrics model returned no text")
                return "Error: No lyrics received from API"

            return lyrics
        except asyncio.TimeoutError:
            logger.error("Lyrics generation timed out after %s seconds", timeout)
            return f"Error: Lyrics generation timed out after {timeout} seconds"
        except Exception as exc:
            logger.error("Lyrics generation failed: %s", exc, exc_info=True)
            return f"Error generating lyrics: {exc}"

    def _get_model(self) -> AIModel:
        if self._model is not None:
            return self._model

        max_tokens = self.config.get("lyrics_max_output_tokens", 8192)
        if self.provider == "gemini":
            self._model = GeminiAIModel(
                self.config,
                model_id=self.model,
                max_tokens=max_tokens,
                thinking_level=self.config.get("lyrics_reasoning_level", "low"),
                request_timeout_sec=self.config.get("lyrics_timeout", 120),
            )
        elif self.provider == "openrouter":
            self._model = OpenRouterModel(
                self.config,
                model_id=self.model,
                max_tokens=max_tokens,
                reasoning_effort=self.config.get("lyrics_reasoning_effort"),
            )
        elif self.provider == "openai":
            self._model = OpenAIModel(
                self.config,
                model_id=self.model,
                reasoning_effort=self.config.get("lyrics_reasoning_effort"),
                max_tokens=max_tokens,
            )
        else:
            self._model = ClaudeAIModel(
                self.config,
                model_id=self.model,
                max_tokens=max_tokens,
                reasoning_effort=self.config.get("lyrics_reasoning_effort"),
            )
        return self._model


def main(argv=None) -> int:
    """Generate lyrics from a command-line prompt and print only the result."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate or revise song lyrics")
    parser.add_argument("prompt", help="Complete songwriting request")
    parser.add_argument(
        "--existing-lyrics",
        default="",
        help="Optional existing lyrics to revise",
    )
    args = parser.parse_args(argv)

    load_dotenv()
    tool = LyricsTool(Config())
    result = asyncio.run(
        tool.generate_lyrics_async(
            {
                "prompt": args.prompt,
                "existing_lyrics": args.existing_lyrics,
            }
        )
    )
    if result.startswith("Error:"):
        parser.exit(1, f"{result}\n")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
