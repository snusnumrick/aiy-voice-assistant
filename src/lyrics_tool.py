import asyncio
import logging
import re
import time

from dotenv import load_dotenv

from src.ai_models import (
    AIModel,
    ClaudeAIModel,
    GeminiAIModel,
    OpenAIModel,
    OpenRouterModel,
    TruncatedResponseError,
)
from src.config import Config
from src.tools import NonRetryableError, retry_async

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

В русском тексте обязательно пиши «ё» там, где она требуется по смыслу и нормативному
написанию слова; не заменяй её на «е». Например: идёт, бьётся, вьётся, гололёд, перекрёсток,
пёс, жёлтый. При разметке ударения сохраняй букву «ё»: иДЁТ, БЬЁТся, голоЛЁД, пЁС.
Не заменяй все «е» на «ё» механически: выбирай написание по контексту (все люди / всё готово).
Перед ответом проверь весь текст на пропущенные «ё», в том числе при редактировании.

В русском тексте песни отмечай ударение в многосложных словах заглавными буквами ударного
слога, например: поДАрок, доРОга, моЛОко. Остальные слоги оставляй в обычном регистре.
Не используй знак + или диакритические знаки для ударений. Это разметка произношения для
пения, а не выделение громкости. Метки разделов вроде [Verse] и [Chorus] не меняй.

Следуй языку, теме, точке зрения, жанру, настроению, структуре, длине и другим ограничениям из
запроса пользователя. При редактировании существующего текста меняй только то, что попросил
пользователь, и сохраняй указанные им части. Пиши оригинальный текст и не воспроизводи текст
существующих песен. Верни только готовый текст песни без анализа, предисловия, кавычек и блоков
кода. Где уместно, используй короткие метки разделов: [Verse], [Chorus], [Bridge], [Outro]."""


LYRICS_REVIEW_INSTRUCTION = """Перечитай текущую версию песни и исходный запрос. Проверь:
соответствие замыслу и аудитории; связность сюжета и образов; естественность языка;
смысл каждой строки; отсутствие натянутых рифм и лишних деталей; пригодность для пения;
правильность ударений и букв «ё»; завершённость всех строк и финала.
Если есть проблемы, создай новую улучшенную версию. Сохраняй удачные строки и ограничения
пользователя. Верни только полный новый текст песни, без объяснений и анализа.
Если существенных проблем больше нет и ты удовлетворён результатом, верни ровно
LYRICS_APPROVED вместо текста. Это служебный сигнал проверки, а не строка песни."""


class ChildInappropriateLyricsError(RuntimeError):
    """Generated lyrics contain a smoking reference for a child or unknown audience."""


def _has_smoking_reference(lyrics: str) -> bool:
    text = lyrics.casefold()
    return bool(
        re.search(r"\b(?:сигарет\w*|папирос\w*|окур\w*|табак\w*|кури[тл]\w*|курени\w*|вейп\w*|cigarette\w*|tobacco|smoking|smokes?|vaping)\b", text)
        or (re.search(r"\bбыч(?:ок|к\w*)\b", text) and re.search(r"урн|брос|гас|туш|дым", text))
    )


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
        logger.info(
            "Lyrics tool configured: provider=%s model=%s reasoning_level=%s",
            self.provider,
            self.model or "not configured",
            config.get("lyrics_reasoning_level", "provider default"),
        )

    def tool_definition(self):
        from src.ai_models_with_tools import Tool, ToolParameter

        return Tool(
            name="generate_lyrics",
            description=(
                "Write, rewrite, translate, or refine original song lyrics using the configured "
                "specialist lyrics model. Normalize the user's intent with minimal interpretation; "
                "do not create a detailed songwriting brief. Include the intended audience "
                "and known speaker age from conversation context in the intent. "
                "Use the returned lyrics as the exact lyrics input for generate_music."
            ),
            iterative=True,
            parameters=[
                ToolParameter(
                    name="intent",
                    type="string",
                    description=(
                        "A minimal normalization of the user's songwriting intent. Preserve every "
                        "person, object, relationship, count, genuine semantic ambiguity, style "
                        "request, emotional cue, and explicit constraint. Remove conversational "
                        "wrapper text, obvious repetitions, false starts, filler, and likely "
                        "speech-recognition artifacts when they do not change the most probable "
                        "meaning. Do not mention such cleanup in the normalized intent. Do not "
                        "invent a setting, plot, imagery, song references, structure, rhyme, meter, "
                        "genre conventions, or other creative details the user did not say. "
                        "Include known audience context: if the speaker is a child or the song "
                        "is for children, explicitly say it is for a child and include the age "
                        "when known. For example: 'Песня для ребёнка 8 лет про девочку с пуделем "
                        "и её маму; русский постпанк; текст должен быть подходящим детям.' "
                        "Do not invent an age or infer adulthood from the requested music style."
                    ),
                ),
                ToolParameter(
                    name="audience",
                    type="string",
                    description=(
                        "Audience: child, adult, or unknown. Use child when the current speaker is "
                        "known to be a child or the song is for children. Use adult only when known "
                        "from conversation context; do not infer it from musical style. "
                        "Unknown defaults to child-appropriate content."
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
            required=["intent"],
            processor=self.generate_lyrics_async,
            rule_instructions={
                "russian": (
                    "Когда пользователь просит написать, переписать, улучшить или перевести текст "
                    "песни, используй generate_lyrics. В параметре intent только нормализуй намерение "
                    "пользователя с минимальной интерпретацией: сохрани всех людей, предметы, связи, "
                    "количество, подлинную смысловую неоднозначность, стиль, эмоцию и явно заданные "
                    "ограничения. Удали разговорную обёртку, очевидные повторы, самоперебивы, "
                    "слова-паразиты и вероятные артефакты распознавания речи, если они не меняют "
                    "наиболее вероятный смысл. Не упоминай эту очистку в intent. Не пиши творческий "
                    "бриф и не добавляй место действия, сюжет, образы, ссылки на песни, структуру, "
                    "рифму, размер или жанровые детали, которых не было у пользователя. "
                    "Добавь в intent известный контекст аудитории: если собеседник ребёнок или "
                    "песня предназначена детям, явно укажи это и известный возраст. Например: "
                    "'Песня для ребёнка 8 лет про девочку с пуделем и её маму; русский постпанк; "
                    "текст должен быть подходящим детям'. Не выдумывай возраст. "
                    "Передай audience=child, если собеседник ребёнок или песня для детей; adult — "
                    "только если взрослый возраст известен из контекста, иначе unknown. "
                    "Для новой песни с текстом, если пользователь не дал окончательный текст "
                    "дословно, сначала вызови generate_lyrics, затем передай полученный текст без "
                    "изменений в параметре lyrics инструмента generate_music. Для инструментальной "
                    "музыки generate_lyrics не используй."
                ),
                "english": (
                    "When the user asks to write, rewrite, improve, or translate song lyrics, use "
                    "generate_lyrics. In intent, only normalize the user's intent with minimal "
                    "interpretation: preserve every person, object, relationship, count, genuine "
                    "semantic ambiguity, style request, emotional cue, and explicit constraint. "
                    "Remove conversational wrapper text, obvious repetitions, false starts, filler, "
                    "and likely speech-recognition artifacts when they do not change the most "
                    "probable meaning. Do not mention such cleanup in the normalized intent. Do not "
                    "write a creative brief or add a setting, plot, imagery, song references, "
                    "structure, rhyme, meter, or genre details the user did not provide. Set audience "
                    "to child for a child speaker or children’s song, adult only for a known adult, "
                    "otherwise unknown. Also include the known audience and age in intent; "
                    "explicitly request child-appropriate lyrics for a child speaker or children's "
                    "song. Do not invent an age. For a new "
                    "song with lyrics, unless "
                    "the user supplied final lyrics verbatim, call generate_lyrics first and then "
                    "pass its returned text unchanged as the lyrics parameter of generate_music. "
                    "Do not call generate_lyrics for instrumental music."
                ),
            },
        )

    async def generate_lyrics_async(self, parameters: dict) -> str:
        intent = str(parameters.get("intent") or parameters.get("prompt", "") or "").strip()
        if not intent:
            return "Error: 'intent' is required"
        if len(intent) < 10:
            return "Error: 'intent' should be at least 10 characters"
        if not self.model:
            return "Error: lyrics_model is not configured"

        audience = str(parameters.get("audience", "unknown") or "unknown").strip().lower()
        if audience not in {"child", "adult", "unknown"}:
            return "Error: audience must be child, adult, or unknown"
        child_appropriate = audience != "adult"
        existing_lyrics = str(parameters.get("existing_lyrics", "") or "").strip()
        timeout = self.config.get("lyrics_timeout", 120)
        started_at = time.monotonic()

        logger.info(
            "Lyrics generation started: provider=%s model=%s intent_chars=%d "
            "existing_lyrics_chars=%d timeout_sec=%s",
            self.provider,
            self.model,
            len(intent),
            len(existing_lyrics),
            timeout,
        )
        if self.config.get("lyrics_log_content", False):
            logger.info("Normalized lyrics intent:\n%s", intent)
            if existing_lyrics:
                logger.info("Existing lyrics supplied for revision:\n%s", existing_lyrics)

        request_text = f"Audience: {audience}\n\nUser songwriting intent:\n{intent}"
        if existing_lyrics:
            request_text += f"\n\nExisting lyrics:\n{existing_lyrics}"

        logger.info("Generating lyrics: provider=%s model=%s", self.provider, self.model)

        try:
            model = self._get_model()
            logger.info("Lyrics model ready: %s", type(model).__name__)
            system_instruction = LYRICS_SYSTEM_INSTRUCTION
            if child_appropriate:
                system_instruction += (
                    "\nАудитория: ребёнок или возраст неизвестен. Пиши текст, подходящий детям. "
                    "Не включай курение, сигареты, окурки, бычки от сигарет, вейпы, алкоголь, "
                    "наркотики, обсценную лексику и взрослые сексуальные темы. "
                    "Не добавляй эти детали ради мрачного стиля. Если они есть в исходном тексте, "
                    "замени их нейтральными бытовыми действиями, сохранив сюжет."
                )
            else:
                system_instruction += "\nАудитория: взрослый."
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": request_text},
            ]
            reasoning_effort = self.config.get("lyrics_reasoning_effort")
            token_budget = self.config.get("lyrics_max_output_tokens", 32768)

            @retry_async(max_retries=3)
            async def request_lyrics():
                nonlocal token_budget
                original_budget = getattr(model, "max_tokens", token_budget)
                try:
                    model.max_tokens = token_budget
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            model.get_response, messages, reasoning_effort=reasoning_effort,
                        ),
                        timeout=timeout,
                    )
                    if child_appropriate and _has_smoking_reference(str(result or "")):
                        messages.append({
                            "role": "user",
                            "content": "Перепиши песню без любых упоминаний курения и табака. Аудитория — дети.",
                        })
                        raise ChildInappropriateLyricsError("Smoking reference in child-audience lyrics")
                    return result
                except ChildInappropriateLyricsError:
                    raise
                except TruncatedResponseError:
                    token_budget = min(token_budget * 2, 65536)
                    logger.warning("Retrying truncated lyrics with token budget %s", token_budget)
                    raise
                except asyncio.TimeoutError as exc:
                    raise NonRetryableError(
                        f"Lyrics generation timed out after {timeout} seconds"
                    ) from exc
                except Exception as exc:
                    raise NonRetryableError(str(exc)) from exc
                finally:
                    model.max_tokens = original_budget

            lyrics = await request_lyrics()
            lyrics = str(lyrics or "").strip()
            if not lyrics:
                logger.error("Lyrics model returned no text")
                return "Error: No lyrics received from API"

            review_passes = self.config.get("lyrics_review_max_passes", 5)
            if isinstance(review_passes, bool) or not isinstance(review_passes, int) or not 0 <= review_passes <= 10:
                return "Error: lyrics_review_max_passes must be an integer between 0 and 10"
            for review_pass in range(review_passes):
                messages.extend([
                    {"role": "assistant", "content": lyrics},
                    {"role": "user", "content": LYRICS_REVIEW_INSTRUCTION},
                ])
                logger.info("Lyrics review pass %s/%s", review_pass + 1, review_passes)
                reviewed = str(await request_lyrics() or "").strip()
                if reviewed == "LYRICS_APPROVED":
                    logger.info("Lyrics approved after %s review passes", review_pass + 1)
                    break
                if not reviewed:
                    return "Error: Lyrics review returned no text"
                lyrics = reviewed
            else:
                if review_passes:
                    return f"Error: Lyrics were not approved after {review_passes} review passes"

            logger.info(
                "Lyrics generation completed: provider=%s model=%s chars=%d elapsed_sec=%.2f",
                self.provider,
                self.model,
                len(lyrics),
                time.monotonic() - started_at,
            )
            if self.config.get("lyrics_log_content", False):
                logger.info("Generated lyrics:\n%s", lyrics)
            return lyrics
        except asyncio.TimeoutError:
            logger.error(
                "Lyrics generation timed out: timeout_sec=%s elapsed_sec=%.2f",
                timeout,
                time.monotonic() - started_at,
            )
            return f"Error: Lyrics generation timed out after {timeout} seconds"
        except Exception as exc:
            logger.error(
                "Lyrics generation failed after %.2f seconds: %s",
                time.monotonic() - started_at,
                exc,
                exc_info=True,
            )
            return f"Error generating lyrics: {exc}"

    def _get_model(self) -> AIModel:
        if self._model is not None:
            return self._model

        max_tokens = self.config.get("lyrics_max_output_tokens", 32768)
        if self.provider == "gemini":
            self._model = GeminiAIModel(
                self.config,
                model_id=self.model,
                max_tokens=max_tokens,
                thinking_level=self.config.get("lyrics_reasoning_level", "medium"),
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
    parser.add_argument("intent", help="User's songwriting intent")
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
                "intent": args.intent,
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
