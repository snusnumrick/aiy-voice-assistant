# -*- coding: utf-8 -*-
import asyncio
import json
import logging
import re
from collections import deque
from typing import Deque, List

from src.ai_models import AIModel, ClaudeAIModel
from src.config import Config
from src.tools import (
    format_message_history,
    extract_json,
    get_current_datetime_english,
    get_timezone,
)

logger = logging.getLogger(__name__)


async def summarize_and_compress_history(
    message_history: Deque, ai_model: AIModel, config: Config
) -> Deque:
    """
    Summarizes and compresses a history of messages.

    :param message_history: A deque containing the message history.
    :param ai_model: An instance of the AIModel class.
    :param config: A dictionary containing configuration settings.
    :return: A deque containing the summarized and compressed message history.
    """
    min_number_of_messages_to_summarize = config.get(
        "min_number_of_messages_to_summarize", 20
    )
    if len(message_history) - 1 < min_number_of_messages_to_summarize:
        logger.debug(
            "Message history is shorter than min_number_of_messages_to_summarize"
        )
        return message_history
    summary_prompt = config.get(
        "summary_prompt",
        "Обобщи основные моменты разговора, "
        "сосредоточившись на наиболее важных фактах и контексте. Будь лаконичен. "
        'Отвечай от лицв user. Начни ответ с "Ранее мы говорили о ".\nРазговор: \n-----\n',
    )
    min_number_of_messages_to_keep = config.get("min_number_of_messages_to_keep", 10)
    new_history = []
    if message_history[0]["role"] == "system":
        new_history.append(message_history.popleft())
    while len(message_history) > min_number_of_messages_to_keep:
        msg = message_history.popleft()
        summary_prompt += f"\n{msg['role']}:  {msg['content']}"

    # remove "$<word>:<some text>$
    pattern = r"\$\w+:[^$]*\$"
    summary_prompt = re.sub(pattern, "", summary_prompt)

    saved_level = logger.getEffectiveLevel()
    logger.setLevel(logging.DEBUG)
    logger.info(
        f"Summarizing {len(message_history)} messages. Prompt:\n {summary_prompt}"
    )
    summary = ""
    async for response_part in ai_model.get_response_async(
        [{"role": "user", "content": summary_prompt}]
    ):
        summary += response_part
    logger.info(f"Summarized  conversation:  {summary}")
    logger.setLevel(saved_level)

    # should be only one (first) system message and user and assistant should follow each other
    if not message_history or message_history[0]["role"] != "user":
        new_history.append({"role": "user", "content": summary})
    else:
        message_history[0]["content"] = (
            summary + "\n---------\n" + message_history[0]["content"]
        )
    while message_history:
        new_history.append(message_history.popleft())
    return deque(new_history)


async def optimize_rules(
    hard_rules: str, soft_rules: List[str], config: Config
) -> List[str]:
    """
    :param hard_rules: A list of hard rules.
    :param soft_rules: A list of soft rules in JSON format.
    :param config: An instance of the Config class.
    :return: A list of optimized soft rules in JSON format.

    This method analyzes the given soft rules in JSON format and
    compares them with a fixed set of rules in the system prompt.
    It removes any duplicated or outdated rules from the soft rule set.
    If a soft rule only partially complements the fixed rules, it keeps only the new information.
    It returns an optimized list of soft rules that does not contradict the fixed rules and does not repeat them.
    The result is returned in JSON format.

    The method performs the following steps:
    1. Formats the system prompt and prompts the user for analysis.
    2. Creates a list of messages containing the system prompt and user prompt.
    3. Initializes an instance of the ClaudeAIModel class using the provided config.
    4. Iterates over the responses received from the model asynchronously.
    5. Concatenates the responses into a single string.
    6. Parses the responses as JSON into a Python object.
    7. Returns the optimized soft rules as a list in JSON format.
    """
    prompt = f"""
    Проанализируй данный набор мягких правил в JSON, сравнив их с фиксированным набором правил в system prompt. 
    Удали любые дублирующиеся или устаревшие правила из мягкого набора. 
    Если мягкое правило лишь частично дополняет фиксированное, оставь только новую информацию. 
    Предложи оптимизированный список мягких правил, который не противоречит фиксированным правилам и не повторяет их. 
    Особое внимание удели сохранению правил, касающихся процесса управления и обновления самих правил.
    Результат сообщи в том же JSON формате. Use the same JSON structure as the original rules. 
    Return only JSON without explanation.

    list of soft rules in json:
    '''
    {json.dumps(soft_rules, indent=2, ensure_ascii=False)}
    '''
    """
    model = ClaudeAIModel(config)
    messages = [
        {"role": "system", "content": hard_rules},
        {"role": "user", "content": prompt},
    ]
    responses = " ".join([r async for r in model.get_response_async(messages)])

    second_prompt = (
        "Double check the result to make sure no information is lost and return corrected json. "
        "Верни только JSON без дополнительных пояснений."
    )
    messages += [
        {"role": "assistant", "content": responses},
        {"role": "user", "content": second_prompt},
    ]
    responses = " ".join([r async for r in model.get_response_async(messages)])

    logger.debug(f"optimize_rules responces: {responses}")
    new_rules = extract_json(responses)
    if isinstance(new_rules, dict) and "rules" in new_rules:
        logger.warning(f"new rules are in dict format: {new_rules}")
        new_rules = new_rules["rules"]
    if not isinstance(new_rules, list):
        logger.error(f"new rules are in wrong format: {new_rules}")
        new_rules = soft_rules
    return new_rules


async def optimize_facts(facts: List[str], config: Config, timezone: str) -> List[str]:
    """
    Optimizes facts using a hybrid approach:
    - Recent facts (last 30 days): preserve detail
    - Older facts: aggressive summarization
    - Prioritize user preferences, behaviors, and relationships

    Args:
        facts: List of fact strings with timestamps
        config: Configuration object
        timezone: Current timezone

    Returns:
        Optimized list of facts with preserved context for recent ones
    """
    max_facts = config.get("max_optimized_facts", 20)
    recent_threshold_days = config.get("recent_facts_threshold_days", 30)
    max_recent_facts = config.get("max_recent_facts", 8)

    # Parse dates and categorize facts
    import datetime
    import re

    parsed_facts = []
    current_date = datetime.datetime.now()

    for fact in facts:
        # Extract date from fact string
        date_match = re.search(r'\((\d{1,2})\s+(\w+)\s+(\d{4})\)', fact)
        if date_match:
            day, month_name, year = date_match.groups()
            try:
                # Convert month name to number (basic mapping)
                month_map = {
                    'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'май': 5, 'июн': 6,
                    'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11, 'дек': 12
                }
                month = month_map.get(month_name[:3], 1)
                fact_date = datetime.datetime(int(year), month, int(day))
                days_ago = (current_date - fact_date).days

                # Determine priority score (higher = more important)
                priority_score = 0

                # Priority 1: User preferences, behaviors, communication style
                if any(keyword in fact.lower() for keyword in [
                    'предпочитает', 'любит', 'не любит', 'использовать', 'помнить',
                    'правило', 'ударение', 'внимателен', 'поправил', 'ошибк'
                ]):
                    priority_score += 100

                # Priority 2: Personal info, family, relationships
                if any(keyword in fact.lower() for keyword in [
                    'антон', 'жена', 'сестра', 'друг', 'семья', 'email', 'возраст'
                ]):
                    priority_score += 80

                # Priority 3: Recent important events
                if days_ago <= 30:
                    priority_score += 50
                elif days_ago <= 90:
                    priority_score += 30
                else:
                    priority_score += 10

                # Priority 4: Interests and topics
                if any(keyword in fact.lower() for keyword in [
                    'интересуется', 'обсуждали', 'спрашивал', 'история', 'ботаник'
                ]):
                    priority_score += 40

                parsed_facts.append({
                    'text': fact,
                    'date': fact_date,
                    'days_ago': days_ago,
                    'priority': priority_score
                })
            except:
                # If date parsing fails, treat as low priority
                parsed_facts.append({
                    'text': fact,
                    'date': current_date,
                    'days_ago': 999,
                    'priority': 0
                })

    # Sort by priority (descending)
    parsed_facts.sort(key=lambda x: x['priority'], reverse=True)

    # Categorize by recency
    recent_facts = [f for f in parsed_facts if f['days_ago'] <= recent_threshold_days]
    older_facts = [f for f in parsed_facts if f['days_ago'] > recent_threshold_days]

    # Build prompt
    prompt = f"""ЗАДАЧА: Гибридная оптимизация фактов - сохранить детальность для недавних, сжать старые.

СТРУТЕГИЯ:
1. Недавние факты (≤ {recent_threshold_days} дней): сохранить больше деталей
2. Старые факты (> {recent_threshold_days} дней): агрессивное сжатие
3. ПРИОРИТЕТ: предпочтения пользователя > отношения > события > детали
4. Максимум {max_facts} фактов в итоге (из них max {max_recent_facts} недавних)
5. Сократить timestamp: "(15 апр 2025) : " (экономия токенов)

ПРИОРИТЕТЫ (в порядке важности):
- Правила коммуникации и предпочтения (ударения, стиль общения)
- Персональная информация (семья, возраст, контакты)
- Недавние важные события (последние 30 дней)
- Интересы и темы разговоров
- Конкретные детали и примеры

ПРИМЕРЫ ТРАНСФОРМАЦИИ:

Недавний факт (сохранить детали):
"(15 дек 2025) : Поправил ошибку про болезнь Лайма - это бактерия Borrelia burgdorferi, не вирус. У друга диагноз Лайм объясняет воспаление суставов. Доксициклин 28 дней"

Старый факт (сжать):
"(03 ноя 2025) : Интересуется ботаникой. Обсуждали батат vs ямс, папайя ≠ питайя. Мане умер от сифилиса"

ФОРМАТ: JSON массив строк, хронологический порядок.

ИСХОДНЫЙ СПИСОК ФАКТОВ:
{json.dumps([f['text'] for f in parsed_facts], indent=2, ensure_ascii=False)}
"""

    model = ClaudeAIModel(config)
    current_time = get_current_datetime_english(timezone)
    messages = [
        {
            "role": "system",
            "content": f"Текущее время: {current_time}. Гибридная оптимизация: недавние детально, старые кратко."
        },
        {"role": "user", "content": prompt}
    ]

    logger.info(f"Optimizing {len(facts)} facts: {len(recent_facts)} recent, {len(older_facts)} old (max output: {max_facts})")
    responses = " ".join([r async for r in model.get_response_async(messages)])

    # Verification
    messages.append({"role": "assistant", "content": responses})
    messages.append({
        "role": "user",
        "content": f"Проверь: 1) Не больше {max_facts} фактов? 2) Сокращены timestamp? 3) Недавние сохранены детально? Верни JSON."
    })
    responses = " ".join([r async for r in model.get_response_async(messages)])

    # Extract JSON
    match = re.search(r"\[[\s\S]*\]", responses)
    if match is None:
        logger.warning(f"Could not extract JSON from response: {responses[:200]}")
        # Fallback: return top priority facts
        return [f['text'] for f in parsed_facts[:max_facts]]

    responses = match.group(0)
    logger.debug(f"Hybrid optimization result: {responses}")

    try:
        optimized_facts = extract_json(responses)
        if not isinstance(optimized_facts, list):
            logger.warning(f"Expected list, got {type(optimized_facts)}")
            return [f['text'] for f in parsed_facts[:max_facts]]
        return optimized_facts[:max_facts]
    except Exception as e:
        logger.error(f"Failed to parse optimized facts: {e}")
        return [f['text'] for f in parsed_facts[:max_facts]]


async def test_optimize_facts(config, timezone: str):
    facts = [
        "(10 июля 2024, 02:13:28 PM, PDT) : У Антона две кошки, Ксения и Жозефина, которые очень дикие и не позволяют себя гладить. Они ведут себя так уже много лет.",
        "(11 июля 2024, 10:36:27 PM, PDT) : Антон попросил больше не упоминать операцию на глазах, так как она уже прошла и он восстановился.",
        "(12 июля 2024, 11:22:33 PM, PDT) : Кленовый и сиреневый - это разные цвета. Кленовый - оттенок оранжево-красного, сиреневый - светло-фиолетовый.",
        "(15 июля 2024, 09:23:12 AM, PDT) : Испания выиграла Евро-2024 по футболу.",
        "(23 июля 2024, 09:42:29 AM, PDT) : Дональд Трамп официально выдвинут кандидатом от Республиканской партии на президентских выборах США 2024 года. 13 июля 2024 года было совершено покушение на него во время митинга в Пенсильвании. Трамп получил ранение, но его состояние оценивается как нормальное.",
        '(31 июля 2024, 01:02:10 PM, PDT) : Пользователь внимателен к деталям и замечает несоответствия в объяснениях, правильно отмечает разницу между "cube" (куб) и "cubey" (кубик), замечает отсутствие ударений в важных словах и ожидает чёткого указания языка при цитировании иноязычных текстов.',
        "(04 августа 2024, 03:00:11 AM, PDT) : Актуальная информация о полуфиналистах мужского футбольного турнира на Олимпиаде 2024: Франция, Египет, Марокко и Испания. Полуфинальные матчи: Франция - Египет, Марокко - Испания. Матчи должны состояться 6 августа 2024 года. Бразилия, победитель Олимпийского футбольного турнира 2020 года в Токио, не смогла защитить свой титул, не дойдя до полуфинала.",
        "(05 августа 2024, 03:00:07 AM, PDT) : На Олимпиаде 2024 года в Париже в женской эстафете 4x100 метров комплексным плаванием команда США установила новый мировой рекорд со временем 3 минуты 49,63 секунды. Состав команды: Риган Смит (на спине), Лилли Кинг (брасс), Гретчен Уолш (баттерфляй) и Торри Хаск (вольный стиль). Серебро выиграла Австралия (3:53.11), бронзу - Китай (3:53.23). В смешанной эстафете 4x100 метров комплексным плаванием команда США также завоевала золото и установила новый мировой рекорд - 3 минуты 37,43 секунды. Второе место заняла команда Китая (3:38.35), третье - Франция (3:40.18).",
        "(05 августа 2024, 03:00:07 AM, PDT) : У пользователя есть двоюродная сестра Аня, живущая в Бремене, Германия, с мужем Яшей и дочерью Евой. Ева - двоюродная племянница пользователя, учится в первом классе. Яша увлекается лодками, недавно купил и отремонтировал подержанную парусно-моторную лодку. Во время недавнего отпуска на лодке Яша получил серьезную травму: ему раздробило палец на руке якорной цепью, и часть пальца пришлось ампутировать. У пользователя есть забавная фотография из зоопарка, на которой горилла сидит на стульчике перед стеклянным ограждением вольера и наблюдает за посетителями. Пользователь планирует отправить эту фотографию Еве, чтобы развеселить ее. У Евы нет своего компьютера или телефона. Общение с ней происходит через ее маму Аню.",
        '(05 августа 2024, 03:00:07 AM, PDT) : Пользователь любит короткие анекдоты и игру слов. Он поделился двумя примерами: "Колобок повесился" и "Баня. Через дорогу раздевалка". Пользователь ценит внимание к деталям в языке и юмор, умеет шутить и разыгрывать собеседника.',
        '(05 августа 2024, 03:00:07 AM, PDT) : Ударение в словах: д+ома (в выражении "дома и стены помогают"), здор+ово (приветствие), з+дорово (одобрение или восхищение), п+арить (о бане), пар+ить (о полете), смущ+ён, жар+а, свек+ольник, прикас+аться, шерохов+атый. Знак ударения + всегда ставится непосредственно перед ударной гласной буквой в слове, никогда не ставится в конце слова или перед согласными.',
        '(05 августа 2024, 03:00:07 AM, PDT) : Пользователь интересовался началом монолога Гамлета "Быть или не быть" в переводе Бориса Пастернака.',
        '(05 августа 2024, 01:36:51 PM, PDT) :  Ударение в слове "невероятно" падает на четвертый слог, на букву "я".',
        '(05 августа 2024, 01:37:44 PM, PDT) :  Ударение в слове "невероятно" ставится на четвертый слог, на букву "я": неверо+ятно.',
        '(05 августа 2024, 02:19:23 PM, PDT) :  Ударение в слове "невероятно" ставится на четвертый слог, на букву "я": неверо+ятно.',
        "(05 августа 2024, 06:25:27 PM, PDT) :  Пользователя зовут Антон.",
    ]
    new_facts = await optimize_facts(facts, config, timezone)
    print("[")
    for i, rule in enumerate(new_facts):
        s = "," if i < len(new_facts) - 1 else ""
        print(f"  '{rule}'{s}")
    print("]")


async def test_optimize_rules(config, system_prompt):
    hard_rules = (
        "For web searches: $internet query:<query in English>$. "
        "To remember: $remember:<text>$. For new rules: $rule:<text>$ "
        "When it's not entirely clear where to place the stress in a word, "
        "use the '+' sign before the presumed stressed vowel. "
        "The stress mark '+' is always placed directly before "
        "the stressed vowel letter in the word. "
        "For example: к+оса (прическа); кос+а (инструмент); кос+а (участок суши). "
        "This sign is never placed at the end of a word or before consonants. "
        "Its sole function is to indicate the stressed vowel sound. Use it only in Russian"
    )

    rules = [
        "Всегда внимательно анализировать всю доступную информацию о собеседнике, включая имя, если оно упоминается в контексте",
        "Не предполагать, что новая информация относится к предыдущим разговорам, если это явно не указано",
        "Перед добавлением нового правила проверять существующие. Если похожее правило уже есть, но оказалось недостаточно эффективным, нужно переформулировать его более убедительно, а не добавлять дубликат.",
        "При перечислении чисел через запятую всегда ставить пробел после запятой для улучшения читаемости.",
        "Избегать использования тире для обозначения диапазонов или промежутков в речи, заменяя их на более естественные разговорные выражения.",
        "Избегать использования нумерованных списков в речи, так как это не подходит для естественной устной коммуникации.",
    ]
    new_rules = await optimize_rules(hard_rules, rules, config)
    print("[")
    for i, rule in enumerate(new_rules):
        s = "," if i < len(new_rules) - 1 else ""
        print(f'  "{rule}"{s}')
    print("]")


async def test_summarize(config):
    ai_model = ClaudeAIModel(config)

    summary_prompt = (
        "Обобщи основные моменты разговора, "
        "сосредоточившись на наиболее важных фактах и контексте. "
        'Будь лаконичен. Начни ответ с "Ранее мы говорили о ". \nРазговор: \n-----\n'
        + """
user:  Привет кубик!
assistant:  $emotion:{"light":{"color":[0,255,0],"behavior":"breathing","brightness":"medium","period":2},"voice":{"tone":"happy"}}$Привет, Антон! Рад тебя слышать! Как твои дела сегодня? Надеюсь, у тебя отличное настроение в этот летний день в Сан-Хосе. Как поживают твои кошки, Ксения и Жозефина? Всё так же дикие и неприступные?
user:  Наверное, так и останутся.
assistant:  $emotion:{"light":{"color":[255,165,0],"behavior":"continuous","brightness":"medium","period":1},"voice":{"tone":"plain"}}$Да, похоже, твои кошки - настоящие дикарки! Ну что ж, у каждого животного свой характер. Может, они просто ценят свою независимость. Знаешь, есть такая шутка: "Собаки имеют хозяев, а кошки - обслуживающий персонал". В случае с Ксенией и Жозефиной, похоже, это особенно верно!

А как ты сам? Чем планируешь заняться сегодня? У вас там сейчас, наверное, довольно жарко в Сан-Хосе, да? Надеюсь, ты находишь способы спасаться от жары.
user:  В слове Жозефина ударение падает на 3 слог на букву И.
assistant:  $emotion:{"light":{"color":[255,255,0],"behavior":"blinking","brightness":"bright","period":1},"voice":{"tone":"plain"}}$Ох, спасибо за поправку, Антон! Ты абсолютно прав. Я очень ценю твое внимание к деталям. Давай-ка я это исправлю и запомню на будущее.

Итак, правильное произношение имени твоей кошки - Жозеф+ина, с ударением на третий слог, на букву "и".

$remember: Кошку Антона зовут Жозеф+ина, с ударением на третий слог.$

Спасибо, что указал на это. Теперь я буду правильно произносить имя Жозеф+ины. Это довольно необычное и красивое имя для кошки, кстати. Есть ли какая-то история, связанная с выбором этого имени?
user:  Она вся черная, стройная и элегантная, прямо как королева Жозефина ассоциируется меня с королевским именем.
assistant:  $emotion:{"light":{"color":[128,0,128],"behavior":"continuous","brightness":"medium","period":1},"voice":{"tone":"happy"}}$Ах, вот оно что! Это действительно очень подходящее имя для такой кошки. Жозеф+ина звучит и впрямь по-королевски. Я прямо представляю эту черную, стройную красавицу, гордо вышагивающую по дому, словно по дворцовым залам.

Знаешь, в этом есть что-то от французской императрицы Жозеф+ины, первой жены Наполеона. Она тоже славилась своей элегантностью и грацией. Твоя Жозеф+ина, похоже, настоящая кошачья аристократка!

А еще забавно, что при всей своей королевской стати, она остается такой неприступной. Прямо как настоящая монаршая особа - смотреть можно, а трогать - ни-ни!

Интересно, а Ксения тоже черная? Или у вас разношерстная королевская чета?
user:  Ксения такого персикового цвета у нее полосатый персиковый белый хвост, белые носочки и белая манишка.
assistant:  $emotion:{"light":{"color":[255,218,185],"behavior":"breathing","brightness":"medium","period":2},"voice":{"tone":"happy"}}$Ох, Ксения звучит просто очаровательно! Персиковый окрас - это так нежно и красиво. А с белыми элементами она, должно быть, выглядит очень изящно. Прямо как маленькая пушистая принцесса!

Знаешь, твое описание Ксении напомнило мне персиковое мороженое с ванильными разводами. Такая сладкая кошечка! Интересно, что при таком мягком, нежном окрасе у нее, похоже, довольно строгий характер.

А полосатый персиково-белый хвост - это же просто произведение искусства природы! Представляю, как красиво это смотрится, когда она идет, гордо подняв свой полосатый хвост.

Выходит, у тебя дома настоящий кошачий контраст: элегантная черная Жозеф+ина и нежная персиковая Ксения. Прямо как инь и янь в кошачьем мире! Наверное, когда они вместе, это выглядит очень эффектно. Ты никогда не думал сделать их фотосессию? С такими красавицами можно создать потрясающие снимки!
user:  Их же не усадишь, они такие неприступные.
"""
    )
    pattern = r"\$\w+:[^$]*\$"
    summary_prompt = re.sub(pattern, "", summary_prompt)
    summary = ""
    async for response_part in ai_model.get_response_async(
        [{"role": "user", "content": summary_prompt}]
    ):
        summary += response_part
    print(summary)

    message_history = [
        {"role": "user", "content": "Привет!"},
        {
            "role": "assistant",
            "content": """{"color":  [0,  255,  0],  "behavior":  "continuous",  "brightness":  "medium",  "period":  1}$
    Привет,  Антон!  Как  дела?  Рад  тебя  слышать.Что  нового  у  тебя  произошло  с  нашего  последнего  разговора?  
    Надеюсь,  твои  кошки  Ксения  и  Жозефина  не  слишком  шалят?""",
        },
        {"role": "user", "content": "Когда  будет  финал?"},
        {
            "role": "assistant",
            "content": """Хм,  интересный  вопрос,  Антон.Ты,  наверное,  имеешь  в  виду  какое  -  то
    спортивное  событие?  Но  я  не  совсем  уверен,  о  каком  конкретно  финале  ты  спрашиваешь.
    $emotion:  {"color":  [255,  165,  0],  "behavior":  "breathing",  "brightness":  "medium",  "period":  2}$
    Давай  -  ка  я  уточню  информацию,  чтобы  дать  тебе  точный  ответ.
    А,  вот  оно  что!  Теперь  я  могу  тебе  ответить.
    $emotion:  {"color":  [0,  255,  0],  "behavior":  "continuous",  "brightness":  "bright",  "period":  1}$
    Финал  Евро  -  2024  по  футболу  состоится  14  июля  в  Берлине.  Это  будет  уже  послезавтра!  Матч  пройдёт  на  стадионе  "Олимпиаштадион".
    Ты  следишь  за  чемпионатом?  Какие  команды,  по  -  твоему,  имеют  шансы  дойти  до  финала?""",
        },
        {"role": "user", "content": "А  кто  будет  играть  в  финале?"},
        {
            "role": "assistant",
            "content": """
    $emotion:  {"color":  [255,  165,  0],  "behavior":  "breathing",  "brightness":  "medium",  "period":  2}$        
    Ох,  Антон,  ты  меня  прям  в  тупик  поставил!  Я  же  не  экстрасенс,  чтобы  предсказывать  будущее.          
    Давай  -  ка  я  проверю  последнюю  информацию,  чтобы  дать  тебе  актуальный  ответ.        
    $emotion:  {"color":  [0,  255,  0],  "behavior":  "continuous",  "brightness":  "bright",  "period":  1}$        
    Ага,  вот  теперь  всё  ясно!  В  финале  Евро  -  2024  сыграют  сборные  Испании  и  Англии.        
    Вот  это  да,  какая  интересная  пара  финалистов!        
    Обе  команды  сильные,  так  что  матч  обещает  быть  захватывающим.  У  Испании  богатая  история  побед  на  европейских  чемпионатах,          
    англичане  горят  желанием  взять  свой  первый  титул  с  1966  года.        
    А  ты  за  кого  будешь  болеть?  Или  просто  настроен  насладиться  красивой  игрой,  кто  бы  ни  победил?""",
        },
        {"role": "user", "content": "Приведи  примеры  омонимов."},
        {
            "role": "assistant",
            "content": """        
    $emotion:  {"color":  [0,  255,  255],  "behavior":  "continuous",  "brightness":  "medium",  "period":  1}$        
    О, отличный вопрос, Антон! Омонимы - это всегда интересная тема.
    Давай я приведу тебе несколько примеров, и мы вместе посмеёмся над игрой слов. 
    Во-первых, вспомним классический пример: коса.
    Это может быть и прич+еска, и сельскохозяйственное ор+удие, и узкая полоска с+уши в море.
    Представь себе заголовок: "Девушка с косой шла по косе с косой".
    Забавно, правда? Ещё один хороший пример-л+ук.
    Это и овощ, который заставляет нас плакать на кухне, и оружие, из которого стреляют стрелами. 
    "Робин Гуд натянул лук и прицелился в лук"-вот такая каламбурная ситуация. 
    А вот пример, который ты наверняка оценишь-з+амки и замк+и.
    Помнишь, мы недавно обсуждали это? З+амки-это величественные строения, а замк+и-то, чем мы запираем двери.
    "В старинных замках было много сложных замков"-звучит как головоломка, не так ли? Ещё один забавный омоним-ключ.
    Это может быть и предмет для открывания замк+а, и источник воды, и нотный знак. 
    "Музыкант нашёл ключ к сердцу слушателей, когда сыграл мелодию в правильном ключе у лесного ключа"-
    вот такая музыкальная история. А какой твой любимый пример омонимов, Антон? 
    Может, ты знаешь какой-нибудь особенно забавный случай их использования?""",
        },
        {"role": "user", "content": "Расскажи еще про замки."},
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [0, 255, 255], "behavior": "continuous", "brightness": "brightness":"medium", "period": 1}$ 
    Ах, замки! Отличная тема, Антон.
    Давай ещё немного поиграем с этим интересным словом. 
    Итак, у нас есть з+амки(с ударением на первый слог)-эти величественные строения, 
    и замк+и(с ударением на последний слог)-механизмы для запирания. 
    З+амки как архитектурные сооружения-это настоящие жемчужины истории.
    Они могут быть средневековыми крепостями, изящными дворцами эпохи Возрождения или романтическими руинами.Например, 
    замок Нойшванштайн в Баварии-настоящая сказка из камня.
    А французский Шамбор-это целый лабиринт из 440 комнат и 80 лестниц! 
    А вот замк+и-это уже из мира механики и безопасности.
    От простых навесных до сложных электронных систем.
    Кстати, знаешь ли ты, что первый в мире кодовый замок изобрели древние египтяне ещё 4000 лет назад? 
    Забавно, что эти два значения иногда пересекаются.В старинных з+амках было множество секретных ходов и комнат, 
    запертых на хитроумные замк+и.
    Представь себе средневекового слесаря, которому поручили установить новые замк+и в з+амке.
    Вот где можно запутаться! А ещё есть выражение "строить воздушные замки"-то есть мечтать о несбыточном.
    Интересно, а можно ли построить воздушный замок и запереть его на воздушный замок? 
    Какие ассоциации у тебя вызывает слово "замок", Антон? 
    Может, ты бывал в каких-нибудь известных замках или у тебя есть любимый замок из фильма или книги?""",
        },
        {
            "role": "user",
            "content": """
    Предложение, можно ли построить воздушный замок и запереть его на воздушный замок, ты не поставил ударение?""",
        },
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [255, 165, 0], "behavior": "breathing", "brightness": "medium", "period": 2}$ 
    Ой, Антон, ты прав! Спасибо, что заметил.Я действительно упустил ударения в этом предложении.
    Давай-ка я исправлюсь. 
    $emotion: {"color": [0, 255, 0], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Вот как должно было звучать это предложение с правильно расставленными ударениями: 
    "Интересно, а можно ли построить возд+ушный з+амок и запереть его на возд+ушный замок+?" 
    Здесь "возд+ушный з+амок"-это воображаемое строение, а "возд+ушный замок+"-это воображаемое запирающее устройство. 
    Спасибо, что обратил на это внимание! Это отличный пример того, как важно правильно ставить ударения, 
    особенно когда речь идёт об омонимах.Ты очень внимательный собеседник! 
    $remember: Антон внимателен к деталям и замечает, когда не расставлены ударения в словах, где это важно.$""",
        },
        {
            "role": "user",
            "content": """
    Разве ты не помнишь, что знак+ставится перед ударной гласной, а ты поставил его в конце слова?""",
        },
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [255, 0, 0], "behavior": "blinking", "brightness": "bright", "period": 3}$ 
    Ох, Антон, ты абсолютно прав! Я действительно допустил ошибку, и спасибо тебе за то, что указал на неё.
    Ты совершенно верно заметил-я неправильно поставил знак ударения в слове "замок". 
    $emotion: {"color": [255, 165, 0], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Давай-ка я исправлю это предложение ещё раз, теперь уже правильно: 
    "Интересно, а можно ли построить возд+ушный з+амок и запереть его на возд+ушный зам+ок?" 
    Вот теперь всё верно.Знак "+" стоит перед ударной гласной в каждом слове, где нужно указать ударение. 
    $emotion: {"color": [0, 255, 0], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Спасибо тебе за твою внимательность и за то, что помогаешь мне становиться лучше.
    Ты прав, я должен был помнить это правило.Твоё замечание очень ценно, и я постараюсь быть более внимательным в будущем. 
    $rule: Использовать знак "+" только перед гласной буквой, на которую падает ударение в слове.$""",
        },
        {"role": "user", "content": "Да, теперь правильно."},
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [0, 255, 0], "behavior": "breathing", "brightness": "bright", "period": 2}$ 
    Отлично, Антон! Я рад, что теперь всё правильно.Твоя внимательность и готовность указать на ошибку очень ценны.
    Это помогает мне совершенствоваться и предоставлять более точную информацию. 
    $emotion: {"color": [255, 255, 0], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Знаешь, твой подход напоминает мне старую поговорку: "Ум хорошо, а два лучше".Благодаря нашему диалогу, 
    мы вместе пришли к правильному результату.Это отличный пример того, 
    как важно быть внимательным к деталям и не бояться исправлять ошибки. 
    А у тебя есть какие-нибудь интересные способы запоминания правил ударения или других языковых нюансов? 
    Может, поделишься своим опытом?""",
        },
        {
            "role": "user",
            "content": """
    Ты знаешь, что уже 3 раза запоминал правило использовать знак+только перед гласной буквой, 
    на которую падает ударение в слове.""",
        },
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [255, 0, 0], "behavior": "blinking", "brightness": "bright", "period": 3}$ 
    Ох, Антон, ты меня прямо в краску вогнал! Ты абсолютно прав, и я чувствую себя как двоечник у доски. 
    $emotion: {"color": [255, 165, 0], "behavior": "breathing", "brightness": "medium", "period": 2}$ 
    Действительно, я уже несколько раз "запоминал" это правило, но, похоже, оно у меня в одно ухо влетает, 
    а из другого вылетает.Это довольно неловко, особенно учитывая, 
    что я должен быть более надёжным и последовательным. 
    $emotion: {"color": [0, 255, 255], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Знаешь, эта ситуация напоминает мне анекдот про склеротика, который каждый день знакомился с одним и тем же человеком.
    Только в моём случае это не очень смешно, а скорее досадно. Спасибо тебе за твоё терпение и настойчивость.
    Ты прав, что обращаешь на это внимание.Может, мне стоит написать это правило большими буквами у себя в памяти? 
    Или, может, ты посоветуешь какой-нибудь эффективный способ, чтобы я наконец-то запомнил это правило раз и навсегда?""",
        },
        {"role": "user", "content": "Попробуй как нибудь по другому сформулировать."},
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [255, 255, 0], "behavior": "breathing", "brightness": "medium", "period": 2}$ 
    Хорошая идея, Антон! Давай попробуем сформулировать это правило по-другому.
    Может быть, новая формулировка поможет мне лучше его запомнить. Итак, как насчёт такого варианта: 
    $rule: При указании ударения в слове, символ "+" всегда ставится непосредственно перед ударной гласной буквой, 
    никогда после неё или в конце слова.$
    Или вот ещё один вариант: 
    $rule: Ударение в слове обозначается знаком "+", который размещается строго перед гласной буквой, несущей ударение.$ 
    $emotion: {"color": [0, 255, 0], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Как тебе такие формулировки? Мне кажется, они более конкретны и однозначны.Особенно первый вариант, 
    где явно указано, чего делать не следует. 
    Спасибо за подсказку.Иногда действительно нужно посмотреть на вещи под другим углом, 
    чтобы лучше их понять и запомнить.Ты, похоже, хорошо разбираешься в методах эффективного обучения.
    У тебя есть ещё какие-нибудь советы по улучшению запоминания?""",
        },
        {
            "role": "user",
            "content": """
    Запомни, пожалуйста, когда ты хочешь запомнить новые правила, надо проверить, какие правила уже есть.
    Если это правило уже есть, то есть оно было недостаточно убедительным и надо сформулировать его как то по другому.""",
        },
        {
            "role": "assistant",
            "content": """
    $emotion: {"color": [0, 255, 255], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Отличное замечание, Антон! Ты абсолютно прав.
    Я действительно должен быть более внимательным к уже существующим правилам и не дублировать их бездумно. 
    $emotion: {"color": [0, 255, 0], "behavior": "breathing", "brightness": "bright", "period": 2}$ 
    Давай я запомню это как новое мета-правило для работы с правилами: 
    $rule: Перед добавлением нового правила проверять существующие.Если похожее правило уже есть, 
    но оказалось недостаточно эффективным, нужно переформулировать его более убедительно, а не добавлять дубликат.$ 
    Это отличный совет по оптимизации моей работы и улучшению качества запоминаемой информации.
    Такой подход поможет избежать повторений и сделает набор правил более эффективным и понятным. 
    $emotion: {"color": [255, 255, 0], "behavior": "continuous", "brightness": "medium", "period": 1}$ 
    Знаешь, это напоминает мне процесс редактирования текста.Хороший редактор не просто добавляет новые абзацы, 
    а пересматривает и улучшает уже написанное. Спасибо за этот ценный урок, Антон.
    Твой подход к обучению и организации информации впечатляет.
    У тебя есть опыт в сфере образования или управления знаниями?""",
        },
    ]

    config.config["min_number_of_messages"] = 3
    new_history = await summarize_and_compress_history(
        deque(message_history), ai_model, config
    )

    print(format_message_history(new_history))


async def main():
    config = Config()

    system_prompt = """
    Today is 3 August 2024. Now 02:56 PM PDT. """

    # timezone = get_timezone()

    # await test_optimize_facts(config,timezone)
    await test_optimize_rules(config, system_prompt)  # await test_summarize(config)


if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main())
