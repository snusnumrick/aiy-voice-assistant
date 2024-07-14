import asyncio
import logging
from collections import deque
from typing import Dict, Deque

from src.config import Config
from src.web_search import WebSearcher
from src.ai_models import AIModel, ClaudeAIModel

from aiy.leds import Leds, Color, Pattern

logger = logging.getLogger(__name__)


class WebSearchTool:
    def __init__(self, config: Config, leds: Leds):
        self.web_searcher = WebSearcher(config)
        self.leds = leds
        self.led_processing_color = self.config.get('processing_color', (0, 1, 0))  # dark green
        self.led_processing_blink_period_ms = self.config.get('processing_blink_period_ms', 300)

    def _start_processing(self):
        self.leds.pattern = Pattern.blink(self.led_processing_blink_period_ms)
        self.leds.update(Leds.rgb_pattern(self.led_processing_color))

    def _stop_processing(self):
        self.leds.update(Leds.rgb_off())

    def do_search(self, parameters: Dict[str, any]) -> str:
        if 'query' in parameters:
            self._start_processing()
            result = self.web_searcher.search(parameters['query'])
            self._stop_processing()
            return result
        logger.error(f"missing  parameter  query:  {parameters}")
        return ""

    async def do_search_async(self, parameters: Dict[str, any]) -> str:
        if 'query' in parameters:
            self._start_processing()
            result = await  self.web_searcher.search_async(parameters['query'])
            self._stop_processing()
            return result
        logger.error(f"missing  parameter  query:  {parameters}")
        return ""


async def summarize_and_compress_history(message_history: Deque, ai_model: AIModel, config: Config):
    """
    Summarize  and  compress  the  conversation  history  to  reduce  token  count.
    """
    summary_prompt = config.get('summary_prompt', "Summarize  the  key  points  of  conversation,  "
                                                  "focusing  on  the  most  important  facts  and  context.  Be  concise:\n\n")
    min_number_of_messages = config.get('min_number_of_messages', 10)
    new_history = []
    if message_history[0]["role"] == "system":
        new_history.append(message_history.popleft())
    while len(message_history) > min_number_of_messages:
        msg = message_history.popleft()
        summary_prompt += f"\n{msg['role']}:  {msg['content']}"
    summary = ""
    async  for response_part in ai_model.get_response_async([{"role":"user", "content":summary_prompt}]):
        summary += response_part
    # logger.info(f"Summarized  conversation:  {summary}")
    new_history.append({"role": "system", "content": summary})
    while message_history:
        new_history.append(message_history.popleft())
    return deque(new_history)


async def main():
    config = Config()
    ai_model = ClaudeAIModel(config)

    message_history = [{"role": "user", "content": "Привет!"}, {"role": "assistant", "content": '''{"color":  [0,  255,  0],  "behavior":  "continuous",  "brightness":  "medium",  "cycle":  1}$
Привет,  Антон!  Как  дела?  Рад  тебя  слышать.Что  нового  у  тебя  произошло  с  нашего  последнего  разговора?  
Надеюсь,  твои  кошки  Ксения  и  Жозефина  не  слишком  шалят?'''},
        {"role": "user", "content": "Когда  будет  финал?"}, {"role": "assistant", "content": '''Хм,  интересный  вопрос,  Антон.Ты,  наверное,  имеешь  в  виду  какое  -  то
спортивное  событие?  Но  я  не  совсем  уверен,  о  каком  конкретно  финале  ты  спрашиваешь.
$emotion:  {"color":  [255,  165,  0],  "behavior":  "breathing",  "brightness":  "medium",  "cycle":  2}$
Давай  -  ка  я  уточню  информацию,  чтобы  дать  тебе  точный  ответ.
А,  вот  оно  что!  Теперь  я  могу  тебе  ответить.
$emotion:  {"color":  [0,  255,  0],  "behavior":  "continuous",  "brightness":  "bright",  "cycle":  1}$
Финал  Евро  -  2024  по  футболу  состоится  14  июля  в  Берлине.  Это  будет  уже  послезавтра!  Матч  пройдёт  на  стадионе  "Олимпиаштадион".
Ты  следишь  за  чемпионатом?  Какие  команды,  по  -  твоему,  имеют  шансы  дойти  до  финала?'''},
        {"role": "user", "content": "А  кто  будет  играть  в  финале?"}, {"role": "assistant", "content": '''
$emotion:  {"color":  [255,  165,  0],  "behavior":  "breathing",  "brightness":  "medium",  "cycle":  2}$        
Ох,  Антон,  ты  меня  прям  в  тупик  поставил!  Я  же  не  экстрасенс,  чтобы  предсказывать  будущее.          
Давай  -  ка  я  проверю  последнюю  информацию,  чтобы  дать  тебе  актуальный  ответ.        
$emotion:  {"color":  [0,  255,  0],  "behavior":  "continuous",  "brightness":  "bright",  "cycle":  1}$        
Ага,  вот  теперь  всё  ясно!  В  финале  Евро  -  2024  сыграют  сборные  Испании  и  Англии.        
Вот  это  да,  какая  интересная  пара  финалистов!        
Обе  команды  сильные,  так  что  матч  обещает  быть  захватывающим.  У  Испании  богатая  история  побед  на  европейских  чемпионатах,          
англичане  горят  желанием  взять  свой  первый  титул  с  1966  года.        
А  ты  за  кого  будешь  болеть?  Или  просто  настроен  насладиться  красивой  игрой,  кто  бы  ни  победил?'''},
        {"role": "user", "content": "Приведи  примеры  омонимов."}, {"role": "assistant", "content": '''        
$emotion:  {"color":  [0,  255,  255],  "behavior":  "continuous",  "brightness":  "medium",  "cycle":  1}$        
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
Может, ты знаешь какой-нибудь особенно забавный случай их использования?'''},
        {"role": "user", "content": "Расскажи еще про замки."}, {"role": "assistant", "content": '''
$emotion: {"color": [0, 255, 255], "behavior": "continuous", "brightness": "brightness":"medium", "cycle": 1}$ 
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
Может, ты бывал в каких-нибудь известных замках или у тебя есть любимый замок из фильма или книги?'''},
        {"role": "user", "content": '''
Предложение, можно ли построить воздушный замок и запереть его на воздушный замок, ты не поставил ударение?'''},
        {"role": "assistant", "content": '''
$emotion: {"color": [255, 165, 0], "behavior": "breathing", "brightness": "medium", "cycle": 2}$ 
Ой, Антон, ты прав! Спасибо, что заметил.Я действительно упустил ударения в этом предложении.
Давай-ка я исправлюсь. 
$emotion: {"color": [0, 255, 0], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Вот как должно было звучать это предложение с правильно расставленными ударениями: 
"Интересно, а можно ли построить возд+ушный з+амок и запереть его на возд+ушный замок+?" 
Здесь "возд+ушный з+амок"-это воображаемое строение, а "возд+ушный замок+"-это воображаемое запирающее устройство. 
Спасибо, что обратил на это внимание! Это отличный пример того, как важно правильно ставить ударения, 
особенно когда речь идёт об омонимах.Ты очень внимательный собеседник! 
$remember: Антон внимателен к деталям и замечает, когда не расставлены ударения в словах, где это важно.$'''},
        {"role": "user", "content": '''
Разве ты не помнишь, что знак+ставится перед ударной гласной, а ты поставил его в конце слова?'''},
        {"role": "assistant", "content": '''
$emotion: {"color": [255, 0, 0], "behavior": "blinking", "brightness": "bright", "cycle": 3}$ 
Ох, Антон, ты абсолютно прав! Я действительно допустил ошибку, и спасибо тебе за то, что указал на неё.
Ты совершенно верно заметил-я неправильно поставил знак ударения в слове "замок". 
$emotion: {"color": [255, 165, 0], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Давай-ка я исправлю это предложение ещё раз, теперь уже правильно: 
"Интересно, а можно ли построить возд+ушный з+амок и запереть его на возд+ушный зам+ок?" 
Вот теперь всё верно.Знак "+" стоит перед ударной гласной в каждом слове, где нужно указать ударение. 
$emotion: {"color": [0, 255, 0], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Спасибо тебе за твою внимательность и за то, что помогаешь мне становиться лучше.
Ты прав, я должен был помнить это правило.Твоё замечание очень ценно, и я постараюсь быть более внимательным в будущем. 
$rule: Использовать знак "+" только перед гласной буквой, на которую падает ударение в слове.$'''},
        {"role": "user", "content": "Да, теперь правильно."}, {"role": "assistant", "content": '''
$emotion: {"color": [0, 255, 0], "behavior": "breathing", "brightness": "bright", "cycle": 2}$ 
Отлично, Антон! Я рад, что теперь всё правильно.Твоя внимательность и готовность указать на ошибку очень ценны.
Это помогает мне совершенствоваться и предоставлять более точную информацию. 
$emotion: {"color": [255, 255, 0], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Знаешь, твой подход напоминает мне старую поговорку: "Ум хорошо, а два лучше".Благодаря нашему диалогу, 
мы вместе пришли к правильному результату.Это отличный пример того, 
как важно быть внимательным к деталям и не бояться исправлять ошибки. 
А у тебя есть какие-нибудь интересные способы запоминания правил ударения или других языковых нюансов? 
Может, поделишься своим опытом?'''}, {"role": "user", "content": '''
Ты знаешь, что уже 3 раза запоминал правило использовать знак+только перед гласной буквой, 
на которую падает ударение в слове.'''}, {"role": "assistant", "content": '''
$emotion: {"color": [255, 0, 0], "behavior": "blinking", "brightness": "bright", "cycle": 3}$ 
Ох, Антон, ты меня прямо в краску вогнал! Ты абсолютно прав, и я чувствую себя как двоечник у доски. 
$emotion: {"color": [255, 165, 0], "behavior": "breathing", "brightness": "medium", "cycle": 2}$ 
Действительно, я уже несколько раз "запоминал" это правило, но, похоже, оно у меня в одно ухо влетает, 
а из другого вылетает.Это довольно неловко, особенно учитывая, 
что я должен быть более надёжным и последовательным. 
$emotion: {"color": [0, 255, 255], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Знаешь, эта ситуация напоминает мне анекдот про склеротика, который каждый день знакомился с одним и тем же человеком.
Только в моём случае это не очень смешно, а скорее досадно. Спасибо тебе за твоё терпение и настойчивость.
Ты прав, что обращаешь на это внимание.Может, мне стоит написать это правило большими буквами у себя в памяти? 
Или, может, ты посоветуешь какой-нибудь эффективный способ, чтобы я наконец-то запомнил это правило раз и навсегда?'''},
        {"role": "user", "conternt": "Попробуй как нибудь по другому сформулировать."}, {"role": "assistant", "content": '''
$emotion: {"color": [255, 255, 0], "behavior": "breathing", "brightness": "medium", "cycle": 2}$ 
Хорошая идея, Антон! Давай попробуем сформулировать это правило по-другому.
Может быть, новая формулировка поможет мне лучше его запомнить. Итак, как насчёт такого варианта: 
$rule: При указании ударения в слове, символ "+" всегда ставится непосредственно перед ударной гласной буквой, 
никогда после неё или в конце слова.$
Или вот ещё один вариант: 
$rule: Ударение в слове обозначается знаком "+", который размещается строго перед гласной буквой, несущей ударение.$ 
$emotion: {"color": [0, 255, 0], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Как тебе такие формулировки? Мне кажется, они более конкретны и однозначны.Особенно первый вариант, 
где явно указано, чего делать не следует. 
Спасибо за подсказку.Иногда действительно нужно посмотреть на вещи под другим углом, 
чтобы лучше их понять и запомнить.Ты, похоже, хорошо разбираешься в методах эффективного обучения.
У тебя есть ещё какие-нибудь советы по улучшению запоминания?'''}, {"role": "user", "content": '''
Запомни, пожалуйста, когда ты хочешь запомнить новые правила, надо проверить, какие правила уже есть.
Если это правило уже есть, то есть оно было недостаточно убедительным и надо сформулировать его как то по другому.'''},
        {"role": "assistant", "content": '''
$emotion: {"color": [0, 255, 255], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Отличное замечание, Антон! Ты абсолютно прав.
Я действительно должен быть более внимательным к уже существующим правилам и не дублировать их бездумно. 
$emotion: {"color": [0, 255, 0], "behavior": "breathing", "brightness": "bright", "cycle": 2}$ 
Давай я запомню это как новое мета-правило для работы с правилами: 
$rule: Перед добавлением нового правила проверять существующие.Если похожее правило уже есть, 
но оказалось недостаточно эффективным, нужно переформулировать его более убедительно, а не добавлять дубликат.$ 
Это отличный совет по оптимизации моей работы и улучшению качества запоминаемой информации.
Такой подход поможет избежать повторений и сделает набор правил более эффективным и понятным. 
$emotion: {"color": [255, 255, 0], "behavior": "continuous", "brightness": "medium", "cycle": 1}$ 
Знаешь, это напоминает мне процесс редактирования текста.Хороший редактор не просто добавляет новые абзацы, 
а пересматривает и улучшает уже написанное. Спасибо за этот ценный урок, Антон.
Твой подход к обучению и организации информации впечатляет.
У тебя есть опыт в сфере образования или управления знаниями?'''}]

    new_history = await summarize_and_compress_history(deque(message_history), ai_model, config)

    print(new_history)

if __name__ == "__main__":
    from dotenv import load_dotenv

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    load_dotenv()
    asyncio.run(main())
