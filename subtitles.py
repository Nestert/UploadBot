# subtitles.py — создание стилизованных субтитров (ASS/SSA)

import os
import uuid
import logging

logger = logging.getLogger(__name__)

DEFAULT_STYLES = {
    "bold": {
        "name": "Bold",
        "font": "Arial",
        "font_size": 48,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H80000000",
        "bold": -1,
        "italic": 0,
        "outline": 2,
        "shadow": 1,
        "alignment": 2,
        "margin_l": 10,
        "margin_r": 10,
        "margin_v": 10
    },
    "compact": {
        "name": "Compact",
        "font": "Roboto",
        "font_size": 36,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H40000000",
        "bold": 0,
        "italic": 0,
        "outline": 2,
        "shadow": 0,
        "alignment": 2,
        "margin_l": 20,
        "margin_r": 20,
        "margin_v": 20
    },
    "subtitle": {
        "name": "Subtitle",
        "font": "Helvetica",
        "font_size": 32,
        "primary_color": "&H00FFFFFF",
        "outline_color": "&H00000000",
        "back_color": "&H00000000",
        "bold": 0,
        "italic": 0,
        "outline": 3,
        "shadow": 2,
        "alignment": 2,
        "margin_l": 0,
        "margin_r": 0,
        "margin_v": 30
    },
    "modern": {
        "name": "Modern",
        "font": "Montserrat",
        "font_size": 40,
        "primary_color": "&H00FFD700",
        "outline_color": "&H00000000",
        "back_color": "&H00000000",
        "bold": -1,
        "italic": 0,
        "outline": 2,
        "shadow": 0,
        "alignment": 2,
        "margin_l": 15,
        "margin_r": 15,
        "margin_v": 15
    }
}


def seconds_to_ass_time(seconds):
    """Конвертирует секунды в формат времени ASS (H:MM:SS.cc)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centisecs = int((seconds % 1) * 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{centisecs:02d}"


def format_ass_color(color_hex, alpha=255):
    """Конвертирует HEX цвет в формат ASS (AABBGGRR)."""
    if isinstance(color_hex, str) and color_hex.startswith("&H"):
        return color_hex

    if color_hex.startswith("#"):
        color_hex = color_hex[1:]

    if len(color_hex) == 6:
        r = color_hex[0:2]
        g = color_hex[2:4]
        b = color_hex[4:6]
        return f"&H{alpha:02X}{b}{g}{r}"
    elif len(color_hex) == 8:
        a = color_hex[0:2]
        b = color_hex[2:4]
        g = color_hex[4:6]
        r = color_hex[6:8]
        return f"&H{a}{b}{g}{r}"

    return "&H00FFFFFF"


def create_ass_header():
    """Создает заголовок файла ASS."""
    return """[Script Info]
Title: UploadBot Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: None

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def style_to_ass_format(style):
    """Конвертирует стиль в формат строки ASS."""
    return f"Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"


def create_ass_style(style_name="Default", **kwargs):
    """Создает строку стиля ASS."""
    style = DEFAULT_STYLES.get(style_name, DEFAULT_STYLES["subtitle"]).copy()
    style.update(kwargs)

    primary_color = format_ass_color(style.get("primary_color", "#FFFFFF"))
    outline_color = format_ass_color(style.get("outline_color", "#000000"))
    back_color = format_ass_color(style.get("back_color", "#000000"))

    return f"""Style: {style['name']},{style['font']},{style['font_size']},{primary_color},{primary_color},{outline_color},{back_color},{style['bold']},{style['italic']},0,0,100,100,0,0,1,{style['outline']},{style['shadow']},{style['alignment']},{style['margin_l']},{style['margin_r']},{style['margin_v']},1"""


def get_style_config(style_name="subtitle", **kwargs):
    """Возвращает итоговый конфиг стиля с переопределениями."""
    style = DEFAULT_STYLES.get(style_name, DEFAULT_STYLES["subtitle"]).copy()
    style.update(kwargs)
    return style


def get_alignment_by_position(position):
    """Возвращает ASS alignment по позиции субтитров."""
    mapping = {
        "bottom": 2,
        "center": 5,
        "top": 8
    }
    return mapping.get(position, 2)


def split_text_into_lines(text, max_chars=40):
    """Разбивает текст на строки по максимальной длине."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 <= max_chars:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else [text]


def create_ass_dialogue(start_time, end_time, text, style="Default", **kwargs):
    """Создает строку диалога ASS."""
    start = seconds_to_ass_time(start_time)
    end = seconds_to_ass_time(end_time)

    lines = split_text_into_lines(text, max_chars=kwargs.get("max_chars", 40))
    text_formatted = "\\N".join(lines)

    return f"Dialogue: 0,{start},{end},{style},,0,0,0,,{text_formatted}"


def escape_ass_text(text):
    """Экранирует спецсимволы для ASS."""
    return (
        (text or "")
        .replace("\\", r"\\")
        .replace("{", r"\{")
        .replace("}", r"\}")
    )


def _has_strong_punctuation(word):
    """Проверяет окончание слова на сильную пунктуацию."""
    cleaned = (word or "").strip()
    return bool(cleaned) and cleaned[-1] in ".!?"


def group_words_into_phrases(
    words,
    max_words_per_phrase=3,
    max_gap_sec=0.45,
    max_phrase_duration=1.2
):
    """
    Группирует слова с таймкодами в короткие фразы.
    """
    normalized = []
    for item in words or []:
        try:
            word_text = (item.get("word") or "").strip()
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError, AttributeError):
            continue
        if not word_text:
            continue
        if end <= start:
            end = start + 0.05
        normalized.append({"word": word_text, "start": start, "end": end})

    normalized.sort(key=lambda x: (x["start"], x["end"]))

    phrases = []
    current_words = []
    phrase_start = None
    prev_end = None

    def flush_phrase():
        nonlocal current_words, phrase_start, prev_end
        if not current_words:
            return
        phrase_end = current_words[-1]["end"]
        text = " ".join(w["word"] for w in current_words).strip()
        if text:
            phrases.append({
                "start": phrase_start,
                "end": max(phrase_end, phrase_start + 0.05),
                "text": text
            })
        current_words = []
        phrase_start = None
        prev_end = None

    for w in normalized:
        if not current_words:
            current_words = [w]
            phrase_start = w["start"]
            prev_end = w["end"]
            if _has_strong_punctuation(w["word"]):
                flush_phrase()
            continue

        gap = max(0.0, w["start"] - prev_end)
        phrase_duration_if_added = w["end"] - phrase_start
        should_split_before = (
            len(current_words) >= max_words_per_phrase
            or gap > max_gap_sec
            or phrase_duration_if_added > max_phrase_duration
        )

        if should_split_before:
            flush_phrase()
            current_words = [w]
            phrase_start = w["start"]
            prev_end = w["end"]
            if _has_strong_punctuation(w["word"]):
                flush_phrase()
            continue

        current_words.append(w)
        prev_end = w["end"]
        if _has_strong_punctuation(w["word"]):
            flush_phrase()

    flush_phrase()
    return phrases


def create_ass_subtitles_from_words(
    words,
    output_dir=None,
    style_name="subtitle",
    position="bottom",
    max_words_per_phrase=3,
    max_gap_sec=0.45,
    max_phrase_duration=1.2
):
    """
    Создает ASS субтитры из слов с таймкодами.
    """
    ass_id = str(uuid.uuid4())[:8]
    if output_dir:
        ass_file = os.path.join(output_dir, f"subs_{ass_id}.ass")
    else:
        ass_file = f"subs_{ass_id}.ass"

    try:
        alignment = get_alignment_by_position(position)
        style_cfg = get_style_config(style_name, alignment=alignment)
        style_line = create_ass_style(style_name, alignment=alignment)
        style_ass_name = style_cfg["name"]

        phrases = group_words_into_phrases(
            words,
            max_words_per_phrase=max_words_per_phrase,
            max_gap_sec=max_gap_sec,
            max_phrase_duration=max_phrase_duration
        )

        if not phrases:
            return None

        with open(ass_file, "w", encoding="utf-8") as f:
            f.write(create_ass_header())
            f.write(style_line + "\n")
            for phrase in phrases:
                text = escape_ass_text(phrase["text"])
                dialogue = create_ass_dialogue(
                    phrase["start"],
                    phrase["end"],
                    text,
                    style=style_ass_name,
                    max_chars=40
                )
                f.write(dialogue + "\n")

        logger.info("ASS субтитры из таймкодов созданы: %s", ass_file)
        return ass_file
    except Exception as e:
        logger.error("Ошибка создания ASS субтитров из слов: %s", e)
        return None


def create_ass_subtitles(transcript, output_dir=None, style_name="subtitle", duration_per_char=0.08):
    """
    Создает файл субтитров в формате ASS из транскрипта.

    Args:
        transcript: Текст транскрипции
        output_dir: Директория для сохранения
        style_name: Имя стиля (subtitle, bold, compact, modern)
        duration_per_char: Длительность на символ в секундах

    Returns:
        Путь к файлу субтитров
    """
    ass_id = str(uuid.uuid4())[:8]
    if output_dir:
        ass_file = os.path.join(output_dir, f"subs_{ass_id}.ass")
    else:
        ass_file = f"subs_{ass_id}.ass"

    try:
        with open(ass_file, 'w', encoding='utf-8') as f:
            f.write(create_ass_header())

            style_cfg = get_style_config(style_name)
            f.write(create_ass_style(style_name) + "\n")

            paragraphs = transcript.split('\n\n')
            current_time = 0.0

            for para in paragraphs:
                if not para.strip():
                    continue

                sentences = para.replace('.', ' ').replace('!', ' ').replace('?', ' ').split('\n')
                current_time += 0.5

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    duration = max(2.0, len(sentence) * duration_per_char)
                    end_time = current_time + duration

                    dialogue = create_ass_dialogue(current_time, end_time, sentence, style_cfg["name"])
                    f.write(dialogue + "\n")

                    current_time = end_time

        logger.info(f"ASS субтитры созданы: {ass_file}")
        return ass_file

    except Exception as e:
        logger.error(f"Ошибка создания ASS субтитров: {e}")
        return None


def create_srt_subtitles(transcript, output_dir=None, style_name="subtitle"):
    """
    Создает файл субтитров в формате SRT.

    Args:
        transcript: Текст транскрипции
        output_dir: Директория для сохранения
        style_name: Имя стиля (для совместимости)

    Returns:
        Путь к файлу субтитров
    """
    srt_id = str(uuid.uuid4())[:8]
    if output_dir:
        srt_file = os.path.join(output_dir, f"subs_{srt_id}.srt")
    else:
        srt_file = f"subs_{srt_id}.srt"

    try:
        with open(srt_file, 'w', encoding='utf-8') as f:
            paragraphs = transcript.split('\n\n')
            current_time = 0.0
            subtitle_num = 1

            for para in paragraphs:
                if not para.strip():
                    continue

                sentences = para.replace('.', ' ').replace('!', ' ').replace('?', ' ').split('\n')

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    duration = max(2.0, len(sentence) * 0.06)
                    end_time = current_time + duration

                    start_srt = format_srt_time(current_time)
                    end_srt = format_srt_time(end_time)

                    f.write(f"{subtitle_num}\n")
                    f.write(f"{start_srt} --> {end_srt}\n")
                    f.write(f"{sentence}\n\n")

                    subtitle_num += 1
                    current_time = end_time

        logger.info(f"SRT субтитры созданы: {srt_file}")
        return srt_file

    except Exception as e:
        logger.error(f"Ошибка создания SRT субтитров: {e}")
        return None


def format_srt_time(seconds):
    """Конвертирует секунды в формат времени SRT (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millisecs = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"


def get_available_styles():
    """Возвращает список доступных стилей."""
    return list(DEFAULT_STYLES.keys())


def get_style_preview(style_name):
    """Возвращает preview текст для стиля."""
    return {
        "name": DEFAULT_STYLES.get(style_name, DEFAULT_STYLES["subtitle"])["name"],
        "preview": "Пример стилизованных субтитров для вашего видео"
    }
