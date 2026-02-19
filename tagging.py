# tagging.py — генерация хештегов и подписей для коротких видео

import re
from collections import Counter

def extract_keywords(transcript, top_n=7):
    """
    Выделяет топ-ключевые слова из текста (транскрипта).
    Возвращает список ключевых слов.
    """
    text = transcript.lower()
    text = re.sub(r'[^a-zа-яё0-9\s]', '', text)
    words = text.split()
    stop_words = set(['это','он','она','мы','вы','они','что','как','или','для','к','с','и','а','the','and','in','on','with','from','to','is','of','by','at','as'])
    filtered = [w for w in words if w not in stop_words and len(w) > 2]

    freq = Counter(filtered)
    return [w for w, c in freq.most_common(top_n)]


def generate_hashtags(keywords):
    """
    Создаёт список хештегов по ключевым словам.
    """
    hashtags = []
    for w in keywords:
        w_clean = re.sub(r'[^a-zа-яё0-9]', '', w)
        if w_clean:
            hashtags.append(f'#{w_clean}')
    return hashtags


def generate_tags(transcript, hashtag_count=7):
    """
    Главная функция для генерации итогового списка хештегов и текста для публикации.
    """
    keywords = extract_keywords(transcript, top_n=hashtag_count)
    hashtags = generate_hashtags(keywords)
    desc = " ".join(keywords)
    tags_str = " ".join(hashtags)
    return f"{desc}\n{tags_str}"
