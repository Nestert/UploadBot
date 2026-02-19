# moments.py — AI-выделение лучших моментов в видео

import os
import subprocess
import json
import logging
import sys
import numpy as np

logger = logging.getLogger(__name__)

MOMENT_TYPES = {
    "action": {"weight": 1.0, "name": "Экшен"},
    "dialogue": {"weight": 0.5, "name": "Диалог"},
    "laughter": {"weight": 1.5, "name": "Смех"},
    "applause": {"weight": 1.5, "name": "Аплодисменты"},
    "music": {"weight": 0.3, "name": "Музыка"},
    "silence": {"weight": -0.5, "name": "Тишина"}
}


def analyze_audio_levels(video_path, output_dir=None):
    """
    Анализирует уровни громкости в видео.
    Возвращает сегменты с разной громкостью.
    """
    if output_dir:
        analysis_file = os.path.join(output_dir, "audio_analysis.json")
    else:
        analysis_file = "audio_analysis.json"

    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-af", "volumedetect",
            "-f", "null",
            "-"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        volume_info = []
        for line in result.stderr.split('\n'):
            if 'mean_volume' in line.lower() or 'max_volume' in line.lower():
                volume_info.append(line)

        return {
            "raw": volume_info,
            "has_audio": len(volume_info) > 0
        }

    except Exception as e:
        logger.error(f"Ошибка анализа аудио: {e}")
        return {"error": str(e)}


def extract_audio_features(video_path, output_dir=None):
    """
    Извлекает аудио-фичи для анализа.
    """
    if output_dir:
        audio_file = os.path.join(output_dir, "audio_features.wav")
    else:
        audio_file = "audio_features.wav"

    features = {
        "duration": 0,
        "max_volume": 0,
        "avg_volume": 0,
        "dynamic_range": 0,
        "speech_probability": 0.5
    }

    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        features["duration"] = float(data['format']['duration'])

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ar", "16000",
            "-ac", "1",
            audio_file,
            "-y"
        ]
        subprocess.run(cmd, capture_output=True)

        return features

    except Exception as e:
        logger.error(f"Ошибка извлечения фич: {e}")
        return features


def detect_moments_by_scenes(video_path, output_dir=None, min_scene_len=3.0, max_scene_len=30.0):
    """
    Детектирует потенциально интересные моменты на основе сцен.
    """
    import uuid

    moments = []

    try:
        cmd = [
            sys.executable,
            "-m",
            "scenedetect",
            "--input", video_path,
            "detect-content",
            "--min-scene-len", str(min_scene_len),
            "list-scenes",
            "--output", output_dir or ".",
            "-of", "json"
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            try:
                scenes_data = json.loads(result.stdout)
                for scene in scenes_data.get('scenes', []):
                    start = scene.get('start_time', 0)
                    end = scene.get('end_time', start + min_scene_len)
                    duration = end - start

                    if duration >= min_scene_len and duration <= max_scene_len:
                        moments.append({
                            "start": start,
                            "end": end,
                            "duration": duration,
                            "type": "scene",
                            "score": 0.5 + (duration / max_scene_len) * 0.5
                        })
            except json.JSONDecodeError:
                pass

    except Exception as e:
        logger.error(f"Ошибка детекции сцен: {e}")

    return moments


def detect_interesting_moments(video_path, output_dir=None):
    """
    Комплексный анализ для поиска интересных моментов.
    """
    if output_dir:
        moments_dir = os.path.join(output_dir, "moments")
        os.makedirs(moments_dir, exist_ok=True)
    else:
        moments_dir = "moments"

    moments = []

    try:
        scenes = detect_moments_by_scenes(video_path, moments_dir)
        moments.extend(scenes)

        audio_features = extract_audio_features(video_path, moments_dir)

        for moment in moments:
            start = moment.get("start", 0)
            end = moment.get("end", start + 5)

            duration = end - start
            if duration < 5:
                moment["type"] = "short_scene"
            elif duration < 15:
                moment["type"] = "dialogue"
            elif duration < 30:
                moment["type"] = "action"
            else:
                moment["type"] = "long_scene"

            moment["score"] = calculate_moment_score(moment, audio_features)

        moments.sort(key=lambda x: x.get("score", 0), reverse=True)

        logger.info(f"Найдено {len(moments)} потенциально интересных моментов")
        return moments

    except Exception as e:
        logger.error(f"Ошибка анализа моментов: {e}")
        return []


def calculate_moment_score(moment, audio_features):
    """
    Вычисляет оценку интересности момента.
    """
    base_score = 0.5
    moment_type = moment.get("type", "scene")

    type_info = MOMENT_TYPES.get(moment_type, {"weight": 0.5})
    base_score *= type_info.get("weight", 0.5)

    duration = moment.get("duration", 10)
    optimal_duration = 15

    if duration < 5:
        duration_score = 0.6
    elif duration > optimal_duration * 2:
        duration_score = 0.5
    else:
        duration_score = min(1.0, duration / optimal_duration)

    base_score *= (0.7 + duration_score * 0.3)

    return min(1.0, base_score)


def extract_moment(video_path, start_time, end_time, output_dir=None):
    """
    Извлекает фрагмент видео.
    """
    import uuid

    moment_id = str(uuid.uuid4())[:8]
    if output_dir:
        output_file = os.path.join(output_dir, f"moment_{moment_id}.mp4")
    else:
        output_file = f"moment_{moment_id}.mp4"

    try:
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_time),
            "-to", str(end_time),
            "-c:v", "copy",
            "-c:a", "copy",
            output_file,
            "-y"
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        if os.path.exists(output_file):
            return output_file

    except Exception as e:
        logger.error(f"Ошибка извлечения момента: {e}")

    return None


def get_best_moments(video_path, num_moments=3, max_duration=30):
    """
    Возвращает N лучших моментов из видео.
    """
    moments = detect_interesting_moments(video_path)

    if not moments:
        duration = extract_audio_features(video_path).get("duration", 60)
        step = duration / num_moments
        for i in range(num_moments):
            moments.append({
                "start": i * step,
                "end": min((i + 1) * step, duration),
                "duration": step,
                "type": "auto",
                "score": 0.5
            })

    best_moments = []
    total_duration = 0

    for moment in moments:
        if len(best_moments) >= num_moments:
            break
        if total_duration + moment.get("duration", 0) > max_duration:
            continue

        best_moments.append(moment)
        total_duration += moment.get("duration", 0)

    return best_moments


def create_montage(video_path, moments, output_dir=None):
    """
    Создает монтаж из лучших моментов.
    """
    import uuid

    if not moments:
        return None

    moment_id = str(uuid.uuid4())[:8]
    if output_dir:
        concat_file = os.path.join(output_dir, f"concat_{moment_id}.txt")
        output_file = os.path.join(output_dir, f"montage_{moment_id}.mp4")
    else:
        concat_file = f"concat_{moment_id}.txt"
        output_file = f"montage_{moment_id}.mp4"

    try:
        with open(concat_file, 'w') as f:
            for moment in moments:
                clip = extract_moment(video_path, moment["start"], moment["end"], output_dir)
                if clip:
                    f.write(f"file '{clip}'\n")
                    f.write(f"duration {moment['duration']}\n")

        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c:v", "copy",
            "-c:a", "copy",
            output_file,
            "-y"
        ]

        subprocess.run(cmd, check=True, capture_output=True)

        if os.path.exists(output_file):
            try:
                os.remove(concat_file)
            except:
                pass
            return output_file

    except Exception as e:
        logger.error(f"Ошибка создания монтажа: {e}")

    return None


def analyze_video_quality(video_path):
    """
    Анализирует качество видео.
    """
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,r_frame_rate,codec_name",
            "-of", "json",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        video_info = data.get('streams', [{}])[0] if data.get('streams') else {}

        cmd = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration,size,bit_rate",
            "-of", "json",
            video_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        format_data = json.loads(result.stdout)
        format_info = format_data.get('format', {})

        return {
            "width": video_info.get("width", 0),
            "height": video_info.get("height", 0),
            "fps": video_info.get("r_frame_rate", "0/1"),
            "codec": video_info.get("codec_name", "unknown"),
            "duration": float(format_info.get("duration", 0)),
            "size": int(format_info.get("size", 0)),
            "bitrate": int(format_info.get("bit_rate", 0))
        }

    except Exception as e:
        logger.error(f"Ошибка анализа качества: {e}")
        return None
