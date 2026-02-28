# scenes.py — нарезка видео на сцены (PySceneDetect)

import subprocess
import os
import uuid
import glob
import logging
import json
import sys

from utils import get_video_duration
from errors import SceneDetectionError

# Видео длиннее этого порога (в секундах) предварительно фильтруются
# через find_audio_hotspots перед нарезкой на сцены.
HOTSPOT_PREFILTER_THRESHOLD_SEC = 300  # 5 минут


def find_audio_hotspots(video_path, window_sec=60, top_n=10):
    """
    Делит видео на окна по window_sec секунд и оценивает каждое окно
    по громкости (mean_volume) и активности звука (доля не-тишины).
    Возвращает список (start, end) лучших top_n окон, отсортированных
    по убыванию оценки.
    """
    duration = get_video_duration(video_path)
    if duration is None or duration <= 0:
        return []

    windows = []
    t = 0.0
    while t < duration:
        end = min(t + window_sec, duration)
        windows.append((t, end))
        t += window_sec

    scored_windows = []
    for (start, end) in windows:
        window_dur = end - start
        try:
            # volumedetect на конкретном отрезке
            cmd = [
                "ffmpeg",
                "-ss", str(start),
                "-t", str(window_dur),
                "-i", video_path,
                "-af", "volumedetect",
                "-f", "null",
                "-"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

            mean_db = -99.0
            for line in result.stderr.split("\n"):
                if "mean_volume" in line.lower():
                    try:
                        mean_db = float(line.split(":")[1].replace("dB", "").strip())
                    except ValueError:
                        pass

            # Доля не-тишины через silencedetect
            cmd_silence = [
                "ffmpeg",
                "-ss", str(start),
                "-t", str(window_dur),
                "-i", video_path,
                "-af", "silencedetect=n=-35dB:d=0.3",
                "-f", "null",
                "-"
            ]
            res_silence = subprocess.run(cmd_silence, capture_output=True, text=True)
            silence_sec = 0.0
            sil_start = None
            for line in res_silence.stderr.split("\n"):
                if "silence_start" in line:
                    try:
                        sil_start = float(line.split("silence_start:")[1].strip())
                    except ValueError:
                        pass
                elif "silence_end" in line and sil_start is not None:
                    try:
                        seg_end = float(line.split("silence_end:")[1].split("|")[0].strip())
                        silence_sec += seg_end - sil_start
                        sil_start = None
                    except ValueError:
                        pass

            speech_ratio = max(0.0, 1.0 - (silence_sec / window_dur)) if window_dur > 0 else 0.0
            # Нормализуем mean_db из диапазона [-50, -10] → [0, 1]
            normalized_volume = max(0.0, min(1.0, (mean_db + 50) / 40))
            score = normalized_volume * 0.6 + speech_ratio * 0.4

            scored_windows.append({
                "start": start,
                "end": end,
                "score": score,
                "mean_db": mean_db,
                "speech_ratio": speech_ratio,
            })

        except Exception as e:
            logging.warning("Ошибка оценки окна %.1f-%.1f: %s", start, end, e)
            scored_windows.append({"start": start, "end": end, "score": 0.0})

    scored_windows.sort(key=lambda x: x["score"], reverse=True)
    best = scored_windows[:top_n]
    # Возвращаем отсортированными по времени для удобства нарезки
    best.sort(key=lambda x: x["start"])
    logging.info(
        "find_audio_hotspots: видео %.0fс, окна %dс, топ %d из %d",
        duration, window_sec, len(best), len(scored_windows)
    )
    return [(w["start"], w["end"]) for w in best]


def extract_hotspot_segment(video_path, start, end, output_dir):
    """
    Вырезает отрезок [start, end] из video_path без перекодирования.
    Возвращает путь к файлу или None при ошибке.
    """
    seg_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(output_dir, f"hotspot_{seg_id}.mp4")
    cmd = [
        "ffmpeg",
        "-ss", str(start),
        "-t", str(end - start),
        "-i", video_path,
        "-c", "copy",
        "-avoid_negative_ts", "1",
        output_file,
        "-y"
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_file):
            return output_file
    except Exception as e:
        logging.error("Ошибка нарезки hotspot %.1f-%.1f: %s", start, end, e)
    return None

def split_long_clip(video_path, output_dir, max_duration=90):
    """
    Разбивает длинное видео на сегменты по max_duration секунд.
    Возвращает список путей к сегментам.
    """
    duration = get_video_duration(video_path)
    if duration is None or duration <= max_duration:
        return [video_path]
    
    segments = []
    num_segments = int(duration / max_duration) + (1 if duration % max_duration > 0 else 0)
    
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    
    for i in range(num_segments):
        start_time = i * max_duration
        segment_id = str(uuid.uuid4())[:8]
        output_file = os.path.join(output_dir, f"{base_name}_seg{i+1}_{segment_id}.mp4")
        
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start_time),
            "-t", str(max_duration),
            "-c", "copy",
            "-avoid_negative_ts", "1",
            output_file,
            "-y"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            if os.path.exists(output_file):
                segments.append(output_file)
            else:
                logging.warning(f"Сегмент {output_file} не был создан")
        except Exception as e:
            logging.error(f"Ошибка при разбиении видео на сегмент {i+1}: {e}")
    
    return segments if segments else [video_path]

def filter_clips_by_duration(clip_files, output_dir, min_duration=20, max_duration=90):
    """
    Фильтрует клипы по длительности:
    - Удаляет клипы короче min_duration секунд
    - Разбивает клипы длиннее max_duration секунд на части
    
    Возвращает список клипов, соответствующих требованиям по длительности.
    """
    filtered_clips = []
    
    for clip in clip_files:
        duration = get_video_duration(clip)
        
        if duration is None:
            logging.warning(f"Пропускаю клип {clip} (не удалось определить длительность)")
            continue
        
        if duration < min_duration:
            logging.info(f"Клип {clip} слишком короткий ({duration:.1f}с < {min_duration}с), пропускаю")
            continue
        
        if duration > max_duration:
            logging.info(f"Клип {clip} слишком длинный ({duration:.1f}с > {max_duration}с), разбиваю на сегменты")
            segments = split_long_clip(clip, output_dir, max_duration)
            filtered_clips.extend(segments)
        else:
            filtered_clips.append(clip)
    
    return filtered_clips

def _run_scenedetect(input_video, scene_dir, min_duration):
    """
    Запускает PySceneDetect (detect-adaptive) на input_video,
    режет сцены в scene_dir. Возвращает список .mp4 файлов.
    Поднимает SceneDetectionError при ошибке.
    """
    cmd = [
        sys.executable,
        "-m",
        "scenedetect",
        "--input", input_video,
        "detect-adaptive",
        "--min-scene-len", str(min_duration),
        "--adaptive-threshold", "3.0",
        "list-scenes",
        "split-video",
        "--output", scene_dir,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        stderr = ""
        if hasattr(e, "stderr") and e.stderr:
            stderr = f" | stderr: {e.stderr.strip()[:300]}"
        raise SceneDetectionError(f"Ошибка при распознании сцен: {e}{stderr}")
    return sorted(glob.glob(os.path.join(scene_dir, "*.mp4")))


def detect_scenes(input_video, output_dir=None, min_duration=20, max_duration=90):
    """
    Детектирует сцены в видео и автоматически режет их на отдельные файлы.
    Для длинных видео (> HOTSPOT_PREFILTER_THRESHOLD_SEC) сначала находит
    аудио-горячие точки, затем детектирует сцены только в них.
    Фильтрует результаты по заданным параметрам длительности.
    Возвращает список путей к фрагментам/сценам.
    """
    scene_id = str(uuid.uuid4())

    if output_dir:
        scene_dir = os.path.join(output_dir, f"scenes_{scene_id}")
    else:
        scene_dir = f"scenes_{scene_id}"
    os.makedirs(scene_dir, exist_ok=True)

    duration = get_video_duration(input_video) or 0

    # Для длинных видео (стримов) предварительно фильтруем по аудиоактивности
    if duration > HOTSPOT_PREFILTER_THRESHOLD_SEC:
        logging.info(
            "Длинное видео (%.0fс > %ds): запускаем find_audio_hotspots перед нарезкой",
            duration, HOTSPOT_PREFILTER_THRESHOLD_SEC
        )
        hotspot_dir = os.path.join(scene_dir, "hotspots")
        os.makedirs(hotspot_dir, exist_ok=True)

        # Количество окон = ~20, размер окна адаптивен к длине
        window_sec = max(60, int(duration / 20))
        top_n = 10
        hotspots = find_audio_hotspots(input_video, window_sec=window_sec, top_n=top_n)

        if hotspots:
            all_clips = []
            for (hs_start, hs_end) in hotspots:
                seg_file = extract_hotspot_segment(input_video, hs_start, hs_end, hotspot_dir)
                if seg_file is None:
                    continue
                # Детектируем сцены внутри каждого выбранного сегмента
                seg_scene_dir = os.path.join(scene_dir, f"seg_{str(uuid.uuid4())[:6]}")
                os.makedirs(seg_scene_dir, exist_ok=True)
                try:
                    clips = _run_scenedetect(seg_file, seg_scene_dir, min_duration)
                    if clips:
                        all_clips.extend(clips)
                    else:
                        # PySceneDetect не нашёл сцен — используем сам сегмент
                        all_clips.append(seg_file)
                except SceneDetectionError as e:
                    logging.warning("SceneDetect для сегмента не удался, fallback: %s", e)
                    all_clips.append(seg_file)

            if all_clips:
                filtered_clips = filter_clips_by_duration(
                    all_clips, scene_dir, min_duration=min_duration, max_duration=max_duration
                )
                return filtered_clips if filtered_clips else [input_video]

        # Fallback: hotspots не найдены, обрабатываем целиком
        logging.warning("Hotspots не найдены, обрабатываем видео целиком")

    # Стандартный путь для коротких видео или fallback
    clip_files = _run_scenedetect(input_video, scene_dir, min_duration)

    if not clip_files:
        return [input_video]

    filtered_clips = filter_clips_by_duration(
        clip_files, scene_dir, min_duration=min_duration, max_duration=max_duration
    )
    return filtered_clips if filtered_clips else [input_video]
