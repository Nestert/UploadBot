# scenes.py — нарезка видео на сцены (PySceneDetect)

import subprocess
import os
import uuid
import glob
import logging
import json
import sys

def get_video_duration(video_path):
    """
    Получает длительность видео в секундах с помощью ffprobe.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json",
        video_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except Exception as e:
        logging.warning(f"Не удалось определить длительность видео {video_path}: {e}")
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

def detect_scenes(input_video, output_dir=None, min_duration=20, max_duration=90):
    """
    Детектирует сцены в видео и автоматически режет их на отдельные файлы.
    Фильтрует результаты по заданным параметрам длительности.
    Возвращает список путей к фрагментам/сценам.
    """

    scene_id = str(uuid.uuid4())

    if output_dir:
        scene_dir = os.path.join(output_dir, f"scenes_{scene_id}")
    else:
        scene_dir = f"scenes_{scene_id}"
    os.makedirs(scene_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "scenedetect",
        "--input", input_video,
        "detect-content",
        "--min-scene-len", str(min_duration),
        "list-scenes",
        "split-video",
        "--output", scene_dir
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        stderr = ""
        if hasattr(e, "stderr") and e.stderr:
            stderr = f" | stderr: {e.stderr.strip()[:300]}"
        raise Exception(f"Ошибка при распознании сцен: {e}{stderr}")

    clip_files = sorted(glob.glob(os.path.join(scene_dir, "*.mp4")))

    if not clip_files:
        return [input_video]

    filtered_clips = filter_clips_by_duration(clip_files, scene_dir, min_duration=min_duration, max_duration=max_duration)

    return filtered_clips if filtered_clips else [input_video]
