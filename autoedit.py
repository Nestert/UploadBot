# autoedit.py — автоматическая нарезка по динамике (тихие моменты вырезаются)

import subprocess
import os
import uuid
import sys

from errors import VideoProcessingError

def cut_silence(input_video, output_dir=None, silent_speed=99999, min_cut=0.2):
    """
    Удаляет тишину или малодинамичные фрагменты из видео с помощью auto-editor.
    Возвращает имя файла нарезанного ролика.
    
    Параметры:
        silent_speed — во сколько раз ускоряются "тихие" части (99999 значит вырезать полностью)
        min_cut — минимальная длина (в секундах) для нарезки
    """
    cut_id = str(uuid.uuid4())
    if output_dir:
        output_file = os.path.join(output_dir, f"autoedit_{cut_id}.mp4")
    else:
        output_file = f"autoedit_{cut_id}.mp4"

    cmd = [
        sys.executable,
        "-m",
        "auto_editor",
        input_video,
        "--silent-speed", str(silent_speed),
        "--margin", f"{min_cut}s",
        "-o", output_file
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        if os.path.exists(output_file):
            return output_file
        else:
            raise VideoProcessingError("Auto-Editor не создал файл.")
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        stdout = (e.stdout or "").strip()
        details = stderr[:500] if stderr else stdout[:500]
        raise VideoProcessingError(f"Ошибка Auto-Editor: {e}. {details}")
    except Exception as e:
        raise VideoProcessingError(f"Ошибка Auto-Editor: {e}")

# Пример вызова:
# cut_video = cut_silence("scene_01.mp4") 
