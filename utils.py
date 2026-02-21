# utils.py — вспомогательные функции для проекта
import tempfile

import os
import glob
import shutil
import logging
import hashlib

class TempFileManager:
    """
    Менеджер временных файлов для централизованного управления временными данными.
    Создает уникальную директорию для каждой сессии обработки и автоматически очищает её.
    
    Использование:
        with TempFileManager() as temp_mgr:
            video_path = temp_mgr.get_path('video', 'downloaded.mp4')
            # ... работа с файлами ...
        # Все файлы автоматически удалены после выхода из контекста
    """
    
    def __init__(self, base_dir=None, prefix='uploadbot_', keep_files=False):
        """
        Инициализация менеджера временных файлов.
        
        Аргументы:
            base_dir — базовая директория для временных файлов (None = системная temp)
            prefix — префикс для имени временной директории
            keep_files — если True, файлы не удаляются автоматически (для отладки)
        """
        self.base_dir = base_dir
        self.prefix = prefix
        self.keep_files = keep_files
        self.temp_dir = None
        self._temp_dir_obj = None
        self._created_files = []
        self._created_dirs = []
    
    def __enter__(self):
        """Вход в контекст: создание временной директории."""
        self._temp_dir_obj = tempfile.TemporaryDirectory(
            prefix=self.prefix,
            dir=self.base_dir
        )
        self.temp_dir = self._temp_dir_obj.__enter__()
        logging.info(f"Создана временная директория: {self.temp_dir}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Выход из контекста: очистка временной директории."""
        if not self.keep_files and self._temp_dir_obj:
            try:
                self._temp_dir_obj.__exit__(exc_type, exc_val, exc_tb)
                logging.info(f"Временная директория удалена: {self.temp_dir}")
            except Exception as ex:
                logging.warning(f"Не удалось удалить временную директорию {self.temp_dir}: {ex}")
        elif self.keep_files:
            logging.info(f"Временные файлы сохранены в: {self.temp_dir}")
        return False
    
    def get_path(self, file_type='file', filename=None):
        """
        Генерирует путь к временному файлу внутри временной директории.
        
        Аргументы:
            file_type — тип файла ('video', 'audio', 'text', 'scene', 'subtitle', 'file')
            filename — имя файла (если None, генерируется уникальное)
        
        Возвращает:
            Полный путь к временному файлу
        """
        if not self.temp_dir:
            raise RuntimeError("TempFileManager должен использоваться в контексте with")
        
        if filename:
            path = os.path.join(self.temp_dir, filename)
        else:
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            extensions = {
                'video': '.mp4',
                'audio': '.wav',
                'text': '.txt',
                'subtitle': '.srt',
                'scene': '.mp4',
                'file': ''
            }
            ext = extensions.get(file_type, '')
            path = os.path.join(self.temp_dir, f"{file_type}_{unique_id}{ext}")
        
        self._created_files.append(path)
        return path
    
    def get_subdir(self, dirname):
        """
        Создает поддиректорию внутри временной директории.
        
        Возвращает:
            Полный путь к поддиректории
        """
        if not self.temp_dir:
            raise RuntimeError("TempFileManager должен использоваться в контексте with")
        
        subdir = os.path.join(self.temp_dir, dirname)
        os.makedirs(subdir, exist_ok=True)
        self._created_dirs.append(subdir)
        return subdir

def cleanup_files(patterns=None, dirs=None):
    """
    Удаляет временные файлы по шаблону (или список расширений),
    а также временные папки, созданные для нарезки сцен.

    Аргументы:
        patterns — список шаблонов файлов, например ['*.mp4', '*.wav', '*.txt', '*.srt']
        dirs — список директорий для удаления полностью, например ['scenes_xxx', 'vertical_xxx']

    Используйте в конце пайплайна для очистки временных данных.
    """
    if patterns is None:
        patterns = ['*.mp4', '*.wav', '*.txt', '*.srt', '*.ass']

    for p in patterns:
        for f in glob.glob(p):
            try:
                os.remove(f)
            except Exception as ex:
                logging.warning(f"Файл {f} не удалён: {ex}")

    if dirs:
        for d in dirs:
            try:
                shutil.rmtree(d)
            except Exception as ex:
                logging.warning(f"Папка {d} не удалена: {ex}")

def ensure_dir(path):
    """
    Проверяет, существует ли папка, и создаёт её если нужно.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as ex:
        logging.warning(f"Не удалось создать папку {path}: {ex}")

def list_temp_files(ext='.mp4'):
    """
    Возвращает список временных файлов с расширением ext.
    """
    return glob.glob(f"*{ext}")

def human_size(size_bytes):
    """
    Преобразует размер файла в человекочитаемый вид (МБ, КБ, ГБ)
    """
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB")
    i = int(min(len(size_name)-1, (size_bytes.bit_length() / 10)))
    p = pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

def safe_remove(path):
    """
    Безопасно удаляет файл по указанному пути.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception as ex:
        logging.warning(f"Ошибка удаления {path}: {ex}")

def split_list(lst, chunk_size):
    """
    Разбивает список на части (например, для отправки по кнопкам в Telegram).
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


VIDEOS_DIR = "videos"
RAW_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "raw")
READY_VIDEOS_DIR = os.path.join(VIDEOS_DIR, "ready")


def ensure_video_storage_dirs():
    """Создает структуру директорий для сырого и готового видео."""
    ensure_dir(VIDEOS_DIR)
    ensure_dir(RAW_VIDEOS_DIR)
    ensure_dir(READY_VIDEOS_DIR)
    _migrate_legacy_video_layout()


def _migrate_legacy_video_layout():
    """
    Переносит старые видео из videos/ в новую структуру:
    - result_* -> videos/ready
    - остальные -> videos/raw
    """
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        pattern = os.path.join(VIDEOS_DIR, f"*{ext}")
        for legacy_path in glob.glob(pattern):
            if not os.path.isfile(legacy_path):
                continue
            name = os.path.basename(legacy_path)
            target_dir = READY_VIDEOS_DIR if name.startswith("result_") else RAW_VIDEOS_DIR
            target_path = os.path.join(target_dir, name)
            if os.path.abspath(legacy_path) == os.path.abspath(target_path):
                continue
            if os.path.exists(target_path):
                continue
            try:
                shutil.move(legacy_path, target_path)
                logging.info("Legacy video moved: %s -> %s", legacy_path, target_path)
            except Exception as ex:
                logging.warning("Не удалось перенести legacy видео %s: %s", legacy_path, ex)


def file_md5(path, chunk_size=1024 * 1024):
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def persist_unprocessed_video(source_path, source_name=None):
    """
    Сохраняет исходник в videos/raw.
    Если файл с тем же хэшем уже есть, возвращает существующий путь.
    """
    if not source_path or not os.path.exists(source_path):
        return None

    ensure_video_storage_dirs()
    try:
        source_abs = os.path.abspath(source_path)
        raw_abs = os.path.abspath(RAW_VIDEOS_DIR)
        if source_abs.startswith(raw_abs + os.sep):
            return source_path
    except Exception:
        pass

    try:
        file_hash = file_md5(source_path)
    except Exception as ex:
        logging.warning(f"Не удалось вычислить хэш для исходника {source_path}: {ex}")
        file_hash = None

    base_name = source_name or os.path.basename(source_path) or "source.mp4"
    name_root, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".mp4"
    safe_root = name_root[:60] if name_root else "source"
    prefix = f"{file_hash[:10]}_" if file_hash else ""
    target_name = f"{prefix}{safe_root}{ext}"
    target_path = os.path.join(RAW_VIDEOS_DIR, target_name)

    if os.path.exists(target_path):
        return target_path

    # Если хэш известен, ищем дубликат с тем же префиксом.
    if file_hash:
        for existing in glob.glob(os.path.join(RAW_VIDEOS_DIR, f"{file_hash[:10]}_*{ext}")):
            if os.path.exists(existing):
                return existing

    try:
        shutil.copy2(source_path, target_path)
        return target_path
    except Exception as ex:
        logging.warning(f"Не удалось сохранить исходник в {target_path}: {ex}")
        return source_path


def list_available_videos():
    """
    Возвращает список доступных исходных видео файлов в папке videos/raw.
    Возвращает список словарей с ключами: path, name, size, mtime.
    """
    ensure_video_storage_dirs()
    
    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(glob.glob(os.path.join(RAW_VIDEOS_DIR, f"*{ext}")))
    
    result = []
    for vf in video_files:
        try:
            size = os.path.getsize(vf)
            mtime = os.path.getmtime(vf)
            basename = os.path.basename(vf)
            result.append({
                'path': vf,
                'name': basename,
                'size': human_size(size),
                'mtime': mtime
            })
        except Exception as ex:
            logging.warning(f"Ошибка при получении информации о файле {vf}: {ex}")
    
    # Сортируем по времени модификации (новые первые)
    result.sort(key=lambda x: x['mtime'], reverse=True)
    return result


def list_ready_videos():
    """
    Возвращает список готовых видео файлов в папке videos/ready.
    Возвращает список словарей с ключами: path, name, size, mtime.
    """
    ensure_video_storage_dirs()

    video_files = []
    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
        video_files.extend(glob.glob(os.path.join(READY_VIDEOS_DIR, f"*{ext}")))

    result = []
    for vf in video_files:
        try:
            size = os.path.getsize(vf)
            mtime = os.path.getmtime(vf)
            basename = os.path.basename(vf)
            result.append({
                'path': vf,
                'name': basename,
                'size': human_size(size),
                'mtime': mtime
            })
        except Exception as ex:
            logging.warning(f"Ошибка при получении информации о файле {vf}: {ex}")

    result.sort(key=lambda x: x['mtime'], reverse=True)
    return result


def get_video_path(video_name):
    """
    Возвращает полный путь к исходному видео файлу по имени.
    """
    ensure_video_storage_dirs()
    return os.path.join(RAW_VIDEOS_DIR, video_name)


def get_ready_video_path(video_name):
    """Возвращает полный путь к готовому видео по имени."""
    ensure_video_storage_dirs()
    return os.path.join(READY_VIDEOS_DIR, video_name)

def ensure_videos_dir():
    """
    Создает директорию для хранения готовых видео, если её нет.
    """
    ensure_video_storage_dirs()
    return READY_VIDEOS_DIR


def create_preview(input_video, output_dir=None, duration=30):
    """
    Создает превью видео (первые N секунд).
    """
    import subprocess
    import uuid

    if not output_dir:
        output_dir = os.path.dirname(input_video)

    preview_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(output_dir, f"preview_{preview_id}.mp4")

    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", "0",
        "-t", str(duration),
        "-c:v", "copy",
        "-c:a", "copy",
        output_file,
        "-y"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_file):
            return output_file
    except Exception as e:
        logging.warning(f"Не удалось создать превью: {e}")

    return None


def get_video_duration(input_video):
    """
    Возвращает длительность видео в секундах.
    """
    import subprocess
    import json
    
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        input_video
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])
        return duration
    except Exception as e:
        logging.error(f"Ошибка получения длительности видео {input_video}: {e}")
        return None


def cut_video_chunk(input_video, output_video, start_time, duration):
    """
    Вырезает фрагмент видео без перекодирования.
    
    Args:
        input_video: Путь к исходному видео
        output_video: Путь к сохранению фрагмента
        start_time: Время начала (сек)
        duration: Длительность (сек)
        
    Returns:
        True если успешно
    """
    import subprocess
    import os
    
    # Сначала seek (-ss) перед -i для быстрого перехода
    cmd = [
        "ffmpeg",
        "-ss", str(start_time),
        "-i", input_video,
        "-t", str(duration),
        "-c", "copy",
        "-avoid_negative_ts", "make_zero",
        output_video,
        "-y"
    ]
    
    try:
        logging.info(f"Вырезаю фрагмент: start={start_time}, duration={duration}")
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_video) and os.path.getsize(output_video) > 0:
            return True
        return False
    except Exception as e:
        logging.error(f"Ошибка нарезки видео: {e}")
        return False


def get_video_thumbnail(input_video, output_dir=None, timestamp=5):
    """
    Создает thumbnail из видео.
    """
    import subprocess
    import uuid

    if not output_dir:
        output_dir = os.path.dirname(input_video)

    thumb_id = str(uuid.uuid4())[:8]
    output_file = os.path.join(output_dir, f"thumb_{thumb_id}.jpg")

    cmd = [
        "ffmpeg",
        "-i", input_video,
        "-ss", str(timestamp),
        "-vframes", "1",
        "-q:v", "2",
        output_file,
        "-y"
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True)
        if os.path.exists(output_file):
            return output_file
    except Exception as e:
        logging.warning(f"Не удалось создать thumbnail: {e}")

    return None
