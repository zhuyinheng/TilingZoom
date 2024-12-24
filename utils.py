import logging
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor

import imageio.v3 as iio
import numpy as np
import tqdm
import zarr

FFMPEG_BINARY = "/usr/bin/ffmpeg"  # or just "ffmpeg" if it's in your PATH


def read_video_metadata(fn):
    """
    Extracts metadata from a single video file.

    Parameters:
        fn (str): Path to the video file.

    Returns:
        list: A list containing the video's frame rate (fps), duration, height, and width.
    """
    meta = iio.immeta(fn)
    list_meta = [meta[k] for k in ["fps", "duration"]]
    list_meta += [meta["size"][1], meta["size"][0]]
    return list_meta


def batch_read_video_metadata(filelist):
    """
    Extracts metadata from a batch of video files in parallel.

    Parameters:
        filelist (list): A list of file paths to video files.

    Returns:
        np.ndarray: A 2D array where each row contains metadata for one video file:
                    [frame_count, height, width, fps, duration].
    """
    result = np.zeros((len(filelist), 5))

    def process_file(idx_fn):
        """
        Processes a single video file to extract metadata.

        Parameters:
            idx_fn (tuple): A tuple containing the index and file path.

        Returns:
            tuple: The index and an array of metadata.
        """
        idx, fn = idx_fn
        meta = iio.immeta(fn)
        return idx, np.array(
            [
                meta["fps"] * meta["duration"],  # Total frame count
                meta["size"][1],  # Height
                meta["size"][0],  # Width
                meta["fps"],  # Frames per second
                meta["duration"],  # Duration in seconds
            ]
        )

    # NOTE: Use ThreadPoolExecutor for parallel I/O-bound operations. Use ProcessPoolExecutor for CPU-bound operations.
    with ThreadPoolExecutor() as executor:
        tasks = ((i, fn) for i, fn in enumerate(filelist))
        for idx, values in tqdm.tqdm(
            executor.map(process_file, tasks),
            total=len(filelist),
            desc="Batch reading video metadata (parallel)",
        ):
            result[idx] = values

    return result


def setup_logger(name="app", log_file="app.log", level=logging.INFO):
    """
    配置并获取一个日志器

    Parameters:
        name (str): 日志器名称
        log_file (str): 日志文件路径
        level (int): 日志级别

    Returns:
        logging.Logger: 配置好的日志器
    """
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # 控制台日志处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # 文件日志处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # 配置日志器
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def cal_scale_config(
    H,
    W,
    T,
    space_scale_times,
    space_scale_factor,
    temporal_scale_factor,
):

    scale_config = []
    h, w = H, W
    t = T // temporal_scale_factor
    for scale in range(space_scale_times):
        scale_config.append((t, h, w, f"{t:04d}x{h:04d}x{w:04d}"))
        h = h // space_scale_factor
        w = w // space_scale_factor
    return scale_config


def process_video(video_path, output_folder, spatial_size, frame_count):
    """
    NOTE: moivepy is problematic and slow, switch to ffmpeg instead
    Process a single video by resizing and adjusting the frame count.

    Parameters:
    - video_path: Input video file path.
    - output_folder: Folder to save the processed video.
    - spatial_size: Tuple (width, height) for resizing the video.
    - frame_count: Number of frames to retain in the video.
    """

    raise DeprecationWarning

    from moviepy.editor import VideoFileClip

    try:
        # Load video
        video = VideoFileClip(video_path)

        # Resize video
        resized_video = video.resize(newsize=spatial_size)

        # Adjust frame rate to achieve desired frame count
        new_fps = frame_count / video.duration  # Calculate new FPS
        adjusted_video = resized_video.set_fps(new_fps)

        # Output path
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_folder, video_name)

        # Save processed video
        adjusted_video.write_videofile(
            output_path,
            codec="libx264",
            # codec="h264_nvenc",
            fps=new_fps,
            audio=False,
            # preset="ultrafast",
            # ffmpeg_params=["-hwaccel", "cuda"],
            threads=3,
        )

        # print(f"Processed and saved: {output_path}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def process_video_ffmpeg(video_path, output_folder, spatial_size, frame_count):
    """
    Process a single video by resizing and adjusting the frame count using FFmpeg.

    Parameters:
    - video_path: Input video file path.
    - output_folder: Folder to save the processed video.
    - spatial_size: Tuple (width, height) for resizing the video.
    - frame_count: Number of frames to retain in the video.
    """
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        # Extract output video name and path
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_folder, video_name)

        # Calculate new FPS
        # Get video duration using ffprobe
        ffprobe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        duration = float(subprocess.check_output(ffprobe_cmd).decode().strip())
        new_fps = frame_count / duration

        # FFmpeg command to process video
        ffmpeg_cmd = [
            FFMPEG_BINARY,
            "-y",  # Overwrite output file if it exists
            "-hide_banner",  # Hide FFmpeg banner
            "-loglevel",
            "error",  # Suppress FFmpeg logs
            "-threads",
            "1",  # Number of threads to use,
            # it's better to use 1 for FFmpeg and call it multiple times in parallel for different videos(2~3x acceleration)
            # "-hwaccel", # It's not stable
            # "cuvid",
            "-i",
            video_path,  # Input video
            "-vf",
            f"scale={spatial_size[0]}:{spatial_size[1]}",  # Resize filter
            "-r",
            f"{new_fps}",  # Adjust FPS
            "-c:v",
            "libx264",  # Use H.264 codec
            # "h264_nvenc", # Use NVIDIA NVENC H.264 codec
            # "-preset",
            # "ultrafast",  # Encoding preset
            "-an",  # Disable audio
            output_path,  # Output file
        ]

        # Execute FFmpeg command
        subprocess.run(ffmpeg_cmd, check=True)
        # print(f"Processed and saved: {output_path}")

    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e}")
    except Exception as e:
        print(f"Error processing {video_path}: {e}")


def resize_video_T_H_W_parallel(
    input_videos,
    output_folder,
    spatial_size=(128, 128),
    frame_count=30,
    max_workers=11,
):
    """
    Process a list of videos by resizing and adjusting the frame count using parallel processing.

    Parameters:
    - input_videos: List of input video file paths.
    - output_folder: Folder to save the processed videos.
    - spatial_size: Tuple (width, height) for resizing the videos.
    - frame_count: Number of frames to retain in each video.
    - max_workers: Maximum number of threads to use for parallel processing.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm.tqdm(
                executor.map(
                    # lambda video_path: process_video(
                    lambda video_path: process_video_ffmpeg(
                        video_path, output_folder, spatial_size, frame_count
                    ),
                    input_videos,
                ),
                total=len(input_videos),
                desc="Processing videos",
            )
        )


def chunk_based_looping_cache_video(
    fns,
    n_cols,
    n_rows,
    zarr_fn,
    max_chunk_size=4096,
    temporal_chunk_size=60,
):
    fps, duration, H, W = read_video_metadata(fns[0])
    T = int(np.round(fps * duration))
    # print(fps, duration, H, W, T)
    C = 3
    chunk_size_H = min(max_chunk_size, n_rows * H)
    chunk_size_W = min(max_chunk_size, n_cols * W)
    n_pix_H = n_rows * H
    n_pix_W = n_cols * W
    n_chunk_H = int(np.ceil(n_pix_H / chunk_size_H))
    n_chunk_W = int(np.ceil(n_pix_W / chunk_size_W))
    print(
        f"fps: {fps}, duration: {duration}, H: {H}, W: {W}, T: {T},"
        f"n_rows: {n_rows}, n_cols: {n_cols},"
        f"n_pix_H: {n_pix_H}, n_pix_W: {n_pix_W},"
        f"chunk_size_H: {chunk_size_H}, chunk_size_W: {chunk_size_W},"
        f" n_chunk_H: {n_chunk_H}, n_chunk_W: {n_chunk_W}"
    )
    cache_array = zarr.open(
        f"{zarr_fn}",
        "w",
        shape=(T, n_pix_H, n_pix_W, C),
        chunks=(temporal_chunk_size, chunk_size_H, chunk_size_W, C),
        dtype="u1",
    )

    # Iterate over chunks, then rows, then columns for chunk locality in writing
    for chunk_idx_H in tqdm.tqdm(
        range(n_chunk_H), desc="chunk_idx_H", position=0, leave=False
    ):

        min_row_idx = int(max(np.floor(chunk_idx_H * chunk_size_H / H), 0))
        max_row_idx = int(min(np.ceil((chunk_idx_H + 1) * chunk_size_H / H), n_rows))

        for chunk_idx_W in tqdm.tqdm(
            range(n_chunk_W), desc="chunk_idx_W", position=1, leave=False
        ):
            min_col_idx = int(max(np.floor(chunk_idx_W * chunk_size_W / W), 0))
            max_col_idx = int(
                min(np.ceil((chunk_idx_W + 1) * chunk_size_W / W), n_cols)
            )
            xs = []
            for row_idx in tqdm.tqdm(
                range(min_row_idx, max_row_idx),
                desc="row_idx",
                position=2,
                leave=False,
            ):
                for col_idx in tqdm.tqdm(
                    range(min_col_idx, max_col_idx),
                    desc="col_idx",
                    position=3,
                    leave=False,
                ):
                    idx = row_idx * n_cols + col_idx
                    fn = fns[idx]
                    x = iio.imread(fn)
                    # print(len(x))
                    xs.append(np.asarray(x))
                    # xs.append(x)
            xs = np.concatenate(xs, axis=0)
            # print(len(xs))
            xs = (
                xs.reshape(
                    int(max_row_idx - min_row_idx),
                    int(max_col_idx - min_col_idx),
                    T,
                    H,
                    W,
                    C,
                ).transpose(2, 0, 3, 1, 4, 5)
            ).reshape(
                T,
                int((max_row_idx - min_row_idx) * H),
                int((max_col_idx - min_col_idx) * W),
                C,
            )
            cache_array[
                :,
                int(min_row_idx * H) : int(max_row_idx * H),
                int(min_col_idx * W) : int(max_col_idx * W),
                :,
            ] = xs[:, :, :, :]


def compute_best_tiling(N):
    """
    Compute the best tiling (n_cols, n_rows) for arranging N images.
    Ensures that aspect ratio (n_cols / n_rows) and (n_rows / n_cols) are both less than or equal to 3.
    Prioritizes configurations with no empty spaces, and if no such configuration exists,
    chooses the one with the fewest empty spaces.

    Parameters:
    - N: Number of images to arrange.

    Returns:
    - (n_cols, n_rows): A tuple representing the number of columns and rows.
    """
    best_cols, best_rows = N, 1  # Default to 1 row, all columns
    best_aspect_diff = float("inf")  # Track the best aspect ratio difference from 1
    min_empty_spaces = float("inf")  # Track the minimum number of empty spaces

    for rows in range(1, N + 1):  # Try all possible rows from 1 to N
        cols = int(
            np.ceil(N / rows)
        )  # Calculate the number of columns (allow empty spaces)
        aspect_ratio = cols / rows
        empty_spaces = (cols * rows) - N  # Calculate the number of empty spaces

        # Check if the aspect ratio is within the bounds
        if aspect_ratio <= 3 and rows / cols <= 3:
            # Prioritize configurations with no empty spaces
            if empty_spaces == 0:
                aspect_diff = abs(aspect_ratio - 1)
                if aspect_diff < best_aspect_diff:
                    best_cols, best_rows = cols, rows
                    best_aspect_diff = aspect_diff
            # Otherwise, choose the configuration with the fewest empty spaces
            elif empty_spaces < min_empty_spaces:
                aspect_diff = abs(aspect_ratio - 1)
                if aspect_diff < best_aspect_diff:
                    best_cols, best_rows = cols, rows
                    best_aspect_diff = aspect_diff
                    min_empty_spaces = empty_spaces
    if best_cols < best_rows:
        best_cols, best_rows = best_rows, best_cols
    return best_cols, best_rows
