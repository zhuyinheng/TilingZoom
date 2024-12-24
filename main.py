import json
import logging
import os

import dask
import imageio.v3 as iio
import napari
import numpy as np
import tqdm
import zarr
from napari_animation import Animation
from napari_animation.easing import Easing


class MultiscaleTiledVideos:
    """
    MultiscaleTiledVideos
    folder structure
    - tmpdir
     - tiled_videos_zoom_in_animation.log
     - multiscale_videos
       - 0032x0256x0256
       - ...
    """

    tmpdir: str
    N: int
    H: int
    W: int
    T: int
    n_cols: int
    n_rows: int
    scale_config: list[tuple[int, int, int, str]]

    # constant
    required_keys = [
        "tmpdir",
        "H",
        "W",
        "T",
        "n_cols",
        "n_rows",
        "scale_config",
    ]
    optional_keys = [
        "N",
        "enforce_resize",
        "max_chunk_size",
        "temporal_chunk_size",
        "filelist",
    ]

    def __init__(self, tmpdir, clean=False):
        self.tmpdir = tmpdir
        if clean:
            os.system(f"rm -rf {tmpdir}")

        os.makedirs(f"{tmpdir}", exist_ok=True)
        self.logger = setup_logger(
            "tiled_videos_zoom_in_animation",
            f"{tmpdir}/tiled_videos_zoom_in_animation.log",
        )

    @classmethod
    def from_filelist(
        cls,
        tmpdir,
        filelist,
        enforce_resize=False,
        H=1024,
        W=1024,
        T=32,
        space_scale_times=6,
        space_scale_factor=2,
        temporal_scale_factor=1,
        n_cols=None,
        n_rows=None,
        max_chunk_size=4096,
        temporal_chunk_size=60,
    ):
        mst_video = cls(tmpdir)
        mst_video.setup_meta(
            filelist,
            enforce_resize,
            H,
            W,
            T,
            space_scale_times,
            space_scale_factor,
            temporal_scale_factor,
            n_cols,
            n_rows,
            max_chunk_size,
            temporal_chunk_size,
        )
        mst_video.cache_multiscale_videos()
        mst_video.cache_multiscale_zarr()
        mst_video.save_meta()
        return mst_video

    @classmethod
    def from_tmpdir(cls, tmpdir):
        mst_video = cls(tmpdir)
        mst_video.load_from_meta()
        return mst_video

    def setup_meta(
        self,
        filelist,
        enforce_resize=False,
        H=1024,
        W=1024,
        T=32,
        space_scale_times=6,
        space_scale_factor=2,
        temporal_scale_factor=1,
        n_cols=None,
        n_rows=None,
        max_chunk_size=4096,
        temporal_chunk_size=60,
    ):

        self.logger.info(f"Start to initialize from filelist")
        self.logger.info(f"Checking the existence of the files")
        for video_path in filelist:
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File not found: {video_path}")
        self.filelist = filelist
        self.N = len(self.filelist)

        self.logger.info("Checking H, W, T")
        if enforce_resize:
            self.H, self.W, self.T = H, W, T
            self.logger.info(f"Enforced Resize{self.H}, {self.W}, {self.T}")
        else:
            video_metadata = batch_read_video_metadata(self.filelist)
            is_same_T_H_W = (video_metadata[0:1, :] == video_metadata).all()

            float_T = video_metadata[0, 0] * video_metadata[0, 1]
            is_integer_T = float_T.round() == float_T
            if (is_same_T_H_W) and (is_integer_T):
                self.H, self.W, self.T = (
                    video_metadata[0, 1],
                    video_metadata[0, 2],
                    video_metadata[0, 0],
                )
                self.logger.info(
                    "NO RESIZE REQUIRED. The video's H, W, T are used."
                    f"{self.H}, {self.W}, {self.T}"
                )
            else:
                enforce_resize = True
                self.H, self.W, self.T = H, W, T
                self.logger.warning(
                    f"""
                    RESIZE REQUIRED.The specified H, W, T are used for resizing the videos.
                    Is all the videos have the same number of frames, height, width, and frame rate?: {is_same_T_H_W}
                    Is the number of frames an integer?: {is_integer_T}
                    Is the height the same as the specified height?: {video_metadata[0, 1] == H}
                    Is the width the same as the specified width?: {video_metadata[0, 2] == W}
                    Is the frame rate the same as the specified frame rate?: {video_metadata[0, 0] == T}
                    Used H,W,T: {self.H}, {self.W}, {self.T}                  
                """
                )
        self.scale_config = cal_scale_config(
            H,
            W,
            T,
            space_scale_times,
            space_scale_factor,
            temporal_scale_factor,
        )
        self.enforce_resize = enforce_resize

        self.logger.info(f"Setting tile configuration")
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.max_chunk_size = max_chunk_size
        self.temporal_chunk_size = temporal_chunk_size

        # Adjust the number of columns and rows
        if self.n_cols is None and self.n_rows is None:
            self.n_cols, self.n_rows = compute_best_tiling(self.N)
        elif self.n_cols is None:
            if self.N % self.n_cols == 0:
                self.n_cols = self.n_cols
                self.n_rows = self.N // self.n_cols
            else:
                self.logger.warning(
                    "n_cols is not a factor of N.",
                    "use the automated calculated n_cols and n_rows",
                )
        elif self.n_rows is None:
            if self.N % self.n_rows == 0:
                self.n_rows = self.n_rows
                self.n_cols = self.N // self.n_rows
            else:
                self.logger.warning(
                    "n_rows is not a factor of N.",
                    "use the automated calculated n_cols and n_rows",
                )
        else:
            if self.n_cols * self.n_rows != self.N:
                self.logger.warning(
                    "n_cols * n_rows != N.",
                    "use the automated calculated n_cols and n_rows",
                )
                self.n_cols, self.n_rows = compute_best_tiling(self.N)
            else:
                self.n_cols, self.n_rows = self.n_cols, self.n_rows

        self.logger.info(
            f"Using {self.n_cols} columns and {self.n_rows} rows for tiling"
        )

    def cache_multiscale_videos(self):
        multiscale_video_folder = self.multiscale_video_folder
        for scale_idx, (t, h, w, name) in enumerate(self.scale_config):

            self.logger.info(f"Start to cache the videos with scale: {name}")
            os.makedirs(multiscale_video_folder[scale_idx], exist_ok=True)

            if not self.enforce_resize and scale_idx == 0:
                logging.info(
                    "No need to resize the videos.Just create the symbolic"
                    " links"
                    " from fns to"
                    f" {self.tmpdir}/multiscale_videos/{self.scale_config[0][3]}"
                )

                for fn in self.filelist:
                    basename = os.path.basename(fn)
                    os.symlink(
                        fn,
                        f"{self.tmpdir}/multiscale_videos/{self.scale_config[0][3]}/{basename}",
                    )
            else:
                resize_video_T_H_W_parallel(
                    self.filelist,
                    f"{self.tmpdir}/multiscale_videos/{name}",
                    (h, w),
                    t,
                )

    def cache_multiscale_zarr(self):

        for scale_idx, (t, h, w, name) in enumerate(self.scale_config):
            self.logger.info(f"Start to tileize the videos with scale: {name}")

            chunk_based_looping_cache_video(
                self.multiscale_video_filelist[scale_idx],
                self.n_cols,
                self.n_rows,
                self.multiscale_tiled_videos_fn[scale_idx],
                self.max_chunk_size,
                self.temporal_chunk_size,
            )

    def __repr__(self):
        return (
            "MultiscaleTiledVideos\n"
            + json.dumps(
                [
                    f"{key}: {getattr(self, key,None)}"
                    for key in (self.required_keys + self.optional_keys)
                ],
                indent=4,
            )
            + json.dumps(
                {
                    k: getattr(self, k)
                    for k in [
                        "has_cache_multiscale_videos",
                        "has_cache_multiscale_zarr",
                    ]
                },
                indent=4,
            )
        )

    @property
    def has_cache_multiscale_videos(self):
        for folder in self.multiscale_video_folder:
            if not os.path.exists(folder):
                return False
        return True

    @property
    def has_cache_multiscale_zarr(self):
        for fn in self.multiscale_tiled_videos_fn:
            if not os.path.exists(fn):
                return False
        return True

    @property
    def video(self):
        return [zarr.open(fn, "r") for fn in self.multiscale_tiled_videos_fn]

    def load_from_meta(self):
        with open(f"{self.tmpdir}/meta.json", "r") as f:
            dir_meta_json = json.load(f)
        for key in self.required_keys:
            if key not in dir_meta_json.keys():
                raise KeyError(f"Key {key} not found in {self.tmpdir}/meta.json")
            setattr(self, key, dir_meta_json[key])

        for key in self.optional_keys:
            if key in dir_meta_json.keys():
                setattr(self, key, dir_meta_json[key])

    def save_meta(self):
        meta = {key: getattr(self, key) for key in self.required_keys}
        meta.update({key: getattr(self, key) for key in self.optional_keys})
        with open(f"{self.tmpdir}/meta.json", "w") as f:
            json.dump(meta, f, indent=4)

    @property
    def multiscale_tiled_videos_fn(self):
        rt = []
        for t, h, w, name in self.scale_config:
            rt.append(f"{self.tmpdir}/cache_{name}.zarr")
        return rt

    @property
    def multiscale_video_folder(self):
        rt = []
        for t, h, w, name in self.scale_config:
            rt.append(f"{self.tmpdir}/multiscale_videos/{name}/")
        return rt

    @property
    def multiscale_video_filelist(self):
        rt = []
        for t, h, w, name in self.scale_config:
            rt.append(
                [
                    f"{self.tmpdir}/multiscale_videos/{name}/{os.path.basename(fn)}"
                    for fn in self.filelist
                ]
            )
        return rt


def get_multiscale_tiled_video_viewer(tmpdir):
    multiscale_tiled_videos = MultiscaleTiledVideos.from_tmpdir(tmpdir)

    viewer = napari.Viewer()
    dask_multiscale_video = [
        # zarr.open(fn.replace(".zarr", "_sp.zarr"), "r") for fn in fns
        # zarr.open(fn.replace(".zarr", "_sp.zarr"), "r")
        dask.array.from_zarr(
            fn,
            chunk_size=(
                multiscale_tiled_videos.temporal_chunk_size,
                multiscale_tiled_videos.max_chunk_size,
                multiscale_tiled_videos.max_chunk_size,
                3,  # RGB
            ),
        )
        for fn in multiscale_tiled_videos.multiscale_tiled_videos_fn
    ]
    viewer.add_image(dask_multiscale_video, multiscale=True, cache=True)
    return viewer, multiscale_tiled_videos


def run_multiscale_tiled_video_viewer(tmpdir):
    viewer, multiscale_tiled_videos = get_multiscale_tiled_video_viewer(tmpdir)
    napari.run()


def animation_trajectory_1(
    viewer,
    H,
    W,
    T,
    n_rows,
    n_cols,
    target_row_idx,
    target_col_idx,
    animation_duration=10,
    animation_fps=60,
):
    """
    While video is playing, zoom in to a single video, keep for a while and then zoom out.
    """
    animation = Animation(viewer)

    zoom_in_duration = 0.6 * animation_duration
    keeping_duration = 0.2 * animation_duration
    zoom_out_duration = 0.2 * animation_duration

    # Set the initial view

    full_screen_point = (H * n_rows / 2, W * n_cols / 2)
    full_screen_zoom = 1 / (max(n_rows, n_cols) / 2)
    target_zoom = 1
    target_point = (H * target_row_idx / 2, W * target_col_idx / 2)

    viewer.camera.center = full_screen_point
    viewer.camera.zoom = full_screen_zoom
    viewer.dims.set_point(0, 0)  # set time axis to 0
    animation.capture_keyframe(steps=1)

    # zoom in to a singel video
    viewer.camera.center = target_point
    viewer.camera.zoom = target_zoom
    viewer.dims.set_point(0, T - 1)
    animation.capture_keyframe(
        steps=int(zoom_in_duration * animation_fps), ease=Easing.SINE
    )

    # keeping for a while
    viewer.dims.set_point(0, T // 2)
    animation.capture_keyframe(
        steps=int(keeping_duration * animation_fps), ease=Easing.LINEAR
    )

    # zoom out

    viewer.camera.center = full_screen_point
    viewer.camera.zoom = full_screen_zoom
    viewer.dims.set_point(0, 0)
    animation.capture_keyframe(
        steps=int(zoom_out_duration * animation_fps), ease=Easing.SINE
    )
    return animation


def run_animation(
    tmpdir,
    save_to,
    animation_trajectory_func,
    window_size=(2560, 1440),
):
    viewer, multiscale_tiled_videos = get_multiscale_tiled_video_viewer(tmpdir)

    viewer.dims.ndisplay = 2
    viewer.window.resize(window_size[0], window_size[1])

    # hide the layer control and layer selection
    viewer.window.window_menu.children()[2].trigger()  # layer control
    viewer.window.window_menu.children()[3].trigger()  # layer selection
    # viewer.window.window_menu.children()[1].trigger()  # console

    animation: Animation = animation_trajectory_func(viewer)

    animation.animate(
        save_to,
        canvas_only=True,
        quality=10,
        fps=60,
    )


if __name__ == "__main__":
    import glob

    filelist = sorted(
        glob.glob(
            "data/imagecas/cross_section_vis/*_1k__ctaVolR_surface_1SlidingPlane_turntable.mp4"
        )
    )
    mst_video = MultiscaleTiledVideos.from_filelist(
        "./.tmp_clean",
        filelist,
        enforce_resize=True,
        H=1024,
        W=1024,
        # H=512,
        # W=512,
        T=120,
        space_scale_times=6,
        space_scale_factor=2,
        temporal_scale_factor=1,
        n_cols=None,
        n_rows=None,
        max_chunk_size=1024 * 2,
        temporal_chunk_size=60,
    )

    from functools import partial

    animation_trajectory_1_partial = partial(
        animation_trajectory_1,
        H=mst_video.H,
        W=mst_video.W,
        T=mst_video.T,
        n_rows=mst_video.n_rows,
        n_cols=mst_video.n_cols,
        target_row_idx=10,
        target_col_idx=10,
        animation_duration=20,
        animation_fps=60,
    )

    run_animation("./.tmp_clean", "animation_2.mp4", animation_trajectory_1_partial)
