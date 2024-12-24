import glob

from functools import partial
from main import MultiscaleTiledVideos, run_animation, animation_trajectory_1


def test_get_input():

    filelist: list[str] = sorted(
        glob.glob(
            "data/imagecas/cross_section_vis/*_1k__ctaVolR_surface_1SlidingPlane_turntable.mp4"
        )
    )[:30]
    return filelist


def test_MultiscaleTiledVideos_setup_meta():
    filelist = test_get_input()
    mst_video = MultiscaleTiledVideos("./.tmp_test", clean=True)
    mst_video.setup_meta(
        filelist,
        enforce_resize=True,
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
    )
    mst_video.save_meta()

    mst_video = MultiscaleTiledVideos("./.tmp_test")
    mst_video.load_from_meta()
    print(mst_video)


def test_MultiscaleTiledVideos_cache_multiscale_video():
    filelist = test_get_input()
    mst_video = MultiscaleTiledVideos("./.tmp_test_cuda", clean=True)
    mst_video.setup_meta(
        filelist,
        enforce_resize=True,
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
    )
    mst_video.cache_multiscale_videos()
    mst_video.save_meta()

    mst_video = MultiscaleTiledVideos("./.tmp_test")
    mst_video.load_from_meta()
    print(mst_video)


def test_MultiscaleTiledVideos_cache_multiscale_zarr():
    filelist = test_get_input()
    mst_video = MultiscaleTiledVideos("./.tmp_test_cuda")
    mst_video.setup_meta(
        filelist,
        enforce_resize=True,
        H=1024,
        W=1024,
        T=32,
        space_scale_times=6,
        space_scale_factor=2,
        temporal_scale_factor=1,
        n_cols=None,
        n_rows=None,
        max_chunk_size=4096,
        temporal_chunk_size=16,
    )
    # mst_video.cache_multiscale_videos()
    mst_video.save_meta()

    mst_video = MultiscaleTiledVideos("./.tmp_test_cuda")
    mst_video.load_from_meta()
    print(mst_video)

    mst_video.cache_multiscale_zarr()
    mst_video.save_meta()
    print(mst_video)


def test_MultiscaleTiledVideos_animation():

    mst_video = MultiscaleTiledVideos("./.tmp_test_cuda")
    mst_video.load_from_meta()

    animation_trajectory_1_partial = partial(
        animation_trajectory_1,
        H=mst_video.H,
        W=mst_video.W,
        T=mst_video.T,
        n_rows=mst_video.n_rows,
        n_cols=mst_video.n_cols,
        target_row_idx=3,
        target_col_idx=3,
        animation_duration=20,
        animation_fps=60,
    )

    run_animation("./.tmp_test_cuda", "animation_1.mp4", animation_trajectory_1_partial)
