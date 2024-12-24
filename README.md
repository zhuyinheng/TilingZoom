# TilingZoom


https://github.com/user-attachments/assets/214e91bd-4245-4d78-b9d3-d921a6f22d18

<video width="100%" autoplay loop muted playsinline>
  <source src="https://github.com/zhuyinheng/TilingZoom/raw/refs/heads/master/assets/preview.mp4" type="video/mp4" />
</video>

**TilingZoom** is a scalable tool for arranging 1,000+ videos (e.g., 1080P) on a single canvas, generating smooth zooming or sliding effects as shown above. To handle the memory demands (100GB+ for such video sets), videos are dynamically loaded instead of being fully loaded into memory.

This is achieved using:

- **[napari viewer](https://napari.org/)**: Used as the multiscale 2D canvas to manage large-scale video tiling.
- **[dask](https://www.dask.org/)**: Enables lazy loading for efficient handling of large datasets.
- **[ffmpeg](https://ffmpeg.org/)**: Generates multiscale representations of each video, optimizing them for zooming and tiling.
- **[zarr](https://zarr.readthedocs.io/)**: Reorders and chunks video data for better data locality and access performance.

more docs, and test funcs to be done.
