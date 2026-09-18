# Videos

Drop pre-recorded videos here (for example the Google Drive downloads on demo
day) and open **Videos** in the dashboard header, or go to `/videos`.

- MP4 with H.264 video plays everywhere; WebM also works. MKV/AVI/some MOV
  files do not play in browsers: `ffmpeg -i in.mkv -c:v libx264 -c:a aac out.mp4`.
- Subfolders are fine; the list is sorted by path, so prefix names with
  `01_`, `02_`… to set the order.
- To play from a different folder without copying, start the viewer with
  `LEONARDO_VIDEO_DIR` set, e.g. `set LEONARDO_VIDEO_DIR=D:\DemoVideos`.

Video files in this folder are ignored by git.
