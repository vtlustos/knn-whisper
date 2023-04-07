# KNN WHISPER

## MetaCentrum

### Prebuilt image
Download the provided singularity image (built based on the /docker/pytorch-ffmpeg-opencv.def), invoke the singularity shell and then do whatever you want to do.

```bash
singularity exec --nv <path.SIF> bash
```

### Build your own image
Note you must be member of builders!

```bash
export SINGULARITY_TMPDIR="/scratch/<user>/"
export SINGULARITY_CACHEDIR="/scratch/<user>/"
singularity build --fakeroot <dst_path.SIF> <def_file_path.def>
```