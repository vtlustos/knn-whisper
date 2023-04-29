# KNN WHISPER

## MetaCentrum
All the computation was done at the MetaCentrum.

### Prebuilt image
Download the provided singularity image (built based on the /docker/pytorch-ffmpeg-opencv.def), invoke the singularity shell and then do whatever you want to do.

```bash
pip install gdown
gdown https://drive.google.com/file/d/1YdXr3iwlSx0-Bnt6JkCrvcRctPxZPt2h/view?usp=sharing
singularity exec --bind $SCRATCHDIR:$SCRATCHDIR --nv /storage/brno12-cerit/home/xtlust05/images/pytorch-ffmpeg-opencv.SIF bash
```

### Build your own image
Note you must be member of builders!

```bash
export SINGULARITY_TMPDIR="/scratch/<user>/"
export SINGULARITY_CACHEDIR="/scratch/<user>/"
singularity build --fakeroot <dst_path.SIF> <def_file_path.def>
```

## Training
First preprocess the data.
```bash
python src/train.py -o /storage/brno12-cerit/home/xtlust05/whisper/small/ -c $SCRATCHDIR -b 64 -s openai/whisper-small
python src/train.py -o /storage/brno12-cerit/home/xtlust05/whisper/large_v2/ -c $SCRATCHDIR -b 16 -s openai/whisper-large-v2
```