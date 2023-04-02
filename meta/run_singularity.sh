# wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
# tar xJfv ffmpeg-release-amd64-static.tar.xz

REPODIR=/storage/brno12-cerit/home/xvlasa15/knn-whisper
DATASETDIR=/storage/brno12-cerit/home/xvlasa15/common-voice-13
FFMPEGDIR=/storage/brno12-cerit/home/xvlasa15/ffmpeg-6.0-amd64-static

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR"

cp -r $REPODIR $FFMPEGDIR $DATASETDIR "$SCRATCHDIR"


# chmod +x ffmpeg-6.0-amd64-static/ffmpeg
export PATH=$PATH:$SCRATCHDIR/ffmpeg-6.0-amd64-static/

cd $SCRATCHDIR/knn-whisper
pip install -r requirements.txt

# -- do your amazing stuf here
python ./src/fine_tune_whisper.py