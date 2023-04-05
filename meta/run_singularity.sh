# wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
# tar xJfv ffmpeg-release-amd64-static.tar.xz


TOPDIR=/storage/brno12-cerit/home/xvlasa15
REPODIR=$TOPDIR/knn-whisper
DATASETDIR=$TOPDIR/common-voice-13
FFMPEGDIR=$TOPDIR/ffmpeg-6.0-amd64-static


cp -r $REPODIR $FFMPEGDIR $DATASETDIR "$SCRATCHDIR"


# chmod +x ffmpeg-6.0-amd64-static/ffmpeg
export PATH=$PATH:$TOPDIR/ffmpeg-6.0-amd64-static/

cd $SCRATCHDIR/knn-whisper
pip install -r requirements.txt

# -- do your amazing stuf here
python ./src/fine_tune_whisper.py