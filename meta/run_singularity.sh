TOPDIR=/storage/brno12-cerit/home/xtlust05/

wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xJfv ffmpeg-release-amd64-static.tar.xz
cd ffmpeg-6.0-amd64-static/
chmod +x ffmpeg
export PATH=$PATH:$TOPDIR/ffmpeg-6.0-amd64-static/

pip install transformers

# -- do your amazing stuf here
python $TOPDIR/knn-whisper/src/test.py