TOPDIR=/storage/brno12-cerit/home/xvlasa15/

# wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
# tar xJfv ffmpeg-release-amd64-static.tar.xz
chmod +x ffmpeg-6.0-amd64-static/ffmpeg
export PATH=$PATH:$TOPDIR/ffmpeg-6.0-amd64-static/

cd ./knn-whisper
pip install -r requirements.txt

# -- do your amazing stuf here
python ./src/process_dataset.py