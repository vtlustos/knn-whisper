TOPDIR=/storage/brno12-cerit/home/xkotou06/


cd $TOPDIR/knn-whisper
pip install -r requirements.txt

# -- do your amazing stuf here
python ./src/process_parliament_dataset.py