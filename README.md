# KNN WHISPER

## Setup
First of all the environment must be setuped. Either use the provided Dockerfile to build a docker container 
(local machine) or use prepared singularity image with the following modifications (Metacentrum).

### Metacentrum
First invoke the singularity shell.
```bash
# invoke shell
singularity run /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.03-py3.SIF
```

Install required packages.
```bash
# install FFMPEG
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar xJfv ffmpeg-release-amd64-static.tar.xz
cd ffmpeg-6.0-amd64-static/
chmod +x ffmpeg
export PATH=$PATH:<path> # e.g. <path> = /storage/brno12-cerit/home/xtlust05/ffmpeg-6.0-amd64-static/

# install transformers
pip install transformers
```