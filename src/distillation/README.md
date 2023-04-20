singularity exec --nv /storage/brno12-cerit/home/xtlust05/images/pytorch-ffmpeg-opencv_gcc.SIF bash
python preprocess.py -d common_voice -o /storage/brno12-cerit/home/xtlust05/datasets/common_voice/


python3.8 src/distillation/train.py -d dataset/ -t ./runs/

singularity exec --nv /storage/brno12-cerit/home/xtlust05/images/pytorch-ffmpeg-opencv_gcc.SIF bash
export PYTHONPATH=$PYTHONPATH:.
python src/distillation/train.py -d ../../datasets/common_voice/ -o runs/ -t distill 

CUDA_VISIBLE_DEVICES=0,1,2,3
python -m torch.distributed.launch --nproc_per_node 4 src/distillation/train.py -d dataset/ -t ./runs/