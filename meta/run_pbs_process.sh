#!/bin/bash
#PBS -q default@cerit-pbs.cerit-sc.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:mem=16gb:scratch_local=40gb
#PBS -N processCommonVoice

REPODIR=/storage/brno12-cerit/home/xvlasa15/knn-whisper


singularity exec --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.10-py3.SIF bash \
    $REPODIR/meta/process_common_voice.sh

clean_scratch