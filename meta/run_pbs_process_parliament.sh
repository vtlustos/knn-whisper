#!/bin/bash
#PBS -q default@cerit-pbs.cerit-sc.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=4:mem=160gb:scratch_local=100gb
#PBS -N processParliamentHearings

REPODIR=/storage/brno12-cerit/home/xkotou06/knn-whisper


singularity exec --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.10-py3.SIF bash \
    $REPODIR/meta/process_parliament_hearings.sh

clean_scratch