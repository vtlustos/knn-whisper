#PBS -N CommonVoiceFinetuneWhisperSmallCZ
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=100gb:scratch_shm=True:cl_galdor=True
#PBS -m a

REPODIR=/storage/brno12-cerit/home/xvlasa15/knn-whisper

echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" 

singularity exec --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:22.10-py3.SIF bash \
    $REPODIR/meta/run_singularity.sh

clean_scratch
