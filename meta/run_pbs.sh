#PBS -N name
#PBS -q gpu@cerit-pbs.cerit-sc.cz 
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=32gb
#PBS -m a

REPODIR=/storage/brno12-cerit/home/xtlust05/ai-surveillance-center

singularity exec --nv /cvmfs/singularity.metacentrum.cz/NGC/PyTorch\:23.03-py3.SIF bash \
    $REPODIR/meta/export_singularity.sh

clean_scratch
