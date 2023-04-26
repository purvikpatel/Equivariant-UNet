#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=07:59:59
#SBATCH --job-name=unet
#SBATCH --mem=64GB
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=myjob.%j.out
#SBATCH --error=myjob.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=patel.purvi@northeastern.edu


module load cuda/11.7
source /home/patel.purvi/miniconda3/bin/activate
conda activate lightning

python -c'import torch; print(torch.cuda.is_available())'
python /home/patel.purvi/Equivariant-UNet/scripts/train.py
