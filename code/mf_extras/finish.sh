#!/bin/sh
#$ -N finish
#$ -cwd
#$ -l h_rt=00:01:00
#$ -pe sharedmem 1
#$ -l h_vmem=500M
#$ -o /exports/eddie/scratch/jcarr3/logs/
#$ -e /exports/eddie/scratch/jcarr3/logs/

. /etc/profile.d/modules.sh

curl "https://joncarr.net/qnext.php?username=jcarr3&iteration=$1"