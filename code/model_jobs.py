'''
Generates jobs scripts (one for each model parameter combinatiobs)
that can be submitted to the cluster.
'''

sim_job = '''#!/bin/sh
#$ -N {jobid}
#$ -cwd
#$ -l h_rt=2:00:00
#$ -pe sharedmem 1
#$ -l h_vmem=1G
#$ -o ~/scratch/logs/
#$ -e ~/scratch/logs/

ulimit -c 0
. /etc/profile.d/modules.sh

module load anaconda
source activate modpy

python model.py scratch/model_results/{path}/ $SGE_TASK_ID --chains 1 --generations 50 --mincats 1 --maxcats 4 --prior simplicity --weight {weight} --noise {noise} --bottleneck {bottleneck} --exposures {exposures} --mcmc_iterations 5000
'''

inf_job = '''#!/bin/sh
#$ -N {jobid}
#$ -cwd
#$ -l h_rt=0:20:00
#$ -pe sharedmem 1
#$ -l h_vmem=600M
#$ -o ~/scratch/logs/
#$ -e ~/scratch/logs/

ulimit -c 0
export OMP_NUM_THREADS=1
. /etc/profile.d/modules.sh

module load anaconda
source activate modpy

python model.py scratch/model_results/{path}/ $SGE_TASK_ID --chains 1 --generations 50 --mincats 1 --maxcats 4 --prior informativeness --weight {weight} --noise {noise} --bottleneck {bottleneck} --exposures {exposures} --mcmc_iterations 5000
'''

def write_script(jobid, script):
	with open('jobs/%s.sh' % jobid, 'w') as file:
		file.write(script)
	print('qsub -t 1-100 jobs/%s.sh' % jobid)


for e, noise in enumerate(['0.01', '0.05', '0.1'], 1):
	for b, bottleneck in enumerate(['1', '2', '3', '4'], 1):
		for x, exposures in enumerate(['1', '2', '3', '4'], 1):

			jobid = 's' + ''.join(map(str, [e,b,x]))
			path = 's_1.0_%s_%s_%s' % (str(noise), str(bottleneck), str(exposures))
			script = sim_job.format(jobid=jobid, path=path, weight='1.0', noise=noise, bottleneck=bottleneck, exposures=exposures)
			write_script(jobid, script)

			jobid = 'i' + ''.join(map(str, [e,b,x]))
			path = 'i_1.0_%s_%s_%s' % (str(noise), str(bottleneck), str(exposures))
			script = inf_job.format(jobid=jobid, path=path, weight='1.0', noise=noise, bottleneck=bottleneck, exposures=exposures)
			write_script(jobid, script)

			jobid = 'si' + ''.join(map(str, [e,b,x]))
			path = 'i_500.0_%s_%s_%s' % (str(noise), str(bottleneck), str(exposures))
			script = inf_job.format(jobid=jobid, path=path, weight='500.0', noise=noise, bottleneck=bottleneck, exposures=exposures)
			write_script(jobid, script)
