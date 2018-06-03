from subprocess import call
import argparse
import os.path

def find_missing_files(iteration):
	missing = []
	for participant in range(1, 169):
		if not os.path.isfile('scratch/ifit/cand/%i/%i' % (iteration, participant)):
			missing.append(participant)
	return missing

def resubmit(participant):
	call(['qsub', '-t', str(participant), 'opt.sh'])

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('iteration', action='store', type=int, help='iteration number')
	args = parser.parse_args()

	missing = find_missing_files(args.iteration)
	for participant in missing:
		resubmit(participant)