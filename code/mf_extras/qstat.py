from subprocess import call

call(['ssh', 's1153197@blake.ppls.ed.ac.uk', '"python"', '"qstat.py"', '"%i"'%iteration])