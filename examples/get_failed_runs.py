import argparse
import glob
from natsort import natsorted

parser = argparse.ArgumentParser(description="Gets all files with the word error in them")
parser.add_argument('--pattern', type=str)

args = parser.parse_args()

print(f'ARG Pattern: {args.pattern} \\n ')

err_files = natsorted(glob.glob(str(args.pattern)))


# keep those files that were either cancelled or killed. 
unsuccessful_runs = []
for each in err_files:
    ff = open(each, 'r')
    for line in ff:
        error_present = 'error' in line
        if  error_present:
            unsuccessful_runs.append(each)
            break


print(natsorted(unsuccessful_runs))

