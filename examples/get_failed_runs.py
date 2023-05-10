import argparse
import glob
from natsort import natsorted

parser = argparse.ArgumentParser(help="Prints out all files with the word error in them. Useful to filter out job runs that \
didnt run due to an error of some kind")
parser.add_argument('-pattern', type=str)

args = parser.parse_args()

err_files = natsorted(glob.glob(args.pattern))
print(err_files)

# keep those files that were either cancelled or killed. 
unsuccessful_runs = []
for each in err_files:
    ff = open(each, 'r')
    for line in ff:
        error_present = 'error' in line
        if  error_present:
            unsuccessful_runs.append(each)
            break

print(unsuccessful_runs)

