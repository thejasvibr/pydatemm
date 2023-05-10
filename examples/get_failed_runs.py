import argparse
import glob
import string 
import random
from natsort import natsorted

parser = argparse.ArgumentParser(description="Gets all files with the word error in them. Use * to set the variable portion")
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

# now get the matter in between the common text on the left and right. 
lhs, rhs = str(args.pattern).split('*')

def get_index_number(fullstring, lhs, rhs):
    
    test_char_present = False
    test_char = 'Q'
    i = 0
    while not test_char_present:
        if test_char not in fullstring:
            test_char_present = True
        else:
            test_char = random.choice(string.ascii_lowercase.swapcase(), 1)
        i +=1 
        if i>5000:
            raise ValueError('Too many trials - replace and split method wont work!')
    only_variable = fullstring.replace(lhs, test_char).replace(rhs, test_char).replace(test_char, '')
    return only_variable
    
error_indices = [ get_index_number(each, lhs, rhs) for each in unsuccessful_runs]
index_filename = f'errinds_{lhs}.indices'
with open(index_filename, 'w') as ff:
    ff.write(','.join(error_indices))