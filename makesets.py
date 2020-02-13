import subprocess
from glob import glob
import os.path

all_tens = glob('huge_hc_set/10_*')

dirs = ['one_set', 'two_set', 'three_set', 'four_set']

for dire in dirs:
    subprocess.check_output(f'rm {dire}/*', shell=True)

all_tens.sort(key=os.path.getsize)

subprocess.check_output(f'cp {all_tens[0]} one_set/', shell=True)
subprocess.check_output(f'cp {all_tens[0]} {all_tens[1]} two_set/', shell=True)
subprocess.check_output(f'cp {all_tens[0]} {all_tens[1]} {all_tens[2]} three_set/', shell=True)
subprocess.check_output(f'cp {all_tens[0]} {all_tens[1]} {all_tens[2]} {all_tens[3]} four_set/', shell=True)
