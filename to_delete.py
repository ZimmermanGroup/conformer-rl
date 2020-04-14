import subprocess
import os

with open('to_delete.txt') as to_delete:
    [subprocess.check_output(f'rm -rf tf_log/{x}', shell=True) for x in to_delete]
