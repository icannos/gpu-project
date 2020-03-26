from matplotlib import pyplot as plt
import numpy as np
import subprocess
import pandas as pd


max_N, max_d, stepN, stepD = 100, 1000, 5, 100

def get_times(s):
    s = s.splitlines()

    facto_time = float(s[0].split(" ")[2])
    solve_time = float(s[1].split(" ")[2])

    return facto_time, solve_time

for n in range(1, max_N, stepN+1):
    for d in range(2, max_d, stepD+2):
        output = subprocess.run(["./build/full", str(n), str(d)], check=True, stdout=subprocess.PIPE, universal_newlines=True)

        facto_time, solve_time = get_times(output.stdout)

        print(facto_time, solve_time)
