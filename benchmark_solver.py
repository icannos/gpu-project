from matplotlib import pyplot as plt
import numpy as np
import subprocess
import pandas as pd


max_N, max_d, stepN, stepD = 10, 1000, 1, 100

def get_times(s):
    s = s.splitlines()
    solve_time = float(s[0].split(" ")[2])

    return solve_time

data = []
N = []
for n in range(1, max_N, stepN+1):
    data.append([])
    D = []
    for d in range(2, max_d, stepD+2):
        output = subprocess.run(["./build/solve_benchmark", str(n), str(d), "16"], check=True, stdout=subprocess.PIPE, universal_newlines=True)
        print(output.stdout)
        solve_time = get_times(output.stdout)
        data[-1].append(solve_time)
        D.append(d)

df = pd.DataFrame(data, columns = D)
df.to_csv("exports/benchmark_solver.csv")
print(df)





