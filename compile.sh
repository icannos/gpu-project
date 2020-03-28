nvcc benchmark_solve.cu parallel_solver.cu -o build/solve_benchmark
nvcc main_solve.cu parallel_solver.cu -o build/solve
nvcc benchmark.cu parallel_solver.cu LDLt.cu -o build/full
nvcc main_fact.cu LDLt.cu -o build/fact
