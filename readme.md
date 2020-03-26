### GPU project -- LDLT factorization


#### Structure du projet
'''
.
├── build
│   ├── full			// Factorize and solve
│   ├── solve			// Solve the system
│   ├── solve_benchmark		// Solve the systems for benchmarking
│   └── test
├── benchmark.cu		// Used to make a benchmark of the factorization and the solver
├── benchmark.py		// Pyton script which run multiple time the benchmark to make a csv file
├── benchmark_solve.cu		// Benchmark of the solver only
├── benchmark_solver.py		// Pyton script which run multiple time the benchmark to make a csv file
├── CMakeLists.txt
├── compile.sh			// Compile all the needed files
├── ide_params.h		// Just something to help IDE to work
├── LDLt.cu			// The factorization source code
├── LDLt.h			// Facto header
├── main_solve.cu		// Solve a system and display the result
├── parallel_solver.cu		// Solver code
├── parallel_solver.h		// Solver code
├── readme.md			// This file
├── verify_facto.py		// Use python to solve the problem and compare it to the computed solution
├── verify_solver.py		// Use python to inverse the system and compare it to the computed solution
└── verify_solver.sh		// Combine verify_solver.py and main_solver.cu
'''
