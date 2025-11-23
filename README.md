# Sparse Matrix-Vector Multiplication (MVM) Experiments

This repository contains multiple implementations of sparse matrix-vector multiplication (SpMV) in C:

1. **Sequential**: single-threaded reference implementation.
2. **OpenMP Parallel**: parallelized using standard OpenMP loops.
3. **OpenMP Parallel with Atomic**: same as above, but uses atomic operations for accumulation.
4. **SELL-C-σ (Standalone)**: modern sparse matrix format designed for vectorization and efficient parallelization.

A unified **Bash experiment driver** (`run_experiments.sh`) runs all codes, measures timings, and generates a speedup plot.

---

## Compilation

All programs are in C. Compile them with GCC and OpenMP:

```bash
gcc -O2 -fopenmp -o MVM_sequential MVM_sequential.c
gcc -O2 -fopenmp -o MVM_parallel MVM_parallel.c
gcc -O2 -fopenmp -o MVM_parallel_atomic MVM_parallel_atomic.c
gcc -O2 -fopenmp -o MVM_parallel_sellc MVM_parallel_sellc.c
```
Running Individually
Sequential
```bash
./MVM_sequential <matrix_file> -r <runs>
<matrix_file>: path to Matrix Market .txt file.

-r: number of repeated runs (default example: 12).
```
Parallel OpenMP
```bash

./MVM_parallel <matrix_file> -r <runs> -t <threads> -s <schedule> -c <chunk>
./MVM_parallel_atomic <matrix_file> -r <runs> -t <threads> -s <schedule> -c <chunk>
-t: number of OpenMP threads.

-s: schedule type: static, dynamic, guided.

-c: chunk size for schedule.
```

SELL-C-σ
```bash
./MVM_parallel_sellc <matrix_file> -r <runs> -c <chunk> -s <sigma> -t <threads>
-c: chunk height (C).

-s: sigma (sort block size, σ).

-t: number of OpenMP threads.

-r: number of repeated runs.
```
Unified Experiment Bash Script
The run_experiments.sh script automates running all codes on multiple matrices, threads, chunks, schedules, and σ values.

How to run
```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

Configuration
At the top of run_experiments.sh, you can adjust:

Matrices:
```bash
matrices=( "./1138_bus.txt" "./adder_dcop_32.txt" "./bcsstk14.txt" "./rail_5177.txt" "./chimera_matrix.txt")
```

Schedules (for OpenMP parallel codes):
```bash
schedules=("static" "dynamic" "guided")
```
Threads:
```bash
threads_list=(2 4 8 16 32)
```
Chunk sizes:
```bash
chunks=(1 2 4 8 16 32)
```

Sigma values (for SELL-C):
```bash
sigmas=(64 128 256 512)
```

Number of runs:
```bash
runs=12
```
Adjust these values to explore different configurations.

Output
Per-run timings: logged in /tmp/mvm_runs/run_<ID>_<matrix>_<config>/stdout.txt and times.txt.

Aggregated results: RESULTS.txt contains the 90th percentile timing for each configuration.

Speedup data: speedup_data.dat contains normalized speedup values (Sequential = 1).

Plot: speedup_plot.png shows a histogram comparing speedup across all programs.

Notes
SELL-C-σ is particularly advantageous for matrices with irregular row lengths and high sparsity; small matrices may not show performance benefits.

Ensure that all matrices are in Matrix Market format, 1-based indexing.

The bash script automatically handles repeated runs, logs, and percentile calculations.

You can safely modify the script to add/remove matrices, change thread counts, or experiment with different chunk heights and σ values for SELL-C-σ.

Example Usage
Run experiments on two matrices with reduced configurations:

```bash
matrices=( "./1138_bus.txt" "./chimera_matrix.txt" )
threads_list=(4 8)
chunks=(8 16)
sigmas=(128 256)
runs=5
```

This will generate RESULTS.txt, speedup_data.dat, and speedup_plot.png reflecting the selected parameters.

