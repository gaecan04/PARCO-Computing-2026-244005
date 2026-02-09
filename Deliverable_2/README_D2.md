# Sparse Matrix-Vector Multiplication with MPI+OpenMP (Deliverable 2)

A high-performance implementation of sparse matrix-vector multiplication (SpMV) using MPI for distributed computing and OpenMP for shared-memory parallelism. This project evaluates performance through weak and strong scaling experiments on a university cluster.

## Table of Contents

- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Building the Project](#building-the-project)
- [Running the Project](#running-the-project)
- [Results Interpretation](#results-interpretation)

## Project Overview

This project implements a parallel sparse matrix-vector multiplication (SpMV) kernel using hybrid parallelism with MPI (Message Passing Interface) and OpenMP. The implementation is designed to run on distributed-memory clusters and evaluates scalability through two types of experiments:

- **Weak Scaling**: Increases problem size proportionally to the number of processors to maintain constant work per processor
- **Strong Scaling**: Keeps problem size fixed while increasing the number of processors to measure speedup

The project includes a synthetic matrix generator for creating test matrices and automated scripts for running experiments on high-performance computing (HPC) clusters using PBS (Portable Batch System).

## Project Structure

```
Deliverable_2/
├── src/
│   └── spmv_mpi_openmp.c                   # Main SpMV implementation with MPI+OpenMP
├── matrix_generator/
│   └── gen_random_mtx.c                    # Synthetic random sparse matrix generator
├── scripts/
│   ├── run_spmv_weak_scaling.sh            # Bash script for weak scaling experiments
│   └── run_spmv_strong_scaling.sh          # Bash script for strong scaling experiments
├── pbs_script/
│   └── run_experiments.pbs                 # PBS script for cluster execution
└── README.md                                # This file
```

### Key Files Explained

| File | Purpose |
|---|---|
| `src/spmv_mpi_openmp.c` | Core SpMV implementation. Distributes matrix across MPI processes and uses OpenMP for thread-level parallelism within each process. |
| `matrix_generator/gen_random_mtx.c` | Generates synthetic sparse matrices for benchmarking. Creates random matrices with controlled sparsity patterns for reproducible testing. |
| `scripts/run_spmv_weak_scaling.sh` | Automates weak scaling experiments: runs SpMV with increasing problem size and processor count to evaluate scaling efficiency. |
| `scripts/run_spmv_strong_scaling.sh` | Automates strong scaling experiments: runs SpMV with fixed problem size and increasing processor count to measure speedup. |
| `pbs_script/run_experiments.pbs` | Cluster job submission script. Submits both scaling experiments to the PBS queue and manages resource allocation and output collection. |

## Dependencies

- **Language**: C (C99 or later)
- **Compiler**: GCC 11.3.0+ (with OpenMP support)
- **MPI**: OpenMPI 4.1.4+ or MPICH
- **Build Tools**: Standard Unix tools (bash, make, etc.)

### HPC3 Cluster Modules

On HPC3, required modules can be loaded via:

```bash
# Recommended: Load FOSS toolchain (GCC + OpenMPI bundle)
module load foss/2023a

# Or load individually:
module load GCC/12.3.0          # or GCC/12.2.0, GCC/11.3.0
module load OpenMPI/4.1.5       # or OpenMPI/4.1.4, or just OpenMPI
```

Fallback options if above are unavailable:
```bash
# Check what's available
module avail | grep -E 'GCC|OpenMPI|MPICH'

# Load MPICH if OpenMPI unavailable
module load MPICH
```

### Local Installation (Non-Cluster)

#### On Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install build-essential libopenmpi-dev openmpi-bin
```

#### On macOS:
```bash
brew install open-mpi gcc
```

### Verifying Installation

After loading modules, verify the compiler and MPI are available:

```bash
# Check GCC
gcc --version

# Check MPI compiler
which mpicc
mpicc --version

# Check OpenMP (should print "OPENMP" if available)
echo | gcc -E -dM - | grep OPENMP
```

## Building the Project

### On HPC3 Cluster (Using PBS Script)

The PBS script automatically compiles both the SpMV implementation and the matrix generator. Simply submit the job:

```bash
# Submit job to cluster
qsub pbs_script/run_experiments.pbs

# Monitor job status
qstat
```

**Important:** The PBS script assumes all source files (`spmv_mpi_openmp.c`, `gen_random_mtx.c`, and the bash scripts) are in the **same working directory** when the job runs. Ensure the required files are located in the directory from which you submit the PBS job.

The PBS script handles:
1. Loading required modules (GCC, OpenMPI/MPICH)
2. Compiling the matrix generator: `spmv_gen_random_mm`
3. Compiling the SpMV code: `spmv_mpi_openmp`
4. Running both weak and strong scaling experiments
5. Collecting results in a timestamped output directory (`results_<timestamp>/`)

### Manual Compilation (Local Testing)

When compiling manually, assume all source files are in the same working directory (as they are when the PBS script runs):

#### Step 1: Load Required Modules
```bash
module load foss/2023a          # Or similar toolchain
module load GCC/12.3.0          # Or try GCC/12.2.0, GCC/11.3.0
module load OpenMPI/4.1.5       # Or try OpenMPI/4.1.4, OpenMPI
```

#### Step 2: Compile the Matrix Generator
```bash
gcc -O3 gen_random_mtx.c -o spmv_gen_random_mm
```

#### Step 3: Compile the SpMV Implementation
```bash
mpicc -O3 -fopenmp spmv_mpi_openmp.c -o spmv_mpi_openmp
```

**Compiler Flags:**
- `-O3` — Maximum optimization level for performance
- `-fopenmp` — Enable OpenMP for shared-memory parallelism

### Verifying MPI Installation

Before compiling, verify mpicc is available:

```bash
# Check if mpicc is in PATH
which mpicc

# Check MPI version
mpicc --version
```

If `mpicc` is not found, load the appropriate module:

```bash
# List available MPI modules
module avail | grep -i mpi

# Load OpenMPI
module load OpenMPI/4.1.5

# Or load MPICH if OpenMPI is unavailable
module load MPICH
```

### Troubleshooting Build Issues

| Problem | Solution |
|---|---|
| `mpicc: command not found` | Run `module load OpenMPI/4.1.5` or check `module avail` for available versions |
| `-fopenmp not recognized` | Update GCC module: `module load GCC/12.3.0` |
| Module loading fails | Check available modules: `module avail \| grep -E 'GCC\|OpenMPI'` |
| PBS script fails | Check output file: `cat spmv_exp.<JOBID>.out` |
| Compilation errors in PBS job | Review `results_<timestamp>/env.txt` to see which modules were loaded |

## Running the Project

### Submitting to HPC3 Cluster (Recommended)

The recommended way to run the project is via the PBS script, which compiles and runs all experiments:

```bash
# Submit the job to the cluster queue
qsub pbs_script/run_experiments.pbs

# Check job status
qstat

# View detailed job information
qstat -f <JOBID>

# Cancel a job if needed
qdel <JOBID>
```

**PBS Job Configuration:**
- Queue: `shortCPUQ`
- Resources: 2 nodes with 32 cores each (64 cores total)
- Walltime: 45 minutes
- Output: `spmv_exp.<JOBID>.out` (in submit directory)

After the job completes, results are saved in a timestamped directory:
```
results_YYYYMMDD_HHMMSS/
├── strong.out              # Strong scaling results and logs
├── weak.out                # Weak scaling results and logs
└── env.txt                 # Environment/module information
```

### Local Testing (Small-Scale, No Cluster)

#### Step 1: Compile Manually
```bash
module load GCC/12.3.0 OpenMPI/4.1.5
gcc -O3 matrix_generator/gen_random_mtx.c -o gen_random_mtx
mpicc -O3 -fopenmp src/spmv_mpi_openmp.c -o spmv_mpi_openmp
```

#### Step 2: Generate a Test Matrix
```bash
./gen_random_mtx 1000 10000 test_matrix.mtx
```

#### Step 3: Run SpMV Manually
```bash
# Run on 4 MPI processes
mpirun -np 4 ./spmv_mpi_openmp test_matrix.mtx result.txt

# Or with OpenMP threads per process
export OMP_NUM_THREADS=2
mpirun -np 4 ./spmv_mpi_openmp test_matrix.mtx result.txt
```

### Running Scaling Experiments Locally

If you want to run the scaling experiments without the cluster, you can run the bash scripts directly:

```bash
# Make sure binaries are compiled first
chmod +x scripts/run_spmv_weak_scaling.sh
chmod +x scripts/run_spmv_strong_scaling.sh

# Run weak scaling
bash scripts/run_spmv_weak_scaling.sh

# Run strong scaling
bash scripts/run_spmv_strong_scaling.sh
```

**Note:** These scripts may need adjustment for smaller local systems (fewer cores, less memory than HPC3's 64 cores).

### Monitoring PBS Jobs

```bash
# List all your jobs
qstat

# View full job details (shows nodes, walltime, etc.)
qstat -f <JOBID>

# Tail the output file as it runs
tail -f spmv_exp.<JOBID>.out

# Check job error/output after completion
cat spmv_exp.<JOBID>.out
```

### Environment Variables

Control MPI and OpenMP behavior:

```bash
# Set number of OpenMP threads per MPI process
export OMP_NUM_THREADS=4

# Disable dynamic thread adjustment
export OMP_DYNAMIC=FALSE

# Show OpenMP environment at runtime
export OMP_DISPLAY_ENV=true

# MPI-specific settings (OpenMPI)
export OMPI_MCA_btl=self,vader  # Use self and vader BTL
```

## Results Interpretation

This section explains how to understand and interpret the SpMV performance results from scaling experiments.

### Output Format

#### Console Output (During Execution)
```
=== SpMV Execution ===
Matrix file: test_matrix.mtx
Matrix dimensions: 10000 x 10000
Non-zero elements: 100000
Number of MPI processes: 4
Threads per process: 2
Local matrix rows per process: 2500

Execution time: 0.1245 seconds
GFLOPS (floating-point operations per second): 3210.5
Memory bandwidth: 2145.3 MB/s
Verification: PASSED
```

#### Scaling Results (from scripts)
```
Weak Scaling Results:
Processes | Matrix_Size | Time(s) | GFLOPS | Efficiency
2         | 5000        | 0.0512  | 2891.2 | 100%
4         | 7071        | 0.0924  | 3102.4 | 107%
8         | 10000       | 0.1567  | 3245.1 | 112%
16        | 14142       | 0.2314  | 3401.3 | 117%

Strong Scaling Results:
Processes | Time(s) | GFLOPS | Speedup | Efficiency
1         | 0.8234  | 975.6  | 1.00    | 100%
2         | 0.4321  | 1859.2 | 1.90    | 95%
4         | 0.2156  | 3745.8 | 3.82    | 95%
8         | 0.1134  | 7102.1 | 7.25    | 91%
16        | 0.0687  | 11705.2| 11.97   | 75%
```

### Key Metrics Explained

- **Execution Time** — Wall-clock time in seconds to complete SpMV computation
- **GFLOPS** — Gigaflops per second (billions of floating-point operations per second). Higher is better.
- **Speedup** — Ratio of single-processor time to multi-processor time. Ideal speedup = number of processors.
- **Efficiency** — Speedup divided by number of processors (as percentage). 100% = ideal parallel efficiency.
- **Memory Bandwidth** — Amount of data transferred per second (MB/s or GB/s)

### Example Interpretation

**Weak Scaling Results Above:**
- Efficiency stays near 100% or slightly above, indicating the code maintains performance as problem and processor count grow together
- This suggests good **weak scaling** behavior — the algorithm handles larger problems efficiently with more processors
- Slight super-linear speedup (>100%) may occur due to cache effects

**Strong Scaling Results Above:**
- Speedup reaches ~12x on 16 processors (ideal would be 16x)
- Efficiency drops from 95% to 75% as processor count increases
- This is typical: parallelization overhead becomes more significant at higher processor counts
- Speedup still remains reasonable, indicating the code **scales well** to 16 processors

### Common Results Patterns

| Pattern | Interpretation |
|---|---|
| Efficiency ~100% (weak scaling) | Excellent weak scaling; algorithm maintains performance with larger problems |
| Speedup close to processor count (strong) | Excellent strong scaling; minimal parallelization overhead |
| Efficiency drops rapidly (strong) | Poor memory locality or communication bottleneck; may need optimization |
| Non-linear weak scaling | Load imbalance or increasing cache misses with problem size |
| Speedup saturates (levels off) | Hit memory bandwidth limit or computation-to-communication ratio becomes unfavorable |

### Analyzing Performance Bottlenecks

If efficiency is low, check:
1. **Communication Overhead** — MPI_Allgather or MPI_Reduce calls in the code
2. **Load Imbalance** — Uneven matrix distribution across processes
3. **Memory Bandwidth** — Measure with actual GB/s; bandwidth-limited operations cannot exceed hardware limits
4. **Cache Efficiency** — Check L1/L2/L3 cache misses with profiling tools (perf, papi)

## Input Matrices

This project uses sparse matrices for benchmarking. The real-world matrices could not be included in the repository due to their dimensions:

**Real-World Matrices (Not Included):**
- `bmwcra_1.mtx` — Large sparse matrix from real application
- `uni_chimera_i5_A_23.mtx` — Large sparse matrix from real application

These matrices are too large to upload to the repository but can be obtained from sparse matrix collections (e.g., SuiteSparse Matrix Collection).

**Synthetic Matrices:**
Synthetic test matrices are generated automatically during the scaling experiment runs using the matrix generator (`gen_random_mtx.c`). These are created on-the-fly by the PBS script and scaling bash scripts with controlled dimensions and sparsity patterns for reproducible benchmarking.

---

**Last Updated**: February 2026  
**Project Type**: High-Performance Computing (HPC)  
**Contact**: Your Name / Lab Group
