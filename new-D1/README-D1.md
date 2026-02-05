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

## Cluster compilation changes
Here I will show an example of .pbs file used to run the code on the cluster and the bash script with the changes needed to run on the cluster to avoid problems such as the absence of gnuplot, or the difference in the path refernces for the files used. These configuration may still present some issues with some configurations of *MVM_parallel_sellc* because of the amount of memory the codes uses to run

*Bash script*
```bash
#!/bin/bash
# ============================================================
# Unified MVM experiment driver
# Sequential, OpenMP, and SELL-C (fully standalone)
# Outputs: RESULTS.txt, speedup_data.dat, speedup_plot.png
# Logs per-run data under /tmp/mvm_runs
# ============================================================

# ----------------------------
# CONFIGURATION
# ----------------------------
CLI_DIR="/home/gaetano.cannone/cli"
sequential_programs=( "$CLI_DIR/MVM_sequential" )
parallel_programs=( "$CLI_DIR/MVM_parallel" 
                    "$CLI_DIR/MVM_parallel_atomic"
                    "$CLI_DIR/MVM_parallel_sellc" )

RESULTS_FILE="./RESULTS.txt"
RUNS_DIR="/tmp/mvm_runs"
mkdir -p "$RUNS_DIR"
> "$RESULTS_FILE"

# Absolute paths to matrices
matrices=( "$CLI_DIR/1138_bus.txt"
           "$CLI_DIR/adder_dcop_32.txt"
           "$CLI_DIR/bcsstk14.txt"
           "$CLI_DIR/rail_5177.txt"
           "$CLI_DIR/chimera_matrix.txt" )

schedules=("static" "dynamic" "guided")
threads_list=(2 4 8 16 32)
chunks=(1 2 4 8 16 32)
sigmas=(64 128 256 512)  # sigma values for SELL-C
runs=12

# ----------------------------
# 90th percentile helper
# ----------------------------
percentile_90() {
    awk 'BEGIN{c=0} {a[c++]=$1} END{if(c==0){print "N/A"; exit} asort(a); idx=int(0.9*(c-1)); if(idx<0) idx=0; print a[idx]}' "$1"
}

run_counter=0
declare -A best_seq
declare -A best_par
declare -A best_par_config

# ----------------------------
# Helper: parse "Run N: X ms" lines
# ----------------------------
extract_times_from_stdout() {
    local stdout_file="$1"
    local times_file="$2"
    grep -Eo "Run[[:space:]]+[0-9]+:[[:space:]]+[0-9.]+ ms" "$stdout_file" \
        | awk '{print $3}' | sed 's/ms//g' > "$times_file"
}

# ========================================
# Experiment function
# ========================================
run_experiments() {
    local C_PROGRAM="$1"
    local MODE="$2"

    echo "" >> "$RESULTS_FILE"
    echo "#################################################################" >> "$RESULTS_FILE"
    echo "Running experiments for: $C_PROGRAM  ($MODE mode)" >> "$RESULTS_FILE"
    echo "#################################################################" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"

    for matrix in "${matrices[@]}"; do
        matrix_name=$(basename "$matrix" .txt)
        echo "========================================" >> "$RESULTS_FILE"
        echo "Matrix: $matrix" >> "$RESULTS_FILE"
        echo "========================================" >> "$RESULTS_FILE"

        if [[ "$MODE" == "sequential" ]]; then
            run_id=$(printf "%05d" $run_counter)
            current_run_dir="$RUNS_DIR/run_${run_id}_${matrix_name}_seq"
            mkdir -p "$current_run_dir"
            ((run_counter++))

            stdout_file="$current_run_dir/stdout.txt"
            output=$("$C_PROGRAM" "$matrix" -r "$runs" 2>&1)
            echo "$output" > "$stdout_file"

            extract_times_from_stdout "$stdout_file" "$current_run_dir/times.txt"
            p90=$(percentile_90 "$current_run_dir/times.txt")
            printf "Sequential | 90th percentile: %s ms\n" "$p90" >> "$RESULTS_FILE"
            [[ "$p90" != "N/A" && "$p90" != "" ]] && best_seq[$matrix_name]="$p90"

        else
            # Parallel programs
            if [[ "$C_PROGRAM" == *"sellc"* ]]; then
                # SELL-C: iterate over threads, chunks, sigma
                for th in "${threads_list[@]}"; do
                    for ch in "${chunks[@]}"; do
                        for sig in "${sigmas[@]}"; do
                            run_id=$(printf "%05d" $run_counter)
                            current_run_dir="$RUNS_DIR/run_${run_id}_${matrix_name}_sellc_t${th}_c${ch}_s${sig}"
                            mkdir -p "$current_run_dir"
                            ((run_counter++))

                            stdout_file="$current_run_dir/stdout.txt"

                            "$C_PROGRAM" "$matrix" -r "$runs" -c "$ch" -s "$sig" -t "$th" > "$stdout_file" 2>&1
                            status=$?

                            if [[ $status -ne 0 ]]; then
                                printf "SELL-C | Threads: %-2d | Chunk: %-3d | Sigma: %-3d | ERROR: exit %d\n" \
                                    "$th" "$ch" "$sig" "$status" >> "$RESULTS_FILE"
                                printf "Raw output saved: %s\n" "$stdout_file" >> "$RESULTS_FILE"
                                continue
                            fi

                            extract_times_from_stdout "$stdout_file" "$current_run_dir/times.txt"
                            p90=$(percentile_90 "$current_run_dir/times.txt")

                            printf "SELL-C | Threads: %-2d | Chunk: %-3d | Sigma: %-3d | 90th percentile: %s ms\n" \
                                "$th" "$ch" "$sig" "$p90" >> "$RESULTS_FILE"

                            # Store best timing for SELL-C
                            if [[ "$p90" != "N/A" && "$p90" != "" ]]; then
                                key="${matrix_name}:$(basename $C_PROGRAM)"
                                if [[ -z "${best_par[$key]}" || 1 -eq "$(echo "$p90 < ${best_par[$key]}" | bc)" ]]; then
                                    best_par[$key]="$p90"
                                    best_par_config[$key]="${th},${ch},${sig}"
                                fi
                            fi
                        done
                    done
                done
            else
                # Regular OpenMP parallel programs
                for sched in "${schedules[@]}"; do
                    for th in "${threads_list[@]}"; do
                        for ch in "${chunks[@]}"; do
                            run_id=$(printf "%05d" $run_counter)
                            current_run_dir="$RUNS_DIR/run_${run_id}_${matrix_name}_${sched}_t${th}_c${ch}"
                            mkdir -p "$current_run_dir"
                            ((run_counter++))

                            stdout_file="$current_run_dir/stdout.txt"
                            output=$("$C_PROGRAM" "$matrix" -r "$runs" -t "$th" -s "$sched" -c "$ch" 2>&1)
                            echo "$output" > "$stdout_file"

                            extract_times_from_stdout "$stdout_file" "$current_run_dir/times.txt"
                            p90=$(percentile_90 "$current_run_dir/times.txt")

                            printf "Schedule: %-7s | Threads: %-2d | Chunk: %-3d | 90th percentile: %s ms\n" \
                                "$sched" "$th" "$ch" "$p90" >> "$RESULTS_FILE"

                            # Store best timing for this program/matrix
                            if [[ "$p90" != "N/A" && "$p90" != "" ]]; then
                                key="${matrix_name}:$(basename $C_PROGRAM)"
                                if [[ -z "${best_par[$key]}" || 1 -eq "$(echo "$p90 < ${best_par[$key]}" | bc)" ]]; then
                                    best_par[$key]="$p90"
                                    best_par_config[$key]="${sched},${th},${ch}"
                                fi
                            fi
                        done
                    done
                done
            fi
        fi
        echo "" >> "$RESULTS_FILE"
    done
}

# ========================================
# Run Programs
# ========================================
for prog in "${sequential_programs[@]}"; do
    [[ -x "$prog" ]] && run_experiments "$prog" "sequential"
done

for prog in "${parallel_programs[@]}"; do
    [[ -x "$prog" ]] && run_experiments "$prog" "parallel"
done

# ========================================
# SPEEDUP SUMMARY
# ========================================
echo "" >> "$RESULTS_FILE"
echo "#################################################################" >> "$RESULTS_FILE"
echo "                     SPEEDUP SUMMARY                            " >> "$RESULTS_FILE"
echo "#################################################################" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "Matrix" "Program" "PAR(ms)" "SEQ(ms)" "Speedup" >> "$RESULTS_FILE"
SPEEDUP_DAT="speedup_data.dat"
> "$SPEEDUP_DAT"
echo "#Matrix Sequential MVM_parallel MVM_parallel_atomic MVM_parallel_sellc" >> "$SPEEDUP_DAT"

for matrix in "${matrices[@]}"; do
    m=$(basename "$matrix" .txt)
    seq="${best_seq[$m]}"
    [[ -z "$seq" ]] && continue

    par_vals=()
    par_names=("MVM_parallel" "MVM_parallel_atomic" "MVM_parallel_sellc")

    for prog_name in "${par_names[@]}"; do
        key="${m}:${prog_name}"
        par="${best_par[$key]}"
        [[ -z "$par" ]] && par="0"
        par_vals+=("$par")
    done

    printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "$m" "Sequential" "-" "$seq" "1.0" >> "$RESULTS_FILE"

    for i in ${!par_names[@]}; do
        par_val="${par_vals[$i]}"
        par_val_float=$(echo "$par_val" | awk '{printf "%.6f", $0}')
        speedup=$(echo "scale=6; $seq / $par_val" | bc 2>/dev/null)
        speedup=$(printf "%.6f" "$speedup")
        [[ "$par_val" == "0" ]] && speedup="0"
        printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "$m" "${par_names[$i]}" "$par_val_float" "$seq" "$speedup" >> "$RESULTS_FILE"
    done

    echo "$m 1 ${speedup[@]}" >> "$SPEEDUP_DAT"
done

# ========================================
# PLOT (optional)
# ========================================
rm -f speedup_plot.png
cat << 'EOF' > plot_speedup.gnuplot
set terminal pngcairo size 1400,800 noenhanced font "Arial,12"
set output "speedup_plot.png"
set title "Speedup Comparison (Sequential = 1)"
set style data histograms
set style histogram cluster gap 2
set style fill solid border -1
set boxwidth 0.9
set grid ytics
set ylabel "Speedup"
set yrange [0:*]
plot "speedup_data.dat" using 2:xtic(1) title "Sequential" linecolor rgb "#4e79a7", \
     "" using 3 title "MVM_parallel" linecolor rgb "#f28e2b", \
     "" using 4 title "MVM_parallel_atomic" linecolor rgb "#e15759", \
     "" using 5 title "MVM_parallel_sellc" linecolor rgb "#76b7b2"
EOF
# Uncomment the next line if gnuplot is available
# gnuplot plot_speedup.gnuplot

echo "" >> "$RESULTS_FILE"
echo "Run data saved in: $RUNS_DIR" >> "$RESULTS_FILE"
echo "Plot generated: speedup_plot.png" >> "$RESULTS_FILE"
echo "All experiments finished. Results saved in $RESULTS_FILE"
cat "$RESULTS_FILE"

```
*.pbs file*
```bash
#!/bin/bash
# ========================================
# PBS script to run MVM experiments via bash
# Optimized for OpenMP parallel execution
# ========================================

# Job name
#PBS -N MVM_experiment

# Output and error files
#PBS -o ./MVM_experiment.out
#PBS -e ./MVM_experiment.err

# Queue name (replace with a valid queue for your cluster)
#PBS -q short_cpuQ

# Maximum walltime
#PBS -l walltime=00:10:00

# Resources: 1 node, 16 CPUs, 1 GB memory
#PBS -l select=1:ncpus=16:mem=1gb

# Export environment variables for OpenMP
export OMP_NUM_THREADS=16


# Load required modules
module load gcc91
# If gnuplot becomes available, you can uncomment:
# module load gnuplot

# Go to the directory from which the job is submitted
cd "$PBS_O_WORKDIR"

# Run the bash script
bash ~/cli/run_experiments_90p.sh

```


