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
sequential_programs=( "./MVM_sequential" )
parallel_programs=( "./MVM_parallel" 
                    "./MVM_parallel_atomic"
                    "./MVM_parallel_sellc" )

RESULTS_FILE="./RESULTS.txt"
RUNS_DIR="/tmp/mvm_runs"
mkdir -p "$RUNS_DIR"
> "$RESULTS_FILE"

matrices=( "./1138_bus.txt" 
           "./adder_dcop_32.txt"
           "./bcsstk14.txt"
           "./rail_5177.txt"
           "./chimera_matrix.txt")
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

                            if [[ "$p90" != "N/A" && "$p90" != "" ]]; then
                                key="${matrix_name}:${C_PROGRAM}"
                                if [[ -z "${best_par[$key]}" || 1 -eq "$(echo "$p90 < ${best_par[$key]}" | bc)" ]]; then
                                    best_par[$key]="$p90"
                                    best_par_config[$key]="${th},${ch},${sig}"
                                fi
                            fi
                        done
                    done
                done
            else
                # Regular parallel OpenMP codes
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

                            if [[ "$p90" != "N/A" && "$p90" != "" ]]; then
                                key="${matrix_name}:${C_PROGRAM}"
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

    par1="-"; par1_s="0"; key1="${m}:./MVM_parallel"
    [[ -n "${best_par[$key1]}" ]] && par1="${best_par[$key1]}" && par1_s=$(echo "scale=4; $seq / $par1" | bc)

    par2="-"; par2_s="0"; key2="${m}:./MVM_parallel_atomic"
    [[ -n "${best_par[$key2]}" ]] && par2="${best_par[$key2]}" && par2_s=$(echo "scale=4; $seq / $par2" | bc)

    par3="-"; par3_s="0"; key3="${m}:./MVM_parallel_sellc"
    [[ -n "${best_par[$key3]}" ]] && par3="${best_par[$key3]}" && par3_s=$(echo "scale=4; $seq / $par3" | bc)

    printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "$m" "Sequential" "-" "$seq" "1.0" >> "$RESULTS_FILE"
    printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "$m" "MVM_parallel" "$par1" "$seq" "$par1_s" >> "$RESULTS_FILE"
    printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "$m" "MVM_parallel_atomic" "$par2" "$seq" "$par2_s" >> "$RESULTS_FILE"
    printf "%-20s | %-18s | %-10s | %-10s | %-10s\n" "$m" "MVM_parallel_sellc" "$par3" "$seq" "$par3_s" >> "$RESULTS_FILE"

    echo "$m 1 $par1_s $par2_s $par3_s" >> "$SPEEDUP_DAT"
done

# ========================================
# PLOT
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
gnuplot plot_speedup.gnuplot

echo "" >> "$RESULTS_FILE"
echo "Run data saved in: $RUNS_DIR" >> "$RESULTS_FILE"
echo "Plot generated: speedup_plot.png" >> "$RESULTS_FILE"
echo "All experiments finished. Results saved in $RESULTS_FILE"
cat "$RESULTS_FILE"
