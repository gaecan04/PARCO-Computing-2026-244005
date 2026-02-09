#!/usr/bin/env bash
set -euo pipefail

EXEC=${EXEC:-./spmv_mpi_openmp}
OUTCSV=${OUTCSV:-strong_results.csv}
MATRIX=${MATRIX:-./uni_chimera_i5_A_23.mtx}

THREADS_LIST=(${THREADS_LIST:-1})
NP_LIST=(${NP_LIST:-1 2 4 8 16 32 64 128})

# On this PBS setup, PBS_NODEFILE counts nodes (not cores). Use TOTAL_CORES from PBS script.
SLOTS=${TOTAL_CORES:-128}

echo "Detected slots (from TOTAL_CORES): $SLOTS"

echo "mode,matrix,nrows,ncols,nnz,np,threads,time_s,dist_comm_s,x_comm_s,gflops,nnz_min,nnz_avg,nnz_max,mem_mb_max,speedup,efficiency" > "$OUTCSV"

for th in "${THREADS_LIST[@]}"; do
  export OMP_NUM_THREADS="$th"
  export OMP_PROC_BIND=true
  export OMP_PLACES=cores

  BASE_T=""

  for np in "${NP_LIST[@]}"; do
    if (( np > SLOTS )); then
      echo "SKIP np=$np (requested > allocated slots=$SLOTS)" >&2
      continue
    fi

    line=$(mpiexec -n "$np" "$EXEC" "$MATRIX" --csv | tail -n 1)
    time_s=$(echo "$line" | awk -F, '{print $7}')

    if [[ -z "$BASE_T" && "$np" -eq 1 ]]; then
      BASE_T="$time_s"
    fi

    speedup="NA"
    eff="NA"
    if [[ -n "$BASE_T" ]]; then
      speedup=$(awk -v b="$BASE_T" -v t="$time_s" 'BEGIN{ if(t>0) printf "%.6f", b/t; else print "NA"}')
      eff=$(awk -v s="$speedup" -v np="$np" 'BEGIN{ if(np>0 && s!="NA") printf "%.6f", s/np; else print "NA"}')
    fi

    echo "strong,$line,$speedup,$eff" >> "$OUTCSV"
    echo "OK strong np=$np th=$th time=$time_s speedup=$speedup eff=$eff"
  done
done

echo "Wrote: $OUTCSV"
