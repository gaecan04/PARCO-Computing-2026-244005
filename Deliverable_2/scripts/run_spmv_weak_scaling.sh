#!/usr/bin/env bash
set -euo pipefail

EXEC=${EXEC:-./spmv_mpi_openmp}
GEN=${GEN:-./spmv_gen_random_mm}
OUTCSV=${OUTCSV:-weak_results.csv}

BASE_N=${BASE_N:-5000}
NNZPR=${NNZPR:-16}
SEED=${SEED:-12345}

THREADS_LIST=(${THREADS_LIST:-1})
NP_LIST=(${NP_LIST:-1 2 4 8 16 32 64 128})

SLOTS=${TOTAL_CORES:-128}
echo "Detected slots (from TOTAL_CORES): $SLOTS"

echo "mode,matrix,nrows,ncols,nnz,np,threads,time_s,dist_comm_s,x_comm_s,gflops,nnz_min,nnz_avg,nnz_max,mem_mb_max,weak_speedup,weak_efficiency" > "$OUTCSV"

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

    n=$(( BASE_N * np ))
    mat="synthetic_n${n}_np${np}_nnzpr${NNZPR}.mtx"

    if [[ ! -f "$mat" ]]; then
      "$GEN" "$mat" "$n" "$n" "$NNZPR" "$SEED"
    fi

    line=$(mpiexec -n "$np" "$EXEC" "$mat" --csv | tail -n 1)
    time_s=$(echo "$line" | awk -F, '{print $7}')

    if [[ -z "$BASE_T" && "$np" -eq 1 ]]; then
      BASE_T="$time_s"
    fi

    weak_speedup="NA"
    weak_eff="NA"
    if [[ -n "$BASE_T" ]]; then
      weak_speedup=$(awk -v b="$BASE_T" -v t="$time_s" 'BEGIN{ if(t>0) printf "%.6f", b/t; else print "NA"}')
      weak_eff="$weak_speedup"
    fi

    echo "weak,$line,$weak_speedup,$weak_eff" >> "$OUTCSV"
    echo "OK weak np=$np th=$th time=$time_s weakE=$weak_eff"
  done
done

echo "Wrote: $OUTCSV"
