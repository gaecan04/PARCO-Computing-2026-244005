// spmv_mpi_openmp.c
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/*
  SPMV MPI+OpenMP with required distribution:
    owner(global_row) = global_row % P
    local_row_index   = global_row / P

  Metrics (rank0):
    - SpMV time per call (max over ranks)  => time_s
    - FLOPs, GFLOPs/s
    - dist_comm_s (read+pack+scatterv) and x_comm_s (bcast)
    - memory footprint estimate per rank (max)
    - NNZ balance min/avg/max
  Output:
    - human readable + optional one-line CSV (when --csv is used)

  Matrix Market support:
    - matrix coordinate real general
    - matrix coordinate real symmetric  (expanded by mirroring off-diagonals)
*/

static void *xmalloc(size_t n) {
  void *p = malloc(n);
  if (!p) { perror("malloc"); exit(1); }
  return p;
}

static void *xcalloc(size_t n, size_t s) {
  void *p = calloc(n, s);
  if (!p) { perror("calloc"); exit(1); }
  return p;
}

static void die_mpi(const char *msg, int code) {
  fprintf(stderr, "%s\n", msg);
  MPI_Abort(MPI_COMM_WORLD, code);
}

/* ----- MatrixMarket reader (rank0 only) ----- */
/* Supports: matrix coordinate real general OR real symmetric.
   For symmetric: expands off-diagonal entries (i,j) -> (i,j) and (j,i). */
static void read_mm_rank0(const char *fname, int *nrows, int *ncols, int *nnz,
                          int **rows, int **cols, double **vals) {
  FILE *f = fopen(fname, "r");
  if (!f) { perror("fopen"); exit(1); }

  char line[512];

  // Read banner line
  if (!fgets(line, sizeof(line), f)) {
    fprintf(stderr, "Error reading MatrixMarket banner\n");
    exit(1);
  }
  if (strncmp(line, "%%MatrixMarket", 14) != 0) {
    fprintf(stderr, "Not a MatrixMarket file (missing banner)\n");
    exit(1);
  }

  // Determine type
  int is_coordinate = (strstr(line, "coordinate") != NULL);
  int is_real       = (strstr(line, "real") != NULL);
  int is_general    = (strstr(line, "general") != NULL);
  int is_symmetric  = (strstr(line, "symmetric") != NULL);

  if (!is_coordinate || !is_real || !(is_general || is_symmetric)) {
    fprintf(stderr,
      "Unsupported MatrixMarket type.\n"
      "Supported: 'matrix coordinate real general' or 'matrix coordinate real symmetric'.\n"
      "Banner was: %s", line);
    exit(1);
  }

  // Skip comments
  do {
    if (!fgets(line, sizeof(line), f)) {
      fprintf(stderr, "Error reading size line\n");
      exit(1);
    }
  } while (line[0] == '%');

  int nnz_in = 0;
  if (sscanf(line, "%d %d %d", nrows, ncols, &nnz_in) != 3) {
    fprintf(stderr, "Error reading matrix size line\n");
    exit(1);
  }

  if (is_symmetric && *nrows != *ncols) {
    fprintf(stderr, "Invalid symmetric MatrixMarket: nrows != ncols (%d vs %d)\n", *nrows, *ncols);
    exit(1);
  }

  // Read input entries into temporary arrays (0-based)
  int *r_in = (int*)xmalloc((size_t)nnz_in * sizeof(int));
  int *c_in = (int*)xmalloc((size_t)nnz_in * sizeof(int));
  double *v_in = (double*)xmalloc((size_t)nnz_in * sizeof(double));

  for (int k = 0; k < nnz_in; k++) {
    int r, c;
    double v;
    if (fscanf(f, "%d %d %lf", &r, &c, &v) != 3) {
      fprintf(stderr, "Error reading entry %d\n", k);
      exit(1);
    }
    r_in[k] = r - 1;
    c_in[k] = c - 1;
    v_in[k] = v;
  }
  fclose(f);

  if (is_symmetric) {
    // Expand: duplicate off-diagonals
    int extra = 0;
    for (int k = 0; k < nnz_in; k++) {
      if (r_in[k] != c_in[k]) extra++;
    }
    *nnz = nnz_in + extra;

    *rows = (int*)xmalloc((size_t)(*nnz) * sizeof(int));
    *cols = (int*)xmalloc((size_t)(*nnz) * sizeof(int));
    *vals = (double*)xmalloc((size_t)(*nnz) * sizeof(double));

    int out = 0;
    for (int k = 0; k < nnz_in; k++) {
      int r = r_in[k], c = c_in[k];
      double v = v_in[k];

      (*rows)[out] = r;
      (*cols)[out] = c;
      (*vals)[out] = v;
      out++;

      if (r != c) {
        (*rows)[out] = c;
        (*cols)[out] = r;
        (*vals)[out] = v;
        out++;
      }
    }
  } else {
    *nnz = nnz_in;

    *rows = (int*)xmalloc((size_t)(*nnz) * sizeof(int));
    *cols = (int*)xmalloc((size_t)(*nnz) * sizeof(int));
    *vals = (double*)xmalloc((size_t)(*nnz) * sizeof(double));

    for (int k = 0; k < nnz_in; k++) {
      (*rows)[k] = r_in[k];
      (*cols)[k] = c_in[k];
      (*vals)[k] = v_in[k];
    }
  }

  free(r_in);
  free(c_in);
  free(v_in);
}

/* local rows count for cyclic distribution */
static inline int local_nrows_cyclic(int nrows, int rank, int P) {
  if (rank >= nrows) return 0;
  return (nrows - 1 - rank) / P + 1;
}

/* build CSR from local COO (with local row index already) */
static void coo_to_csr(int local_nrows, int local_nnz,
                       const int *coo_r, const int *coo_c, const double *coo_v,
                       int **rowptr_out, int **colind_out, double **vals_out) {
  int *rowptr = (int*)xcalloc((size_t)local_nrows + 1, sizeof(int));
  int *colind = (int*)xmalloc((size_t)local_nnz * sizeof(int));
  double *vals = (double*)xmalloc((size_t)local_nnz * sizeof(double));

  // count
  for (int k = 0; k < local_nnz; k++) {
    int r = coo_r[k];
    if (r < 0 || r >= local_nrows) {
      fprintf(stderr, "Invalid local row index in COO: r=%d local_nrows=%d\n", r, local_nrows);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    rowptr[r + 1]++;
  }

  // prefix sum
  for (int i = 0; i < local_nrows; i++) rowptr[i + 1] += rowptr[i];

  // fill using offset
  int *ofs = (int*)xcalloc((size_t)local_nrows, sizeof(int));
  for (int k = 0; k < local_nnz; k++) {
    int r = coo_r[k];
    if (r < 0 || r >= local_nrows) {
      fprintf(stderr, "Invalid local row index in COO fill: r=%d local_nrows=%d\n", r, local_nrows);
      MPI_Abort(MPI_COMM_WORLD, 2);
    }
    int pos = rowptr[r] + ofs[r]++;
    colind[pos] = coo_c[k];
    vals[pos]   = coo_v[k];
  }
  free(ofs);

  *rowptr_out = rowptr;
  *colind_out = colind;
  *vals_out   = vals;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, P;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &P);

  if (argc < 2) {
    if (rank == 0) fprintf(stderr, "Usage: %s matrix.mtx [--csv]\n", argv[0]);
    MPI_Finalize();
    return 1;
  }

  const char *matrix = argv[1];
  int csv_mode = 0;
  for (int i = 2; i < argc; i++) {
    if (strcmp(argv[i], "--csv") == 0) csv_mode = 1;
  }

  int nrows=0, ncols=0, nnz=0;
  int *rows_g=NULL, *cols_g=NULL;
  double *vals_g=NULL;

  // ----- Read + distribute (timed as "dist_comm") -----
  MPI_Barrier(MPI_COMM_WORLD);
  double t_dist0 = MPI_Wtime();

  if (rank == 0) {
    read_mm_rank0(matrix, &nrows, &ncols, &nnz, &rows_g, &cols_g, &vals_g);
  }

  MPI_Bcast(&nrows, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&ncols, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&nnz,   1, MPI_INT, 0, MPI_COMM_WORLD);

  int *sendcounts = NULL, *displs = NULL;
  int local_nnz = 0;

  if (rank == 0) {
    sendcounts = (int*)xcalloc((size_t)P, sizeof(int));
    displs     = (int*)xcalloc((size_t)P, sizeof(int));
    for (int k = 0; k < nnz; k++) {
      int owner = rows_g[k] % P;
      if (owner < 0) owner += P;
      sendcounts[owner]++;
    }
    displs[0] = 0;
    for (int p = 1; p < P; p++) displs[p] = displs[p-1] + sendcounts[p-1];
  }

  MPI_Scatter(sendcounts, 1, MPI_INT, &local_nnz, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int   *lrows = (int*)xmalloc((size_t)local_nnz * sizeof(int));
  int   *lcols = (int*)xmalloc((size_t)local_nnz * sizeof(int));
  double*lvals = (double*)xmalloc((size_t)local_nnz * sizeof(double));

  // rank0 packs into contiguous arrays by owner
  int *pack_r=NULL, *pack_c=NULL;
  double *pack_v=NULL;

  if (rank == 0) {
    pack_r = (int*)xmalloc((size_t)nnz * sizeof(int));
    pack_c = (int*)xmalloc((size_t)nnz * sizeof(int));
    pack_v = (double*)xmalloc((size_t)nnz * sizeof(double));

    int *cursor = (int*)xmalloc((size_t)P * sizeof(int));
    for (int p = 0; p < P; p++) cursor[p] = displs[p];

    for (int k = 0; k < nnz; k++) {
      int owner = rows_g[k] % P;
      if (owner < 0) owner += P;
      int pos = cursor[owner]++;
      pack_r[pos] = rows_g[k];
      pack_c[pos] = cols_g[k];
      pack_v[pos] = vals_g[k];
    }
    free(cursor);
  }

  MPI_Scatterv(pack_r, sendcounts, displs, MPI_INT,    lrows, local_nnz, MPI_INT,    0, MPI_COMM_WORLD);
  MPI_Scatterv(pack_c, sendcounts, displs, MPI_INT,    lcols, local_nnz, MPI_INT,    0, MPI_COMM_WORLD);
  MPI_Scatterv(pack_v, sendcounts, displs, MPI_DOUBLE, lvals, local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    free(rows_g); free(cols_g); free(vals_g);
    free(pack_r); free(pack_c); free(pack_v);
    free(sendcounts); free(displs);
  }

  // convert global row -> local row index = global_row / P
  for (int k = 0; k < local_nnz; k++) {
    lrows[k] = lrows[k] / P;
  }

  double t_dist1 = MPI_Wtime();
  double dist_comm = t_dist1 - t_dist0;

  // ----- Build CSR -----
  int local_nrows = local_nrows_cyclic(nrows, rank, P);
  int *rowptr=NULL, *colind=NULL;
  double *vals=NULL;
  coo_to_csr(local_nrows, local_nnz, lrows, lcols, lvals, &rowptr, &colind, &vals);
  free(lrows); free(lcols); free(lvals);

  // ----- Vectors -----
  double *x = (double*)xmalloc((size_t)ncols * sizeof(double));
  double *y = (double*)xcalloc((size_t)local_nrows, sizeof(double));

  // init x on rank0 then broadcast (timed as x_comm)
  MPI_Barrier(MPI_COMM_WORLD);
  double t_x0 = MPI_Wtime();
  if (rank == 0) {
    for (int i = 0; i < ncols; i++) x[i] = 1.0;
  }
  MPI_Bcast(x, ncols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  double t_x1 = MPI_Wtime();
  double x_comm = t_x1 - t_x0;

  // ----- Compute SpMV (timed as time_s) -----
  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  #pragma omp parallel for schedule(static)
  for (int i = 0; i < local_nrows; i++) {
    double sum = 0.0;
    for (int j = rowptr[i]; j < rowptr[i+1]; j++) {
      sum += vals[j] * x[colind[j]];
    }
    y[i] = sum;
  }

  double t1 = MPI_Wtime();
  double comp_local = t1 - t0;

  // global time (max)
  double spmv_time = 0.0;
  MPI_Reduce(&comp_local, &spmv_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // comm times (max) for consistency
  double dist_comm_max=0.0, x_comm_max=0.0;
  MPI_Reduce(&dist_comm, &dist_comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&x_comm,    &x_comm_max,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  // load balance nnz stats
  int nnz_min=0, nnz_max=0;
  long long nnz_sum=0;
  MPI_Reduce(&local_nnz, &nnz_min, 1, MPI_INT, MPI_MIN, 0, MPI_COMM_WORLD);
  MPI_Reduce(&local_nnz, &nnz_max, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
  long long lnnz = (long long)local_nnz;
  MPI_Reduce(&lnnz, &nnz_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

  // FLOPs (SpMV ~ 2*nnz)
  double flops = 2.0 * (double)nnz;
  double gflops = 0.0;
  if (rank == 0 && spmv_time > 0.0) gflops = flops / (spmv_time * 1e9);

  // memory estimate per rank (bytes) => reduce max
  size_t mem_bytes =
      (size_t)(local_nrows + 1) * sizeof(int) +
      (size_t)local_nnz * sizeof(int) +
      (size_t)local_nnz * sizeof(double) +
      (size_t)local_nrows * sizeof(double) +
      (size_t)ncols * sizeof(double);
  double mem_mb = (double)mem_bytes / (1024.0 * 1024.0);
  double mem_mb_max=0.0;
  MPI_Reduce(&mem_mb, &mem_mb_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  int threads = omp_get_max_threads();

  if (rank == 0) {
    double nnz_avg = (P > 0) ? ((double)nnz_sum / (double)P) : 0.0;

    if (csv_mode) {
      // CSV line:
      // matrix,nrows,ncols,nnz,np,threads,time_s,dist_comm_s,x_comm_s,gflops,nnz_min,nnz_avg,nnz_max,mem_mb_max
      printf("%s,%d,%d,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%d,%.2f,%d,%.2f\n",
        matrix, nrows, ncols, nnz, P, threads,
        spmv_time, dist_comm_max, x_comm_max, gflops,
        nnz_min, nnz_avg, nnz_max, mem_mb_max
      );
    } else {
      printf("Matrix: %s\n", matrix);
      printf("Dims: %d x %d   NNZ: %d\n", nrows, ncols, nnz);
      printf("MPI ranks: %d   OMP threads: %d\n", P, threads);
      printf("SpMV time (max): %.6f s\n", spmv_time);
      printf("Comm (dist max): %.6f s\n", dist_comm_max);
      printf("Comm (x-bcast):  %.6f s\n", x_comm_max);
      printf("FLOPs: %.0f   GFLOP/s: %.6f\n", flops, gflops);
      printf("NNZ per rank min/avg/max: %d / %.2f / %d\n", nnz_min, nnz_avg, nnz_max);
      printf("Mem per rank (max est): %.2f MB\n", mem_mb_max);
    }
  }

  free(rowptr); free(colind); free(vals);
  free(x); free(y);

  MPI_Finalize();
  return 0;
}
