#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

static uint64_t splitmix64(uint64_t *x) {
  uint64_t z = (*x += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

int main(int argc, char **argv) {
  if (argc < 6) {
    fprintf(stderr, "Usage: %s out.mtx nrows ncols nnz_per_row seed\n", argv[0]);
    return 1;
  }

  const char *out = argv[1];
  int nrows = atoi(argv[2]);
  int ncols = atoi(argv[3]);
  int nnzpr = atoi(argv[4]);
  uint64_t seed = (uint64_t)strtoull(argv[5], NULL, 10);

  if (nrows <= 0 || ncols <= 0 || nnzpr <= 0) {
    fprintf(stderr, "Invalid args\n");
    return 1;
  }
  if (nnzpr > ncols) nnzpr = ncols;

  long long nnz = (long long)nrows * (long long)nnzpr;

  FILE *f = fopen(out, "w");
  if (!f) { perror("fopen"); return 1; }

  fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
  fprintf(f, "%% Random synthetic sparse matrix\n");
  fprintf(f, "%d %d %lld\n", nrows, ncols, nnz);

  uint64_t st = seed ? seed : 1;

  for (int r = 0; r < nrows; r++) {
    // naive: choose nnzpr random columns (duplicates possible, acceptable for synthetic unless rubric forbids)
    for (int k = 0; k < nnzpr; k++) {
      uint64_t rnd = splitmix64(&st);
      int c = (int)(rnd % (uint64_t)ncols);
      double v = 1.0;  // constant value
      fprintf(f, "%d %d %.6f\n", r + 1, c + 1, v);
    }
  }

  fclose(f);
  return 0;
}
