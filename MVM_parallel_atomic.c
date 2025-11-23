// Build with -fopenmp (gcc) or /openmp (MSVC)
// METHOD 2: Atomic operations - Parallelize over non-zero elements instead of rows

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <omp.h>

typedef struct {
    int row;
    int col;
    double val;
} Triplet;

int cmpTriplet(const void *a, const void *b) {
    const Triplet *ta = (const Triplet *)a;
    const Triplet *tb = (const Triplet *)b;
    if (ta->row != tb->row) return ta->row - tb->row;
    return ta->col - tb->col;
}

void convertToCSR(Triplet *triplets, int nnz, int rows, int cols,
                  double **values, int **colIndex, int **rowPtr) {
    *values = (double *)malloc(nnz * sizeof(double));
    *colIndex = (int *)malloc(nnz * sizeof(int));
    *rowPtr = (int *)calloc((rows + 1), sizeof(int));

    if (!(*values) || !(*colIndex) || !(*rowPtr)) {
        printf("Error: memory allocation failed in CSR conversion.\n");
        fflush(stdout);
        exit(1);
    }

    for (int i = 0; i < nnz; i++) {
        (*rowPtr)[triplets[i].row + 1]++;
    }
    for (int i = 0; i < rows; i++) {
        (*rowPtr)[i + 1] += (*rowPtr)[i];
    }

    int *writePtr = (int *)malloc((rows + 1) * sizeof(int));
    if (!writePtr) {
        printf("Error: memory allocation failed in CSR conversion (writePtr).\n");
        fflush(stdout);
        exit(1);
    }
    for (int i = 0; i <= rows; i++) writePtr[i] = (*rowPtr)[i];

    for (int i = 0; i < nnz; i++) {
        int row = triplets[i].row;
        int dest = writePtr[row];
        (*values)[dest] = triplets[i].val;
        (*colIndex)[dest] = triplets[i].col;
        writePtr[row]++;
    }

    free(writePtr);
}

// Helper function: find which row a given non-zero element belongs to (binary search)
int findRow(int k, int *rowPtr, int rows) {
    int low = 0, high = rows - 1;
    while (low <= high) {
        int mid = (low + high) / 2;
        if (rowPtr[mid] <= k && k < rowPtr[mid + 1]) {
            return mid;
        } else if (k < rowPtr[mid]) {
            high = mid - 1;
        } else {
            low = mid + 1;
        }
    }
    return 0;
}

// ---------- METHOD 2: Atomic operations ----------
// Parallelize over non-zero elements instead of rows
// Each thread processes one element and atomically updates y[row]
// Better load balancing for very sparse rows, but has synchronization overhead
void csrMatVecMultiply(int rows, double *values, int *colIndex, int *rowPtr,
                       double *x, double *y) {
    // Initialize y
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < rows; i++) {
        y[i] = 0.0;
    }

    // Parallelize over all non-zero elements
    int total_nnz = rowPtr[rows];
    #pragma omp parallel for schedule(runtime)
    for (int k = 0; k < total_nnz; k++) {
        // Find which row this element belongs to using binary search
        int row = findRow(k, rowPtr, rows);
        
        // Compute product
        double product = values[k] * x[colIndex[k]];
        
        // Atomically update y[row] to avoid race conditions
        #pragma omp atomic
        y[row] += product;
    }
}

double getMilliseconds() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

void printUsage(const char *prog) {
    printf("Usage: %s <matrix_file> [-r runs] [-t threads] [-s schedule] [-c chunk]\n", prog);
    printf("  -r runs      : number of runs (default 10)\n");
    printf("  -t threads   : number of OpenMP threads (default = hardware)\n");
    printf("  -s schedule  : schedule: static | dynamic | guided | auto (default guided)\n");
    printf("  -c chunk     : chunk size for schedule (integer, default 0)\n");
}

int parseSchedule(const char *s, omp_sched_t *outKind) {
    if (!s) return 0;
    if (strcmp(s, "static") == 0) { *outKind = omp_sched_static; return 1; }
    if (strcmp(s, "dynamic") == 0) { *outKind = omp_sched_dynamic; return 1; }
    if (strcmp(s, "guided") == 0) { *outKind = omp_sched_guided; return 1; }
    if (strcmp(s, "auto") == 0)   { *outKind = omp_sched_auto;   return 1; }
    return 0;
}

int main(int argc, char *argv[]) {
    printf("=== Sparse Matrix Program (METHOD 2: Atomic Operations) ===\n");
    fflush(stdout);

    if (argc < 2) {
        printUsage(argv[0]);
        return 1;
    }

    char *filename = argv[1];
    int runs = 10;
    int threads = 0;
    const char *schedStr = "guided";
    int chunk = 0;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            runs = atoi(argv[++i]);
            if (runs <= 0) runs = 10;
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threads = atoi(argv[++i]);
            if (threads < 0) threads = 0;
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            schedStr = argv[++i];
        } else if (strcmp(argv[i], "-c") == 0 && i + 1 < argc) {
            chunk = atoi(argv[++i]);
            if (chunk < 0) chunk = 0;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        }
    }

    printf("Attempting to open file: %s\n", filename);
    fflush(stdout);

    FILE *fin = fopen(filename, "r");
    if (!fin) {
        printf("Error: cannot open file '%s'\n", filename);
        fflush(stdout);
        return 1;
    }

    printf("File opened successfully!\n");
    fflush(stdout);

    int comment_count = 0;
    int ch;
    while ((ch = fgetc(fin)) != EOF) {
        if (ch == '%') {
            while ((ch = fgetc(fin)) != EOF && ch != '\n');
            comment_count++;
        } else {
            ungetc(ch, fin);
            break;
        }
    }

    printf("Skipped %d comment line(s).\n", comment_count);
    fflush(stdout);

    if (ch == EOF) {
        printf("Error: file contains only comments or is empty.\n");
        fflush(stdout);
        fclose(fin);
        return 1;
    }

    printf("Reading matrix header...\n");
    fflush(stdout);

    int rows, cols, nnz;
    if (fscanf(fin, "%d %d %d", &rows, &cols, &nnz) != 3) {
        printf("Error: invalid matrix header.\n");
        fflush(stdout);
        fclose(fin);
        return 1;
    }

    printf("Matrix dimensions: %d x %d with %d non-zero elements\n", rows, cols, nnz);
    fflush(stdout);

    if (rows <= 0 || cols <= 0 || nnz <= 0) {
        printf("Error: invalid matrix dimensions (must be positive).\n");
        fflush(stdout);
        fclose(fin);
        return 1;
    }

    Triplet *triplets = (Triplet *)malloc(nnz * sizeof(Triplet));
    if (!triplets) {
        printf("Error: memory allocation failed for triplets.\n");
        fflush(stdout);
        fclose(fin);
        return 1;
    }

    printf("Reading matrix elements...\n");
    fflush(stdout);

    int maxRow = 0, maxCol = 0;

    for (int i = 0; i < nnz; i++) {
        if (fscanf(fin, "%d %d %lf", &triplets[i].row, &triplets[i].col, &triplets[i].val) != 3) {
            printf("Error: invalid matrix element at entry %d.\n", i + 1);
            fflush(stdout);
            fclose(fin);
            free(triplets);
            return 1;
        }
        if (triplets[i].row > maxRow) maxRow = triplets[i].row;
        if (triplets[i].col > maxCol) maxCol = triplets[i].col;
    }

    if (maxRow == rows || maxCol == cols) {
        printf("Detected 1-based indexing, converting to 0-based...\n");
        fflush(stdout);
        for (int i = 0; i < nnz; i++) {
            triplets[i].row--;
            triplets[i].col--;
        }
    }

    for (int i = 0; i < nnz; i++) {
        if (triplets[i].row < 0 || triplets[i].row >= rows ||
            triplets[i].col < 0 || triplets[i].col >= cols) {
            printf("Error: invalid indices at entry %d (row=%d, col=%d)\n",
                   i + 1, triplets[i].row, triplets[i].col);
            fflush(stdout);
            fclose(fin);
            free(triplets);
            return 1;
        }
    }
    fclose(fin);
    printf("Matrix data loaded successfully!\n");
    fflush(stdout);

    printf("Sorting triplets using qsort...\n");
    fflush(stdout);
    qsort(triplets, nnz, sizeof(Triplet), cmpTriplet);

    printf("Converting to CSR format...\n");
    fflush(stdout);
    double *values;
    int *colIndex, *rowPtr;
    convertToCSR(triplets, nnz, rows, cols, &values, &colIndex, &rowPtr);

    printf("Allocating vectors...\n");
    fflush(stdout);
    double *x = (double *)malloc(cols * sizeof(double));
    double *y = (double *)malloc(rows * sizeof(double));
    double *times = (double *)malloc(runs * sizeof(double));

    if (!x || !y || !times) {
        printf("Error: memory allocation failed for vectors.\n");
        fflush(stdout);
        free(triplets);
        free(values);
        free(colIndex);
        free(rowPtr);
        return 1;
    }

    omp_sched_t schedKind;
    if (!parseSchedule(schedStr, &schedKind)) {
        printf("Unknown schedule '%s'. Valid: static, dynamic, guided, auto\n", schedStr);
        free(triplets);
        free(values);
        free(colIndex);
        free(rowPtr);
        free(x);
        free(y);
        free(times);
        return 1;
    }

    if (threads > 0) {
        omp_set_num_threads(threads);
    }
    omp_set_schedule(schedKind, chunk);

    int usedThreads = omp_get_max_threads();
    printf("\nRuntime configuration:\n");
    printf("  Method: 2 (Atomic Operations)\n");
    printf("  Runs: %d\n", runs);
    printf("  Threads: %d\n", usedThreads);
    printf("  Schedule: %s  chunk=%d\n", schedStr, chunk);
    fflush(stdout);

    srand((unsigned int)time(NULL));

    printf("\nRunning %d matrix-vector multiplications...\n", runs);
    fflush(stdout);

    for (int i = 0; i < runs; i++) {
        for (int j = 0; j < cols; j++)
            x[j] = (double)rand() / RAND_MAX;

        double start = getMilliseconds();
        csrMatVecMultiply(rows, values, colIndex, rowPtr, x, y);
        double end = getMilliseconds();

        times[i] = end - start;
        printf("Run %d: %.6f ms\n", i + 1, times[i]);
        fflush(stdout);
    }

    printf("Saving all %d runs to file...\n", runs);
    fflush(stdout);

    FILE *fp = fopen("all_runs.txt", "w");
    if (fp) {
        fprintf(fp, "All %d runs (in ms) - Method 2 (Atomic Operations):\n", runs);
        for (int i = 0; i < runs; i++) {
            fprintf(fp, "%.6f\n", times[i]);
        }
        fclose(fp);
        printf("\n=== Success! ===\n");
        printf("All %d runs saved to all_runs.txt\n", runs);
        fflush(stdout);
    } else {
        printf("Error: could not create output file all_runs.txt\n");
        fflush(stdout);
    }

    free(triplets);
    free(values);
    free(colIndex);
    free(rowPtr);
    free(x);
    free(y);
    free(times);

    printf("Program completed successfully.\n");
    fflush(stdout);
    return 0;
}