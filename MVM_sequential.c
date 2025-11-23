#include <stdio.h>
#include <stdlib.h>
#include <time.h>


typedef struct {
    int row;
    int col;
    double val;
} Triplet;

// ---------- Quick Sort ----------
    int cmpTriplet(const void *a, const void *b) {
    const Triplet *ta = (const Triplet *)a;
    const Triplet *tb = (const Triplet *)b;

    // Sort by row first, then by column for deterministic CSR
    if (ta->row != tb->row)
        return ta->row - tb->row;
    return ta->col - tb->col;
}

// ---------- CSR Conversion ----------
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

    for (int i = 0; i < nnz; i++) {
        int row = triplets[i].row;
        int dest = (*rowPtr)[row + 1] - 1;
        (*values)[dest] = triplets[i].val;
        (*colIndex)[dest] = triplets[i].col;
        (*rowPtr)[row + 1]--;
    }
}

// ---------- Matrix-Vector Multiplication ----------
void csrMatVecMultiply(int rows, double *values, int *colIndex, int *rowPtr,
                       double *x, double *y) {
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        for (int j = rowPtr[i]; j < rowPtr[i + 1]; j++) {
            sum += values[j] * x[colIndex[j]];
        }
        y[i] = sum;
    }
}

// ---------- Time in milliseconds ----------
double getMilliseconds() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);  // get current time in UTC
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6; // convert to milliseconds
}

// ---------- Main ----------
int main(int argc, char *argv[]) {
    printf("=== Sparse Matrix Program Starting ===\n");
    fflush(stdout);

    if (argc < 2) {
        printf("Usage: %s <matrix_file> [runs]\n", argv[0]);
        printf("Example: %s matrix.txt 10\n", argv[0]);
        fflush(stdout);
        return 1;
    }

    char *filename = argv[1];
    int runs = (argc >= 3) ? atoi(argv[2]) : 10;
    if (runs <= 0) runs = 10;

    printf("Attempting to open file: %s\n", filename);
    fflush(stdout);

    FILE *fin = fopen(filename, "r");
    if (!fin) {
        printf("Error: cannot open file '%s'\n", filename);
        printf("Make sure the file exists in the current directory.\n");
        fflush(stdout);
        return 1;
    }

    printf("File opened successfully!\n");
    fflush(stdout);

    // Skip all comment lines starting with %
    printf("Skipping comment lines (starting with %%)...\n");
    fflush(stdout);
    
    int comment_count = 0;
    int ch;
    while ((ch = fgetc(fin)) != EOF) {
        if (ch == '%') {
            // Skip entire line
            while ((ch = fgetc(fin)) != EOF && ch != '\n');
            comment_count++;
        } else {
            // First non-comment character found, put it back
            ungetc(ch, fin);
            break;
        }
    }

    printf("Skipped %d comment line(s).\n", comment_count);
    fflush(stdout);

    // Check if we reached end of file
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
        printf("Expected format: rows cols nnz\n");
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

    printf("Allocating memory for triplets...\n");
    fflush(stdout);

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
            printf("Expected format: row col value\n");
            fflush(stdout);
            fclose(fin);
            free(triplets);
            return 1;
        }

        if (triplets[i].row > maxRow) maxRow = triplets[i].row;
        if (triplets[i].col > maxCol) maxCol = triplets[i].col;
    }

// If indices are 1-based, subtract 1 from all
    if (maxRow == rows || maxCol == cols) {
        printf("Detected 1-based indexing, converting to 0-based...\n");
        fflush(stdout);
        for (int i = 0; i < nnz; i++) {
            triplets[i].row--;
            triplets[i].col--;
        }
    }

// Now validate indices
    for (int i = 0; i < nnz; i++) {
        if (triplets[i].row < 0 || triplets[i].row >= rows || 
            triplets[i].col < 0 || triplets[i].col >= cols) {
            printf("Error: invalid indices at entry %d (row=%d, col=%d)\n", 
                   i + 1, triplets[i].row, triplets[i].col);
            printf("Valid ranges: row [0,%d), col [0,%d)\n", rows, cols);
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
        printf("Run %d: %.3f ms\n", i + 1, times[i]);
        fflush(stdout);
    }

    // Sort timings ascending
    printf("\nSorting results...\n");
    fflush(stdout);
    
    for (int i = 0; i < runs - 1; i++) {
        for (int j = i + 1; j < runs; j++) {
            if (times[j] < times[i]) {
                double tmp = times[i];
                times[i] = times[j];
                times[j] = tmp;
            }
        }
    }

    int keep = (int)(0.9 * runs);
    printf("Saving best %d runs to file...\n", keep);
    fflush(stdout);

    FILE *fp = fopen("best_runs.txt", "w");
    if (fp) {
        fprintf(fp, "Best %d of %d runs (in ms):\n", keep, runs);
        for (int i = 0; i < keep; i++) {
            fprintf(fp, "%.3f\n", times[i]);
        }
        fclose(fp);
        printf("\n=== Success! ===\n");
        printf("Best %d runs saved to best_runs.txt\n", keep);
        fflush(stdout);
    } else {
        printf("Error: could not create output file best_runs.txt\n");
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