// ================================================================
// Modern SELL-C-σ SpMV (fully standalone, no wrapper needed)
// Flags order: -r <runs> -c <chunk> -s <sigma> -t <threads>
// ================================================================

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>

// ------------------- SELL-C-σ structure -------------------
typedef struct {
    int C;                // chunk height
    int sigma;            // sort block size
    int rows;
    int cols;
    int slices;
    int *slice_ptr;
    int *col_idx;
    double *values;
    int *slice_lengths;
} SELL_CS;

// ------------------- Timing utility ------------------------
double get_ms() {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return ts.tv_sec*1000.0 + ts.tv_nsec/1e6;
}

// ------------------- CSR → SELL-C-σ ------------------------
SELL_CS *csr_to_sellcs(int rows, int cols, int nnz,
                       double *csr_val, int *csr_col, int *csr_rowptr,
                       int C, int sigma)
{
    SELL_CS *S = calloc(1,sizeof(*S));
    S->C = C; S->sigma = sigma; S->rows = rows; S->cols = cols;
    S->slices = (rows + C - 1)/C;
    S->slice_ptr = calloc(S->slices + 1, sizeof(int));
    S->slice_lengths = calloc(S->slices, sizeof(int));

    int *row_len = malloc(rows*sizeof(int));
    for(int i=0;i<rows;i++) row_len[i] = csr_rowptr[i+1]-csr_rowptr[i];

    // sort in blocks of sigma
    for(int b=0;b<rows;b+=sigma) {
        int end = b+sigma < rows ? b+sigma : rows;
        for(int i=b;i<end;i++)
            for(int j=i+1;j<end;j++)
                if(row_len[j]>row_len[i]){
                    int tmp=row_len[i]; row_len[i]=row_len[j]; row_len[j]=tmp;
                }
    }

    // slice lengths
    for(int s=0;s<S->slices;s++){
        int start = s*C;
        int end = (start+C<rows?start+C:rows);
        int max_len=0;
        for(int r=start;r<end;r++) if(row_len[r]>max_len) max_len=row_len[r];
        S->slice_lengths[s] = max_len;
    }

    // slice_ptr prefix sum
    for(int s=0;s<S->slices;s++)
        S->slice_ptr[s+1] = S->slice_ptr[s] + S->slice_lengths[s]*C;

    int total_nnz_sell = S->slice_ptr[S->slices];
    S->col_idx = malloc(total_nnz_sell*sizeof(int));
    S->values  = malloc(total_nnz_sell*sizeof(double));

    for(int s=0;s<S->slices;s++){
        int start=s*C, end=(start+C<rows?start+C:rows);
        int slice_len = S->slice_lengths[s];
        int base = S->slice_ptr[s];
        for(int r=start;r<end;r++){
            int csr_start = csr_rowptr[r], csr_end = csr_rowptr[r+1], row_nnz = csr_end - csr_start;
            int k=0;
            for(int j=csr_start;j<csr_end;j++,k++){
                S->values[base + k*C + (r-start)] = csr_val[j];
                S->col_idx[base + k*C + (r-start)] = csr_col[j];
            }
            for(;k<slice_len;k++){
                S->values[base + k*C + (r-start)] = 0.0;
                S->col_idx[base + k*C + (r-start)] = 0;
            }
        }
    }

    free(row_len);
    return S;
}

// ------------------- SELL-C SpMV --------------------------
void sellcs_spmv(const SELL_CS *S, const double *x, double *y){
    int C=S->C;
#pragma omp parallel for schedule(runtime)
    for(int r=0;r<S->rows;r++) y[r]=0.0;

#pragma omp parallel for schedule(runtime)
    for(int s=0;s<S->slices;s++){
        int start=s*C, end=(start+C<S->rows?start+C:S->rows);
        int slice_len=S->slice_lengths[s], base=S->slice_ptr[s];
        for(int k=0;k<slice_len;k++){
            int offset=base+k*C;
            for(int r=start;r<end;r++){
                int idx=offset+(r-start);
                y[r]+=S->values[idx]*x[S->col_idx[idx]];
            }
        }
    }
}

// ------------------- Main -------------------------------
int main(int argc, char **argv){
    if(argc<10){
        printf("Usage: %s <matrix_file> -r <runs> -c <chunk> -s <sigma> -t <threads>\n",argv[0]);
        return 1;
    }

    char *matrix_file = argv[1];
    int runs = atoi(argv[3]);
    int chunk = atoi(argv[5]);
    int sigma = atoi(argv[7]);
    int threads = atoi(argv[9]);
    omp_set_num_threads(threads);

    // ------------------ Load Matrix Market ----------------
    FILE *f = fopen(matrix_file,"r");
    if(!f){printf("Error opening matrix.\n"); return 1;}

    // skip comments
    char line[512];
    while(fgets(line,sizeof(line),f))
        if(line[0]!='%') break;
    int rows,cols,nnz;
    if(sscanf(line,"%d %d %d",&rows,&cols,&nnz)!=3){
        printf("Error: invalid matrix header.\n"); fclose(f); return 1;
    }

    int *row = malloc(nnz*sizeof(int));
    int *col = malloc(nnz*sizeof(int));
    double *val = malloc(nnz*sizeof(double));
    for(int i=0;i<nnz;i++){
        if(fscanf(f,"%d %d %lf",&row[i],&col[i],&val[i])!=3){
            printf("Error reading matrix entry %d\n",i); fclose(f); return 1;
        }
        row[i]--; col[i]--; // convert to 0-based
    }
    fclose(f);

    // ------------------ Convert to CSR -------------------
    int *rowptr = calloc(rows+1,sizeof(int));
    for(int i=0;i<nnz;i++) rowptr[row[i]+1]++;
    for(int i=0;i<rows;i++) rowptr[i+1]+=rowptr[i];

    int *csr_col = malloc(nnz*sizeof(int));
    double *csr_val = malloc(nnz*sizeof(double));
    int *tmp = malloc((rows+1)*sizeof(int));
    memcpy(tmp,rowptr,(rows+1)*sizeof(int));
    for(int i=0;i<nnz;i++){
        int r=row[i]; int pos=tmp[r]++;
        csr_col[pos]=col[i];
        csr_val[pos]=val[i];
    }

    free(row); free(col); free(val); free(tmp);

    // ------------------ Convert CSR → SELL-C ----------------
    SELL_CS *S = csr_to_sellcs(rows,cols,nnz,csr_val,csr_col,rowptr,chunk,sigma);

    double *x = malloc(cols*sizeof(double));
    double *y = malloc(rows*sizeof(double));
    double *times = malloc(runs*sizeof(double));

    // ------------------ Run SpMV -------------------------
    for(int r=0;r<runs;r++){
        for(int j=0;j<cols;j++) x[j]=(double)rand()/RAND_MAX;
        double t0=get_ms();
        sellcs_spmv(S,x,y);
        double t1=get_ms();
        times[r]=t1-t0;
        printf("Run %d: %.6f ms\n",r+1,times[r]);
    }

    // ------------------ Save all_runs.txt ----------------
    FILE *fp = fopen("all_runs.txt","w");
    for(int r=0;r<runs;r++) fprintf(fp,"%.6f\n",times[r]);
    fclose(fp);

    // ------------------ Cleanup ------------------------
    free(rowptr); free(csr_col); free(csr_val);
    free(S->slice_ptr); free(S->slice_lengths);
    free(S->col_idx); free(S->values); free(S);
    free(x); free(y); free(times);

    return 0;
}
