#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h> 

void verify_correctness(double *C, double *D, int dim)
{
    double epsilon = 1e-9;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            if (fabs(C[i * dim + j] - D[i * dim + j]) > epsilon)
            {
                printf("%f & %f at (%d %d)\n", C[i * dim + j], D[i * dim + j], i, j);
                printf("The two matrices are NOT identical\n");
                return;
            }
        }
    }
    printf("The matrix operation is correct!\n");
    return;
}

// Naive Matrix Transpose
void naiveMatrixTranspose(double *matrix, double *transpose, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            transpose[j * size + i] = matrix[i * size + j];
        }
    }
}

// Cache-Aware tiled Matrix Transpose
void tiledMatrixTranspose(double *matrix, double *transpose, int size, int blockSize) {
    // Students need to implement this function
    int numberTiles=size/blockSize;
    for(int i=0;i<numberTiles;i++){
        for(int j=0;j<numberTiles;j++){
            for(int u=0;u<blockSize;u++){
                for(int v=0;v<blockSize;v++){
                    transpose[(j*blockSize+v)*size+(i*blockSize+u)]=matrix[(i*blockSize+u)*size+(j*blockSize+v)];
                }
            }
        }
    }
}


// Prefetch Matrix Transpose
void prefetchMatrixTranspose(double *matrix, double *transpose, int size) {
    // Students need to implement this function
    int cache_line_size=8; //a parameter depending on the size of a cache line (to be used for prefetching)
    int prefetch_distance=2;//prefetch distance for L1
    int prefetch_distance_2 = 4;//prefetch distance for L2
    int prefetch_distance_3 = 18;//prefetch distance for L3
    int prefetch_degree=3;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if(j+1<size)_mm_prefetch(reinterpret_cast<const char*>(&transpose[(j+1)*size+i]), _MM_HINT_T2);
            transpose[j * size + i] = matrix[i * size + j];
            if((i*size+j)%cache_line_size==0){
                    _mm_prefetch(reinterpret_cast<const char*>(&matrix[i*size+j+cache_line_size*1*prefetch_distance]), _MM_HINT_T0);
                    //_mm_prefetch(reinterpret_cast<const char*>(&transpose[j*size+i+cache_line_size*1*prefetch_distance]), _MM_HINT_T0);
                for(int k=1;k<prefetch_degree+1;k++){
                    _mm_prefetch(reinterpret_cast<const char*>(&matrix[i*size+j+cache_line_size*k*prefetch_distance_2]), _MM_HINT_T1);//prefetching for L2/L3 only so that we can retain it for longer time
                    //_mm_prefetch(reinterpret_cast<const char*>(&transpose[j*size+i+cache_line_size*k*prefetch_distance_2]), _MM_HINT_T1);//prefetching for L2/L3 only so that we can retain it for longer time
                    _mm_prefetch(reinterpret_cast<const char*>(&matrix[i*size+j+cache_line_size*k*prefetch_distance_3]), _MM_HINT_T2);//prefetching for L2/L3 only so that we can retain it for longer time
                    //_mm_prefetch(reinterpret_cast<const char*>(&transpose[j*size+i+cache_line_size*k*prefetch_distance_3]), _MM_HINT_T2);//prefetching for L2/L3 only so that we can retain it for longer time
                }
                // _mm_prefetch(reinterpret_cast<const char*>(&matrix[i*size+j+cache_line_size*prefetch_distance]), _MM_HINT_T0);

            }
            // if(j+2<size)_mm_prefetch(reinterpret_cast<const char*>(&transpose[(j+2)*size+i]), _MM_HINT_T1);
        }
    }
}


// Tiled Prefetch Matrix Transpose
void tiledPrefetchedMatrixTranspose(double *matrix, double *transpose, int size, int tileSize) {
    // Students need to implement this function
    int blockSize=tileSize;
    int numberTiles=size/blockSize;
    int cache_line_size=8; //a parameter depending on the size of a cache line (to be used for prefetching)
    int prefetch_distance=2;//prefetch distance for L1
    int prefetch_distance_2 = 4;//prefetch distance for L2
    int prefetch_distance_3 = 18;//prefetch distance for L3
    int prefetch_degree=1;
    for(int i=0;i<numberTiles;i++){
        for(int j=0;j<numberTiles;j++){
            for(int u=0;u<blockSize;u++){
                for(int v=0;v<blockSize;v++){
                    if((i*blockSize+u+1)<size)_mm_prefetch(reinterpret_cast<const char*>(&transpose[(i*blockSize+u+1)*size+(j*blockSize+v)]), _MM_HINT_T2);
                    transpose[(j*blockSize+v)*size+(i*blockSize+u)]=matrix[(i*blockSize+u)*size+(j*blockSize+v)];
                    if((u*size+j)%cache_line_size==0){
                        _mm_prefetch(reinterpret_cast<const char*>(&matrix[(i*blockSize+u)*size+(j*blockSize+v)+cache_line_size*1*prefetch_distance]), _MM_HINT_T0);
                        //_mm_prefetch(reinterpret_cast<const char*>(&transpose[(j*blockSize+v)*size+(i*blockSize+u)+cache_line_size*1*prefetch_distance]), _MM_HINT_T0);
                        for(int k=1;k<prefetch_degree+1;k++){
                            _mm_prefetch(reinterpret_cast<const char*>(&matrix[(i*blockSize+u)*size+(j*blockSize+v)+cache_line_size*k*prefetch_distance_2]), _MM_HINT_T2);//prefetching for L2/L3 only so that we can retain it for longer time
                            //_mm_prefetch(reinterpret_cast<const char*>(&transpose[(j*blockSize+v)*size+(i*blockSize+u)+cache_line_size*k*prefetch_distance_2]), _MM_HINT_T1);//prefetching for L2/L3 only so that we can retain it for longer time
                            _mm_prefetch(reinterpret_cast<const char*>(&matrix[(i*blockSize+u)*size+(j*blockSize+v)+cache_line_size*k*prefetch_distance_3]), _MM_HINT_T2);//prefetching for L2/L3 only so that we can retain it for longer time
                           // _mm_prefetch(reinterpret_cast<const char*>(&transpose[(j*blockSize+v)*size+(i*blockSize+u)+cache_line_size*k*prefetch_distance_3]), _MM_HINT_T2);//prefetching for L2/L3 only so that we can retain it for longer time
                        }
                        // _mm_prefetch(reinterpret_cast<const char*>(&matrix[i*size+j+cache_line_size*prefetch_distance]), _MM_HINT_T0);
                        // _mm_prefetch(reinterpret_cast<const char*>(&transpose[j*size+i+cache_line_size*prefetch_distance]), _MM_HINT_T0);
                    }
                }
            }
        }
    }    
}



double naive(double * matrix, double *transpose, int size) {
    // Run and time the naive matrix transpose
    clock_t start = clock();
    naiveMatrixTranspose(matrix, transpose, size);

    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by naive matrix transpose: %f seconds\n", time_taken);

    return time_taken;
}



double tiled(double * matrix, double *transpose, int size, int blockSize) {
    // Run and time the tiled matrix transpose
    clock_t start = clock();
    tiledMatrixTranspose(matrix, transpose, size, blockSize);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by tiled matrix transpose: %f seconds\n", time_taken);

    return time_taken;
}


double prefetched(double * matrix, double *transpose, int size) {
    // Run and time the prefetch matrix transpose
    clock_t start = clock();
    prefetchMatrixTranspose(matrix, transpose, size);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by prefetch matrix transpose: %f seconds\n", time_taken);

    return time_taken;
}


double tiled_prefetched(double * matrix, double *transpose, int size, int tileSize) {
    // Run and time the prefetch matrix transpose
    clock_t start = clock();
    tiledPrefetchedMatrixTranspose(matrix, transpose, size, tileSize);
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    printf("Time taken by tiled prefetch matrix transpose: %f seconds\n", time_taken);

    return time_taken;
}


// Function to initialize the matrix with random values
void initializeMatrix(double *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = rand() % 100;
    }
}


// Function to initialize the matrix with random values
void initializeResultMatrix(double *matrix, int size) {
    for (int i = 0; i < size * size; i++) {
        matrix[i] = 0.0;
    }
}


int main(int argc, char *argv[]) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <matrix_size> <block_size>\n", argv[0]);
        return 1;
    }

    int size = atoi(argv[1]);
    int blockSize = atoi(argv[2]);

    // Allocate memory for the matrices
    double *matrix = (double *)malloc(size * size * sizeof(double));
    double *naive_transpose = (double *)malloc(size * size * sizeof(double));
    double *optimized_transpose = (double *)malloc(size * size * sizeof(double));

    // Check if memory allocation was successful
    if (matrix == NULL || naive_transpose == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    // Seed the random number generator
    srand(time(NULL));

    // Initialize the matrix with random values
    initializeMatrix(matrix, size);

    // Initialize the result matrix with zeros
    initializeResultMatrix(naive_transpose, size);


#ifdef NAIVE
    naive(matrix, transpose, size);

#endif


// TASK 1A
#ifdef OPTIMIZE_TILING
    initializeResultMatrix(optimized_transpose, size);    
    
    double tiled_time1 = tiled(matrix, optimized_transpose, size, blockSize);
    // double naive_time1 = naive(matrix, naive_transpose, size);

    // verify_correctness(naive_transpose, optimized_transpose, size);

    // printf("The speedup obtained by blocking is %f\n", naive_time1/tiled_time1);

#endif


// TASK 1B
#ifdef OPTIMIZE_PREFETCH
    initializeResultMatrix(optimized_transpose, size);
    
    
    double prefetched_time2 = prefetched(matrix, optimized_transpose, size);
    // double naive_time2 = naive(matrix, naive_transpose, size);
    
    // verify_correctness(naive_transpose, optimized_transpose, size);

    // printf("The speedup obtained by software prefetching is %f\n", naive_time2/prefetched_time2);
    

#endif


// TASK 1C
#ifdef OPTIMIZE_TILING_PREFETCH
    initializeResultMatrix(optimized_transpose, size);
    
    
    double prefetched_time3 = tiled_prefetched(matrix, optimized_transpose, size, blockSize);
    // double naive_time3 = naive(matrix, naive_transpose, size);
    
    // verify_correctness(naive_transpose, optimized_transpose, size);

    // printf("The speedup obtained by tiling and software prefetching combined is %f\n", naive_time3/prefetched_time3);
    

#endif

    // Free the allocated memory
    free(matrix);
    free(naive_transpose);
    free(optimized_transpose);

    return 0;
}
