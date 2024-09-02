#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <immintrin.h> // For SIMD intrinsics


//global vars for tuning
int param1_ = 0;
int param2_ = 0;
int param3_ = 0;

void naive_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void tiled_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void simd_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);


void tiled_simd_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void simd_prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void tiled_prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);
void simd_tiled_prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);

/**
 * @brief 		Generates random numbers between values fMin and fMax.
 * @param 		fMin 	lower range
 * @param 		fMax 	upper range
 * @return 		random floating point number
 */
double fRand(double fMin, double fMax)
{

    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_kernel(double *matrix, int rows, int cols)
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = ceil(fRand(0.0001, 1.0000)); // random values between 0 and 1
        }
    }
}

/**
 * @brief 		Initialize a matrix of given dimension with random values.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_matrix(double *matrix, int rows, int cols)
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = fRand(0.0001, 1.0000); // random values between 0 and 1
        }
    }
}

/**
 * @brief 		Initialize result matrix of given dimension with 0.
 * @param 		matrix 		pointer to the matrix
 * @param 		rows 		number of rows in the matrix
 * @param 		cols 		number of columns in the matrix
 */
void initialize_result_matrix(double *matrix, int rows, int cols)
{

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            matrix[i * cols + j] = 0.0;
        }
    }
}


/**
 * @brief 		Compare if two matrices of same dimension are identical
 * @param 		C 		first matrix to compare
 * @param 		D 		second matrix to compare
 * @param 		dim 	dimension of the matrices
 */
void verify_correctness(double *C, double *D, int dim)
{
    double epsilon = 1e-9;
    for (int i = 0; i < dim; i++)
    {
        for (int j = 0; j < dim; j++)
        {
            if (fabs(C[i * dim + j] - D[i * dim + j]) > epsilon)
            {
                printf("%f & %f at location (%d %d)\n", C[i * dim + j], D[i * dim + j], i, j);
                printf("Matrix convolution is incorrect!\n");
                return;
            }
        }
    }
    printf("Matrix convolution is correct!\n");
    return;
}

double measure_execution_time(void (*func)(double *, double *, double *, int, int, int), double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size);

int main(int argc, char **argv)
{
    if (argc <= 2)
    {
        printf("Usage: matrix-dimension kernel-size\n\n");
        return 0;
    }

    int dim = atoi(argv[1]);
    int kernel_size = atoi(argv[2]);
    int output_dim = dim - kernel_size + 1;
    // param tuning example usage 
    // int param1 = argv[2];
    // int param2 = argv[3];
    // int param3 = argv[4];

    // Allocate memory for the input and output images
    double *input_image = (double *)malloc(dim * dim * sizeof(double));
    double *output_image = (double *)malloc(output_dim * output_dim * sizeof(double));
    double *kernel = (double *)malloc(kernel_size * kernel_size * sizeof(double));
    double *optimized_op = (double *)malloc(output_dim * output_dim * sizeof(double));

    // Initialize the input image and kernel
    initialize_matrix(input_image, dim, dim);

    // Initialize the kernel
    initialize_kernel(kernel, kernel_size, kernel_size);

    // Initialize the output image
    initialize_result_matrix(output_image, output_dim, output_dim);

    // Measure execution time and perform naive convolution
    double naive_time = measure_execution_time(naive_convolution, input_image, output_image, kernel, dim, output_dim, kernel_size);

    // Print the execution times and speedups
    printf("Naive Convolution Time: %f seconds\n", naive_time);


// Measure execution time and perform tiled convolution
#ifdef OPTIMIZE_TILING

    initialize_result_matrix(optimized_op, output_dim, output_dim);

    double tiled_time = measure_execution_time(tiled_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double tiled_speedup = naive_time / tiled_time;
    printf("Tiled Convolution Time: %f seconds, Speedup: %fx\n", tiled_time, tiled_speedup);

    verify_correctness(output_image, optimized_op, output_dim);

#endif

// Measure execution time and perform SIMD convolution
#ifdef OPTIMIZE_SIMD

    initialize_result_matrix(optimized_op, output_dim, output_dim);

    double simd_time = measure_execution_time(simd_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double simd_speedup = naive_time / simd_time;

    printf("SIMD Convolution Time: %f seconds, Speedup: %fx\n", simd_time, simd_speedup);
    verify_correctness(output_image, optimized_op, output_dim);

#endif

// Measure execution time and perform prefetch convolution
#ifdef OPTIMIZE_PREFETCH

    initialize_result_matrix(optimized_op, output_dim, output_dim);

    double prefetch_time = measure_execution_time(prefetch_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double prefetch_speedup = naive_time / prefetch_time;
    printf("Prefetch Convolution Time: %f seconds, Speedup: %fx\n", prefetch_time, prefetch_speedup);

    verify_correctness(output_image, optimized_op, output_dim);

#endif


#ifdef OPTIMIZE_TILING_SIMD
    initialize_result_matrix(optimized_op, output_dim, output_dim);

    // Measure execution time and perform tiled SIMD convolution
    double tiled_simd_time = measure_execution_time(tiled_simd_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double tiled_simd_speedup = naive_time / tiled_simd_time;
    printf("Tiled SIMD Convolution Time: %f seconds, Speedup: %fx\n", tiled_simd_time, tiled_simd_speedup);

    verify_correctness(output_image, optimized_op, output_dim);

#endif

// Measure execution time and perform SIMD prefetch convolution
#ifdef OPTIMIZE_SIMD_PREFETCH
    initialize_result_matrix(optimized_op, output_dim, output_dim);


    double simd_prefetch_time = measure_execution_time(simd_prefetch_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double simd_prefetch_speedup = naive_time / simd_prefetch_time;
    printf("SIMD Prefetch Convolution Time: %f seconds, Speedup: %fx\n", simd_prefetch_time, simd_prefetch_speedup);

    verify_correctness(output_image, optimized_op, output_dim);


#endif

// Measure execution time and perform tiled prefetch convolution
#ifdef OPTIMIZE_TILING_PREFETCH
    initialize_result_matrix(optimized_op, output_dim, output_dim);

    double tiled_prefetch_time = measure_execution_time(tiled_prefetch_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double tiled_prefetch_speedup = naive_time / tiled_prefetch_time;
    printf("Tiled Prefetch Convolution Time: %f seconds, Speedup: %fx\n", tiled_prefetch_time, tiled_prefetch_speedup);

    verify_correctness(output_image, optimized_op, output_dim);


#endif

// Measure execution time and perform SIMD tiled prefetch convolution
#ifdef OPTIMIZE_TILING_SIMD_PREFETCH
    initialize_result_matrix(optimized_op, output_dim, output_dim);

    double simd_tiled_prefetch_time = measure_execution_time(simd_tiled_prefetch_convolution, input_image, optimized_op, kernel, dim, output_dim, kernel_size);
    double simd_tiled_prefetch_speedup = naive_time / simd_tiled_prefetch_time;
    printf("SIMD Tiled Prefetch Convolution Time: %f seconds, Speedup: %fx\n", simd_tiled_prefetch_time, simd_tiled_prefetch_speedup);

    verify_correctness(output_image, optimized_op, output_dim);


#endif

    // Free allocated memory
    free(input_image);
    free(output_image);
    free(optimized_op);

    return 0;
}

// Naive convolution implementation
void naive_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    for (int i = 0; i < output_dim; i++)
    {
        for (int j = 0; j < output_dim; j++)
        {
            double sum = 0.0;
            for (int ki = 0; ki < kernel_size; ki++)
            {
                for (int kj = 0; kj < kernel_size; kj++)
                {
                    int x = i + ki;
                    int y = j + kj;
                    sum += input_image[x * dim + y] * kernel[ki * kernel_size + kj];
                }
            }
            output_image[i * output_dim + j] = sum;
        }
    }
}

// Tiled convolution implementation
void tiled_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
   // Students need to implement this
   int Bx = 1; //L1 // Block Size or Tile Size
   int By = 16;
//    int B2 = 512; //ignore L2
//    int B3 = 1024; //ignore L3
   int N_by_Bx = dim / Bx; // dim = c*B
   int N_by_By = dim / By;
   //kernel always in cache
   for (int i = 0; i < N_by_Bx; i++)
   {
       for (int j = 0; j < N_by_By; j++)
       {
            //beginning coordinates of the block
           int y_ = i*Bx;
           int x_ = j*By;
           //compute over the block
           for(int ii = 0;ii<Bx;ii++){
                if (y_ + ii + kernel_size>dim) break;
                for(int ki = 0;ki<kernel_size;ki++){
                    int y = y_ + ii + ki;
                    for(int jj=0;jj<By;jj++){
                        if (x_ + jj + kernel_size>dim) break;
                        double sum = 0;
                        // if (x >= dim) break;
                        for(int kj = 0;kj<kernel_size;kj++){
                            int x = x_ + jj + kj;
                            // if (y >= dim) break;
                            sum+= input_image[y*dim+x]*kernel[ki*kernel_size+kj];
                            // sum += prod
                            // printf("sum: %d\n", sum);
                        }
                        output_image[(y_ + ii)*output_dim+(x_ + jj)]+= sum;
                    }
                }
           }
       }
   }

}

// SIMD convolution implementation
void simd_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    // Students need to implement this
    //assuming dim is multiple of 8
    //memlaign attribute
    //AVX-512
    // const int vector_size = 8; // AVX-512 processes 8 doubles at a time

    // // Assume dim and output_dim are multiples of vector_size
    // for (int i = 0; i < output_dim; i++)
    // {
    //     for (int j = 0; j < output_dim; j += vector_size)
    //     {
    //         __m512d sum_vec = _mm512_setzero_pd();

    //         for (int ki = 0; ki < kernel_size; ki++)
    //         {
    //             for (int kj = 0; kj < kernel_size; kj++)
    //             {
    //                 int x = i + ki;
    //                 int y = j + kj;

    //                 __m512d input_vec = _mm512_load_pd(&input_image[x * dim + y]);
    //                 __m512d kernel_val = _mm512_set1_pd(kernel[ki * kernel_size + kj]);
    //                 sum_vec = _mm512_fmadd_pd(input_vec, kernel_val, sum_vec);
    //             }
    //         }

    //         _mm512_store_pd(&output_image[i * output_dim + j], sum_vec);
    //     }
    // }

    const int vector_size = 8; // AVX-512 processes 8 doubles at a time

    for (int i = 0; i < output_dim; i++)
    {
        for (int j = 0; j < output_dim; j += vector_size)
        {
            __m512d sum_vec = _mm512_setzero_pd();

            for (int ki = 0; ki < kernel_size; ki++)
            {
                for (int kj = 0; kj < kernel_size; kj++)
                {
                    int x = i + ki;
                    int y = j + kj;

                    __m512d input_vec = _mm512_loadu_pd(&input_image[x * dim + y]);
                    __m512d kernel_val = _mm512_set1_pd(kernel[ki * kernel_size + kj]);
                    sum_vec = _mm512_fmadd_pd(input_vec, kernel_val, sum_vec);
                }
            }

            _mm512_storeu_pd(&output_image[i * output_dim + j], sum_vec);
        }

    
    }
    // const int vector_size = 4; // AVX processes 4 doubles at a time

    // for (int i = 0; i < output_dim; i++)
    // {
    //     for (int j = 0; j < output_dim; j += vector_size)
    //     {
    //         __m256d sum_vec = _mm256_setzero_pd(); // Initialize sum vector to zero

    //         for (int ki = 0; ki < kernel_size; ki++)
    //         {
    //             for (int kj = 0; kj < kernel_size; kj++)
    //             {
    //                 int x = i + ki;
    //                 int y = j + kj;

    //                 __m256d input_vec = _mm256_loadu_pd(&input_image[x * dim + y]); // Load 4 doubles
    //                 __m256d kernel_val = _mm256_set1_pd(kernel[ki * kernel_size + kj]); // Broadcast kernel value
    //                __m256d mul_vec = _mm256_mul_pd(input_vec, kernel_val); // Multiply
    //                 sum_vec = _mm256_add_pd(sum_vec, mul_vec); // Add
    //             }
    //         }

    //         _mm256_storeu_pd(&output_image[i * output_dim + j], sum_vec); // Store the result
    //     }
    // }
}

#define BLOCK_SIZE 4
#define VECTOR_SIZE 8
// Prefetch convolution implementation
void prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    // Students need to implement this
    // for (int i = 0; i < output_dim; i++)
    // {
    //     for (int j = 0; j < output_dim; j += BLOCK_SIZE * VECTOR_SIZE)
    //     {
    //         __m512d sum_vec[BLOCK_SIZE] = {
    //             _mm512_setzero_pd(), _mm512_setzero_pd(),
    //             _mm512_setzero_pd(), _mm512_setzero_pd()
    //         };

    //         for (int ki = 0; ki < kernel_size; ki++)
    //         {
    //             for (int kj = 0; kj < kernel_size; kj++)
    //             {
    //                 int x = i + ki;
    //                 __m512d kernel_val = _mm512_set1_pd(kernel[ki * kernel_size + kj]);

    //                 for (int b = 0; b < BLOCK_SIZE; b++)
    //                 {
    //                     int y = j + kj + b * VECTOR_SIZE;
    //                     __m512d input_vec = _mm512_loadu_pd(&input_image[x * dim + y]);
    //                     sum_vec[b] = _mm512_fmadd_pd(input_vec, kernel_val, sum_vec[b]);
    //                 }
    //             }
    //         }

    //         // Store results
    //         for (int b = 0; b < BLOCK_SIZE; b++)
    //         {
    //             _mm512_storeu_pd(&output_image[i * output_dim + j + b * VECTOR_SIZE], sum_vec[b]);
    //         }
    //     }
    // }
    int prefetch_degree=kernel_size;
    int cache_line_size=8;
    int prefetch_distance=2;
    int prefetch_distance2=2;

    // for (int i = 0; i < output_dim; i++)
    // {
    //     for (int ki = 0; ki < kernel_size; ki++){
    //         for (int j = 0; j < output_dim; j++)
    //         {
    //             // if(ki==0 && i%kernel_size==0){
    //             //     for(int a=3;a<=prefetch_degree;a++){
    //             //         _mm_prefetch(reinterpret_cast<const char*>(&input_image[(i+a)*dim+j]), _MM_HINT_T1);//prefetching the next row in the L2 cache so that kernel multiplication gets fast
    //             //     }
    //             // }   
    //             double sum = 0.0;
    //             for (int kj = 0; kj < kernel_size; kj++)
    //             {   
    //                 // if(kj%8==0){
    //                 //     _mm_prefetch(reinterpret_cast<const char*>(&input_image[(i)*dim+j+prefetch_distance*cache_line_size]), _MM_HINT_T0);//prefetching the next cache line (similar to next line prefetcher) in the L1 cache so that kernel multiplication gets fast
    //                 // }
    //                 // if((j)%cache_line_size==0){
    //                 //     _mm_prefetch(reinterpret_cast<const char*>(&input_image[(i+ki)*dim+j+prefetch_distance2*cache_line_size]), _MM_HINT_T0);
    //                 // }
    //                 int x = i + ki;
    //                 int y = j + kj;
    //                 sum += input_image[x * dim + y] * kernel[ki * kernel_size + kj];
    //             }
    //             output_image[i * output_dim + j]+= sum;
    //             // if((j)%cache_line_size==0){
    //             //     _mm_prefetch(reinterpret_cast<const char*>(&output_image[(i)*output_dim+j+prefetch_distance*cache_line_size]), _MM_HINT_T1);
    //             // }
    //         }
    //     }
    // }

    for (int i = 0; i < output_dim; i++)
    {
        for (int j = 0; j < output_dim; j++)
        {
            double sum = 0.0;
            for (int ki = 0; ki < kernel_size; ki++)
            {
                if(ki==0 && j%cache_line_size==0 && i%kernel_size==0){
                    for(int a=3;a<=prefetch_degree;a++){
                        _mm_prefetch(reinterpret_cast<const char*>(&input_image[(i+a)*dim+j]), _MM_HINT_T1);//prefetching the next row in the L2 cache so that kernel multiplication gets fast
                    }
                }   
                for (int kj = 0; kj < kernel_size; kj++)
                {
                    // if(kj%cache_line_size==0){
                    //     _mm_prefetch(reinterpret_cast<const char*>(&input_image[(i+ki)*dim+j+prefetch_distance*cache_line_size]), _MM_HINT_T2);
                    // }
                    int x = i + ki;
                    int y = j + kj;
                    sum += input_image[x * dim + y] * kernel[ki * kernel_size + kj];
                }
            }
            if((j)%cache_line_size==0){
                _mm_prefetch(reinterpret_cast<const char*>(&output_image[(i)*output_dim+j+prefetch_distance*cache_line_size]), _MM_HINT_T1);
            }
            output_image[i * output_dim + j] = sum;

        }
    }

}

// Bonus Tasks
// Tiled SIMD convolution implementation
void tiled_simd_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    //Students need to implement this
    const int vector_size = 8; // AVX-512 processes 8 doubles at a time

    int Bx = 1; //L1 // Block Size or Tile Size
    int By = 16;
    //    int B2 = 512; //ignore L2
    //    int B3 = 1024; //ignore L3
    int N_by_Bx = dim / Bx; // dim = c*B
    int N_by_By = dim / By;
        //kernel always in cache
        

    for(int ii=0;ii<N_by_Bx;ii++){
        for(int jj=0;jj<N_by_By;jj++){

            int x_ = ii*Bx;
            int y_ = jj*By;

            for (int i = 0; i < Bx; i++)
            {
                if (x_ + i + kernel_size>dim) break;
                for (int j = 0; j < By; j += vector_size)
                {
                    if (y_ + j + kernel_size>dim) break;
                    __m512d sum_vec = _mm512_setzero_pd();

                    for (int ki = 0; ki < kernel_size; ki++)
                    {
                        for (int kj = 0; kj < kernel_size; kj++)
                        {
                            int x = x_ + i + ki;
                            int y = y_ + j + kj;

                            __m512d input_vec = _mm512_loadu_pd(&input_image[x * dim + y]);
                            __m512d kernel_val = _mm512_set1_pd(kernel[ki * kernel_size + kj]);
                            sum_vec = _mm512_fmadd_pd(input_vec, kernel_val, sum_vec);
                        }
                    }

                    _mm512_storeu_pd(&output_image[(x_+i) * output_dim + (y_+j)], sum_vec);
                }    
            }
        }
    }
}

// SIMD prefetch convolution implementation
void simd_prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    // Students need to implement this
    const int vector_size = 8; // AVX-512 processes 8 doubles at a time

    int prefetch_degree=kernel_size;
    int cache_line_size=8;
    int prefetch_distance=2;
    int prefetch_distance2=2;

    for (int i = 0; i < output_dim; i++)
    {
        for (int j = 0; j < output_dim; j += vector_size)
        {
            __m512d sum_vec = _mm512_setzero_pd();

            for (int ki = 0; ki < kernel_size; ki++)
            {
                if(ki==0 && j%cache_line_size==0 && i%kernel_size==0){
                    for(int a=3;a<=prefetch_degree;a++){
                        _mm_prefetch(reinterpret_cast<const char*>(&input_image[(i+a)*dim+j]), _MM_HINT_T1);//prefetching the next row in the L2 cache so that kernel multiplication gets fast
                    }
                }
                for (int kj = 0; kj < kernel_size; kj++)
                {
                    int x = i + ki;
                    int y = j + kj;

                    __m512d input_vec = _mm512_loadu_pd(&input_image[x * dim + y]);
                    __m512d kernel_val = _mm512_set1_pd(kernel[ki * kernel_size + kj]);
                    sum_vec = _mm512_fmadd_pd(input_vec, kernel_val, sum_vec);
                }
            }

            _mm512_storeu_pd(&output_image[i * output_dim + j], sum_vec);
        }
    
    }
}

// Tiled prefetch convolution implementation
void tiled_prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    // Students need to implement this
}

// SIMD tiled prefetch convolution implementation
void simd_tiled_prefetch_convolution(double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    // Students need to implement this
    const int vector_size = 8; // AVX-512 processes 8 doubles at a time

    
    int prefetch_degree=kernel_size;
    int cache_line_size=8;
    int prefetch_distance=2;
    int prefetch_distance2=2;

    int Bx = 1; //L1 // Block Size or Tile Size
    int By = 16;
    //    int B2 = 512; //ignore L2
    //    int B3 = 1024; //ignore L3
    int N_by_Bx = dim / Bx; // dim = c*B
    int N_by_By = dim / By;
        //kernel always in cache
        

    for(int ii=0;ii<N_by_Bx;ii++){
        for(int jj=0;jj<N_by_By;jj++){

            int x_ = ii*Bx;
            int y_ = jj*By;

            for (int i = 0; i < Bx; i++)
            {
                if (x_ + i + kernel_size>dim) break;
                for (int j = 0; j < By; j += vector_size)
                {
                    if (y_ + j + kernel_size>dim) break;
                    __m512d sum_vec = _mm512_setzero_pd();

                    for (int ki = 0; ki < kernel_size; ki++)
                    {
                        if(ki==0 && (y_+j)%cache_line_size==0 && (x_+i)%kernel_size==0){
                            for(int a=3;a<=prefetch_degree;a++){
                                _mm_prefetch(reinterpret_cast<const char*>(&input_image[(x_+i+a)*dim+(y_+j)]), _MM_HINT_T1);//prefetching the next row in the L2 cache so that kernel multiplication gets fast
                            }
                        }
                        for (int kj = 0; kj < kernel_size; kj++)
                        {
                            int x = x_ + i + ki;
                            int y = y_ + j + kj;

                            __m512d input_vec = _mm512_loadu_pd(&input_image[x * dim + y]);
                            __m512d kernel_val = _mm512_set1_pd(kernel[ki * kernel_size + kj]);
                            sum_vec = _mm512_fmadd_pd(input_vec, kernel_val, sum_vec);
                        }
                    }

                    _mm512_storeu_pd(&output_image[(x_+i) * output_dim + (y_+j)], sum_vec);
                }    
            }
        }
    }
}

// Function to measure execution time of a convolution function
double measure_execution_time(void (*func)(double *, double *, double *, int, int, int), double *input_image, double *output_image, double *kernel, int dim, int output_dim, int kernel_size)
{
    clock_t start, end;
    start = clock();
     __builtin___clear_cache((char*)input_image, (char*)(input_image + dim*dim));
    __builtin___clear_cache((char*)output_image, (char*)(output_image + output_dim*output_dim));
    __builtin___clear_cache((char*)kernel, (char*)(kernel + kernel_size * kernel_size));
    func(input_image, output_image, kernel, dim, output_dim, kernel_size);
    end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC;
}



