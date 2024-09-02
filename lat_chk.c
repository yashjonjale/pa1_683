#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define ARRAY_SIZE (1024 * 1024 * 64)  // Adjust size according to cache size
#define ITERATIONS 1000000

int main() {
    int *array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    int stride = 64 / sizeof(int);  // 64-byte cache line size
    clock_t start, end;
    double total_time;
    
    start = clock();
    
    for (int i = 0; i < ITERATIONS; i++) {
        for (int j = 0; j < ARRAY_SIZE; j += stride) {
            array[j]++;
        }
    }
    
    end = clock();
    total_time = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    printf("Time taken: %f seconds\n", total_time / ITERATIONS);
    
    free(array);
    return 0;
}
