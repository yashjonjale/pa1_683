#include <iostream>
#include <immintrin.h>

int main() {
    int zmm_count = 0;

    // Check if AVX-512 is supported
    if (__builtin_cpu_supports("avx512f")) {
        // AVX-512 is supported, now let's count the registers
        __asm__ __volatile__ (
            "mov $0, %%eax \n\t"
            "1: \n\t"
            "vpxorq %%zmm0, %%zmm0, %%zmm0 \n\t"
            "inc %%eax \n\t"
            "cmp $32, %%eax \n\t"
            "jne 1b \n\t"
            "mov %%eax, %0"
            : "=r" (zmm_count)
            :
            : "%eax"
        );
        std::cout << "Number of ZMM registers: " << zmm_count << std::endl;
    } else {
        std::cout << "AVX-512 is not supported on this processor." << std::endl;
    }

    return 0;
}
