#!/bin/bash

# Iterate over matrix sizes from 10,000 to 17,000 with a step of 1,000
for ((matrSize=10000; matrSize<18000; matrSize+=1000)); do
    # Iterate over kernel sizes starting from 32 and doubling each iteration
    for ((kerSize=32; kerSize<1024; kerSize*=2)); do
        # Run the perf command for different build versions and output results to corresponding files
        sudo perf stat -e L1-dcache-load-misses,instructions ./part1/build/naive $matrSize $kerSize 2> part1Res/naive_${matrSize}_${kerSize}.txt 1> part1Res/naive_${matrSize}_${kerSize}_time.txt
        echo "naive done for $matrSize $kerSize"
        sudo perf stat -e L1-dcache-load-misses,instructions ./part1/build/prefetch $matrSize $kerSize 2> part1Res/prefetch_${matrSize}_${kerSize}.txt 1> part1Res/prefetch_${matrSize}_${kerSize}_time.txt
        echo "prefetch done for $matrSize $kerSize"
        sudo perf stat -e L1-dcache-load-misses,instructions ./part1/build/tiling $matrSize $kerSize 2> part1Res/tiling_${matrSize}_${kerSize}.txt 1> part1Res/tiling_${matrSize}_${kerSize}_time.txt
        echo "tiling done for $matrSize $kerSize"
        sudo perf stat -e L1-dcache-load-misses,instructions ./part1/build/tiling-prefetch $matrSize $kerSize 2> part1Res/tiling-prefetch_${matrSize}_${kerSize}.txt 1> part1Res/tiling-prefetch_${matrSize}_${kerSize}_time.txt
        echo "tiling-prefetch done for $matrSize $kerSize"
    done
done
