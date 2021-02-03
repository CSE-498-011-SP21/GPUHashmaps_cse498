/*
 * Copyright (c) 2020-2021 dePaul Miller (dsm220@lehigh.edu)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <StandardSlabDefinitions.cuh>
#include <vector>
#include <Slab.cuh>
#include <cuda_profiler_api.h>

int main() {

    const int size = 1000000;

    std::hash<unsigned> hfn;

    SlabUnified<unsigned long long, unsigned> s(size);
    auto b = new BatchBuffer<unsigned long long, unsigned>();

    float time;
    s.setGPU();
    std::cerr << "Populating" << std::endl;

    unsigned gen_key = 1;

    int inserted = 0;
    while (inserted < size / 2) {
        int k = 0;
        for (; k < size / 2 - inserted && k < THREADS_PER_BLOCK * BLOCKS; ++k) {
            unsigned key = gen_key;
            gen_key++;
            b->getBatchKeys()[k] = key;
            b->getHashValues()[k] = hfn(key);
            b->getBatchRequests()[k] = REQUEST_INSERT;
        }
        for (; k < THREADS_PER_BLOCK * BLOCKS; ++k) {
            b->getBatchRequests()[k] = REQUEST_EMPTY;
        }

        std::cerr << inserted << std::endl;
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));

        k = 0;
        int loopCond = size / 2 - inserted;
        for (; k < loopCond && k < THREADS_PER_BLOCK * BLOCKS; ++k) {
            if (b->getBatchValues()[k] == EMPTY<unsigned>::value)
                inserted++;
        }
    }

    std::cerr << "Populated" << std::endl;

    gpuErrchk(cudaProfilerStart());

    for (int rep = 0; rep < 10; rep++) {

        for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; ++i) {
            unsigned key = rand() / (double) RAND_MAX * (2 * size) + 1;
            b->getBatchKeys()[i] = key;
            b->getHashValues()[i] = hfn(key);
            b->getBatchRequests()[i] = REQUEST_GET;
        }


        auto start = std::chrono::high_resolution_clock::now();
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;

        std::cout << "Standard Uniform test" << std::endl;
        std::cout << "Latency\t" << dur.count() * 1e3 << " ms" << std::endl;
        //std::cout << "Latency 2\t" << time << " ms" << std::endl;
        //std::cout << "Throughput\t" << THREADS_PER_BLOCK * BLOCKS / (time * 1e3) << " Mops" << std::endl;
    }

    std::cout << "\nNo conflict test\n";

    for (int rep = 0; rep < 10; rep++) {

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; ++i) {
            b->getBatchKeys()[i] = i + 1;
            b->getHashValues()[i] = (i + 1) / 32;
            b->getBatchRequests()[i] = REQUEST_GET;
        }
        s.moveBufferToGPU(b, 0x0);
        s.diy_batch(b, BLOCKS, THREADS_PER_BLOCK, 0x0);
        s.moveBufferToCPU(b, 0x0);
        gpuErrchk(cudaStreamSynchronize(0x0));
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> dur = end - start;

        std::cout << "Latency\t" << dur.count() * 1e3 << " ms" << std::endl;
        //std::cout << "Latency 2\t" << time << " ms" << std::endl;
        //std::cout << "Throughput\t" << THREADS_PER_BLOCK * BLOCKS / (time * 1e3) << " Mops" << std::endl;
        std::cout << "Arrival rate handled\t" << THREADS_PER_BLOCK * BLOCKS / (dur.count() * 1e6) << " Mops"
                  << std::endl;

    }
    gpuErrchk(cudaProfilerStop());
    delete b;
}