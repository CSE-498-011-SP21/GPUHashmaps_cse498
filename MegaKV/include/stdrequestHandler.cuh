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

#include "Operations.cuh"
#include "gpuErrchk.cuh"
#include "Request.cuh"

#ifndef GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
#define GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH

namespace megakv {

    int getBlocks() noexcept {
        cudaDeviceProp prop;
        prop.multiProcessorCount = 68;
        gpuAssert_megakv(cudaGetDeviceProperties(&prop, 0), __FILE__, __LINE__, false);
        return 2 * prop.multiProcessorCount;
    }

    const int BLOCKS = getBlocks() / 4; // TODO implement some tuning to run on any machine
    const int THREADS_PER_BLOCK = 512;

    /**
     *
     * @param slabs
     * @param num_of_buckets
     * @param myKey
     * @param myValue
     * @param request
     */
    __global__ void requestHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                                   unsigned *myKey,
                                   unsigned *myValue, int *request) {
        const int tid = threadIdx.x + blockIdx.x * blockDim.x;

        unsigned key = myKey[tid];
        unsigned value = myValue[tid];
        bool activity = (request[tid] == REQUEST_GET);

        warp_operation_search(activity, key, value, slabs, num_of_buckets);

        activity = (request[tid] == REQUEST_INSERT);
        warp_operation_replace(activity, key, value, slabs,
                               num_of_buckets);
        activity = (request[tid] == REQUEST_REMOVE);
        warp_operation_delete(activity, key, value, slabs, num_of_buckets);
        myValue[tid] = value;
    }

}

#endif//GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
