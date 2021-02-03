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

#ifndef GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
#define GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH

const int REQUEST_INSERT = 1;
const int REQUEST_GET = 2;
const int REQUEST_REMOVE = 3;
const int REQUEST_EMPTY = 0;

__global__ void requestHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request, WarpAllocCtx ctx = WarpAllocCtx()) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];
    bool activity = (request[tid] == REQUEST_GET);

    warp_operation_search(activity, key, value, slabs, num_of_buckets);

    activity = (request[tid] == REQUEST_INSERT);
    warp_operation_replace(activity, key, value, slabs,
                           num_of_buckets, ctx);
    activity = (request[tid] == REQUEST_REMOVE);
    warp_operation_delete(activity, key, value, slabs, num_of_buckets);
    myValue[tid] = value;
}

__global__ void getHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];
    bool activity = (request[tid] == REQUEST_GET);

    warp_operation_search(activity, key, value, slabs, num_of_buckets);
    myValue[tid] = value;
}

__global__ void insertHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request, WarpAllocCtx ctx = WarpAllocCtx()) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];

    bool activity = (request[tid] == REQUEST_INSERT);
    warp_operation_replace(activity, key, value, slabs,
                           num_of_buckets, ctx);
    myValue[tid] = value;
}

__global__ void deleteHandler(volatile SlabData **slabs, unsigned num_of_buckets,
                               unsigned *myKey,
                               unsigned *myValue, int *request) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    unsigned key = myKey[tid];
    unsigned value = myValue[tid];
    bool activity = (request[tid] == REQUEST_REMOVE);
    warp_operation_delete(activity, key, value, slabs, num_of_buckets);
    myValue[tid] = value;
}


#endif//GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
