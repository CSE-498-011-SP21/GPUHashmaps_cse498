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

/**
 * @file
 * @brief Has the kernel used for execution.
 */

#include "Operations.cuh"

#ifndef GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
#define GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH

/*
 * turning this into an enum may be worthwhile perhaps
 * especially as more functionality is added.
 */

/**
 * Marks a put.
 */
const int REQUEST_INSERT = 1;
/**
 * Marks a get.
 */
const int REQUEST_GET = 2;
/**
 * Marks a remove.
 */
const int REQUEST_REMOVE = 3;
/**
 * Marks an empty request.
 */
const int REQUEST_EMPTY = 0;

/**
 * Enum that may be used in the future to mark requests.
 */
enum RequestType {
    RQ_NIL = REQUEST_EMPTY,
    RQ_PUT = REQUEST_INSERT,
    RQ_GET = REQUEST_GET,
    RQ_DEL = REQUEST_REMOVE
};


/**
 * mvValue index is set to the value on a GET or EMPTY<V>::value if there is no value
 * It is set to the previous value on a successful INSERT or REMOVE and EMPTY<V>::value on an unsuccessful one
 * @tparam K key type
 * @tparam V value type
 * @param slabs slab data structure
 * @param num_of_buckets the number of buckets in the slab data structure
 * @param myKey array of keys to perform requests with
 * @param myValue array of values to perform requests with
 * @param myHash array of hashes
 * @param request array of request identifiers
 * @param ctx context object for the allocator
 */
template<typename K, typename V>
__global__ void requestHandler(volatile SlabData<K, V> **slabs, unsigned num_of_buckets,
                               K *myKey,
                               V *myValue, const unsigned * myHash, const int *request, WarpAllocCtx<K, V> ctx) {
    const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

    K key = myKey[tid];
    V value = myValue[tid];
    unsigned hash = myHash[tid] % num_of_buckets;
    bool activity = (request[tid] == REQUEST_GET);

    warp_operation_search(activity, key, value, hash, slabs, num_of_buckets);

    activity = (request[tid] == REQUEST_INSERT);
    warp_operation_replace(activity, key, value, hash, slabs,
                           num_of_buckets, ctx);

    activity = (request[tid] == REQUEST_REMOVE);
    warp_operation_delete(activity, key, value, hash, slabs, num_of_buckets);
    myValue[tid] = value;
}

#endif//GPUKEYVALUESTORE_STDREQUESTHANDLER_CUH
