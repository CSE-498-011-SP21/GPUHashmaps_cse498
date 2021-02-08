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
 * @brief Handles general group allocation.
 */

#ifndef GPUHASHMAPS_GROUPALLOCATOR_CUH
#define GPUHASHMAPS_GROUPALLOCATOR_CUH

#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "GroupAllocator_Classes.cuh"

namespace groupallocator {

    static std::mutex groupMapMutex;
    static std::unordered_map<int, std::shared_ptr<GroupAllocator>> allocator;

    /**
     * Context object
     */
    struct Context {
        /**
         * Create context
         */
        Context() {}

        /**
         * Delete context
         */
        ~Context() {}

        /**
         * Size of pages to use
         */
        std::size_t page_size = 4096;
    };

    /**
     * Allocates memory of type T and sets *ptr to this memory of size s. It
     * allocates in group group.
     * Thread Safe!
     * @param ptr
     * @param s
     * @param group
     */
    template<typename T>
    void allocate(T **ptr, size_t s, const Context ctx, int group = -1, bool forceAligned128 = false) {
        groupMapMutex.lock();
        std::shared_ptr<GroupAllocator> g = allocator[group];
        if (g == nullptr) {
            g = std::make_shared<GroupAllocator>(group, ctx.page_size);
            allocator[group] = g;
        }
        groupMapMutex.unlock();

        g->allocate<T>(ptr, s, forceAligned128);
    }

    /**
     * Free T* p from group
     * @tparam T
     * @param p
     * @param group
     */
    template<typename T>
    void free(T *p, int group = -1) {
        groupMapMutex.lock();
        std::shared_ptr<GroupAllocator> g = allocator[group];
        groupMapMutex.unlock();
        if (g != nullptr) {
            g->free(p);
        }
    }

    /**
     * Cleans up the allocator by freeing everything so there is no memory leak. This is no longer needed and is a no-op.
     * Thread safe.
     */
    void freeall() {
        groupMapMutex.lock();
        for (std::pair<const int, std::shared_ptr < groupallocator::GroupAllocator>>
                    &elm : allocator) {
            elm.second->freeall();
        }
        groupMapMutex.unlock();
    }

    /**
     * Moves data to GPU, thread safe
     * @param group
     * @param gpuID
     * @param stream
     */
    void moveToGPU(int group = -1, int gpuID = 0, cudaStream_t stream = cudaStreamDefault) {
        groupMapMutex.lock();
        std::shared_ptr<GroupAllocator> g = allocator[group];
        groupMapMutex.unlock();
        if (g != nullptr) {
            //std::cerr << "Using stream " << stream << std::endl;

            g->moveToDevice(gpuID, stream);
        }
    }

    /**
     * Moves data to CPU, thread safe
     * @param group
     * @param stream
     */
    void moveToCPU(int group = -1, cudaStream_t stream = cudaStreamDefault) {
        groupMapMutex.lock();
        std::shared_ptr<GroupAllocator> g = allocator[group];
        groupMapMutex.unlock();
        if (g != nullptr) {
            //std::cerr << "Using stream " << stream << std::endl;
            g->moveToDevice(cudaCpuDeviceId, stream);
        }
    }

}  // namespace groupallocator

#endif //GPUHASHMAPS_GROUPALLOCATOR_CUH
