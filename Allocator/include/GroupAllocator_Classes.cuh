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
 * @breif Classes to handle allocation by group allocator.
 */

#ifndef GPUHASHMAPS_GROUPALLOCATOR_CLASSES_CUH
#define GPUHASHMAPS_GROUPALLOCATOR_CLASSES_CUH

#include <cmath>
#include <functional>
#include <mutex>
#include <vector>
#include <list>
#include <utility>

#include "ListAllocator.cuh"

namespace groupallocator {

    /**
     * Assert that CUDA returned successful
     * @param code
     * @param file
     * @param line
     * @param abort
     */
    inline void gpuAssert(cudaError_t code, const char *file, int line,
                          bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                    line);
            if (abort)
                exit(code);
        }
    }

    /**
     * Allocates memory in a page set by page_size.
     */
    class InPageAllocator {
    public:
        InPageAllocator() = delete;

        /**
         * Create the in page allocator. Not thread safe w.r.t. operation in the class.
         */
        InPageAllocator(std::size_t page_size)
                : PAGE_SIZE(page_size) {
            gpuAssert(cudaMallocManaged((void **) &mem, PAGE_SIZE), __FILE__, __LINE__);
            l = ListAllocator(mem, PAGE_SIZE);
        }

        /**
         * Deletes in page allocator, cuda context must exist to do so. Not thread safe w.r.t. operation in the class.
         */
        ~InPageAllocator() { gpuAssert(cudaFree(mem), __FILE__, __LINE__); }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s. Thread safe. Blocking.
         * @param ptr
         * @param s
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
#ifdef DEBUGALLOC
            std::clog << "Allocating in IPA " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
            m.lock();
            l.alloc(ptr, s, forceAligned128);
            m.unlock();
        }

        /**
         * Frees memory at ptr. Thread safe. Blocking.
         * @tparam T
         * @param ptr
         */
        template<class T>
        void free(T *ptr) {
            m.lock();
            l.free(ptr);
            m.unlock();
        }

        /**
         * Checks if ptr is contained in the page. Thread safe.
         * @param ptr
         * @return
         */
        bool contains(size_t ptr) {
            return ptr >= (size_t) mem && ptr < (size_t) mem + PAGE_SIZE;
        }

        /**
         * Moves page to device. Thread safe.
         * @param device
         * @param stream
         */
        void moveToDevice(int device, cudaStream_t stream) {
            gpuAssert(cudaMemPrefetchAsync(mem, PAGE_SIZE, device, stream), __FILE__, __LINE__);
        }

        /**
         * Gets the number of pages. Thread safe.
         * @return
         */
        size_t getPages() { return 1; }

        /**
         * Gets the size of the pages. Thread safe.
         * @return
         */
        size_t getPageSize() { return PAGE_SIZE; }

    private:
        char *mem;
        ListAllocator l;
        std::mutex m;
        const size_t PAGE_SIZE;
    };

    /**
     * Allocates multiple pages of memory.
     */
    class MultiPageAllocator {
    public:
        MultiPageAllocator() = delete;

        /**
         * Constructor. Not thread safe w.r.t. operation in the class.
         */
        MultiPageAllocator(std::size_t page_size)
                : PAGE_SIZE(page_size), pagesAllocated(0) {}

        /**
         * Delete function. Not thread safe w.r.t. operation in the class.
         */
        ~MultiPageAllocator() {
            m.lock();
            for (auto &e : mem) {
                gpuAssert(cudaFree((void *) e.first), __FILE__, __LINE__);
            }
            m.unlock();
        }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s. Thread safe. Blocking.
         * @param ptr
         * @param s
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
#ifdef DEBUGALLOC
            std::clog << "Allocating in MPA " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
            size_t pages_needed = (size_t) ceil(s / (double) PAGE_SIZE);
            char *c;
            gpuAssert(cudaMallocManaged((void **) &c, pages_needed * PAGE_SIZE), __FILE__, __LINE__);
            *ptr = (T *) c;
            m.lock();
            pagesAllocated += pages_needed;
            mem.push_back({c, pages_needed * PAGE_SIZE});
            m.unlock();
        }

        /**
         * Frees ptr. Thread safe. Blocking.
         * @tparam T
         * @param ptr
         */
        template<class T>
        void free(T *ptr) {
            m.lock();
            for (auto i = mem.begin(); i != mem.end(); i++) {
                if ((size_t) i->first == (size_t) ptr) {
                    gpuAssert(cudaFree((void *) i->first), __FILE__, __LINE__);
                    mem.erase(i);
                    break;
                }
            }
            m.unlock();
        }

        /**
         * Moves pages to device. Thread safe. Blocking.
         * @param device
         * @param stream
         */
        void moveToDevice(int device, cudaStream_t stream) {
            m.lock();
            for (auto i = mem.begin(); i != mem.end(); i++) {
                gpuAssert(cudaMemPrefetchAsync(i->first, i->second, device, stream), __FILE__, __LINE__);
            }
            m.unlock();
        }

        /**
         * Gets the number of pages. Thread safe. Blocking.
         * @return
         */
        size_t getPages() {
            std::unique_lock<std::mutex> ul(m);
            return pagesAllocated;
        }

        /**
         * Gets the size of pages. Thread safe.
         * @return
         */
        size_t getPageSize() { return PAGE_SIZE; }

    private:
        std::list<std::pair<char *, size_t>> mem;
        std::mutex m;
        const size_t PAGE_SIZE;
        size_t pagesAllocated;
    };

    /**
     * Allocates with group affinity
     */
    class GroupAllocator {
    public:

        GroupAllocator() = delete;

        /**
         * Constructor takes group_num to allocate to. Not thread safe.
         * @param group_num
         */
        GroupAllocator(int group_num, std::size_t page_size)
                : group_num_(group_num),
                  PAGE_SIZE(page_size) {
            mpa = new MultiPageAllocator(page_size);
        }

        /**
         * Delete function. Its a no-op.
         */
        ~GroupAllocator() {}

        /**
         * Function to free all memory of group allocator. Thread safe.
         */
        void freeall() {
            m.lock();
            for (auto &e : ipas) {
                delete e;
            }
            ipas.clear();
            delete mpa;
            mpa = nullptr;
            m.unlock();
        }


        /**
         * Free pointer T* ptr. Thread safe.
         * @tparam T
         * @param ptr
         */
        template<class T>
        void free(T *ptr) {
            mpa->free(ptr);
            m.lock();
            for (auto &e : ipas) {
                if (e->contains((size_t) ptr)) {
                    e->free(ptr);
                    break;
                }
            }
            m.unlock();
        }

        /**
         * Allocates memory of type T and sets *ptr to this memory of size s. Thread safe.
         * @tparam T
         * @param ptr
         * @param s
         * @param forceAligned128
         */
        template<class T>
        void allocate(T **ptr, size_t s, bool forceAligned128) {
            if (ptr == NULL || s == 0) {
                return;
            }

            if (s + alignof(T *) > PAGE_SIZE) {
                mpa->allocate<T>(ptr, s, forceAligned128);
            } else {
                m.lock();
                int lastSize = ipas.size();
                if (lastSize == 0) {
                    InPageAllocator *ipa_new =
                            new InPageAllocator(PAGE_SIZE);
                    ipas.push_back(ipa_new);
                }
                auto ipa3 = ipas[ipas.size() - 1];
                m.unlock();
                ipa3->allocate<T>(ptr, s, forceAligned128);
                while (*ptr == NULL) {
                    InPageAllocator *ipa2 =
                            new InPageAllocator(PAGE_SIZE);
                    m.lock();
                    if (lastSize == ipas.size()) {
                        ipas.push_back(ipa2);
                        lastSize = ipas.size();
                    }
                    m.unlock();
                    m.lock();
                    auto ipa = ipas[ipas.size() - 1];
                    m.unlock();
                    ipa->allocate<T>(ptr, s, forceAligned128);
                }
            }
        }

        /**
         * Move to memory device in stream. Thread safe.
         * @param device
         * @param stream
         */
        void moveToDevice(int device, cudaStream_t stream) {
            mpa->moveToDevice(device, stream);
            m.lock();
            for (auto &e : ipas) {
                e->moveToDevice(device, stream);
            }
            m.unlock();
        }

        /**
         * Gets the pages allocated. Thread safe.
         * @return
         */
        size_t pagesAllocated() {
            auto s = mpa->getPages();
            m.lock();
            s += ipas.size();
            m.unlock();
            return s;
        }

        /**
         * Gets the page size. Thread safe.
         * @return
         */
        size_t getPageSize() {
            return PAGE_SIZE;
        }

    private:

        std::vector<InPageAllocator *> ipas;
        MultiPageAllocator *mpa;
        std::mutex m;
        int group_num_;
        const size_t PAGE_SIZE;
    };

}  // namespace groupallocator


#endif //GPUHASHMAPS_GROUPALLOCATOR_CLASSES_CUH
