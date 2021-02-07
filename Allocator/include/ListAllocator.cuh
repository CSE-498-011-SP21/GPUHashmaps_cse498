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

#include <iostream>
#include <cassert>

#ifndef GPUHASHMAPS_LISTALLOCATOR_CUH
#define GPUHASHMAPS_LISTALLOCATOR_CUH

namespace groupallocator {

    /**
     * Gets padding of a type
     * @param startingptr
     * @param alignment
     * @return
     */
    size_t getPadding(size_t startingptr, size_t alignment) {
        size_t multiplier = startingptr / alignment + 1;
        size_t padding = multiplier * alignment - startingptr;
        return padding;
    }


    /**
     * Not thread safe. There is no compaction.
     */
    class ListAllocator {
    private:

        // when allocating
        // write pointer to next
        // then have the data

        struct MallocData {
            size_t size;
            size_t used;
            void *start;
        };

    public:

        /**
         * Creates a list allocator managing the memory of size s starting at p.
         * @param p
         * @param s
         */
        ListAllocator(void *p, size_t s) : ptr(p), size(s) {
            // needs to maintain p to p + s
            l.push_back({s, 0, p});
        }

        /**
         * Empty list allocator.
         */
        ListAllocator() : ptr(nullptr), size(0) {}

        /**
         * Allocates data in a free area or sets p to nullptr.
         * @tparam T
         * @param p
         * @param s
         * @param forceAligned128
         */
        template<typename T>
        void alloc(T **p, size_t s, bool forceAligned128) {
            size_t alignment = forceAligned128 ? 128 : std::alignment_of<T *>();

            for (auto iter = l.begin(); iter != l.end(); ++iter) {

                if (iter->used == 0 && getPadding((size_t) iter->start, alignment) + s <= iter->size) {

                    *p = (T *) iter->start;

                    size_t prevSize = iter->size;
                    void *prevStart = iter->start;

                    iter->size = s + getPadding((size_t) iter->start, alignment);
                    iter->used = 1;

                    MallocData m = {prevSize - iter->size, 0, (void *) ((size_t) prevStart + iter->size)};
                    iter++;
                    l.insert(iter, m);
                    return;
                }
            }

            *p = nullptr;
        }

        /**
         * Frees the ptr p with no compaction.
         * @tparam T
         * @param p
         */
        template<typename T>
        void free(T *p) {
            for (auto &iter : l) {
                if ((size_t) iter.start == (size_t) p) {
                    iter.used = 0;
                    return;
                }
            }
        }

    private:
        void *ptr;
        size_t size;
        std::list<MallocData> l;

    };
}  // namespace groupallocator


#endif //GPUHASHMAPS_LISTALLOCATOR_CUH
