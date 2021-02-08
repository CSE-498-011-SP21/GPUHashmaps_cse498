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
 * @brief Has important definitions that are used in programming Slab.
 */

#ifndef GPUHASHMAPS_IMPORTANTDEFINITIONS_CUH
#define GPUHASHMAPS_IMPORTANTDEFINITIONS_CUH

/**
 * EMPTY<T>::value is used to denote an empty value in the map.
 * @tparam T
 */
template<typename T>
struct EMPTY {
    static constexpr T value{};
};

/**
 * compare is used to compare two objects to check if they are the same.
 * @tparam T
 * @param lhs
 * @param rhs
 * @return
 */
template<typename T>
__forceinline__ __host__ __device__ unsigned compare(const T &lhs, const T &rhs);

#endif //GPUHASHMAPS_IMPORTANTDEFINITIONS_CUH
