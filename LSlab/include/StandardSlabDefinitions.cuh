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

#include <Operations.cuh>
#include <functional>

#ifndef LSLABEXT_STANDARDSLABS_CUH
#define LSLABEXT_STANDARDSLABS_CUH

struct data_t {

    data_t() : size(0), data(nullptr) {}

    data_t(size_t s) : size(s), data(new char[s]) {}

    /// Note this doesn't free the underlying data
    ~data_t() {}

    size_t size;
    char *data;

    data_t &operator=(const data_t &rhs) {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

    volatile data_t &operator=(const data_t &rhs) volatile {
        this->size = rhs.size;
        this->data = rhs.data;
        return *this;
    }

};

/// For use with shared_ptr
class Data_tDeleter{
    void operator()(data_t* ptr) const noexcept {
        delete[] ptr->data;
        delete ptr;
    }
};

template<>
struct EMPTY<data_t *> {
    static constexpr data_t *value = nullptr;
};

template<>
__forceinline__ __device__ unsigned compare(data_t *const &lhs, data_t *const &rhs) {

    if (lhs == rhs) {
        return 0;
    } else if (lhs == nullptr || rhs == nullptr) {
        return 1;
    }

    if (lhs->size != rhs->size) {
        return (unsigned) (lhs->size - rhs->size);
    }

    for (size_t i = 0; i < lhs->size; i++) {
        unsigned sub = lhs->data[i] - rhs->data[i];
        if (sub != 0)
            return sub;
    }

    return 0;
}

namespace std {
    template<>
    struct std::hash<data_t *> {
        std::size_t operator()(data_t *&x) {
            return std::hash<std::string>{}(x->data) ^ std::hash<std::size_t>{}(x->size);
        }
    };
}

template<>
struct EMPTY<unsigned> {
    static const unsigned value = 0;
};

template<>
__forceinline__ __device__ unsigned compare(const unsigned &lhs, const unsigned &rhs) {
    return lhs - rhs;
}

template<>
__forceinline__ __device__ unsigned compare(const unsigned long long &lhs, const unsigned long long &rhs) {
    return lhs - rhs;
}


#endif //LSLABEXT_STANDARDSLABS_CUH
