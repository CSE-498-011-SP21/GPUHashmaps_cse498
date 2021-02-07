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

#include "gpuErrchk.cuh"
#include <iostream>

#ifndef GPUMEMORY_CUH
#define GPUMEMORY_CUH

/**
 * Maintains a host to device maping of memory. There is minimal encapsulation to allow for many optimizations.
 * @tparam T
 */
template<typename T>
struct GPUCPUMemory {

    /**
     * Creates nullptrs.
     */
    GPUCPUMemory() : host(nullptr), size(0), device(nullptr) {
    }

    /**
     * Creates a host to device mapping of size.
     * @param size
     */
    GPUCPUMemory(size_t size) : GPUCPUMemory(new T[size], size) {}

    /**
     * Creates a host to device mapping of ptr h of size to device ptr of size.
     * h must be allocated with new[] and is owned by this class.
     * @param h
     * @param size
     */
    GPUCPUMemory(T *h, size_t size) : host(h), size(size), device(new T *[1]) {
        gpuErrchk(cudaMalloc(&device[0], sizeof(T) * size))
    }

    /**
     * Move constructor.
     * @param ref
     */
    GPUCPUMemory(GPUCPUMemory<T> &&ref) noexcept {
        host = ref.host;
        device = ref.device;
        size = ref.size;
        ref.host = nullptr;
        ref.device = nullptr;
    }

    /**
     * Destructor frees underlying memory.
     */
    ~GPUCPUMemory() {
        delete[] host;
        if (device != nullptr) {
            std::cerr << "Deleting memory\n";
            gpuErrchk(cudaFree(*device))
            delete[] device;
        }
    }

    /**
     * Move operator=
     * @param other
     * @return
     */
    GPUCPUMemory<T> &operator=(GPUCPUMemory<T> &&other) {
        if (&other != this) {
            delete[] host;
            if (device != nullptr) {
                gpuErrchk(cudaFree(*device))
                delete[] device;
            }
            host = other.host;
            device = other.device;
            size = other.size;
            other.host = nullptr;
            other.device = nullptr;
        }
        return *this;
    }

    /**
     * Moves data to GPU.
     */
    void movetoGPU() {
        gpuErrchk(
                cudaMemcpy(*device, host, sizeof(T) * size, cudaMemcpyHostToDevice))
    }

    /**
     * Moves data to CPU.
     */
    void movetoCPU() {
        gpuErrchk(
                cudaMemcpy(host, *device, sizeof(T) * size, cudaMemcpyDeviceToHost))
    }

    /**
     * Gets the device pointer.
     * @return
     */
    T *getDevice() {
        return *device;
    }

    T *host;
    size_t size;
private:
    T **device;
};

template<typename T>
struct GPUCPU2DArray {

    GPUCPU2DArray(size_t dim1, size_t dim2) : dim1(dim1), dim2(dim2), outer(dim1), inner(new GPUCPUMemory<T>[dim1]) {
        for (size_t i = 0; i < dim1; i++) {
            inner[i] = GPUCPUMemory<T>(dim2);
            outer.host[i] = inner[i].getDevice();
        }
    }

    ~GPUCPU2DArray() {
    }

    void movetoGPU() {
        for (int i = 0; i < dim1; i++) {
            inner[i].movetoGPU();
        }
        outer.movetoGPU();
    }

    void print() {
        for (int i = 0; i < dim1; i++) {
            std::cerr << outer.host[i] << std::endl;
            for (int j = 0; j < dim2; j++)
                std::cerr << "\t" << inner[i].host[j] << std::endl;
        }
    }


    void movetoCPU() {
        for (int i = 0; i < dim1; i++) {
            inner[i].movetoCPU();
        }
        outer.movetoCPU();
    }

    T **getDevice2DArray() {
        return outer.getDevice();
    }

    T *&operator[](int idx) {
        return inner[idx].host;
    }

    size_t dim1;
    size_t dim2;
    GPUCPUMemory<T *> outer;
    GPUCPUMemory<T> *inner;
};

#endif