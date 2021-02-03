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

#include <Slab.cuh>
#include <vector>

int main() {

    SlabUnified *s = new SlabUnified(1000000);

    std::vector<unsigned> keys;
    std::vector<unsigned> values;
    std::vector<unsigned> requests;

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys.push_back((unsigned) (rand() % 100000));
        values.push_back(1);
        requests.push_back(REQUEST_INSERT);
    }

    auto res = s->batch_insert(keys.data(), values.data(), requests.data());

    std::cout << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << std::endl;

    keys = std::vector<unsigned>();
    values = std::vector<unsigned>();
    requests = std::vector<unsigned>();

    for (int i = 0; i < THREADS_PER_BLOCK * BLOCKS; i++) {
        keys.push_back((unsigned) (rand() % 100000));
        values.push_back(1);
        requests.push_back(REQUEST_GET);
    }

    res = s->batch_get(keys.data(), values.data(), requests.data());

    std::cout << std::get<0>(res) << " " << std::get<1>(res) << " " << std::get<2>(res) << std::endl;


    delete s;

    return 0;
}