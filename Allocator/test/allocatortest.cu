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
#include <functional>
#include <groupallocator>
#include "gtest/gtest.h"

TEST(GroupAllocator_test, TestForACrash) {
    groupallocator::Context ctx;

    char *s;
    char *t;

    groupallocator::allocate(&s, 40, ctx);

    for (unsigned long long i = 0; i < 39; i++) {
        s[i] = 'a';
    }
    s[39] = '\0';

    groupallocator::allocate(&t, 40, ctx);

    for (unsigned long long i = 0; i < 39; i++) {
        t[i] = 'b';
    }
    t[39] = '\0';

    groupallocator::free(t);

    groupallocator::freeall();
}

TEST(GroupAllocator_test, TooSmallAllocateInMPA) {
    ASSERT_TRUE(sizeof(int) + alignof(int *) > sizeof(int *));

    groupallocator::GroupAllocator g(0, sizeof(int));
    for (int i = 0; i < 10000; i++) {
        int *j;
        g.allocate(&j, sizeof(int), false);
        ASSERT_FALSE(j == nullptr);
        *j = 1;
        ASSERT_TRUE(g.pagesAllocated() == i + 1) << g.pagesAllocated() << " == " << i + 1;
    }
    g.freeall();
}

TEST(GroupAllocator_test, RepeatAllocateInIPA) {
    ASSERT_FALSE(sizeof(int) + alignof(int *) > 2 * sizeof(int *));
    groupallocator::GroupAllocator g(0, 2 * sizeof(int *));
    for (int i = 0; i < 10000; i++) {
        int *j;
        g.allocate(&j, sizeof(int), false);
        ASSERT_FALSE(j == nullptr);
        *j = 1;
    }
    g.freeall();
}

TEST(GroupAllocator_test, RepeatAllocatePtrToPtrInIPA) {
    groupallocator::GroupAllocator g(0, 128);
    for (int i = 0; i < 10000; i++) {
        int **j;
        g.allocate(&j, sizeof(int *), false);
        ASSERT_FALSE(j == nullptr);
        g.allocate(&j[0], sizeof(int), false);
    }
    g.freeall();
}

TEST(GroupAllocator_test, IPADoesntAllocateSamePtr) {
    groupallocator::GroupAllocator g(0, 128);
    for (int i = 0; i < 10000; i++) {
        int *j, *k;
        g.allocate(&j, sizeof(int), false);
        ASSERT_FALSE(j == nullptr);
        g.allocate(&k, sizeof(int), false);
        ASSERT_FALSE(k == nullptr);
        ASSERT_FALSE(j == k);
    }
    g.freeall();
}
