/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>

#include <omp.h>

#include <gridtools/stencil_composition/backend_mc/tmp_storage_sid.hpp>
#include <gridtools/stencil_composition/extent.hpp>
#include <gridtools/stencil_composition/sid/concept.hpp>

using namespace gridtools;
using namespace gridtools::literals;
using namespace gridtools::mc;

static constexpr std::size_t byte_alignment = 64;

TEST(tmp_storage_sid_mc, allocator) {
    mc::tmp_allocator_mc allocator;

    std::size_t n = 100;
    auto ptr_holder = allocate(allocator, meta::lazy::id<double>(), n);

    double *ptr = ptr_holder();
    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % byte_alignment, 0);

    for (std::size_t i = 0; i < n; ++i) {
        ptr[i] = 0;
        EXPECT_EQ(ptr[i], 0);
    }
}

TEST(tmp_storage_sid_mc, nonzero_k_extents) {
    using extent_t = extent<-1, 2, -2, 3, -1, 2>;
    pos3<std::size_t> block_size{12, 2, 8};

    tmp_allocator_mc allocator;
    auto tmp = make_tmp_storage_mc<double, extent_t, false>(allocator, block_size);

    using tmp_t = decltype(tmp);

    static_assert(is_sid<tmp_t>(), "");
    static_assert(std::is_same<sid::ptr_type<tmp_t>, double *>(), "");

    auto f = [](int_t i, int_t j, int_t k, int_t t) { return i + j * 100 + k * 200 + t * 400; };

    // check write and read
#pragma omp parallel
    {
        const int_t thread = omp_get_thread_num();
        auto strides = sid::get_strides(tmp);

        double *ptr = sid::get_origin(tmp)();
        // shift to origin of thread
        sid::shift(ptr, sid::get_stride<dim::thread>(strides), thread);

        // shift to very first data point in temporary
        sid::shift(ptr, sid::get_stride<dim::i>(strides), extent_t::iminus::value);
        sid::shift(ptr, sid::get_stride<dim::j>(strides), extent_t::jminus::value);
        sid::shift(ptr, sid::get_stride<dim::k>(strides), extent_t::kminus::value);

        const int_t size_i = block_size.i - extent_t::iminus::value + extent_t::iplus::value;
        const int_t size_j = block_size.j - extent_t::jminus::value + extent_t::jplus::value;
        const int_t size_k = block_size.k - extent_t::kminus::value + extent_t::kplus::value;
        for (int_t j = 0; j < size_j; ++j) {
            for (int_t k = 0; k < size_k; ++k) {
                for (int_t i = 0; i < size_i; ++i) {
                    // check alignment of first data point inside domain
                    if (i == -extent_t::iminus::value) {
                        EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % byte_alignment, 0);
                    }
                    *ptr = f(i, j, k, thread);
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
                }
                sid::shift(ptr, sid::get_stride<dim::i>(strides), -size_i);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), 1_c);
            }
            sid::shift(ptr, sid::get_stride<dim::k>(strides), -size_k);
            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
        }
        sid::shift(ptr, sid::get_stride<dim::j>(strides), -size_j);

#pragma omp barrier

        for (int_t j = 0; j < size_j; ++j) {
            for (int_t k = 0; k < size_k; ++k) {
                for (int_t i = 0; i < size_i; ++i) {
                    // check that previously expected value was not overwritten
                    EXPECT_EQ(*ptr, f(i, j, k, thread));
                    sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
                }
                sid::shift(ptr, sid::get_stride<dim::i>(strides), -size_i);
                sid::shift(ptr, sid::get_stride<dim::k>(strides), 1_c);
            }
            sid::shift(ptr, sid::get_stride<dim::k>(strides), -size_k);
            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
        }
    }
}

TEST(tmp_storage_sid_mc, zero_k_extents) {
    using extent_t = extent<-1, 2, -2, 3, 0, 0>;
    pos3<std::size_t> block_size{12, 5, 1};

    tmp_allocator_mc allocator;
    auto tmp = make_tmp_storage_mc<double, extent_t, true>(allocator, block_size);

    using tmp_t = decltype(tmp);

    static_assert(is_sid<tmp_t>(), "");
    static_assert(std::is_same<sid::ptr_type<tmp_t>, double *>(), "");

    auto f = [](int_t i, int_t j, int_t t) { return i + j * 100 + t * 200; };

    // check write and read
#pragma omp parallel
    {
        const int_t thread = omp_get_thread_num();
        auto strides = sid::get_strides(tmp);

        double *ptr = sid::get_origin(tmp)();
        // shift to origin of thread
        sid::shift(ptr, sid::get_stride<dim::thread>(strides), thread);

        // shift to very first data point in temporary
        sid::shift(ptr, sid::get_stride<dim::i>(strides), extent_t::iminus::value);
        sid::shift(ptr, sid::get_stride<dim::j>(strides), extent_t::jminus::value);
        sid::shift(ptr, sid::get_stride<dim::k>(strides), extent_t::kminus::value);

        const int_t size_i = block_size.i - extent_t::iminus::value + extent_t::iplus::value;
        const int_t size_j = block_size.j - extent_t::jminus::value + extent_t::jplus::value;
        for (int_t j = 0; j < size_j; ++j) {
            for (int_t i = 0; i < size_i; ++i) {
                // check alignment of first data point inside domain
                if (i == -extent_t::iminus::value) {
                    EXPECT_EQ(reinterpret_cast<std::uintptr_t>(ptr) % byte_alignment, 0);
                }
                *ptr = f(i, j, thread);
                sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
            }
            sid::shift(ptr, sid::get_stride<dim::i>(strides), -size_i);
            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
        }
        sid::shift(ptr, sid::get_stride<dim::j>(strides), -size_j);

#pragma omp barrier

        for (int_t j = 0; j < size_j; ++j) {
            for (int_t i = 0; i < size_i; ++i) {
                // check that previously expected value was not overwritten
                EXPECT_EQ(*ptr, f(i, j, thread));
                sid::shift(ptr, sid::get_stride<dim::i>(strides), 1_c);
            }
            sid::shift(ptr, sid::get_stride<dim::i>(strides), -size_i);
            sid::shift(ptr, sid::get_stride<dim::j>(strides), 1_c);
        }
    }
}
