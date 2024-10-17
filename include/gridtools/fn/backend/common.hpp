/*
 * GridTools
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "../../common/hymap.hpp"
#include "../../common/tuple_util.hpp"
#include "../../meta/rename.hpp"
#include "../../sid/concept.hpp"
#include "../../sid/loop.hpp"

namespace gridtools::fn::backend {

    namespace common {

        template <class Dims, class Sizes>
        constexpr GT_FUNCTION auto make_loops(Sizes const &sizes) {
            return tuple_util::host_device::fold(
                [&](auto outer, auto dim) {
                    return [outer = std::move(outer),
                               inner = sid::make_loop<decltype(dim)>(host_device::at_key<decltype(dim)>(sizes))](
                               auto &&...args) { return outer(inner(std::forward<decltype(args)>(args)...)); };
                },
                host_device::identity(),
                meta::rename<tuple, Dims>());
        }

        template <class Dims, class Sizes, class UnrollFactors>
        constexpr GT_FUNCTION auto make_unrolled_loops(Sizes const &sizes, UnrollFactors) {
            return tuple_util::host_device::fold(
                [&](auto outer, auto dim) {
                    using unroll_factor = element_at<decltype(dim), UnrollFactors>;
                    return [outer = std::move(outer),
                               inner = sid::make_unrolled_loop<decltype(dim), 1>(
                                   host_device::at_key<decltype(dim)>(sizes))](
                               auto &&...args) { return outer(inner(std::forward<decltype(args)>(args)...)); };
                },
                host_device::identity(),
                meta::rename<tuple, Dims>());
        }

        template <class Sizes>
        constexpr GT_FUNCTION auto make_loops(Sizes const &sizes) {
            return make_loops<get_keys<Sizes>>(sizes);
        }

        template <class Sizes, class UnrollFactors>
        constexpr GT_FUNCTION auto make_unrolled_loops(Sizes const &sizes, UnrollFactors unroll_factors) {
            return make_unrolled_loops<get_keys<Sizes>>(sizes, unroll_factors);
        }
    } // namespace common

    template <class T>
    struct data_type {};

} // namespace gridtools::fn::backend
