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

#include <cstddef>
#include <type_traits>

#include "../common/array.hpp"
#include "../fn/unstructured.hpp"
#include "../sid/concept.hpp"
#include "../sid/as_const.hpp"

namespace gridtools::fn::sid_neighbor_table {
    namespace sid_neighbor_table_impl_ {
        template <class IndexDimension,
            class NeighborDimension,
            std::size_t MaxNumNeighbors,
            class PtrHolder,
            class Strides>
        struct sid_neighbor_table {
            PtrHolder origin;
            Strides strides;
        };

        // #ifdef __CUDA_ARCH__
        // __device__ static gridtools::array<int, 4> prev_neighbors;
        // __device__ static int previous_index = -1;
        // #else
        // static gridtools::array<int, 4> prev_neighbors;
        // static int previous_index = -1;
        // #endif

        template <class IndexDimension,
            class NeighborDimension,
            std::size_t MaxNumNeighbors,
            class PtrHolder,
            class Strides>
        GT_FUNCTION constexpr const gridtools::array<int, MaxNumNeighbors>& neighbor_table_neighbors(
            sid_neighbor_table<IndexDimension, NeighborDimension, MaxNumNeighbors, PtrHolder, Strides> const &table,
            int index) {

            // if (index == previous_index) {
            //     return prev_neighbors;
            // }

            using namespace gridtools::literals;

            auto ptr = table.origin();
            using element_type = std::decay_t<decltype(*ptr)>;

            sid::shift(ptr, sid::get_stride<IndexDimension>(table.strides), index);
            const gridtools::array<element_type, MaxNumNeighbors> neighbors = [&]<std::size_t... Is>(std::index_sequence<Is...>) {
                return gridtools::array<element_type, MaxNumNeighbors>{ptr[Is * sid::get_stride<NeighborDimension>(table.strides)]...};
            }(std::make_index_sequence<MaxNumNeighbors>{});

            // prev_neighbors = neighbors;
            // previous_index = index;

            // for (std::size_t element_idx = 0; element_idx < MaxNumNeighbors; ++element_idx) {
            //     // #ifdef __CUDA_ARCH__
            //     // neighbors[element_idx] = __ldg(ptr);
            //     // #else
            //     neighbors[element_idx] = *ptr;
            //     // #endif
            //     sid::shift(ptr, sid::get_stride<NeighborDimension>(table.strides), 1_c);
            // }
            return neighbors;
        }

        template <class IndexDimension, class NeighborDimension, std::size_t MaxNumNeighbors, class Sid>
        auto as_neighbor_table(Sid &&sid) {

            static_assert(gridtools::tuple_util::size<decltype(sid::get_strides(std::declval<Sid>()))>::value == 2,
                "Neighbor tables must have exactly two dimensions: the index dimension and the neighbor dimension");
            static_assert(!std::is_same_v<IndexDimension, NeighborDimension>,
                "The index dimension and the neighbor dimension must be different.");

            const auto c_sid = sid::as_const(std::forward<Sid>(sid));
            const auto origin = sid::get_origin(c_sid);
            const auto strides = sid::get_strides(c_sid);

            return sid_neighbor_table<IndexDimension,
                NeighborDimension,
                MaxNumNeighbors,
                sid::ptr_holder_type<decltype(c_sid)>,
                sid::strides_type<decltype(c_sid)>>{
                origin, strides}; // Note: putting the return type into the function signature will crash nvcc 12.0
        }
    } // namespace sid_neighbor_table_impl_

    using sid_neighbor_table_impl_::as_neighbor_table;

} // namespace gridtools::fn::sid_neighbor_table
