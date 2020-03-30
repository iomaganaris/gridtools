/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gridtools/stencil_composition/core/level.hpp>

using namespace gridtools;
using namespace core;

namespace test_level_to_index {
    template <class L, class I>
    constexpr bool testee = std::is_same<typename level_to_index<L>::type, I>::value;

    static_assert(testee<level<0, -2, 2>, level_index<0, 2>>, "");
    static_assert(testee<level<0, -1, 2>, level_index<1, 2>>, "");
    static_assert(testee<level<0, 1, 2>, level_index<2, 2>>, "");
    static_assert(testee<level<0, 2, 2>, level_index<3, 2>>, "");
    static_assert(testee<level<1, -2, 2>, level_index<4, 2>>, "");
    static_assert(testee<level<1, -1, 2>, level_index<5, 2>>, "");
    static_assert(testee<level<1, 1, 2>, level_index<6, 2>>, "");
    static_assert(testee<level<1, 2, 2>, level_index<7, 2>>, "");
    static_assert(testee<level<2, -2, 2>, level_index<8, 2>>, "");
    static_assert(testee<level<2, -1, 2>, level_index<9, 2>>, "");
    static_assert(testee<level<2, 1, 2>, level_index<10, 2>>, "");
    static_assert(testee<level<2, 2, 2>, level_index<11, 2>>, "");
} // namespace test_level_to_index

namespace test_index_to_level {
    template <class L, class I>
    constexpr bool testee = std::is_same<L, typename index_to_level<I>::type>::value;

    static_assert(testee<level<0, -2, 2>, level_index<0, 2>>, "");
    static_assert(testee<level<0, -1, 2>, level_index<1, 2>>, "");
    static_assert(testee<level<0, 1, 2>, level_index<2, 2>>, "");
    static_assert(testee<level<0, 2, 2>, level_index<3, 2>>, "");
    static_assert(testee<level<1, -2, 2>, level_index<4, 2>>, "");
    static_assert(testee<level<1, -1, 2>, level_index<5, 2>>, "");
    static_assert(testee<level<1, 1, 2>, level_index<6, 2>>, "");
    static_assert(testee<level<1, 2, 2>, level_index<7, 2>>, "");
    static_assert(testee<level<2, -2, 2>, level_index<8, 2>>, "");
    static_assert(testee<level<2, -1, 2>, level_index<9, 2>>, "");
    static_assert(testee<level<2, 1, 2>, level_index<10, 2>>, "");
    static_assert(testee<level<2, 2, 2>, level_index<11, 2>>, "");
} // namespace test_index_to_level
