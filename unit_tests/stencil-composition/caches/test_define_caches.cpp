/**
   @file
   @brief File containing tests for the define_cache construct
*/


#include "gtest/gtest.h"
#include <boost/mpl/equal.hpp>
#include <stencil-composition/caches/define_caches.hpp>
#include <stencil-composition/accessor.hpp>
#include <stencil-composition/backend.hpp>
#include <common/defs.hpp>

using namespace gridtools;
using namespace enumtype;

#ifdef __CUDACC__
  #define BACKEND backend<Cuda, Block >
#else
  #ifdef BACKEND_BLOCK
    #define BACKEND backend<Host, Block >
  #else
    #define BACKEND backend<Host, Naive >
  #endif
#endif

TEST(define_caches, define_mixed_caches)
{
#ifdef __CUDACC__
    typedef gridtools::layout_map<2,1,0> layout_t;//stride 1 on i
#else
    typedef gridtools::layout_map<0,1,2> layout_t;//stride 1 on k
#endif
    typedef gridtools::BACKEND::storage_type<float_type, layout_t >::type storage_type;

    typedef gridtools::arg<0,storage_type> arg0_t;
    typedef gridtools::arg<1,storage_type> arg1_t;
    typedef gridtools::arg<2,storage_type> arg2_t;

    typedef decltype(gridtools::define_caches(
        cache<IJ, arg0_t, cFill>(),
        cache<IJK, arg1_t, cFlush>(),
        cache<K, arg2_t, cLocal>()
    ) ) cache_sequence_t;

    GRIDTOOLS_STATIC_ASSERT((
        boost::mpl::equal<
            cache_sequence_t,
            boost::mpl::vector3<cache<IJ, arg0_t, cFill>, cache<IJK, arg1_t, cFlush>, cache<K, arg2_t, cLocal> >
        >::value),
        "Failed TEST"
    );

    ASSERT_TRUE(true);
}

