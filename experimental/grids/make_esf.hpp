#pragma once

#include "common/generic_metafunctions/variadic_to_vector.hpp"
#include "mss_metafunctions.hpp"

namespace gridtools {

    template <typename Functor, typename Grid, typename LocationType, typename ... Args>
    esf_descriptor<Functor, Grid, LocationType, boost::mpl::vector<Args ...> >
    make_esf( Args&& ... /*args_*/){
        return esf_descriptor<Functor, Grid, LocationType, boost::mpl::vector<Args ...> >();
    }
} // namespace gridtools
