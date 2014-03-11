#pragma once 

#include <boost/mpl/minus.hpp>
#include <boost/mpl/plus.hpp>
#include "loopintervals.h"
#include "array.h"
#include "gpu_clone.h"

namespace gridtools {
    template <typename t_min_level, typename t_max_level>
    struct make_axis {
        typedef interval<t_min_level, t_max_level> type;
    };

    template <typename t_axis, int I>
    struct extend_by {
        typedef interval<level<t_axis::FromLevel::Splitter::value, t_axis::FromLevel::Offset::value - 1>,
                         level<t_axis::ToLevel::Splitter::value, t_axis::ToLevel::Offset::value + 1> > type;
    };

    template <typename t_axis>
    struct coordinates : public clonable_to_gpu<coordinates<t_axis> > {
        BOOST_STATIC_ASSERT(is_interval<t_axis>::value);

        typedef t_axis axis_type;

    typedef typename boost::mpl::plus<
        typename boost::mpl::minus<typename t_axis::ToLevel::Splitter, 
                                   typename t_axis::FromLevel::Splitter>,
        typename boost::mpl::int_<1> >::type size_type;
    
        gridtools::array<int, size_type::value > value_list;
    
        int _i_low_bound;
        int _i_high_bound;
        int _j_low_bound;
        int _j_high_bound;

        GT_FUNCTION
        explicit coordinates(int il, int ih, int jl, int jh)
            : _i_low_bound(il)
            , _i_high_bound(ih)
            , _j_low_bound(jl)
            , _j_high_bound(jh)
        {}
        
        GT_FUNCTION
        int i_low_bound() const {
            return _i_low_bound;
        }

        GT_FUNCTION
        int i_high_bound() const {
            return _i_high_bound;
        }

        GT_FUNCTION
        int j_low_bound() const {
            return _j_low_bound;
        }

        GT_FUNCTION
        int j_high_bound() const {
            return _j_high_bound;
        }

        template <typename t_level>
        GT_FUNCTION
        int value_at() const {
            BOOST_STATIC_ASSERT(is_level<t_level>::value);
            int offs = t_level::Offset::value;
            if (offs < 0) offs += 1;
            return value_list[t_level::Splitter::value] + offs;
        }

        template <typename t_level>
        GT_FUNCTION
        int& value_at(int val) const {
            BOOST_STATIC_ASSERT(is_level<t_level>::value);
            return value_list[t_level::Splitter::value];
        }

        GT_FUNCTION
        int value_at_top() const {
            return value_list[size_type::value - 1];
        }

        GT_FUNCTION
        int value_at_bottom() const {
            return value_list[0];
        }
    };
} // namespace gridtools
