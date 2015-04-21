#pragma once
#include <gt_for_each/for_each.hpp>
#include "../backend_traits_fwd.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/**@file
@brief type definitions and structures specific for the Host backend
*/
namespace gridtools{
    namespace _impl_host{
        /**forward declaration*/
        template <typename Arguments>
        struct run_functor_host;
    }

    namespace multithreading{
        /**
           Global variable storing the current thread id
        */
#ifdef _OPENMP
        int __attribute__((weak)) gt_thread_id;
#pragma omp threadprivate(gt_thread_id)
#else
        static const int gt_thread_id=1;
#endif
    }//namespace multithreading

    /**forward declaration*/
    template<typename T>
    struct wrap_pointer;

    /**Traits struct, containing the types which are specific for the host backend*/
    template<>
    struct backend_traits_from_id<enumtype::Host>{

        /**
           @brief pointer type associated to the host backend
         */
        template <typename T>
        struct pointer
        {
            typedef wrap_pointer<T> type;
        };

        /**
           @brief storage type associated to the host backend
         */
        template <typename ValueType, typename Layout, bool Temp=false, short_t FieldDim=1>
        struct storage_traits{
            typedef storage<base_storage<typename pointer<ValueType>::type, Layout, Temp, FieldDim > >   storage_t;
        };

        template <typename Arguments>
        struct execute_traits{
            typedef _impl_host::run_functor_host< Arguments > run_functor_t;
        };

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the i-direction, in a 2D
            grid of threads.
        */
        static uint_t n_i_pes(int = 0) {
            return n_threads();
        }

        /** This is the function used by the specific backend to inform the
            generic backend and the temporary storage allocator how to
            compute the number of threads in the j-direction, in a 2D
            grid of threads.
        */
        static uint_t n_j_pes(int = 0) {
            return 1;
        }

        /** This is the function used by the specific backend
         *  that determines the i coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t const& processing_element_i()  {
            return multithreading::gt_thread_id;
        }

        /**@brief set the thread id

           this method when openmp is enabled calls the thread_id() routine.
           NOTE: the reason why the latter routine is not called directly every time
           the thread id is queried is a possibly non-negligible overhead introduced by such call.
           This interface allows to store the thread id in a global variable when a parallel
           region is encountered.
         */
        static void set_thread_id(){
#ifdef _OPENMP
            multithreading::gt_thread_id=thread_id();
#endif
        }
        /** This is the function used by the specific backend
         *  that determines the j coordinate of a processing element.
         *  In the case of the host, a processing element is equivalent to an OpenMP core
         */
        static uint_t  processing_element_j()  {
            return 0;
        }

        //function alias (pre C++11, std::bind or std::mem_fn,
        //using function pointers looks very ugly)
        template<
            typename Sequence
            , typename F
            >

#ifdef CXX11_ENABLED
        //unnecessary copies/indirections if the compiler is not smart
        inline static void for_each(F&& f){
            gridtools::for_each<Sequence>(std::forward<F>(f));
            }
#else
        //unnecessary copies/indirections if the compiler is not smart
        inline static void for_each(F f){
            gridtools::for_each<Sequence>(f);
            }
#endif

        template <uint_t Id>
        struct once_per_block {
            template<typename Left, typename Right>
            GT_FUNCTION//inline
            static void assign(Left& l, Right const& r){
                l=r;
            }
        };

    };

}//namespace gridtools