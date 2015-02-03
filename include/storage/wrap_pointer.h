#pragma once
#include <common/defs.h>

/**
@file
Write my documentation!
*/

/** This class wraps a raw pointer*/
namespace gridtools {

    namespace workaround_ {
        template <typename T>
        struct new_op;

#define NEW_OP(x) template <>                   \
        struct new_op<x> {                      \
	  GT_FUNCTION                           \
            x* operator()(/*u*/int_t size) const {  \
                return new x[size];             \
            }                                   \
        };

        NEW_OP(int)
        NEW_OP(unsigned int)
        NEW_OP(char)
        NEW_OP(float)
        NEW_OP(double)
    }


template <typename T>
struct wrap_pointer{
        typedef T pointee_t;

    //default constructor
    GT_FUNCTION
    wrap_pointer(bool managed=false)
        : m_cpu_p(NULL),
	  m_managed(managed)
        {
	}


    GT_FUNCTION
    wrap_pointer(wrap_pointer const& other)
        : m_cpu_p(other.m_cpu_p),
	  m_managed(false)
        {
	}

    GT_FUNCTION
    wrap_pointer(T* p, bool managed=true)
        : m_cpu_p(p)
        , m_managed(managed)
    {}

    wrap_pointer<T>& operator = (T& p)
        {
            m_cpu_p=p;
	    m_managed=true;
            return *this;
        }

    pointee_t* get() const {return m_cpu_p;}

    void reset(T* cpu_p){m_cpu_p=cpu_p;}

    bool managed(){return m_managed;}

  GT_FUNCTION
  virtual ~wrap_pointer(){
#ifndef NDEBUG
#ifndef __CUDACC__
      std::cout<<"deleting wrap pointer "<<this<<std::endl;
#endif
#endif
  }

    GT_FUNCTION
    void update_gpu() {}//\todo find a way to remove this method

    GT_FUNCTION
    wrap_pointer(uint_t size, bool managed=false): m_managed(managed) {
        allocate_it(size);
#ifndef NDEBUG
            printf(" - %X %d\n", m_cpu_p, size);
#endif
        }

    GT_FUNCTION
    void allocate_it(uint_t size){
/* #if (CUDA_VERSION > 5050) */
        m_cpu_p = new T[size];
/* #else */
/* 	m_cpu_p = workaround_::new_op<T>()(size); */
/* #endif */
    }

    GT_FUNCTION
    void free_it() {
	if(m_cpu_p)
	{
#ifndef NDEBUG
#ifndef __CUDACC__
	    std::cout<<"deleting data pointer "<<m_cpu_p<<std::endl;
#endif
#endif
	    delete [] m_cpu_p  ;
	    m_cpu_p=NULL;
	}
    }


    __host__ __device__
    operator T*() {
        return m_cpu_p;
    }

    __host__ __device__
    operator T const*() const {
        return m_cpu_p;
    }

    __host__ __device__
    T& operator[](uint_t i) {
	return m_cpu_p[i];
    }

    __host__ __device__
    T const& operator[](uint_t i) const {
	return m_cpu_p[i];
        }

    __host__ __device__
    T& operator*() {
        return *m_cpu_p;
    }

    __host__ __device__
    T const& operator*() const {
        return *m_cpu_p;
    }

    __host__ __device__
    T* operator+(uint_t i) {
	return &m_cpu_p[i];
    }

    __host__ __device__
    T* const& operator+(uint_t i) const {
	return &m_cpu_p[i];
    }

    GT_FUNCTION
    const T* get_m_cpu_p(){return m_cpu_p;};

protected:
    T * m_cpu_p;
    bool m_managed;


};
}//namespace gridtools