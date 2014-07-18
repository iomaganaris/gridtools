#pragma once

namespace gridtools {

    /**
       On all boundary the values are copied from the last data field to the first.
       Minimum 2 fields.

     */
    struct copy_boundary {

        template <typename Direction, typename DataField0, typename DataField1>
        GT_FUNCTION
        void operator()(Direction,
                        DataField0 & data_field0,
                        DataField1 const & data_field1,
                        int i, int j, int k) const {
            data_field0(i,j,k) = data_field1(i,j,k);
        }

        template <typename Direction, typename DataField0, typename DataField1, typename DataField2>
        GT_FUNCTION
        void operator()(Direction,
                        DataField0 & data_field0,
                        DataField1 & data_field1,
                        DataField2 const & data_field2,
                        int i, int j, int k) const {
            data_field0(i,j,k) = data_field2(i,j,k);
            data_field1(i,j,k) = data_field2(i,j,k);
        }
    };



} // namespace gridtools