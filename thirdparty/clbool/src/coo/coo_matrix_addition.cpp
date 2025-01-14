#include <program.hpp>
#include "coo_matrix_addition.hpp"

#include "merge_path.h"
#include "prepare_positions.h"
#include "set_positions.h"

void matrix_addition(Controls &controls,
                     matrix_coo &matrix_out,
                     const matrix_coo &a,
                     const matrix_coo &b) {

    cl::Buffer merged_rows;
    cl::Buffer merged_cols;
    uint32_t new_size;

    timer t;
    t.restart();
    merge(controls, merged_rows, merged_cols, a, b);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE)  Log() << "merge routine finished in " <<  t.last_elapsed();

    t.restart();
    reduce_duplicates(controls, merged_rows, merged_cols, new_size, a.nnz() + b.nnz());
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE)  Log() << "reduce_duplicates routine finished in " <<  t.last_elapsed();

    matrix_out = matrix_coo(a.nRows(), a.nCols(), new_size, merged_rows, merged_cols);
}


void merge(Controls &controls,
           cl::Buffer &merged_rows_out,
           cl::Buffer &merged_cols_out,
           const matrix_coo &a,
           const matrix_coo &b) {

    uint32_t merged_size = a.nnz() + b.nnz();

    auto coo_merge = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
                    uint32_t, uint32_t>
#ifndef FPGA
                    (merge_path_kernel, merge_path_kernel_length)
#else
                    ("merge_path")
#endif
                    .set_needed_work_size(merged_size)
                    .set_kernel_name("merge");


    cl::Buffer merged_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);
    cl::Buffer merged_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

    coo_merge.run(controls,
                 merged_rows, merged_cols,
                 a.rows_indices_gpu(), a.cols_indices_gpu(),
                 b.rows_indices_gpu(), b.cols_indices_gpu(),
                 a.nnz(), b.nnz());

//        check_merge_correctness(controls, merged_rows, merged_cols, merged_size);
    merged_rows_out = std::move(merged_rows);
    merged_cols_out = std::move(merged_cols);
}


void reduce_duplicates(Controls &controls,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       uint32_t &new_size,
                       uint32_t merged_size
) {
    // ------------------------------------ prepare array to count positions ----------------------

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * merged_size);

    timer t;

    t.restart();
    prepare_positions(controls, positions, merged_rows, merged_cols, merged_size);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "reduce_duplicates -> prepare_positions routine finished in " << t.last_elapsed();


    // ------------------------------------ calculate positions, get new_size -----------------------------------


    t.restart();
    prefix_sum(controls, positions, new_size, merged_size);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "reduce_duplicates -> prefix_sum routine finished in " << t.last_elapsed();


    cl::Buffer new_rows(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);
    cl::Buffer new_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * new_size);

    t.restart();
    set_positions(controls, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "reduce_duplicates -> set_positions routine finished in " << t.last_elapsed();


    merged_rows = std::move(new_rows);
    merged_cols = std::move(new_cols);
}


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       cl::Buffer &merged_rows,
                       cl::Buffer &merged_cols,
                       uint32_t merged_size
) {
    auto prepare_positions = program<cl::Buffer, cl::Buffer, cl::Buffer, uint32_t>
#ifndef FPGA
            (prepare_positions_kernel, prepare_positions_kernel_length)
#else
            ("prepare_positions")
#endif
            .set_needed_work_size(merged_size)
            .set_kernel_name("prepare_array_for_positions");

    prepare_positions.run(controls, positions, merged_rows, merged_cols, merged_size);
}


void set_positions(Controls &controls,
                   cl::Buffer &new_rows,
                   cl::Buffer &new_cols,
                   cl::Buffer &merged_rows,
                   cl::Buffer &merged_cols,
                   cl::Buffer &positions,
                   uint32_t merged_size) {
    auto set_positions = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, unsigned int>
#ifndef FPGA
            (set_positions_kernel, set_positions_kernel_length)
#else
            ("set_positions")
#endif
            .set_needed_work_size(merged_size)
            .set_kernel_name("set_positions");

    set_positions.run(controls, new_rows, new_cols, merged_rows, merged_cols, positions, merged_size);
}
