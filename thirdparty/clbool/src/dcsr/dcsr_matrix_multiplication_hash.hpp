#pragma once

#include <program.hpp>

struct BinsQueuesAndPrograms;

void build_groups_and_allocate_hash(Controls &controls,
                                    std::vector<cpu_buffer> &cpu_workload_groups,
                                    cl::Buffer &nnz_estimation,
                                    const matrix_dcsr &a,
                                    cl::Buffer &global_hash_tables,
                                    cl::Buffer &global_hash_tables_offset
);

void count_nnz(Controls &controls,
               BinsQueuesAndPrograms &queuesAndPrograms,
               const cpu_buffer &groups_length,
               const cpu_buffer &groups_pointers,

               const cl::Buffer &gpu_workload_groups,
               cl::Buffer &nnz_estimation,

               const matrix_dcsr &a,
               const matrix_dcsr &b,

               cl::Buffer &global_hash_tables,
               const cl::Buffer &global_hash_tables_offset

);

void fill_nnz(Controls &controls,

              BinsQueuesAndPrograms &queuesAndPrograms,
              const cpu_buffer &groups_length,
              const cpu_buffer &groups_pointers,

              const cl::Buffer &gpu_workload_groups,
              cl::Buffer &pre_matrix_rows_pointers,

              matrix_dcsr &c,
              const matrix_dcsr &a,
              const matrix_dcsr &b,

              const cl::Buffer &global_hash_tables,
              cl::Buffer &global_hash_tables_offset
);

void matrix_multiplication_hash(Controls &controls,
                                matrix_dcsr &matrix_out,
                                const matrix_dcsr &a,
                                const matrix_dcsr &b);

struct BinsQueuesAndPrograms {
    cl::CommandQueue hash_tb_512_queue;
    cl::CommandQueue hash_tb_1024_queue;
    cl::CommandQueue hash_global_queue;

    using program_tb_t = program<cl::Buffer, uint32_t, cl::Buffer,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            uint32_t, uint32_t>;

    using program_gl_symb_t = program<cl::Buffer, uint32_t, cl::Buffer,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            uint32_t, cl::Buffer, cl::Buffer>;

    using program_gl_num_t = program<cl::Buffer, uint32_t,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>;

    program_tb_t hash_tb_512;
    program_tb_t hash_tb_1024;
    program_gl_symb_t hash_global_symbolic;
    program_gl_num_t hash_global_numeric;

    BinsQueuesAndPrograms(Controls &controls) {
        hash_tb_512_queue = cl::CommandQueue(controls.context);
        hash_tb_1024_queue = cl::CommandQueue(controls.context);
        hash_global_queue = cl::CommandQueue(controls.context);

        hash_tb_512 = program_tb_t("hash_tb_512")
                .set_kernel_name("hash_tb_512")
                .set_queue(hash_tb_512_queue);
        hash_tb_1024 = program_tb_t("hash_tb_1024")
                .set_kernel_name("hash_tb_1024")
                .set_queue(hash_tb_1024_queue);
        hash_global_symbolic = program_gl_symb_t
                ("hash_symbolic_global")
                .set_kernel_name("hash_symbolic_global")
                .set_queue(hash_global_queue);
        hash_global_numeric = program_gl_num_t
                ("hash_numeric_global")
                .set_kernel_name("hash_numeric_global")
                .set_queue(hash_global_queue);
    }

    auto &get_program(uint32_t bin_id) {
        if (bin_id == 0) return hash_tb_512;
        if (bin_id == 1) return hash_tb_1024;
        throw std::runtime_error("Unknown bin id. error 212421");
    }

};