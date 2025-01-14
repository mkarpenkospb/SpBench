#include <numeric>
#include "dcsr_matrix_multiplication.hpp"
#include "dcsr_matrix_multiplication_hash.hpp"
#include "../coo/coo_matrix_addition.hpp"


const uint32_t BINS_NUM = 3;
const uint32_t MAX_GROUP_ID = BINS_NUM - 1;
#define PWARP 4

namespace hash_details {
    uint32_t get_block_size(uint32_t bin_id) {


#ifdef WIN
        // NOTE: NVIDIA can operate more than 256 threads per group, but AMD cannot
        if (bin_id == 0) return 256;
        if (bin_id == 1) return 256;
        if (bin_id == 2) return 256;
#else
        if (bin_id == 0) return 256;
        if (bin_id == 1) return 256;
        if (bin_id == 2) return 256;
#endif
        throw std::runtime_error("Unknown bin id. error 24642342152");
    }

    uint32_t get_group(uint32_t size) {
        if (size <= 512) return 0;
        if (size <= 1024) return 1;
        return 2;
    }
}

void matrix_multiplication_hash(Controls &controls,
                                matrix_dcsr &matrix_out,
                                const matrix_dcsr &a,
                                const matrix_dcsr &b) {
    if (a.nnz() == 0 || b.nnz() == 0) {
        std::cout << "empty result\n";
        return;
    }

    BinsQueuesAndPrograms queuesAndPrograms(controls);

    // TODO добавтиь rassert на размеры
    cl::Buffer nnz_estimation;
    timer t;
    t.restart();
    count_workload(controls, nnz_estimation, a, b);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "count_workload in " << t.last_elapsed();

    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM + 1);
    cpu_buffer groups_length(BINS_NUM);


    cl::Buffer global_hash_tables;
    cl::Buffer global_hash_tables_offset;

    t.restart();
    build_groups_and_allocate_hash(controls, cpu_workload_groups, nnz_estimation, a,
                                   global_hash_tables, global_hash_tables_offset);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "build_groups_and_allocate_hash in " << t.last_elapsed();


    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

    t.restart();
    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "write_bins_info in " << t.last_elapsed();

    t.restart();
    count_nnz(controls, queuesAndPrograms, groups_length, groups_pointers, gpu_workload_groups, nnz_estimation,
              a, b, global_hash_tables, global_hash_tables_offset);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "count_nnz in " << t.last_elapsed();

    t.restart();
    fill_nnz(controls, queuesAndPrograms, groups_length, groups_pointers, gpu_workload_groups, nnz_estimation,
             matrix_out, a, b, global_hash_tables, global_hash_tables_offset);
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "fill_nnz in " << t.last_elapsed();
}


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

) {
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "hay 11111!";

    std::vector<cl::Event> events;
    for (uint32_t bin_id = 0; bin_id < BINS_NUM; ++bin_id) {
        if (groups_length[bin_id] == 0) continue;

        uint32_t block_size = hash_details::get_block_size(bin_id);

        if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << " [count_nnz] group " << bin_id << ", size " << groups_length[bin_id];

        if (bin_id != MAX_GROUP_ID) {
            auto program = queuesAndPrograms.get_program(bin_id);
            program.set_block_size(block_size);
            program.set_needed_work_size(block_size * groups_length[bin_id]);
            events.push_back(program.run(controls, gpu_workload_groups, groups_pointers[bin_id],
                                         nnz_estimation, cl::Buffer(),
                                         a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                         b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                         b.nzr(), 0
            ));
            continue;
        }

        queuesAndPrograms.hash_global_symbolic.set_block_size(block_size);
        queuesAndPrograms.hash_global_symbolic.set_needed_work_size(block_size * groups_length[bin_id]);
        events.push_back(queuesAndPrograms.hash_global_symbolic.run(controls, gpu_workload_groups, groups_pointers[bin_id],
                                         nnz_estimation, a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                         b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                         b.nzr(),
                                         global_hash_tables, global_hash_tables_offset
        ));
    }

    try {
        cl::Event::waitForEvents(events);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << " in " << "run_kernels" << " \n";
        throw std::runtime_error(exception.str());
    }


}

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
) {
    uint32_t c_nnz;
    prefix_sum(controls, pre_matrix_rows_pointers, c_nnz, a.nzr() + 1);
    cl::Buffer c_cols(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nnz);


    std::vector<cl::Event> events;
    for (uint32_t bin_id = 0; bin_id < BINS_NUM; ++bin_id) {
        if (groups_length[bin_id] == 0) continue;

        uint32_t block_size = hash_details::get_block_size(bin_id);

        if (bin_id != MAX_GROUP_ID) {
            auto program = queuesAndPrograms.get_program(bin_id);
            program.set_block_size(block_size);
            program.set_needed_work_size(block_size * groups_length[bin_id]);
            events.push_back(program.run(controls, gpu_workload_groups, groups_pointers[bin_id],
                                         pre_matrix_rows_pointers, c_cols,
                                         a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                         b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                         b.nzr(), 1
            ));
            continue;
        }

        queuesAndPrograms.hash_global_numeric.set_block_size(block_size);
        queuesAndPrograms.hash_global_numeric.set_needed_work_size(block_size * groups_length[bin_id]);
        events.push_back(queuesAndPrograms.hash_global_numeric.run(controls, gpu_workload_groups, groups_pointers[bin_id],
                                         pre_matrix_rows_pointers, c_cols,
                                         global_hash_tables, global_hash_tables_offset
        ));
    }

    try {
        cl::Event::waitForEvents(events);
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << " in " << "run_kernels" << " \n";
        throw std::runtime_error(exception.str());
    }

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());
    prepare_positions(controls, positions, pre_matrix_rows_pointers, a.nzr(), "prepare_for_shift_empty_rows");

    uint32_t c_nzr;
    prefix_sum(controls, positions, c_nzr, a.nzr());

    cl::Buffer c_rpt = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (c_nzr + 1));
    cl::Buffer c_rows = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nzr);

    set_positions(controls, c_rpt, c_rows, pre_matrix_rows_pointers, a.rows_compressed_gpu(), positions,
                  c_nnz, a.nzr(), c_nzr);

    c = matrix_dcsr(c_rpt, c_rows, c_cols, a.nRows(), b.nCols(), c_nnz, c_nzr);
}

void build_groups_and_allocate_hash(Controls &controls,
                                    std::vector<cpu_buffer> &cpu_workload_groups,
                                    cl::Buffer &nnz_estimation,
                                    const matrix_dcsr &a,

                                    cl::Buffer &global_hash_tables,
                                    cl::Buffer &global_hash_tables_offset
) {

    cpu_buffer global_hash_tables_offset_cpu;
    uint32_t global_hash_mem_size = 0;

    cpu_buffer cpu_workload(a.nzr());
    //!!!!!!!!!!!! memory should be aligned
    controls.queue.enqueueReadBuffer(nnz_estimation, CL_TRUE, 0, sizeof(uint32_t) * a.nzr(), cpu_workload.data()
            /*, nullptr, &event*/);
    uint32_t pre_nnz;
    for (uint32_t i = 0; i < a.nzr(); ++i) {
        uint32_t current_workload = cpu_workload[i];
        uint32_t group = hash_details::get_group(current_workload);
        cpu_workload_groups[group].push_back(i);

        pre_nnz += current_workload;
        if (group == MAX_GROUP_ID) {
            global_hash_tables_offset_cpu.push_back(global_hash_mem_size);
            global_hash_mem_size += current_workload;
        }
    }

    if (pre_nnz == 0) {
        std::cout << "empty result\n";
        return;
    }

    global_hash_tables_offset_cpu.push_back(global_hash_mem_size);

    if (global_hash_mem_size != 0) {

        global_hash_tables_offset = cl::Buffer(controls.context, CL_MEM_READ_WRITE,
                                        sizeof(uint32_t) * global_hash_tables_offset_cpu.size());
        controls.queue.enqueueWriteBuffer(global_hash_tables_offset, true, 0,
                                          sizeof(uint32_t) * global_hash_tables_offset_cpu.size(),
                                          global_hash_tables_offset_cpu.data());
        global_hash_tables = cl::Buffer(controls.context, CL_MEM_READ_WRITE,
                                        sizeof(uint32_t) * global_hash_mem_size);
    }

}

