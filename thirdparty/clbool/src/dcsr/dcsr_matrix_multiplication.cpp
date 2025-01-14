#include <numeric>
#include <program.hpp>
#include "dcsr_matrix_multiplication.hpp"
#include "coo_matrix_addition.hpp"
#include "to_result_matrix_single_thread.h"
#include "to_result_matrix_work_group.h"
#include "heap_merge.h"
#include "copy_one_value.h"
#include "merge_large_rows.h"
#include "bitonic_esc.h"
#include "count_workload.h"
#include "prepare_positions.h"
#include "set_positions.h"

const uint32_t BINS_NUM = 38;
const uint32_t HEAP_MERGE_BLOCK_SIZE = 32;

uint32_t esc_estimation(uint32_t group) {
    switch (group) {
        case 33:
            return 64;
        case 34:
            return 128;
        case 35:
            return 256;
        case 36:
            return 512;
        default:
            throw std::runtime_error("A group should be in range 33-36!");
    }
}

void matrix_multiplication(Controls &controls,
                           matrix_dcsr &matrix_out,
                           const matrix_dcsr &a,
                           const matrix_dcsr &b) {
    timer t;
    if (a.nnz() == 0 || b.nnz() == 0) {
        std::cout << "empty result\n";
        return;
    }
    cl::Buffer nnz_estimation;
    t.restart();
    count_workload(controls, nnz_estimation, a, b);
    double time = t.elapsed();
    if (DEBUG_ENABLE) Log() << "Workload counted in " << time << "\n";

    std::vector<cpu_buffer> cpu_workload_groups(BINS_NUM, cpu_buffer());
    cpu_buffer groups_pointers(BINS_NUM + 1);
    cpu_buffer groups_length(BINS_NUM);


    cl::Buffer aux_37_group_mem_pointers;
    cl::Buffer aux_37_group_mem;

    matrix_dcsr pre;
    build_groups_and_allocate_new_matrix(controls, pre, cpu_workload_groups, nnz_estimation, a, b.nCols(),
                                         aux_37_group_mem_pointers, aux_37_group_mem);

    cl::Buffer gpu_workload_groups(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

    t.restart();
    write_bins_info(controls, gpu_workload_groups, cpu_workload_groups, groups_pointers, groups_length);
    time = t.elapsed();
    if (DEBUG_ENABLE) Log() << "Bins written in " << time << "\n";


    run_kernels(controls, groups_length, groups_pointers,
                gpu_workload_groups, nnz_estimation,
                pre, a, b,
                aux_37_group_mem_pointers, aux_37_group_mem
                );


    create_final_matrix(controls, matrix_out,
                        nnz_estimation, pre,
                        gpu_workload_groups, groups_pointers, groups_length,
                        a
                        );
}


void create_final_matrix(Controls &controls,
                         matrix_dcsr &c,
                         cl::Buffer &nnz_estimation,
                         const matrix_dcsr &pre,

                         const cl::Buffer &gpu_workload_groups,
                         const cpu_buffer &groups_pointers,
                         const cpu_buffer &groups_length,

                         const matrix_dcsr &a
                         ) {
    cl::Buffer c_rows_pointers;
    cl::Buffer c_rows_compressed;
    cl::Buffer c_cols_indices;

    uint32_t c_nnz;
    uint32_t c_nzr;

    prefix_sum(controls, nnz_estimation, c_nnz, a.nzr() + 1);

    c_cols_indices = cl::Buffer(controls.context, CL_TRUE, sizeof(uint32_t) * c_nnz);

    cl::Event e1;
    cl::Event e2;
    if (groups_length[1] != 0) {
        auto single_value_rows = program<cl::Buffer, uint32_t, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
#ifndef FPGA
                (to_result_matrix_single_thread_kernel, to_result_matrix_single_thread_kernel_length)
#else
                ("to_result_matrix_single_thread")
#endif
                .set_block_size(std::min(controls.block_size, std::max(32u, utils::ceil_to_power2(groups_length[1]))))
                .set_needed_work_size(groups_length[1])
                .set_async(true)
                .set_kernel_name("to_result");

        e1 = single_value_rows.run(controls, gpu_workload_groups, groups_pointers[1], groups_length[1],
                                       nnz_estimation, c_cols_indices, pre.rows_pointers_gpu(), pre.cols_indices_gpu());
    }

    uint32_t second_group_length = std::accumulate(groups_length.begin() + 2, groups_length.end(), 0u);

    if (second_group_length != 0) {
        auto ordinary_rows = program<cl::Buffer, uint32_t,
                cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer>
#ifndef FPGA
                (to_result_matrix_work_group_kernel, to_result_matrix_work_group_kernel_length)
#else
                ("to_result_matrix_work_group")
#endif
                .set_needed_work_size(controls.block_size * second_group_length)
                .set_async(true)
                .set_kernel_name("to_result");

        e2 = ordinary_rows.run(controls,
                          gpu_workload_groups, groups_length[0] + groups_length[1],
                          nnz_estimation, c_cols_indices, pre.rows_pointers_gpu(), pre.cols_indices_gpu());
    }

    try {
        if (groups_length[1] != 0) e1.wait();
        if (second_group_length != 0) e2.wait();
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << " in " << "create_final_matrix" << " \n";
        throw std::runtime_error(exception.str());
    }

    cl::Buffer positions(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * a.nzr());

    prepare_positions(controls, positions, nnz_estimation, a.nzr(), "prepare_for_shift_empty_rows");


    // ------------------------------------  get rid of empty rows -------------------------------

    prefix_sum(controls, positions, c_nzr, a.nzr());
    c_rows_pointers = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (c_nzr + 1));
    c_rows_compressed = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * c_nzr);
    set_positions(controls, c_rows_pointers, c_rows_compressed, nnz_estimation, a.rows_compressed_gpu(), positions,
                  c_nnz, a.nzr(), c_nzr);

    c = matrix_dcsr(c_rows_pointers, c_rows_compressed, c_cols_indices, pre.nRows(), pre.nCols(), c_nnz, c_nzr);
}

void write_bins_info(Controls &controls,
                     cl::Buffer &gpu_workload_groups,
                     const std::vector<cpu_buffer> &cpu_workload_groups,
                     cpu_buffer &groups_pointers,
                     cpu_buffer &groups_length
                     ) {

    uint32_t offset = 0;
    uint32_t bins = cpu_workload_groups.size();
    cpu_buffer cpu_workload_groups_for_copy;
//    cl::Event end_write_buffer;
    for (uint32_t workload_group_id = 0; workload_group_id < bins; ++workload_group_id) {
        const cpu_buffer& group = cpu_workload_groups[workload_group_id];
        if (group.empty()) continue;
        groups_pointers[workload_group_id] = offset;
        groups_length[workload_group_id] = group.size();
        cpu_workload_groups_for_copy.insert(
                cpu_workload_groups_for_copy.end(),
                group.begin(),
                group.end()
                );
//        controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, sizeof(uint32_t) * offset,
//                                          sizeof(uint32_t) * group.size(), group.data()
//                                         /* , nullptr, &end_write_buffer*/);
        offset += group.size();
    }
    controls.queue.enqueueWriteBuffer(gpu_workload_groups, CL_TRUE, 0,
                                      sizeof(uint32_t) * cpu_workload_groups_for_copy.size(),
                                      cpu_workload_groups_for_copy.data());


    groups_pointers[bins] = offset;
//    end_write_buffer.wait();
}

void run_kernels(Controls &controls,
                 const cpu_buffer &groups_length,
                 const cpu_buffer &groups_pointers,

                 const cl::Buffer &gpu_workload_groups,
                 cl::Buffer &nnz_estimation,

                 const matrix_dcsr &pre,
                 const matrix_dcsr &a,
                 const matrix_dcsr &b,

                 const cl::Buffer &aux_mem_pointers,
                 cl::Buffer &aux_mem

) {
    timer t;
    t.restart();

#ifdef FPGA
    static int heap_kernels[33] = {-1,-1,
                                   4,4,4, // 2,3,4
                                   8,8,8,8, // 5,6,7,8
                                   16,16,16,16,16,16,16,16, // 9,10,11,12,13,14,15,16
                                   20,20,20,20, // 17,18,19,20
                                   24,24,24,24, // 21,22,23,24
                                   28,28,28,28, // 25,26,27,28
                                   32,32,32,32  // 29,30,31,32
                                   };
#endif
    auto heap_merge = program<cl::Buffer, uint32_t, uint32_t,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t>
#ifndef  FPGA
        (heap_merge_kernel, heap_merge_kernel_length)
        .set_kernel_name("heap_merge")
#else
        ("heap_merge")
#endif
        .set_block_size(HEAP_MERGE_BLOCK_SIZE)
        .set_async(true);

    auto copy_one_value = program<cl::Buffer, uint32_t, uint32_t,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t>
#ifndef  FPGA
            (copy_one_value_kernel, copy_one_value_kernel_length)
#else
            ("copy_one_value")
#endif
        .set_kernel_name("copy_one_value")
        .set_async(true);

    auto merge_large_rows = program<cl::Buffer, uint32_t, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t>
#ifndef  FPGA
            (merge_large_rows_kernel, merge_large_rows_kernel_length)
#else
            ("merge_large_rows")
#endif
        .set_kernel_name("merge_large_rows")
        .set_block_size(controls.block_size)
        .set_async(true);

    auto esc_kernel = program<cl::Buffer, uint32_t, uint32_t,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            uint32_t>
#ifndef  FPGA
            (bitonic_esc_kernel, bitonic_esc_kernel_length)
#else
            ("bitonic_esc")
#endif
            .set_kernel_name("bitonic_esc")
            .set_block_size(controls.block_size)
            .set_async(true);


    std::vector<cl::Event> events;
    for (uint32_t workload_group_id = 1; workload_group_id < BINS_NUM; ++workload_group_id) {
        if (groups_length[workload_group_id] == 0) continue;

        if (workload_group_id == 1) {
            std::cout << "first group!\n";
            copy_one_value.set_needed_work_size(groups_length[workload_group_id])
            .set_block_size(std::min(controls.block_size,
                                     std::max(32u, utils::ceil_to_power2(groups_length[workload_group_id]))));
            events.push_back(
                    copy_one_value.run(controls,
                                       gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                                       pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                                       a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                       b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                       b.nzr()
                             )
            );
            continue;
        }



        if (workload_group_id < 33 ) {
            std::cout << "2 - 32!: " << workload_group_id << "\n";
            heap_merge.set_needed_work_size(groups_length[workload_group_id])
#ifdef FPGA
                        .set_kernel_name("heap_merge_" + std::to_string(heap_kernels[workload_group_id]))
#endif
                        .add_option("NNZ_ESTIMATION", workload_group_id);

            events.push_back(heap_merge.run(controls, gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                                            pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                                            nnz_estimation,
                                            a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                            b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                            b.nzr()));

            continue;
        }

        if (workload_group_id < 37 ) {
            std::cout << "33 - 36!\n";
            uint32_t block_size = std::max(32u, esc_estimation(workload_group_id) / 2);
            esc_kernel.add_option("NNZ_ESTIMATION", esc_estimation(workload_group_id))
            .set_block_size(block_size)
            .set_needed_work_size(block_size * groups_length[workload_group_id]);
            events.push_back(esc_kernel.run(
                    controls,
                    gpu_workload_groups, groups_pointers[workload_group_id], groups_length[workload_group_id],
                    pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                    nnz_estimation,
                    a.rows_pointers_gpu(), a.cols_indices_gpu(),
                    b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                    b.nzr()
            ));
            continue;
        }


        std::cout << "37!\n";
        merge_large_rows.set_needed_work_size(groups_length[workload_group_id] * controls.block_size);
        events.push_back(merge_large_rows.run(controls,
                                              gpu_workload_groups, groups_pointers[workload_group_id],
                                              aux_mem_pointers, aux_mem,
                                              pre.rows_pointers_gpu(), pre.cols_indices_gpu(),
                                              nnz_estimation,
                                              a.rows_pointers_gpu(), a.cols_indices_gpu(),
                                              b.rows_pointers_gpu(), b.rows_compressed_gpu(), b.cols_indices_gpu(),
                                              b.nzr()
                            ));
    }
    try {
        cl::Event::waitForEvents(events);
        double time = t.elapsed();
        if (DEBUG_ENABLE) Log() << "kernels run in " << time << "\n";
    } catch (const cl::Error &e) {
        std::stringstream exception;
        exception << "\n" << e.what() << " : " << utils::error_name(e.err()) << " in " << "run_kernels" << " \n";
        throw std::runtime_error(exception.str());
    }
}

void build_groups_and_allocate_new_matrix(Controls& controls,
                                          matrix_dcsr &pre,
                                          std::vector<cpu_buffer>& cpu_workload_groups,
                                          cl::Buffer& nnz_estimation,
                                          const matrix_dcsr &a,
                                          uint32_t b_cols,

                                          cl::Buffer &aux_pointers,
                                          cl::Buffer &aux_mem
                                          ) {

    cpu_buffer aux_pointers_cpu;
    uint32_t aux = 0;

    cpu_buffer cpu_workload(a.nzr());
   // cl::Event event;
    controls.queue.enqueueReadBuffer(nnz_estimation, CL_TRUE, 0, sizeof(uint32_t) * a.nzr(), cpu_workload.data()
                                     /*, nullptr, &event*/);
 //   event.wait();
    timer t;
    t.restart();
    uint32_t pre_nnz = 0;
    cpu_buffer rows_pointers_cpu(a.nzr() + 1);

    pre_nnz = 0;
    for (uint32_t i = 0; i < a.nzr(); ++i) {

        uint32_t current_workload = cpu_workload[i];
        uint32_t group = get_group(current_workload);
        cpu_workload_groups[group].push_back(i);
        rows_pointers_cpu[i] = pre_nnz;

        // TODO: добавить переаллокацию
        pre_nnz += current_workload;
        if (group == 37) {
            aux_pointers_cpu.push_back(aux);
            aux += current_workload;
        }
    }
    if (pre_nnz == 0) {
        std::cout << "empty result\n";
        return;
    }

    aux_pointers_cpu.push_back(aux);
    rows_pointers_cpu[a.nzr()] = pre_nnz;

    double time = t.elapsed();
    if (DEBUG_ENABLE) Log() << "Groups formed in " << time << "\n";

    t.restart();
    cl::Buffer pre_rows_pointers = cl::Buffer(controls.queue, rows_pointers_cpu.begin(), rows_pointers_cpu.end(), false);
    cl::Buffer pre_cols_indices_gpu = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * pre_nnz);

    if (aux != 0) {
        aux_pointers = cl::Buffer(controls.queue, aux_pointers_cpu.begin(), aux_pointers_cpu.end(),
                                  true);
        aux_mem = cl::Buffer(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * aux);
    }


    pre = matrix_dcsr(pre_rows_pointers, a.rows_compressed_gpu(), pre_cols_indices_gpu,
                      a.nRows(), b_cols, pre_nnz, a.nzr());
    time = t.elapsed();
    if (DEBUG_ENABLE) Log() << "Temporary matrix formed in " << time << "\n";
}


uint32_t get_group(uint32_t size) {
    if (size < 33) return size;
    if (size < 65) return 33;
    if (size < 129) return 34;
    if (size < 257) return 35;
    if (size < 513) return 36;
    return 37;
}


void count_workload(Controls &controls,
                    cl::Buffer &nnz_estimation_out,
                    const matrix_dcsr &a,
                    const matrix_dcsr &b) {

    auto count_workload = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
        uint32_t, uint32_t>
        ("count_workload")
        .set_needed_work_size(a.nzr())
        .set_kernel_name("count_workload")
//#ifndef WIN
//        .set_task(true)
//#endif
        ;

    cl::Buffer nnz_estimation(controls.context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (a.nzr() + 1));

    count_workload.run(controls, nnz_estimation, a.rows_pointers_gpu(), a.cols_indices_gpu(),
                       b.rows_compressed_gpu(), b.rows_pointers_gpu(), a.nzr(), b.nzr())
                       .wait();
    nnz_estimation_out = std::move(nnz_estimation);
}


void prepare_positions(Controls &controls,
                       cl::Buffer &positions,
                       const cl::Buffer &array,
                       uint32_t size,
                       const std::string &program_name
) {
    timer t;
    auto prepare_positions = program<cl::Buffer, cl::Buffer, uint32_t>
            ("prepare_positions")
            .set_kernel_name(program_name)
            .set_needed_work_size(size)
//#ifndef WIN
//            .set_task(true)
//#endif
            ;
    t.restart();
    prepare_positions.run(controls, positions, array, size).wait();
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "Set positions routine finished in " << t.last_elapsed();
}


void set_positions(Controls &controls,
                   cl::Buffer &c_rows_pointers,
                   cl::Buffer &c_rows_compressed,
                   const cl::Buffer &nnz_estimation,
                   const cl::Buffer &a_rows_compressed,
                   const cl::Buffer &positions,
                   uint32_t c_nnz,
                   uint32_t old_nzr,
                   uint32_t c_nzr
) {
    timer t;
    auto set_positions = program<cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer, cl::Buffer,
            uint32_t, uint32_t, uint32_t>
            ("set_positions")
            .set_kernel_name("set_positions_pointers_and_rows")
            .set_needed_work_size(old_nzr)
//#ifndef WIN
//            .set_task(true)
//#endif
            ;
    t.restart();
    set_positions.run(controls, c_rows_pointers, c_rows_compressed,
                  nnz_estimation, a_rows_compressed, positions,
                  c_nnz, old_nzr, c_nzr).wait();
    t.elapsed();
    if (DEBUG_ENABLE && DETAIL_DEBUG_ENABLE) Log() << "Set positions routine finished in " << t.last_elapsed();
//    event.wait();
}