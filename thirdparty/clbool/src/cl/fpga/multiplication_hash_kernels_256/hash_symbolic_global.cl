#define __local local
#define GROUP_SIZE 256
#define WARP 32 // TODO add define for amd to 64, for fpga unknown
#define HASH_SCAL 107


uint ceil_to_power2(uint v) {
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}

void bitonic_sort_global(__global uint *data,
                           uint size) {

    uint half_segment_length, local_line_id, local_twin_id, group_line_id, line_id, twin_id;
    uint local_id_real = get_local_id(0);

    uint outer = ceil_to_power2(size);
    uint threads_needed = outer / 2;
    // local_id < outer / 2

    uint segment_length = 2;
    while (outer != 1) {
        for (uint local_id = local_id_real; local_id < threads_needed; local_id += GROUP_SIZE) {
            half_segment_length = segment_length / 2;
            // id inside a segment
            local_line_id = local_id & (half_segment_length - 1);
            // index to compare and swap
            local_twin_id = segment_length - local_line_id - 1;
            // segment id
            group_line_id = local_id / half_segment_length;
            // индексы элементов в массиве
            line_id = segment_length * group_line_id + local_line_id;
            twin_id = segment_length * group_line_id + local_twin_id;

            if (line_id < size && twin_id < size && data[line_id] > data[twin_id]) {
                uint tmp = data[line_id];
                data[line_id] = data[twin_id];
                data[twin_id] = tmp;
            }
        }
        barrier(CLK_GLOBAL_MEM_FENCE);
        for (uint j = half_segment_length; j > 1; j >>= 1) {
            for (uint local_id = local_id_real; local_id < threads_needed; local_id += GROUP_SIZE) {
                uint half_j = j / 2;
                local_line_id = local_id & (half_j - 1);
                local_twin_id = local_line_id + half_j;
                group_line_id = local_id / half_j;
                line_id = j * group_line_id + local_line_id;
                twin_id = j * group_line_id + local_twin_id;

                if (line_id < size && twin_id < size && data[line_id] > data[twin_id]) {
                    uint tmp = data[line_id];
                    data[line_id] = data[twin_id];
                    data[twin_id] = tmp;
                }
                barrier(CLK_GLOBAL_MEM_FENCE);
            }
        }
        outer >>= 1;
        segment_length <<= 1;
    }
}


uint search_global(__global const uint *array, uint value, uint size)
#ifdef GPU
{
    uint l = 0;
    uint r = size;
    uint m = l + ((r - l) / 2);
    while (l < r) {
        if (array[m] == value) {
            return m;
        }

        if (array[m] < value) {
            l = m + 1;
        } else {
            r = m;
        }

        m = l + ((r - l) / 2);
    }

    return size;
}
#endif
;

// need to monitor table_size

__attribute__((reqd_work_group_size(GROUP_SIZE,1,1)))
__kernel void hash_symbolic_global(__global const uint * restrict indices, // indices -- aka premutation
                                  uint group_start,

                                  __global uint * restrict nnz_estimation,

                                  __global const uint * restrict a_rows_pointers,
                                  __global const uint * restrict a_cols,

                                  __global const uint * restrict b_rows_pointers,
                                  __global const uint * restrict b_rows_compressed,
                                  __global const uint * restrict b_cols,
                                  const uint b_nzr,
                                  __global uint * restrict hash_table_data,
                                  __global const uint * restrict hash_table_offset

) {
    uint hash, old, row_index, a_start, a_end, col_index, b_col, b_rpt;
    uint local_id = get_local_id(0); // 0 - 255
    uint row_id_bin = get_group_id(0); // 0, 1, ...
    uint warps_per_group = GROUP_SIZE / WARP; // 256 / 32 -- how many vals of A row you can process in ones
    // от 0 до 31
    uint id_in_warp = get_local_id(0) & (WARP - 1); // 0 - 31 , get_global_id(0) & (WARP - 1) == get_global_id(0) % WARP
    uint warp_id = local_id / WARP; // (0 - 255) / 32 -- warp id of thread
    uint row_pos = group_start + row_id_bin; //

    uint group_id = get_group_id(0);
    uint table_size = hash_table_offset[group_id + 1] - hash_table_offset[group_id];
    __global uint *hash_table = hash_table_data + hash_table_offset[group_id];
    __local uint nz_count[GROUP_SIZE];
    __local uint *thread_nz = nz_count + local_id;
    thread_nz[0] = 0;

    for (uint i = local_id; i < table_size; i += GROUP_SIZE) {
        hash_table[i] = -1;
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    row_index = indices[row_pos];
    a_start = a_rows_pointers[row_index];
    a_end = a_rows_pointers[row_index + 1];
    for (uint a_prt = a_start + warp_id; a_prt < a_end; a_prt += warps_per_group) {
        col_index = a_cols[a_prt]; // позицию этого будем искать в матрице B
        b_rpt = search_global(b_rows_compressed, col_index, b_nzr);
        if (b_rpt == b_nzr) {
            continue;
        }

        for (uint k = b_rows_pointers[b_rpt] + id_in_warp; k < b_rows_pointers[b_rpt + 1]; k += WARP) {
            b_col = b_cols[k];
            // Now go to hashtable and search for b_col
            hash = (b_col * HASH_SCAL) & (table_size - 1);
            while (true) {
                if (hash_table[hash] == b_col) {
                    break;
                } else if (hash_table[hash] == -1) {
                    old = atom_cmpxchg(hash_table + hash, -1, b_col);
                    if (old == -1) {
                        thread_nz[0] ++;
                        break;
                    }
                } else {
                    hash = (hash + 1) % table_size;
                }
            }
        }
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    // reduce nz values
    int step = GROUP_SIZE / 2;
    while (step > 0) {
        if (local_id < step) {
            nz_count[local_id] = nz_count[local_id] + nz_count[local_id + step];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        step /= 2;
    }
    bitonic_sort_global(hash_table, table_size);
    if (local_id == 0) {
        nnz_estimation[row_index] = nz_count[0];
    }
}
