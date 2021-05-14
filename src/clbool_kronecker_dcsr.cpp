////////////////////////////////////////////////////////////////////////////////////
// MIT License                                                                    //
//                                                                                //
// Copyright (c) 2021 Egor Orachyov                                               //
//                                                                                //
// Permission is hereby granted, free of charge, to any person obtaining a copy   //
// of this software and associated documentation files (the "Software"), to deal  //
// in the Software without restriction, including without limitation the rights   //
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell      //
// copies of the Software, and to permit persons to whom the Software is          //
// furnished to do so, subject to the following conditions:                       //
//                                                                                //
// The above copyright notice and this permission notice shall be included in all //
// copies or substantial portions of the Software.                                //
//                                                                                //
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     //
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,       //
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE    //
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER         //
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,  //
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE  //
// SOFTWARE.                                                                      //
////////////////////////////////////////////////////////////////////////////////////

#include <benchmark_base.hpp>
#include <matrix_loader.hpp>
#include <args_processor.hpp>

// clBool goes here
#include <core/controls.hpp>
#include <common/utils.hpp>
#include <common/env.hpp>
#include <common/matrices_conversions.hpp>
#include <coo/coo_matrix_addition.hpp>
#include <coo_utils.hpp>
#include <dcsr/dcsr.hpp>

#define BENCH_DEBUG

namespace benchmark {
    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            benchmarkName = "Clbool-Kronecker-DCSR";
            experimentsCount = 1;
            coeff = std::atoi(argv[2]);
            itCnt = std::atoi(argv[3]);
        }

    protected:

        void setupBenchmark() override {
            controls = new clbool::Controls(clbool::create_controls(1, 0));
        }

        void tearDownBenchmark() override {
            delete controls;
        }

        void setupExperiment(size_t experimentIdx, size_t &iterationsCount, std::string& name) override {
            iterationsCount = itCnt;
            uint32_t nnz = 1000 * coeff;
            name = "Matrix with nnz = " + std::to_string(nnz);
            uint32_t size_a = 10000;
#ifdef BENCH_DEBUG
            log       << ">   Load A: \nsize: " << size_a << " x " << size_a << " nvals: " << nnz<< std::endl;
#endif // BENCH_DEBUG

            {
                A = clbool::matrix_dcsr_from_cpu(*controls,
                        clbool::coo_utils::coo_pairs_to_dcsr_cpu(
                         clbool::coo_utils::generate_coo_pairs_cpu(nnz, size_a)), size_a);
            }
        }

        void tearDownExperiment(size_t experimentIdx) override {
            A = clbool::matrix_dcsr{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            clbool::dcsr::kronecker_product(*controls, R, A, A);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.nrows() << " x " << R.ncols()
                << " nvals " << R.nnz() << std::endl;
#endif

            R = clbool::matrix_dcsr{};
        }

    protected:
        uint32_t itCnt;
        uint32_t coeff;
        clbool::Controls* controls;
        clbool::matrix_dcsr A;
        clbool::matrix_dcsr R;
    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}

