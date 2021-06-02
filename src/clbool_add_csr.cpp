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
#include <core/cpu_matrices.hpp>
#include <common/utils.hpp>
#include <common/matrices_conversions.hpp>
#include <common/env.hpp>
#include <csr/csr.hpp>

#define BENCH_DEBUG

namespace benchmark {
    class Add: public BenchmarkBase {
    public:

        Add(int argc, const char** argv) {
            argsProcessor.parse(argc, argv);
            assert(argsProcessor.isParsed());

            benchmarkName = "Clbool-Add-CSR";
            experimentsCount = argsProcessor.getExperimentsCount();
        }

    protected:

        void setupBenchmark() override {
            controls = new clbool::Controls(clbool::create_controls(0, 0));
        }

        void tearDownBenchmark() override {
            delete controls;
        }

        void setupExperiment(size_t experimentIdx, size_t &iterationsCount, std::string& name) override {
            auto& entry = argsProcessor.getEntries()[experimentIdx];

            iterationsCount = entry.iterations;
            name = entry.name;

            const auto& file = entry.name;
            const auto& type = entry.isUndirected;

            MatrixLoader loader(file, type);
            loader.loadData();
            input = std::move(loader.getMatrix());

#ifdef BENCH_DEBUG
            log       << ">   Load A: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            {
                size_t n = input.nrows;
                assert(input.nrows == input.ncols);

                A = clbool::csr_from_cpu(*controls,clbool::csr_cpu_from_coo_cpu(
                        clbool::matrix_coo_cpu(input.rows, input.cols),
                                                           input.nrows, input.ncols));
            }

            MatrixLoader2 loader2(file);
            loader2.loadData();
            input = std::move(loader2.getMatrix());

#ifdef BENCH_DEBUG
            log       << ">   Load A2: \"" << file << "\" isUndirected: " << type << std::endl
                      << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals << std::endl;
#endif // BENCH_DEBUG

            {
                size_t n = input.nrows;
                assert(input.nrows == input.ncols);


                A2 = clbool::csr_from_cpu(*controls,clbool::csr_cpu_from_coo_cpu(
                        clbool::matrix_coo_cpu(input.rows, input.cols),
                        input.nrows, input.ncols));
            }
        }

        void tearDownExperiment(size_t experimentIdx) override {
            input = Matrix{};
            A = clbool::matrix_csr{};
            A2 = clbool::matrix_csr{};
        }

        void setupIteration(size_t experimentIdx, size_t iterationIdx) override {

        }

        void execIteration(size_t experimentIdx, size_t iterationIdx) override {
            clbool::csr::matrix_addition(*controls, R, A, A2);
        }

        void tearDownIteration(size_t experimentIdx, size_t iterationIdx) override {
#ifdef BENCH_DEBUG
            log << "   Result matrix: size " << R.nrows() << " x " << R.ncols()
                << " nvals " << R.nnz() << std::endl;
#endif
            R = clbool::matrix_csr{};
        }

    protected:

        clbool::Controls* controls;
        clbool::matrix_csr A;
        clbool::matrix_csr A2;
        clbool::matrix_csr R;

        ArgsProcessor argsProcessor;
        Matrix input;
    };

}

int main(int argc, const char** argv) {
    benchmark::Add add(argc, argv);
    add.runBenchmark();
    return 0;
}