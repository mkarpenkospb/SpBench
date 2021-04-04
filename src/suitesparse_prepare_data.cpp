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

#include <args_processor.hpp>
#include <matrix_loader.hpp>
#include <matrix_writer.hpp>
#include <cubool/cubool.h>

extern "C"
{
#include <GraphBLAS.h>
};

using namespace benchmark;

int main(int argc, const char** argv) {
    ArgsProcessor argsProcessor;
    Matrix input;

    argsProcessor.parse(argc, argv);

    GrB_Matrix matrix = nullptr;
    GrB_Matrix result = nullptr;

    CUBOOL_CHECK(cuBool_Initialize(CUBOOL_HINT_NO));

    for (auto& entry: argsProcessor.getEntries()) {
        const auto& file = entry.name;
        const auto& type = entry.isUndirected;

        MatrixLoader loader(file, type);
        loader.loadData();
        input = std::move(loader.getMatrix());

        size_t n = input.nrows;
        assert(input.nrows == input.ncols);

        // --------------------------- init matrix -----------------------------------------
        GrB_CHECK(GrB_Matrix_new(&matrix, GrB_BOOL, n, n));

        std::vector<GrB_Index> I(input.nvals);
        std::vector<GrB_Index> J(input.nvals);

        bool* X = (bool*) std::malloc(sizeof(bool) * input.nvals);

        for (auto i = 0; i < input.nvals; i++) {
            I[i] = input.rows[i];
            J[i] = input.cols[i];
            X[i] = true;
        }

        GrB_CHECK(GrB_Matrix_build_BOOL(matrix, I.data(), J.data(), X, input.nvals, GrB_FIRST_BOOL));

        std::free(X);

        // --------------------------- init result -----------------------------------------

        GrB_CHECK(GrB_Matrix_new(&result, GrB_BOOL, n, n));

        // -------------------------- execute multiplication ----------------------------------
        GrB_CHECK(GrB_mxm(result, nullptr, nullptr, GrB_LOR_LAND_SEMIRING_BOOL, matrix, matrix, nullptr));

        // -------------------------- read matrix --------------------------------------------
        GrB_Index nvals;
        GrB_Index nrows;
        GrB_Index ncols;

        GrB_CHECK(GrB_Matrix_nrows(&nrows, result));
        GrB_CHECK(GrB_Matrix_ncols(&ncols, result));
        GrB_CHECK(GrB_Matrix_nvals(&nvals, result));

        std::cout << "Result matrix " << file << "2 : size: " << nrows << " x " << ncols << " nvals: " << nvals << std::endl;

        Matrix m2;
        m2.nrows = nrows;
        m2.ncols = ncols;
        m2.nvals = nvals;
        m2.rows.resize(nvals);
        m2.cols.resize(nvals);

        GrB_CHECK(GrB_Matrix_extractTuples(m2.rows.data(), m2.cols.data(), nullptr, &nvals, result));

        m2.nvals = nvals;

        MatrixWriter writer;
        writer.save(file + "2", m2);


        GrB_CHECK(GrB_Matrix_free(result));
        GrB_CHECK(GrB_Matrix_free(matrix));

        result = nullptr;
        matrix = nullptr;
    }

    CUBOOL_CHECK(cuBool_Finalize());

    return 0;
}