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

extern "C"
{
#include <GraphBLAS.h>
};

using namespace benchmark;

#define GrB_CHECK(func) do { auto s = func; assert(s == GrB_SUCCESS); } while(0);

int main(int argc, const char **argv) {
    ArgsProcessor argsProcessor;
    Matrix input;

    argsProcessor.parse(argc, argv);

    GrB_Matrix matrix = nullptr;
    GrB_Matrix result = nullptr;
    bool *X = nullptr;


    { // -------------------------- init input matrix -------------------------------------------
        // https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.coo_matrix.html
        cpu_buffer is{0, 0, 1, 3};
        cpu_buffer js{0, 2, 1, 3};

        size_t n = 4;
        // так совпало, что везде 4
        input.ncols = 4;
        input.nrows = 4;
        input.nvals = 4;
        input.rows = is;
        input.cols = js;
    }
    std::vector<GrB_Index> I(input.nvals);
    std::vector<GrB_Index> J(input.nvals);

    std::cout << ">   Load matrix: \"" << file << "\" isUndirected: " << type << std::endl
              << "                 size: " << input.nrows << " x " << input.ncols << " nvals: " << input.nvals
              << std::endl;


    {// --------------------------- init GrB matrix from input -----------------------------------------

        GrB_CHECK(GrB_Matrix_new(&matrix, GrB_BOOL, n, n));

        X = (bool *) std::malloc(sizeof(bool) * input.nvals);

        for (auto i = 0; i < input.nvals; i++) {
            I[i] = input.rows[i];
            J[i] = input.cols[i];
            X[i] = true;
        }

        GrB_CHECK(GrB_Matrix_build_BOOL(matrix, I.data(), J.data(), X, input.nvals, GrB_FIRST_BOOL));

        std::free(X);
        X = nullptr;

    }


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

    std::cout << "Result matrix " << file << "2 : size: " << nrows << " x " << ncols << " nvals: " << nvals
              << std::endl;

    Matrix m2;
    m2.nrows = nrows;
    m2.ncols = ncols;
    m2.nvals = nvals;
    m2.rows.resize(nvals);
    m2.cols.resize(nvals);
    I.resize(nvals);
    J.resize(nvals);

    std::cout << "Memory successfully allocated " << std::endl;

    GrB_CHECK(GrB_Matrix_extractTuples_UINT32(I.data(), J.data(), nullptr, &nvals, result));

    std::cout << "Data successfully extracted " << std::endl;
    for (auto i = 0; i < nvals; i++) {
        m2.rows[i] = I[i];
        m2.cols[i] = J[i];
        X[i] = true;
    }

    m2.nvals = nvals;

    std::cout << std::endl << "result rows: " << std::endl;
    for (size_t i = 0; i < m2.nrows; ++i) {
        std::cout << m2.rows[i] << ", ";
    }

    std::cout << std::endl << "result cols: " << std::endl;
    for (size_t i = 0; i < m2.ncols; ++i) {
        std::cout << m2.cols[i] << ", ";
    }

    GrB_CHECK(GrB_Matrix_free(&result));
    GrB_CHECK(GrB_Matrix_free(&matrix));

    result = nullptr;
    matrix = nullptr;


    return 0;
}