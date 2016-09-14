
// =================================================================================================
// This file is part of the CLBlast project. The project is licensed under Apache Version 2.0. This
// project loosely follows the Google C++ styleguide and uses a tab-size of two spaces and a max-
// width of 100 characters per line.
//
// Author(s):
//   Cedric Nugteren <www.cedricnugteren.nl>
//
// This file implements the Xgemm class (see the header for information about the class).
//
// =================================================================================================

#include "routines/level3/xgemm.hpp"

#include <string>
#include <vector>

namespace {
// =================================================================================================

using namespace clblast;

// The "database overlay" specifying parameters for the Mali-specific xgemm kernel
const std::vector<Database::DatabaseEntry> MaliXgemm = {
  {
    "MaliXgemm", Precision::kSingle, { {
        Database::kDeviceTypeAll, Database::kDeviceVendorAll, {
          {
            "default", {
              { "MALI_INTERLEAVE_xWIS", 4 }, { "MALI_INTERLEAVE_yWIS", 4 },
              { "MALI_TRANSPOSE_xWIS", 4 }, { "MALI_TRANSPOSE_yWIS", 1 },
              { "MALI_MM_xWIS", 4 }, { "MALI_MM_yWIS", 4 },
              { "MALI_FINALIZE_xWIS", 4 }, { "MALI_FINALIZE_yWIS", 1 },
              { "MALI_MM_xLWS", 4 }, { "MALI_MM_yLWS", 8 }
            }
          },
        }
      }
    }
  },

  {
    "MaliXgemm", Precision::kHalf, { {
        Database::kDeviceTypeAll, Database::kDeviceVendorAll, {
          {
            "default", {
              { "MALI_INTERLEAVE_xWIS", 4 }, { "MALI_INTERLEAVE_yWIS", 4 },
              { "MALI_TRANSPOSE_xWIS", 8 }, { "MALI_TRANSPOSE_yWIS", 1 },
              { "MALI_MM_xWIS", 8 }, { "MALI_MM_yWIS", 4 },
              { "MALI_FINALIZE_xWIS", 8 }, { "MALI_FINALIZE_yWIS", 1 },
              { "MALI_MM_xLWS", 4 }, { "MALI_MM_yLWS", 8 }
            }
          },
        }
      }
    }
  },
};

// =================================================================================================
}

namespace clblast {
// =================================================================================================

// Constructor: forwards to base class constructor
template <typename T>
Xgemm<T>::Xgemm(Queue &queue, EventPointer event, const std::string &name):
    Routine(queue, event, name, {"Copy","Pad","Transpose","Padtranspose","MaliXgemm"}, PrecisionValue<T>(), MaliXgemm) {
  source_string_ =
    #include "../../kernels/level3/level3.opencl"
    #include "../../kernels/level3/copy_fast.opencl"
    #include "../../kernels/level3/copy_pad.opencl"
    #include "../../kernels/level3/transpose_fast.opencl"
    #include "../../kernels/level3/transpose_pad.opencl"
    #include "../../kernels/level3/convert_symmetric.opencl"
    #include "../../kernels/level3/convert_triangular.opencl"
    #include "../../kernels/level3/convert_hermitian.opencl"
    #include "../../kernels/level3/xgemm_mali.opencl"
  ;
}

// =================================================================================================

// The main routine
template <typename T>
StatusCode Xgemm<T>::DoGemm(const Layout layout,
                            const Transpose a_transpose, const Transpose b_transpose,
                            const size_t m, const size_t n, const size_t k,
                            const T alpha,
                            const Buffer<T> &a_buffer, const size_t a_offset, const size_t a_ld,
                            const Buffer<T> &b_buffer, const size_t b_offset, const size_t b_ld,
                            const T beta,
                            const Buffer<T> &c_buffer, const size_t c_offset, const size_t c_ld) {

  // Makes sure all dimensions are larger than zero
  if ((m == 0) || (n == 0) || (k == 0)) { return StatusCode::kInvalidDimension; }

  // Computes whether or not the matrices are transposed in memory. This is based on their layout
  // (row or column-major) and whether or not they are requested to be pre-transposed.
  // NOTE: "rotated" here means row-major.
  // NOTE: the Mali-specific Xgemm kernel expects all three matrices in row-major layout
  // NOTE: the dimensions (m, n, k) are given after transposition.
  const auto a_rotated = (layout == Layout::kColMajor && a_transpose != Transpose::kNo) ||
                         (layout == Layout::kRowMajor && a_transpose == Transpose::kNo);
  const auto b_rotated = (layout == Layout::kColMajor && b_transpose != Transpose::kNo) ||
                         (layout == Layout::kRowMajor && b_transpose == Transpose::kNo);
  const auto c_rotated = (layout == Layout::kRowMajor);
  static const auto a_want_rotated = true;
  static const auto b_want_rotated = true;
  static const auto c_want_rotated = true;
  const auto a_do_transpose = a_rotated != a_want_rotated;
  const auto b_do_transpose = b_rotated != b_want_rotated;
  const auto c_do_transpose = c_rotated != c_want_rotated;

  // In case of complex data-types, the transpose can also become a conjugate transpose
  const auto a_conjugate = (a_transpose == Transpose::kConjugate);
  const auto b_conjugate = (b_transpose == Transpose::kConjugate);

  // Computes the first and second dimensions of the 3 matrices taking into account whether the
  // matrices are rotated or not
  // NOTE: these dimensions are physical, i. e. together with X_ld and X_offset they constitute
  //       the physical input matrix layout in memory.
  // NOTE: "one" is x, "two" is y.
  const auto a_one = (a_rotated) ? k : m;
  const auto a_two = (a_rotated) ? m : k;
  const auto b_one = (b_rotated) ? n : k;
  const auto b_two = (b_rotated) ? k : n;
  const auto c_one = (c_rotated) ? n : m;
  const auto c_two = (c_rotated) ? m : n;

  // Tests three matrices (A, B, C) for validity, first from a perspective of the OpenCL buffers and
  // their sizes, and then from a perspective of parameter values (e.g. m, n, k). Tests whether the
  // OpenCL buffers are valid and non-zero and whether the OpenCL buffers have sufficient storage
  // space. Also tests that the leading dimensions of:
  //    matrix A cannot be less than K when rotated, or less than M when not-rotated
  //    matrix B cannot be less than N when rotated, or less than K when not-rotated
  //    matrix C cannot be less than N when rotated, or less than M when not-rotated
  auto status = TestMatrixA(a_one, a_two, a_buffer, a_offset, a_ld);
  if (ErrorIn(status)) { return status; }
  status = TestMatrixB(b_one, b_two, b_buffer, b_offset, b_ld);
  if (ErrorIn(status)) { return status; }
  status = TestMatrixC(c_one, c_two, c_buffer, c_offset, c_ld);
  if (ErrorIn(status)) { return status; }

  // Calculates the ceiled versions of m, n, and k
  const auto m_ceiled = Ceil(m, db_["MALI_MM_yWIS"] * db_["MALI_MM_yLWS"]); // rows (y) of A and AB
  const auto n_ceiled = Ceil(n, db_["MALI_MM_xWIS"] * db_["MALI_MM_xLWS"]); // cols (x) of B and AB
  const auto ka_ceiled = Ceil(k, db_["MALI_MM_xWIS"]); // columns (x) of A
  const auto kb_ceiled = Ceil(k, db_["MALI_MM_yWIS"]); // rows (y) of B

  // Computes the first and second "internal" (ceiled) dimensions of the 3 matrices taking into account
  // whether the matrices need to be rotated or not for the kernel.
  // NOTE: these dimensions are physical, i. e. they constitute the (intended) physical internal
  // matrix layout in memory.
  // NOTE: "one" is x, "two" is y.
  const auto a_one_i = (a_want_rotated) ? ka_ceiled : m_ceiled;
  const auto a_two_i = (a_want_rotated) ? m_ceiled : ka_ceiled;
  const auto b_one_i = (b_want_rotated) ? n_ceiled : kb_ceiled;
  const auto b_two_i = (b_want_rotated) ? kb_ceiled : n_ceiled;
  const auto c_one_i = (c_want_rotated) ? n_ceiled : m_ceiled;
  const auto c_two_i = (c_want_rotated) ? m_ceiled : n_ceiled;

  // The padded/transposed input/output matrices: if memory allocation fails, throw an exception
  try {

    // Loads the program from the database
    const auto program = GetProgramFromCache(context_, PrecisionValue<T>(), routine_name_);

    // Determines whether or not temporary matrices are needed
    auto a_no_temp = a_one == a_one_i && a_two == a_two_i && a_ld == a_one && a_offset == 0 &&
                     a_do_transpose == false && a_conjugate == false;
    auto b_no_temp = b_one == b_one_i && b_two == b_two_i && b_ld == b_one && b_offset == 0 &&
                     b_do_transpose == false && b_conjugate == false;
    auto c_no_temp = c_one == c_one_i && c_two == c_two_i && c_ld == c_one && c_offset == 0 &&
                     c_do_transpose == false;

    auto no_finalize = alpha == static_cast<T>(1) && beta == static_cast<T>(0);

    // Creates the temporary matrices
    const auto a_temp = (a_no_temp) ? a_buffer : Buffer<T>(context_, a_one_i*a_two_i);
    const auto b_temp = (b_no_temp) ? b_buffer : Buffer<T>(context_, b_one_i*b_two_i);
    const auto c_temp = (c_no_temp) ? c_buffer : Buffer<T>(context_, c_one_i*c_two_i);

    // Create the temporary matrices for Mali-specific transformations
    const auto aI_temp =                          Buffer<T>(context_, a_one_i*a_two_i);
    const auto bT_temp =                          Buffer<T>(context_, b_one_i*b_two_i);
    const auto ab_temp = (no_finalize) ? c_temp : Buffer<T>(context_, c_one_i*c_two_i);

    auto emptyEventList = std::vector<Event>();

    // Runs the pre-processing kernel for matrix A. This transposes the matrix, but also pads zeros
    // to fill it up until it reaches a certain multiple of size (kernel parameter dependent). In
    // case nothing has to be done, these kernels can be skipped.
    auto eventProcessA = Event();
    if (!a_no_temp) {
      status = PadCopyTransposeMatrix(queue_, device_, db_, eventProcessA.pointer(), emptyEventList,
                                      a_one, a_two, a_ld, a_offset, a_buffer,
                                      a_one_i, a_two_i, a_one_i, 0, a_temp,
                                      ConstantOne<T>(), program,
                                      true, a_do_transpose, a_conjugate);
      if (ErrorIn(status)) { return status; }
    }

    // As above, but now for matrix B
    auto eventProcessB = Event();
    if (!b_no_temp) {
      status = PadCopyTransposeMatrix(queue_, device_, db_, eventProcessB.pointer(), emptyEventList,
                                      b_one, b_two, b_ld, b_offset, b_buffer,
                                      b_one_i, b_two_i, b_one_i, 0, b_temp,
                                      ConstantOne<T>(), program,
                                      true, b_do_transpose, b_conjugate);
      if (ErrorIn(status)) { return status; }
    }

    // As above, but now for matrix C. This is only necessary if C is used both as input and output.
    auto eventProcessC = Event();
    if (!c_no_temp && beta != static_cast<T>(0)) {
      status = PadCopyTransposeMatrix(queue_, device_, db_, eventProcessC.pointer(), emptyEventList,
                                      c_one, c_two, c_ld, c_offset, c_buffer,
                                      c_one_i, c_two_i, c_one_i, 0, c_temp,
                                      ConstantOne<T>(), program,
                                      true, c_do_transpose, false);
      if (ErrorIn(status)) { return status; }
    }

    // Retrieves the Xgemm kernel from the compiled binary
    try {
      // First, Mali-specific "interleave" the matrix A
      auto interleaveKernel = Kernel(program, "mali_interleave");
      interleaveKernel.SetArgument(0, a_temp());
      interleaveKernel.SetArgument(1, aI_temp());
      interleaveKernel.SetArgument(2, static_cast<int>(a_one_i)); // COL_MTX_A
      interleaveKernel.SetArgument(3, static_cast<int>(a_one_i * db_["MALI_INTERLEAVE_yWIS"])); // MATRIX_A_INTERLEAVED_STRIDE

      const auto interleaveGlobal = std::vector<size_t> {
        a_one_i / db_["MALI_INTERLEAVE_xWIS"],
        a_two_i / db_["MALI_INTERLEAVE_yWIS"]
      };

      auto interleaveEvent = Event();
      status = RunKernel(interleaveKernel, queue_, device_,
                         interleaveGlobal, {},
                         interleaveEvent.pointer(), std::vector<Event>{ eventProcessA });
      if (ErrorIn(status)) { return status; }

      // Then, Mali-specific "transpose" the matrix B
      auto transposeKernel = Kernel(program, "mali_transpose");
      transposeKernel.SetArgument(0, b_temp());
      transposeKernel.SetArgument(1, bT_temp());
      transposeKernel.SetArgument(2, static_cast<int>(b_one_i)); // COL_MTX_B
      transposeKernel.SetArgument(3, static_cast<int>(b_two_i * db_["MALI_TRANSPOSE_xWIS"])); // MATRIX_B_TRANSPOSED_STRIDE

      const auto transposeGlobal = std::vector<size_t> {
        b_one_i / db_["MALI_TRANSPOSE_xWIS"],
        b_two_i / db_["MALI_TRANSPOSE_yWIS"]
      };

      auto transposeEvent = Event();
      status = RunKernel(transposeKernel, queue_, device_,
                         transposeGlobal, {},
                         transposeEvent.pointer(), std::vector<Event>{ eventProcessB });
      if (ErrorIn(status)) { return status; }

      // Then, Mali-specific "multiply" the temporary matrices
      auto mmKernel = Kernel(program, "mali_mm");
      mmKernel.SetArgument(0, aI_temp());
      mmKernel.SetArgument(1, bT_temp());
      mmKernel.SetArgument(2, ab_temp());
      mmKernel.SetArgument(3, static_cast<int>(a_one_i * db_["MALI_INTERLEAVE_yWIS"])); // MATRIX_A_INTERLEAVED_STRIDE
      mmKernel.SetArgument(4, static_cast<int>(b_two_i * db_["MALI_TRANSPOSE_xWIS"])); // MATRIX_B_TRANSPOSED_STRIDE
      mmKernel.SetArgument(5, static_cast<int>(c_one_i)); // COL_MTX_C

      const auto mmGlobal = std::vector<size_t>{
        c_one_i / db_["MALI_MM_xWIS"],
        c_two_i / db_["MALI_MM_yWIS"]
      };
      const auto mmLocal = std::vector<size_t>{db_["MALI_MM_xLWS"], db_["MALI_MM_yLWS"]};

      auto mmEvent = Event();
      status = RunKernel(mmKernel, queue_, device_,
                         mmGlobal, mmLocal,
                         mmEvent.pointer(), std::vector<Event>{ interleaveEvent, transposeEvent });
      if (ErrorIn(status)) { return status; }

      auto finalizeEvent = Event();
      if (!no_finalize) {
        // Finally, Mali-specific "finalize" the intermediate result
        auto finalizeKernel = Kernel(program, "mali_finalize");
        finalizeKernel.SetArgument(0, ab_temp());
        finalizeKernel.SetArgument(1, c_temp());
        finalizeKernel.SetArgument(2, alpha);
        finalizeKernel.SetArgument(3, beta);
        finalizeKernel.SetArgument(4, static_cast<int>(c_one_i)); // COL_MTX_C

        const auto finalizeGlobal = std::vector<size_t> {
          c_one_i / db_["MALI_FINALIZE_xWIS"],
          c_two_i / db_["MALI_FINALIZE_yWIS"]
        };

        status = RunKernel(finalizeKernel, queue_, device_,
                          finalizeGlobal, {},
                          finalizeEvent.pointer(), std::vector<Event>{ mmEvent });
        if (ErrorIn(status)) { return status; }
      } else {
        finalizeEvent = mmEvent;
      }

      // Runs the post-processing kernel if needed
      if (!c_no_temp) {
        status = PadCopyTransposeMatrix(queue_, device_, db_, event_, { finalizeEvent },
                                        c_one_i, c_two_i, c_one_i, 0, c_temp,
                                        c_one, c_two, c_ld, c_offset, c_buffer,
                                        ConstantOne<T>(), program,
                                        false, c_do_transpose, false);
        if (ErrorIn(status)) { return status; }
      } else {
        if (event_ != nullptr) {
          *event_ = finalizeEvent.release();
        }
      }

      // Successfully finished the computation
      return StatusCode::kSuccess;
    } catch (...) { return StatusCode::kInvalidKernel; }
  } catch (...) { return StatusCode::kTempBufferAllocFailure; }
}

// =================================================================================================

// Compiles the templated class
template class Xgemm<half>;
template class Xgemm<float>;

// =================================================================================================
} // namespace clblast
