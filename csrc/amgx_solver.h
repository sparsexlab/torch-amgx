// torch-amgx -- C++ wrapper for NVIDIA AmgX, designed for zero-copy
// integration with PyTorch CUDA tensors.
//
// Lifecycle (RAII):
//   AmgXSolver solver(config_str);     // creates Config + Resources + Solver
//   solver.setup_csr(...);             // uploads matrix, runs AmgX setup
//   solver.solve(b, x);                // runs N V-cycles + Krylov iterations
//   // destructor: destroy Solver, Matrix, Resources, Config in that order
//
// Initialization is global and handled at module load via `amgx_initialize()`
// / `amgx_finalize()`. We deliberately do NOT register finalize with
// atexit because Python's interpreter shutdown ordering destroys atexit
// callbacks before module globals, leaving solver handles dangling.
#pragma once

// We use <torch/extension.h> (not <torch/torch.h>) to keep the include
// graph small and avoid MSVC's std::-ambiguity errors in torch's
// dynamo / compiled_autograd headers -- this is a binding-only library
// and never references autograd::Function classes.
#include <torch/extension.h>
#include <amgx_c.h>

#include <memory>
#include <string>

namespace torch_amgx {

// ---------------------------------------------------------------------- //
// Global init / finalize
// ---------------------------------------------------------------------- //
void amgx_initialize();
void amgx_finalize_if_initialized();   // for explicit shutdown if needed
bool amgx_is_initialized();
std::string amgx_version();

// ---------------------------------------------------------------------- //
// AmgX resource graph owned by a single solver instance.
// All operations require the caller to hold a torch CUDA context on the
// same device as the resource's GPU.
// ---------------------------------------------------------------------- //
class AmgXSolver {
public:
    // Constructs (Config, Resources, Solver, Matrix, Vector) handles bound
    // to ``device``. Throws std::runtime_error on AmgX API failure.
    explicit AmgXSolver(const std::string& config_str,
                        torch::Device device);
    ~AmgXSolver();

    // Non-copyable, non-movable -- the AmgX handles cannot be safely
    // shared between owners.
    AmgXSolver(const AmgXSolver&) = delete;
    AmgXSolver& operator=(const AmgXSolver&) = delete;
    AmgXSolver(AmgXSolver&&) = delete;
    AmgXSolver& operator=(AmgXSolver&&) = delete;

    // Upload a CSR matrix from torch tensors and run AmgX setup.
    // ``indptr`` / ``indices`` / ``values`` must be 1-D CUDA tensors on
    // the same device the solver was constructed for. ``shape`` is the
    // square matrix dimension.
    //
    // ``indptr``  -- int32 or int64, length n + 1
    // ``indices`` -- int32 or int64, length nnz
    // ``values``  -- float32 or float64, length nnz
    void setup_csr(const torch::Tensor& indptr,
                   const torch::Tensor& indices,
                   const torch::Tensor& values,
                   int64_t n);

    // Solve A x = b. Returns the solution as a new tensor on the same
    // device + dtype as ``b``. Requires ``setup_csr`` to have been
    // called.
    torch::Tensor solve(const torch::Tensor& b);

    // Same as ``solve`` but writes into a caller-provided ``x`` buffer
    // (which must be a CUDA tensor on the same device/dtype as ``b``);
    // useful for warm starts and to avoid an output allocation.
    void solve_into(const torch::Tensor& b, torch::Tensor& x);

    // Diagnostics from the most recent solve() / solve_into() call.
    int       last_iter_count() const { return last_iter_count_; }
    double    last_residual()   const { return last_residual_;   }
    bool      last_converged()  const { return last_converged_;  }
    torch::Device device() const { return device_; }

private:
    void destroy_();
    void check_(AMGX_RC rc, const char* what);

    torch::Device device_;
    bool setup_done_ = false;

    // AmgX opaque handles (declared in amgx_c.h). nullptr until init.
    AMGX_config_handle    config_   = nullptr;
    AMGX_resources_handle resources_ = nullptr;
    AMGX_matrix_handle    matrix_   = nullptr;
    AMGX_solver_handle    solver_   = nullptr;

    // Stats from the last solve(). solver_get_iterations_number / etc.
    int    last_iter_count_ = 0;
    double last_residual_   = std::numeric_limits<double>::quiet_NaN();
    bool   last_converged_  = false;
};

}  // namespace torch_amgx
