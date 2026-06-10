// torch-amgx -- C++ implementation of the AmgX wrapper.
//
// All AmgX calls happen via the public C API declared in <amgx_c.h>.
// Where possible we use AmgX's CSR upload variant that takes raw device
// pointers, so torch CUDA tensors are passed through without an extra
// host round-trip.
#include "amgx_solver.h"

#include <torch/cuda.h>

#include <atomic>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace torch_amgx {
namespace {

// Global initialization state. AmgX requires a single initialize/finalize
// per process; multiple init calls return SUCCESS but are wasted, and
// premature finalize crashes any outstanding handles.
std::atomic<bool> g_initialized{false};
std::mutex        g_init_mutex;

// AmgX's printf-style error callback. We discard noise from the library
// itself (which already prints to stderr) and surface the message in the
// next exception we throw.
//
// Note: AmgX expects ``void (*)(const char*, int)`` -- no AMGX_API on
// the definition (the macro expands to __declspec(dllimport) on
// Windows, which the compiler refuses on a function *body*).
char g_last_error_msg[1024] = {0};
void amgx_print_callback(const char* msg, int /*length*/) {
    std::snprintf(g_last_error_msg, sizeof(g_last_error_msg), "%s", msg);
}

void check_amgx(AMGX_RC rc, const char* what) {
    if (rc == AMGX_RC_OK) return;
    char buf[1024] = {0};
    AMGX_get_error_string(rc, buf, sizeof(buf));
    std::ostringstream oss;
    oss << "AmgX error in " << what << ": " << buf;
    if (g_last_error_msg[0] != 0) {
        oss << " (" << g_last_error_msg << ")";
        g_last_error_msg[0] = 0;
    }
    throw std::runtime_error(oss.str());
}

// Map torch dtypes to AmgX mode enum values. AmgX modes encode (host,
// device, matrix-precision, vector-precision, index-precision) into a
// single int constant. We restrict to dDDI (device-double-double-int32)
// and dFFI (device-float-float-int32) for now -- adding complex modes
// is a straight extension once the underlying AmgX C API stabilises
// them.
//
// We do NOT use ``AMGX_mode_from_str`` -- that helper isn't in the
// public 2.5+ AmgX C API. The enum constants in amgx_c.h are the
// supported way.
AMGX_Mode mode_for_dtype(torch::ScalarType dtype) {
    if (dtype == torch::kFloat64) return AMGX_mode_dDDI;
    if (dtype == torch::kFloat32) return AMGX_mode_dFFI;
    std::ostringstream oss;
    oss << "torch-amgx: unsupported dtype " << dtype
        << "; expected float32 or float64";
    throw std::runtime_error(oss.str());
}

}  // namespace

// ---------------------------------------------------------------------- //
// Global init / finalize
// ---------------------------------------------------------------------- //
void amgx_initialize() {
    std::lock_guard<std::mutex> guard(g_init_mutex);
    if (g_initialized.load()) return;

    check_amgx(AMGX_initialize(), "AMGX_initialize");
    // Plugins are deprecated in AmgX 2.5+ but still required at init.
    AMGX_initialize_plugins();
    AMGX_register_print_callback(&amgx_print_callback);
    AMGX_install_signal_handler();
    g_initialized.store(true);
}

void amgx_finalize_if_initialized() {
    std::lock_guard<std::mutex> guard(g_init_mutex);
    if (!g_initialized.load()) return;
    AMGX_finalize_plugins();
    AMGX_finalize();
    g_initialized.store(false);
}

bool amgx_is_initialized() { return g_initialized.load(); }

std::string amgx_version() {
    int major = 0, minor = 0;
    AMGX_get_api_version(&major, &minor);
    std::ostringstream oss;
    oss << major << "." << minor;
    return oss.str();
}

// ---------------------------------------------------------------------- //
// AmgXSolver
// ---------------------------------------------------------------------- //
AmgXSolver::AmgXSolver(const std::string& config_str, torch::Device device)
    : device_(device)
{
    if (!device.is_cuda()) {
        throw std::runtime_error(
            "torch-amgx: AmgXSolver requires a CUDA device; got " +
            device.str());
    }
    amgx_initialize();

    check_amgx(
        AMGX_config_create(&config_, config_str.c_str()),
        "AMGX_config_create");

    // Resources are bound to a single GPU. AmgX takes a device id list +
    // count; we pass the device index of the torch device.
    int device_id = device.index();
    check_amgx(
        AMGX_resources_create_simple(&resources_, config_),
        "AMGX_resources_create_simple");
}

AmgXSolver::~AmgXSolver() {
    destroy_();
}

void AmgXSolver::destroy_() {
    // Reverse-construction order. Guarded so partially-constructed solvers
    // still clean up correctly.
    if (solver_)    { AMGX_solver_destroy(solver_);       solver_    = nullptr; }
    if (matrix_)    { AMGX_matrix_destroy(matrix_);       matrix_    = nullptr; }
    if (resources_) { AMGX_resources_destroy(resources_); resources_ = nullptr; }
    if (config_)    { AMGX_config_destroy(config_);       config_    = nullptr; }
}

void AmgXSolver::setup_csr(const torch::Tensor& indptr,
                           const torch::Tensor& indices,
                           const torch::Tensor& values,
                           int64_t n) {
    TORCH_CHECK(indptr.device() == device_,
                "indptr must be on the same device as the solver");
    TORCH_CHECK(indices.device() == device_,
                "indices must be on the same device as the solver");
    TORCH_CHECK(values.device() == device_,
                "values must be on the same device as the solver");
    TORCH_CHECK(indptr.is_contiguous() && indices.is_contiguous()
                && values.is_contiguous(),
                "AmgX CSR inputs must be contiguous");
    TORCH_CHECK(indptr.numel() == n + 1,
                "indptr must have length n+1");

    // Cast index tensors to int32 (AmgX expects 32-bit CSR offsets).
    auto idx_dtype = indptr.dtype();
    torch::Tensor indptr32  = indptr.scalar_type()  == torch::kInt32
                              ? indptr  : indptr.to(torch::kInt32);
    torch::Tensor indices32 = indices.scalar_type() == torch::kInt32
                              ? indices : indices.to(torch::kInt32);

    int64_t nnz = indices32.numel();

    AMGX_Mode mode = mode_for_dtype(values.scalar_type());

    // (Re)create the matrix and solver handles each time setup is called
    // so we don't carry stale state. Destroy current ones first.
    if (solver_) { AMGX_solver_destroy(solver_); solver_ = nullptr; }
    if (matrix_) { AMGX_matrix_destroy(matrix_); matrix_ = nullptr; }

    check_amgx(AMGX_matrix_create(&matrix_, resources_, mode),
               "AMGX_matrix_create");
    check_amgx(AMGX_solver_create(&solver_, resources_, mode, config_),
               "AMGX_solver_create");

    // Upload CSR using the device-pointer variant -- zero host copy.
    check_amgx(
        AMGX_matrix_upload_all(
            matrix_,
            /*n=*/static_cast<int>(n),
            /*nnz=*/static_cast<int>(nnz),
            /*block_dimx=*/1,
            /*block_dimy=*/1,
            indptr32.data_ptr<int>(),
            indices32.data_ptr<int>(),
            values.data_ptr(),
            /*diag_data=*/nullptr),
        "AMGX_matrix_upload_all");

    check_amgx(AMGX_solver_setup(solver_, matrix_), "AMGX_solver_setup");
    setup_done_ = true;
}

void AmgXSolver::solve_into(const torch::Tensor& b, torch::Tensor& x) {
    TORCH_CHECK(setup_done_,
                "AmgXSolver::solve called before setup_csr");
    TORCH_CHECK(b.device() == device_ && x.device() == device_,
                "b and x must be on the solver's device");
    TORCH_CHECK(b.scalar_type() == x.scalar_type(),
                "b and x must have the same dtype");
    TORCH_CHECK(b.is_contiguous() && x.is_contiguous(),
                "b and x must be contiguous");
    TORCH_CHECK(b.dim() == 1 && x.dim() == 1,
                "Only 1-D right-hand sides supported in this release");
    TORCH_CHECK(b.numel() == x.numel(),
                "b and x must have the same length");

    AMGX_Mode mode = mode_for_dtype(b.scalar_type());

    AMGX_vector_handle b_vec = nullptr;
    AMGX_vector_handle x_vec = nullptr;
    check_amgx(
        AMGX_vector_create(&b_vec, resources_, mode),
        "AMGX_vector_create(b)");
    check_amgx(
        AMGX_vector_create(&x_vec, resources_, mode),
        "AMGX_vector_create(x)");

    try {
        // upload_raw takes a device pointer and a size -- direct from
        // the torch tensor's storage.
        check_amgx(
            AMGX_vector_upload(b_vec, /*n=*/static_cast<int>(b.numel()),
                               /*block_dim=*/1, b.data_ptr()),
            "AMGX_vector_upload(b)");
        check_amgx(
            AMGX_vector_upload(x_vec, /*n=*/static_cast<int>(x.numel()),
                               /*block_dim=*/1, x.data_ptr()),
            "AMGX_vector_upload(x0)");

        check_amgx(
            AMGX_solver_solve(solver_, b_vec, x_vec),
            "AMGX_solver_solve");

        // Download x in-place into the caller's torch tensor.
        check_amgx(
            AMGX_vector_download(x_vec, x.data_ptr()),
            "AMGX_vector_download(x)");

        // Diagnostics
        AMGX_SOLVE_STATUS status;
        AMGX_solver_get_status(solver_, &status);
        last_converged_ = (status == AMGX_SOLVE_SUCCESS);

        int iters = 0;
        AMGX_solver_get_iterations_number(solver_, &iters);
        last_iter_count_ = iters;

        if (iters > 0) {
            // Final residual norm of the last iteration on the only
            // block (block 0). Failure to fetch leaves the previous nan
            // in place.
            double rn = 0;
            AMGX_RC rc =
                AMGX_solver_get_iteration_residual(solver_, iters - 1, 0, &rn);
            if (rc == AMGX_RC_OK) last_residual_ = rn;
        }
    } catch (...) {
        AMGX_vector_destroy(b_vec);
        AMGX_vector_destroy(x_vec);
        throw;
    }
    AMGX_vector_destroy(b_vec);
    AMGX_vector_destroy(x_vec);
}

torch::Tensor AmgXSolver::solve(const torch::Tensor& b) {
    torch::Tensor x = torch::zeros_like(b);
    solve_into(b, x);
    return x;
}

void AmgXSolver::check_(AMGX_RC rc, const char* what) {
    check_amgx(rc, what);
}

}  // namespace torch_amgx
