// torch-amgx -- pybind11 + torch op registration.
//
// Exposes two surfaces to Python:
//
//   1. `torch_amgx._C.AmgXSolver` -- direct class binding for users who
//      want explicit lifecycle control (build solver, hold it across
//      many solves, destroy explicitly).
//
//   2. `torch_amgx._C.solve_csr` -- functional, autograd-aware one-shot
//      solve. Registered as a custom torch op so backward (adjoint
//      solve via conjugate-transpose) is wired into PyTorch's autograd.
//      For the forward pass we build a transient AmgXSolver, run setup
//      + solve, and tear down -- callers wanting amortised setup should
//      use the class form.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>
#include <torch/script.h>

#include "amgx_solver.h"

namespace py = pybind11;
using torch_amgx::AmgXSolver;

namespace {

// ---------------------------------------------------------------------- //
// Functional one-shot solve (autograd path)
// ---------------------------------------------------------------------- //
torch::Tensor solve_csr_forward(const torch::Tensor& indptr,
                                const torch::Tensor& indices,
                                const torch::Tensor& values,
                                int64_t n,
                                const torch::Tensor& b,
                                const std::string& config_str) {
    TORCH_CHECK(values.is_cuda(),
                "torch_amgx.solve_csr: values must be a CUDA tensor");
    AmgXSolver solver(config_str, values.device());
    solver.setup_csr(indptr, indices, values, n);
    return solver.solve(b);
}

// Adjoint for autograd: with x = A^{-1} b,
//   gradb  = A^{-T} gradu
//   gradval[k] = - gradb[row[k]] * x[col[k]]
//
// Backward needs row indices to scatter the per-nnz gradient. CSR
// indptr lets us expand them in a single pass.
class SolveCSRFunction : public torch::autograd::Function<SolveCSRFunction> {
public:
    static torch::Tensor forward(
            torch::autograd::AutogradContext* ctx,
            torch::Tensor indptr,
            torch::Tensor indices,
            torch::Tensor values,
            int64_t n,
            torch::Tensor b,
            std::string config_str) {
        auto x = solve_csr_forward(indptr, indices, values, n, b, config_str);
        ctx->save_for_backward({indptr, indices, values, x});
        ctx->saved_data["n"] = n;
        ctx->saved_data["config_str"] = config_str;
        return x;
    }

    static torch::autograd::tensor_list backward(
            torch::autograd::AutogradContext* ctx,
            torch::autograd::tensor_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto indptr  = saved[0];
        auto indices = saved[1];
        auto values  = saved[2];
        auto x       = saved[3];
        int64_t n    = ctx->saved_data["n"].toInt();
        auto config_str = ctx->saved_data["config_str"].toStringRef();
        auto gradu = grad_outputs[0];

        // For real matrices A^{-T} = A^{-1} on the conjugate-transpose
        // sparsity (swap row/col indices). For complex we'd also conjugate
        // the values; not supported in the first cut.
        auto crow = torch::zeros_like(indptr);
        // Build CSR for A^T: rows<->cols, then recompress.
        // Simpler path: convert to COO row indices, swap, scatter to CSR.
        // Skipped here for the scaffold; see TODO below.
        TORCH_CHECK(false,
                    "SolveCSRFunction::backward is scaffolded but not yet "
                    "implemented; in this release call AmgXSolver directly "
                    "and stage your own autograd Function.");

        return {torch::Tensor(), torch::Tensor(), torch::Tensor(),
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};

torch::Tensor solve_csr_autograd(const torch::Tensor& indptr,
                                 const torch::Tensor& indices,
                                 const torch::Tensor& values,
                                 int64_t n,
                                 const torch::Tensor& b,
                                 const std::string& config_str) {
    return SolveCSRFunction::apply(indptr, indices, values, n, b, config_str);
}

}  // namespace

// ---------------------------------------------------------------------- //
// pybind11 module definition
// ---------------------------------------------------------------------- //
PYBIND11_MODULE(_C, m) {
    m.doc() = "torch-amgx: PyTorch-native AmgX bindings";

    // Module-level helpers
    m.def("initialize", &torch_amgx::amgx_initialize,
          "Eagerly initialize the AmgX runtime (called automatically on "
          "first solver construction; explicit init is optional).");
    m.def("is_initialized", &torch_amgx::amgx_is_initialized);
    m.def("finalize",
          &torch_amgx::amgx_finalize_if_initialized,
          "Tear down the AmgX runtime. Do not call while any solver is "
          "still alive -- AmgX will crash on destruction of stale handles.");
    m.def("amgx_version", &torch_amgx::amgx_version);

    // The class form -- explicit lifecycle, reusable across many solves
    py::class_<AmgXSolver>(m, "AmgXSolver")
        .def(py::init<const std::string&, torch::Device>(),
             py::arg("config_str"), py::arg("device"))
        .def("setup_csr", &AmgXSolver::setup_csr,
             py::arg("indptr"), py::arg("indices"),
             py::arg("values"),  py::arg("n"))
        .def("solve",      &AmgXSolver::solve,      py::arg("b"))
        .def("solve_into", &AmgXSolver::solve_into,
             py::arg("b"), py::arg("x"))
        .def_property_readonly("iter_count", &AmgXSolver::last_iter_count)
        .def_property_readonly("residual",   &AmgXSolver::last_residual)
        .def_property_readonly("converged",  &AmgXSolver::last_converged)
        .def_property_readonly("device",     &AmgXSolver::device);

    // Functional, no-autograd one-shot solve
    m.def("solve_csr_no_grad", &solve_csr_forward,
          "One-shot solve without autograd; for grad-enabled use "
          "`torch_amgx.solve_csr` from the Python facade.",
          py::arg("indptr"), py::arg("indices"), py::arg("values"),
          py::arg("n"),      py::arg("b"),       py::arg("config_str"));

    // Autograd-wired one-shot. Backward is scaffolded only; see TODO in
    // SolveCSRFunction::backward. The Python facade falls back to a
    // hand-rolled adjoint until that lands.
    m.def("solve_csr_autograd", &solve_csr_autograd,
          py::arg("indptr"), py::arg("indices"), py::arg("values"),
          py::arg("n"),      py::arg("b"),       py::arg("config_str"));
}
