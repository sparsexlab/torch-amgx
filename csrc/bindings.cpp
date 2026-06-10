// torch-amgx -- pybind11 + torch op registration.
//
// Exposes two surfaces to Python:
//
//   1. `torch_amgx._C.AmgXSolver` -- the C++ class for explicit lifecycle
//      control.
//
//   2. `torch_amgx._C.solve_csr` -- one-shot forward solve.
//
// No autograd is registered here on purpose: torch-amgx is the thin
// binding layer. Higher-level libraries (e.g. torch-sla) wrap these
// primitives in their own `torch.autograd.Function` to provide
// backward via the adjoint solve A^T grad_u = grad_b.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <torch/extension.h>

#include "amgx_solver.h"

namespace py = pybind11;
using torch_amgx::AmgXSolver;

namespace {

// ---------------------------------------------------------------------- //
// Functional one-shot solve
// ---------------------------------------------------------------------- //
torch::Tensor solve_csr(const torch::Tensor& indptr,
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

    // Functional one-shot solve (forward only)
    m.def("solve_csr", &solve_csr,
          "One-shot forward solve. No autograd is registered; wrap this "
          "in your own torch.autograd.Function for backward.",
          py::arg("indptr"), py::arg("indices"), py::arg("values"),
          py::arg("n"),      py::arg("b"),       py::arg("config_str"));
}
