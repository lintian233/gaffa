#include "python/dedispersion_bindings.h"
#include "python/filterbank_bindings.h"
#include "python/folding_bindings.h"

#include "gaffa/vector_add.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(_core, module) {
  module.doc() = "gaffa CUDA/C++ extension module";
  gaffa::python::bind_filterbank(module);
  gaffa::python::bind_dedispersion(module);
  gaffa::python::bind_folding(module);
  module.def("vector_add", &gaffa::vector_add, py::arg("lhs"), py::arg("rhs"));
  module.def("cuda_device_count", &gaffa::cuda_device_count);
  module.def("cuda_runtime_version", &gaffa::cuda_runtime_version);
}
