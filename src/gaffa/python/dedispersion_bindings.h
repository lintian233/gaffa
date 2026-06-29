#pragma once

namespace pybind11 {
class module_;
}

namespace gaffa::python {

void bind_dedispersion(pybind11::module_& module);

}  // namespace gaffa::python
