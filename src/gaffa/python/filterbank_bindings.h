#pragma once

#include "gaffa/filterbank.h"

#include <pybind11/numpy.h>

namespace pybind11 {
class module_;
}

namespace gaffa::python {

struct PyFilterbank {
  gaffa::FilterbankHeader header;
  pybind11::array data;
};

void bind_filterbank(pybind11::module_& module);

}  // namespace gaffa::python
