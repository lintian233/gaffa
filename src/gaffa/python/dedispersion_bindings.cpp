#include "dedispersion_bindings.h"

#include "filterbank_bindings.h"
#include "gaffa/dedispersion.h"
#include "gaffa/dedispersion_cuda.h"
#include "gaffa/filterbank_view.h"
#include "gaffa/sample_view.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

enum class PythonDedispersionBackend {
  Cpu,
  Cuda,
};

struct PyDedispersedResult {
  py::array data;
  std::string backend;
  double dm_low = 0.0;
  double dm_step = 0.0;
  std::size_t ndm = 0;
  std::size_t nsamples = 0;
  double tsamp = 0.0;
};

struct PyDedispersedSpectrum {
  py::array data;
  std::string backend;
  double dm = 0.0;
  std::size_t nsamples = 0;
  std::size_t nchans = 0;
  double tsamp = 0.0;
  std::size_t chan_begin = 0;
  std::size_t chan_end = 0;
};

PythonDedispersionBackend parse_backend(const std::string& value) {
  if (value == "cpu") {
    return PythonDedispersionBackend::Cpu;
  }
  if (value == "cuda") {
    return PythonDedispersionBackend::Cuda;
  }
  throw py::value_error("unknown dedispersion backend: " + value);
}

std::string backend_name(PythonDedispersionBackend backend) {
  switch (backend) {
    case PythonDedispersionBackend::Cpu:
      return "cpu";
    case PythonDedispersionBackend::Cuda:
      return "cuda";
  }
  throw std::logic_error("unknown dedispersion backend");
}

void validate_filterbank_header(const gaffa::FilterbankHeader& header) {
  if (header.nsamples <= 0) {
    throw std::invalid_argument("dedispersion requires at least one sample");
  }
  if (header.nifs != 1) {
    throw std::invalid_argument("dedispersion requires nifs == 1");
  }
  if (header.nchans <= 0) {
    throw std::invalid_argument("dedispersion requires at least one channel");
  }
  if (!(header.tsamp > 0.0)) {
    throw std::invalid_argument("filterbank tsamp must be positive");
  }
  if (header.frequency_table.size() != static_cast<std::size_t>(header.nchans)) {
    throw std::invalid_argument(
        "filterbank frequency table length must match nchans");
  }
}

double reference_frequency_mhz(const gaffa::FilterbankHeader& header) {
  validate_filterbank_header(header);
  return *std::max_element(header.frequency_table.begin(),
                           header.frequency_table.end());
}

gaffa::SampleShape sample_shape_from_header(
    const gaffa::FilterbankHeader& header) {
  return gaffa::sample_shape(header);
}

template <typename T>
gaffa::HostSampleView<T> typed_sample_view(
    const gaffa::python::PyFilterbank& filterbank) {
  const gaffa::SampleShape shape = sample_shape_from_header(filterbank.header);
  const py::buffer_info info = filterbank.data.request();

  if (info.ndim != 3) {
    throw std::invalid_argument("filterbank data must be a 3D numpy array");
  }
  if (info.itemsize != static_cast<py::ssize_t>(sizeof(T))) {
    throw std::invalid_argument("filterbank data dtype does not match header");
  }
  if (info.shape[0] != static_cast<py::ssize_t>(shape.nsamples) ||
      info.shape[1] != static_cast<py::ssize_t>(shape.nifs) ||
      info.shape[2] != static_cast<py::ssize_t>(shape.nchans)) {
    throw std::invalid_argument("filterbank data shape does not match header");
  }

  const py::ssize_t item_size = static_cast<py::ssize_t>(sizeof(T));
  const py::ssize_t expected_chan_stride = item_size;
  const py::ssize_t expected_if_stride =
      static_cast<py::ssize_t>(shape.nchans) * item_size;
  const py::ssize_t expected_sample_stride =
      static_cast<py::ssize_t>(shape.nifs * shape.nchans) * item_size;
  if (info.strides[0] != expected_sample_stride ||
      info.strides[1] != expected_if_stride ||
      info.strides[2] != expected_chan_stride) {
    throw std::invalid_argument(
        "filterbank data must be C-contiguous in (time, if, channel) order");
  }

  return gaffa::HostSampleView<T>{
      .data = static_cast<const T*>(info.ptr),
      .shape = shape,
  };
}

template <typename T>
py::array make_numpy_array(gaffa::DedispersedResult<T>&& result) {
  auto owner = std::make_unique<std::vector<T>>(std::move(result.data));
  T* data = owner->data();
  py::capsule capsule(owner.release(), [](void* pointer) {
    delete static_cast<std::vector<T>*>(pointer);
  });

  const auto ndm = static_cast<py::ssize_t>(result.shape.ndm);
  const auto nsamples = static_cast<py::ssize_t>(result.shape.nsamples);
  const auto item_size = static_cast<py::ssize_t>(sizeof(T));
  return py::array_t<T>({ndm, nsamples}, {nsamples * item_size, item_size},
                        data, capsule);
}

template <typename T>
PyDedispersedResult make_python_result(gaffa::DedispersedResult<T>&& result,
                                       PythonDedispersionBackend backend,
                                       double dm_low, double dm_step,
                                       double tsamp) {
  PyDedispersedResult py_result;
  py_result.ndm = result.shape.ndm;
  py_result.nsamples = result.shape.nsamples;
  py_result.data = make_numpy_array(std::move(result));
  py_result.backend = backend_name(backend);
  py_result.dm_low = dm_low;
  py_result.dm_step = dm_step;
  py_result.tsamp = tsamp;
  return py_result;
}

template <typename T>
py::array make_numpy_array(gaffa::DedispersedSpectrum<T>&& result) {
  auto owner = std::make_unique<std::vector<T>>(std::move(result.data));
  T* data = owner->data();
  py::capsule capsule(owner.release(), [](void* pointer) {
    delete static_cast<std::vector<T>*>(pointer);
  });

  const auto nsamples = static_cast<py::ssize_t>(result.shape.nsamples);
  const auto nchans = static_cast<py::ssize_t>(result.shape.nchans);
  const auto item_size = static_cast<py::ssize_t>(sizeof(T));
  return py::array_t<T>({nsamples, nchans}, {nchans * item_size, item_size},
                        data, capsule);
}

template <typename T>
PyDedispersedSpectrum make_python_spectrum(
    gaffa::DedispersedSpectrum<T>&& result, PythonDedispersionBackend backend,
    double tsamp) {
  PyDedispersedSpectrum py_result;
  py_result.nsamples = result.shape.nsamples;
  py_result.nchans = result.shape.nchans;
  py_result.backend = backend_name(backend);
  py_result.dm = result.dm;
  py_result.tsamp = tsamp;
  py_result.chan_begin = result.chan_begin;
  py_result.chan_end = result.chan_end;
  py_result.data = make_numpy_array(std::move(result));
  return py_result;
}

gaffa::SingleDmDedispersionPlan make_single_dm_plan(
    const gaffa::FilterbankHeader& header, double dm) {
  return gaffa::SingleDmDedispersionPlan{
      .dm = dm,
      .ref_frequency_mhz = reference_frequency_mhz(header),
      .tsamp = header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(header.nchans),
  };
}

gaffa::MultiDmDedispersionPlan make_multi_dm_plan(
    const gaffa::FilterbankHeader& header, double dm_low, double dm_step,
    std::size_t ndm) {
  return gaffa::MultiDmDedispersionPlan{
      .dm_low = dm_low,
      .dm_step = dm_step,
      .ndm = ndm,
      .ref_frequency_mhz = reference_frequency_mhz(header),
      .tsamp = header.tsamp,
      .chan_begin = 0,
      .chan_end = static_cast<std::size_t>(header.nchans),
  };
}

template <typename T>
PyDedispersedSpectrum dedisperse_spectrum_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    PythonDedispersionBackend backend,
    const gaffa::CudaDedispersionOptions& cuda_options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = make_single_dm_plan(filterbank.header, dm);
  if (backend == PythonDedispersionBackend::Cpu) {
    decltype(gaffa::dedisperse_spectrum_cpu(
        samples, filterbank.header.frequency_table, plan)) result;
    {
      py::gil_scoped_release release;
      result = gaffa::dedisperse_spectrum_cpu(
          samples, filterbank.header.frequency_table, plan);
    }
    return make_python_spectrum(std::move(result), backend,
                                filterbank.header.tsamp);
  }

  decltype(gaffa::dedisperse_spectrum_cuda(
      samples, filterbank.header.frequency_table, plan, cuda_options)) result;
  {
    py::gil_scoped_release release;
    result = gaffa::dedisperse_spectrum_cuda(
        samples, filterbank.header.frequency_table, plan, cuda_options);
  }
  return make_python_spectrum(std::move(result), backend,
                              filterbank.header.tsamp);
}

template <typename T>
PyDedispersedResult dedisperse_single_dm_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    PythonDedispersionBackend backend,
    const gaffa::CudaDedispersionOptions& cuda_options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = make_single_dm_plan(filterbank.header, dm);
  if (backend == PythonDedispersionBackend::Cpu) {
    decltype(gaffa::dedisperse_single_dm_cpu(
        samples, filterbank.header.frequency_table, plan)) result;
    {
      py::gil_scoped_release release;
      result = gaffa::dedisperse_single_dm_cpu(
          samples, filterbank.header.frequency_table, plan);
    }
    return make_python_result(std::move(result), backend, dm, 0.0,
                              filterbank.header.tsamp);
  }

  decltype(gaffa::dedisperse_single_dm_cuda(
      samples, filterbank.header.frequency_table, plan, cuda_options)) result;
  {
    py::gil_scoped_release release;
    result = gaffa::dedisperse_single_dm_cuda(
        samples, filterbank.header.frequency_table, plan, cuda_options);
  }
  return make_python_result(std::move(result), backend, dm, 0.0,
                            filterbank.header.tsamp);
}

template <typename T>
PyDedispersedResult dedisperse_multi_dm_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, PythonDedispersionBackend backend,
    const gaffa::CudaDedispersionOptions& cuda_options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = make_multi_dm_plan(filterbank.header, dm_low, dm_step, ndm);
  if (backend == PythonDedispersionBackend::Cpu) {
    decltype(gaffa::dedisperse_multi_dm_cpu(
        samples, filterbank.header.frequency_table, plan)) result;
    {
      py::gil_scoped_release release;
      result = gaffa::dedisperse_multi_dm_cpu(
          samples, filterbank.header.frequency_table, plan);
    }
    return make_python_result(std::move(result), backend, dm_low, dm_step,
                              filterbank.header.tsamp);
  }

  decltype(gaffa::dedisperse_multi_dm_cuda(
      samples, filterbank.header.frequency_table, plan, cuda_options)) result;
  {
    py::gil_scoped_release release;
    result = gaffa::dedisperse_multi_dm_cuda(
        samples, filterbank.header.frequency_table, plan, cuda_options);
  }
  return make_python_result(std::move(result), backend, dm_low, dm_step,
                            filterbank.header.tsamp);
}

template <typename T>
PyDedispersedResult dedisperse_subband_typed(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, PythonDedispersionBackend backend,
    const gaffa::SubbandDedispersionOptions& subband_options,
    const gaffa::CudaDedispersionOptions& cuda_options) {
  const auto samples = typed_sample_view<T>(filterbank);
  const auto plan = make_multi_dm_plan(filterbank.header, dm_low, dm_step, ndm);
  if (backend == PythonDedispersionBackend::Cpu) {
    decltype(gaffa::dedisperse_subband_cpu(
        samples, filterbank.header.frequency_table, plan,
        subband_options)) result;
    {
      py::gil_scoped_release release;
      result = gaffa::dedisperse_subband_cpu(
          samples, filterbank.header.frequency_table, plan, subband_options);
    }
    return make_python_result(std::move(result), backend, dm_low, dm_step,
                              filterbank.header.tsamp);
  }

  decltype(gaffa::dedisperse_subband_cuda(
      samples, filterbank.header.frequency_table, plan, subband_options,
      cuda_options)) result;
  {
    py::gil_scoped_release release;
    result = gaffa::dedisperse_subband_cuda(
        samples, filterbank.header.frequency_table, plan, subband_options,
        cuda_options);
  }
  return make_python_result(std::move(result), backend, dm_low, dm_step,
                            filterbank.header.tsamp);
}

template <typename Func>
PyDedispersedResult dispatch_filterbank_dtype(
    const gaffa::python::PyFilterbank& filterbank, Func&& func) {
  validate_filterbank_header(filterbank.header);
  switch (filterbank.header.nbits) {
    case 8:
      return func.template operator()<std::uint8_t>();
    case 16:
      return func.template operator()<std::uint16_t>();
    case 32:
      return func.template operator()<float>();
    default:
      throw py::type_error("unsupported filterbank sample dtype");
  }
}

template <typename Func>
PyDedispersedSpectrum dispatch_filterbank_spectrum_dtype(
    const gaffa::python::PyFilterbank& filterbank, Func&& func) {
  validate_filterbank_header(filterbank.header);
  switch (filterbank.header.nbits) {
    case 8:
      return func.template operator()<std::uint8_t>();
    case 16:
      return func.template operator()<std::uint16_t>();
    case 32:
      return func.template operator()<float>();
    default:
      throw py::type_error("unsupported filterbank sample dtype");
  }
}

PyDedispersedSpectrum dedisperse_spectrum_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    const std::string& backend, int device_id, std::size_t threads_per_block,
    std::size_t time_tile_samples) {
  const PythonDedispersionBackend parsed_backend = parse_backend(backend);
  const gaffa::CudaDedispersionOptions cuda_options{
      .device_id = device_id,
      .threads_per_block = threads_per_block,
      .time_tile_samples = time_tile_samples,
  };

  return dispatch_filterbank_spectrum_dtype(filterbank, [&]<typename T>() {
    return dedisperse_spectrum_typed<T>(filterbank, dm, parsed_backend,
                                        cuda_options);
  });
}

PyDedispersedResult dedisperse_single_dm_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm,
    const std::string& backend, int device_id, std::size_t threads_per_block,
    std::size_t time_tile_samples) {
  const PythonDedispersionBackend parsed_backend = parse_backend(backend);
  const gaffa::CudaDedispersionOptions cuda_options{
      .device_id = device_id,
      .threads_per_block = threads_per_block,
      .time_tile_samples = time_tile_samples,
  };

  return dispatch_filterbank_dtype(filterbank, [&]<typename T>() {
    return dedisperse_single_dm_typed<T>(filterbank, dm, parsed_backend,
                                         cuda_options);
  });
}

PyDedispersedResult dedisperse_multi_dm_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, const std::string& backend, int device_id,
    std::size_t threads_per_block, std::size_t time_tile_samples) {
  const PythonDedispersionBackend parsed_backend = parse_backend(backend);
  const gaffa::CudaDedispersionOptions cuda_options{
      .device_id = device_id,
      .threads_per_block = threads_per_block,
      .time_tile_samples = time_tile_samples,
  };

  return dispatch_filterbank_dtype(filterbank, [&]<typename T>() {
    return dedisperse_multi_dm_typed<T>(filterbank, dm_low, dm_step, ndm,
                                        parsed_backend, cuda_options);
  });
}

PyDedispersedResult dedisperse_subband_for_python(
    const gaffa::python::PyFilterbank& filterbank, double dm_low,
    double dm_step, std::size_t ndm, const std::string& backend,
    std::size_t subband_channels, std::size_t ndm_per_nominal, int device_id,
    std::size_t threads_per_block, std::size_t time_tile_samples) {
  const PythonDedispersionBackend parsed_backend = parse_backend(backend);
  const gaffa::SubbandDedispersionOptions subband_options{
      .subband_channels = subband_channels,
      .ndm_per_nominal = ndm_per_nominal,
  };
  const gaffa::CudaDedispersionOptions cuda_options{
      .device_id = device_id,
      .threads_per_block = threads_per_block,
      .time_tile_samples = time_tile_samples,
  };

  return dispatch_filterbank_dtype(filterbank, [&]<typename T>() {
    return dedisperse_subband_typed<T>(filterbank, dm_low, dm_step, ndm,
                                       parsed_backend, subband_options,
                                       cuda_options);
  });
}

std::string dedispersed_result_repr(const PyDedispersedResult& result) {
  return "<DedispersedResult shape=(" + std::to_string(result.ndm) + ", " +
         std::to_string(result.nsamples) + ") backend=" + result.backend + ">";
}

std::string dedispersed_spectrum_repr(const PyDedispersedSpectrum& result) {
  return "<DedispersedSpectrum shape=(" + std::to_string(result.nsamples) +
         ", " + std::to_string(result.nchans) + ") backend=" + result.backend +
         ">";
}

}  // namespace

namespace gaffa::python {

void bind_dedispersion(py::module_& module) {
  py::class_<PyDedispersedResult>(module, "DedispersedResult")
      .def_readonly("data", &PyDedispersedResult::data)
      .def_readonly("backend", &PyDedispersedResult::backend)
      .def_readonly("dm_low", &PyDedispersedResult::dm_low)
      .def_readonly("dm_step", &PyDedispersedResult::dm_step)
      .def_readonly("ndm", &PyDedispersedResult::ndm)
      .def_readonly("nsamples", &PyDedispersedResult::nsamples)
      .def_readonly("tsamp", &PyDedispersedResult::tsamp)
      .def_property_readonly("shape",
                             [](const PyDedispersedResult& result) {
                               return py::make_tuple(result.ndm,
                                                     result.nsamples);
                             })
      .def_property_readonly("dtype",
                             [](const PyDedispersedResult& result) {
                               return result.data.attr("dtype");
                             })
      .def_property_readonly("nbytes",
                             [](const PyDedispersedResult& result) {
                               return result.data.attr("nbytes");
                             })
      .def("__repr__", &dedispersed_result_repr);

  py::class_<PyDedispersedSpectrum>(module, "DedispersedSpectrum")
      .def_readonly("data", &PyDedispersedSpectrum::data)
      .def_readonly("backend", &PyDedispersedSpectrum::backend)
      .def_readonly("dm", &PyDedispersedSpectrum::dm)
      .def_readonly("nsamples", &PyDedispersedSpectrum::nsamples)
      .def_readonly("nchans", &PyDedispersedSpectrum::nchans)
      .def_readonly("tsamp", &PyDedispersedSpectrum::tsamp)
      .def_readonly("chan_begin", &PyDedispersedSpectrum::chan_begin)
      .def_readonly("chan_end", &PyDedispersedSpectrum::chan_end)
      .def_property_readonly("shape",
                             [](const PyDedispersedSpectrum& result) {
                               return py::make_tuple(result.nsamples,
                                                     result.nchans);
                             })
      .def_property_readonly("dtype",
                             [](const PyDedispersedSpectrum& result) {
                               return result.data.attr("dtype");
                             })
      .def_property_readonly("nbytes",
                             [](const PyDedispersedSpectrum& result) {
                               return result.data.attr("nbytes");
                             })
      .def("__repr__", &dedispersed_spectrum_repr);

  module.def("dedisperse_spectrum", &dedisperse_spectrum_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm"),
             py::arg("backend") = "cpu", py::arg("device_id") = 0,
             py::arg("threads_per_block") = 256,
             py::arg("time_tile_samples") = 81920);

  module.def("dedisperse_single_dm", &dedisperse_single_dm_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm"),
             py::arg("backend") = "cpu", py::arg("device_id") = 0,
             py::arg("threads_per_block") = 256,
             py::arg("time_tile_samples") = 81920);

  module.def("dedisperse_multi_dm", &dedisperse_multi_dm_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm_low"),
             py::arg("dm_step"), py::arg("ndm"), py::arg("backend") = "cpu",
             py::arg("device_id") = 0, py::arg("threads_per_block") = 256,
             py::arg("time_tile_samples") = 81920);

  module.def("dedisperse_subband", &dedisperse_subband_for_python,
             py::arg("filterbank"), py::kw_only(), py::arg("dm_low"),
             py::arg("dm_step"), py::arg("ndm"), py::arg("backend") = "cuda",
             py::arg("subband_channels") = 32,
             py::arg("ndm_per_nominal") = 32, py::arg("device_id") = 0,
             py::arg("threads_per_block") = 256,
             py::arg("time_tile_samples") = 81920);
}

}  // namespace gaffa::python
