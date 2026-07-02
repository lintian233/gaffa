#include "folding_bindings.h"

#include "gaffa/folding.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace {

struct PyFoldResult {
  py::array cube;
  py::array exposure;
  py::array profile;
  py::array freq_phase;
  py::array time_phase;
  py::array phase;
  py::array time;
  std::size_t nsubint = 0;
  std::size_t nchans = 0;
  std::size_t nbin = 0;
  double period = 0.0;
  double tsamp = 0.0;
  double tsubint = 0.0;
};

template <typename T>
py::array make_owned_array(std::vector<T>&& data,
                           std::vector<py::ssize_t> shape) {
  auto owner = std::make_unique<std::vector<T>>(std::move(data));
  T* pointer = owner->data();
  py::capsule capsule(owner.release(), [](void* raw_pointer) {
    delete static_cast<std::vector<T>*>(raw_pointer);
  });

  std::vector<py::ssize_t> strides(shape.size());
  py::ssize_t stride = static_cast<py::ssize_t>(sizeof(T));
  for (std::ptrdiff_t index = static_cast<std::ptrdiff_t>(shape.size()) - 1;
       index >= 0; --index) {
    strides[static_cast<std::size_t>(index)] = stride;
    stride *= shape[static_cast<std::size_t>(index)];
  }

  return py::array_t<T>(shape, strides, pointer, capsule);
}

std::vector<float> make_phase_axis(std::size_t nbin) {
  std::vector<float> phase(nbin);
  for (std::size_t index = 0; index < nbin; ++index) {
    phase[index] = static_cast<float>(static_cast<double>(index) /
                                      static_cast<double>(nbin));
  }
  return phase;
}

std::vector<double> make_time_axis(std::size_t nsubint, double tsubint) {
  std::vector<double> time(nsubint);
  for (std::size_t index = 0; index < nsubint; ++index) {
    time[index] = (static_cast<double>(index) + 0.5) * tsubint;
  }
  return time;
}

void validate_fold_array_info(const py::buffer_info& info) {
  if (info.ndim != 2) {
    throw py::value_error("dedispersed spectrum data must be a 2D array");
  }
  if (info.shape[0] <= 0 || info.shape[1] <= 0) {
    throw py::value_error("dedispersed spectrum data must not be empty");
  }
  if (info.strides[1] != static_cast<py::ssize_t>(info.itemsize) ||
      info.strides[0] != info.shape[1] * info.strides[1]) {
    throw py::value_error(
        "dedispersed spectrum data must be C-contiguous; use "
        "numpy.ascontiguousarray explicitly if a copy is intended");
  }
}

template <typename T>
bool buffer_matches_dtype(const py::buffer_info& info) {
  return info.itemsize == static_cast<py::ssize_t>(sizeof(T)) &&
         info.format == py::format_descriptor<T>::format();
}

template <typename T>
gaffa::FoldDedispersedSpectrumResult fold_array_view(
    const py::buffer_info& info, double tsamp, double period,
    std::size_t nbin, double tsubint, std::size_t output_channels) {
  validate_fold_array_info(info);
  if (!buffer_matches_dtype<T>(info)) {
    throw py::type_error("dedispersed spectrum data dtype mismatch");
  }

  const auto nsamples = static_cast<std::size_t>(info.shape[0]);
  const auto nchans = static_cast<std::size_t>(info.shape[1]);
  const T* const pointer = static_cast<const T*>(info.ptr);
  const gaffa::HostSampleView<T> samples{
      .data = pointer,
      .shape = gaffa::SampleShape{
          .nsamples = nsamples,
          .nifs = 1,
          .nchans = nchans,
      },
  };

  const gaffa::FoldDedispersedSpectrumOptions options{
      .period = period,
      .nbin = nbin,
      .tsubint = tsubint,
      .output_channels = output_channels,
  };

  py::gil_scoped_release release;
  return gaffa::fold_dedispersed_spectrum_view_cpu(samples, tsamp, options);
}

PyFoldResult make_python_fold_result(
    gaffa::FoldDedispersedSpectrumResult&& result) {
  PyFoldResult py_result;
  py_result.nsubint = result.nsubint;
  py_result.nchans = result.nchans;
  py_result.nbin = result.nbin;
  py_result.period = result.period;
  py_result.tsamp = result.tsamp;
  py_result.tsubint = result.tsubint;

  const auto nsubint = static_cast<py::ssize_t>(result.nsubint);
  const auto nchans = static_cast<py::ssize_t>(result.nchans);
  const auto nbin = static_cast<py::ssize_t>(result.nbin);

  py_result.cube = make_owned_array<float>(
      std::move(result.cube.data), {nsubint, nchans, nbin});
  py_result.exposure = make_owned_array<double>(
      std::move(result.cube.exposure), {nsubint, nbin});
  py_result.profile =
      make_owned_array<float>(std::move(result.profile), {nbin});
  py_result.freq_phase = make_owned_array<float>(
      std::move(result.freq_phase), {nchans, nbin});
  py_result.time_phase = make_owned_array<float>(
      std::move(result.time_phase), {nsubint, nbin});
  py_result.phase = make_owned_array<float>(
      make_phase_axis(result.nbin), {nbin});
  py_result.time = make_owned_array<double>(
      make_time_axis(result.nsubint, result.tsubint), {nsubint});

  return py_result;
}

PyFoldResult fold_dedispersed_spectrum_for_python(
    py::array data,
    double tsamp, double period, std::size_t nbin, double tsubint,
    std::size_t output_channels) {
  const py::buffer_info info = data.request();
  validate_fold_array_info(info);

  gaffa::FoldDedispersedSpectrumResult result;
  if (buffer_matches_dtype<std::uint8_t>(info)) {
    result = fold_array_view<std::uint8_t>(
        info, tsamp, period, nbin, tsubint, output_channels);
  } else if (buffer_matches_dtype<std::uint16_t>(info)) {
    result = fold_array_view<std::uint16_t>(
        info, tsamp, period, nbin, tsubint, output_channels);
  } else if (buffer_matches_dtype<float>(info)) {
    result = fold_array_view<float>(
        info, tsamp, period, nbin, tsubint, output_channels);
  } else {
    throw py::type_error(
        "fold supports uint8, uint16, and float32 C-contiguous arrays");
  }
  return make_python_fold_result(std::move(result));
}

std::string fold_result_repr(const PyFoldResult& result) {
  return "<FoldResult cube_shape=(" + std::to_string(result.nsubint) + ", " +
         std::to_string(result.nchans) + ", " +
         std::to_string(result.nbin) + ")>";
}

}  // namespace

namespace gaffa::python {

void bind_folding(py::module_& module) {
  py::class_<PyFoldResult>(module, "FoldResult")
      .def_readonly("cube", &PyFoldResult::cube)
      .def_readonly("exposure", &PyFoldResult::exposure)
      .def_readonly("profile", &PyFoldResult::profile)
      .def_readonly("freq_phase", &PyFoldResult::freq_phase)
      .def_readonly("time_phase", &PyFoldResult::time_phase)
      .def_readonly("phase", &PyFoldResult::phase)
      .def_readonly("time", &PyFoldResult::time)
      .def_readonly("nsubint", &PyFoldResult::nsubint)
      .def_readonly("nchans", &PyFoldResult::nchans)
      .def_readonly("nbin", &PyFoldResult::nbin)
      .def_readonly("period", &PyFoldResult::period)
      .def_readonly("tsamp", &PyFoldResult::tsamp)
      .def_readonly("tsubint", &PyFoldResult::tsubint)
      .def("__repr__", &fold_result_repr);

  module.def("_fold_dedispersed_spectrum", &fold_dedispersed_spectrum_for_python,
             py::arg("data"), py::kw_only(), py::arg("tsamp"),
             py::arg("period"), py::arg("nbin"), py::arg("tsubint"),
             py::arg("output_channels"));
}

}  // namespace gaffa::python
