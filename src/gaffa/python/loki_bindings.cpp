#include "gaffa/cuda_memory.h"
#include "gaffa/loki_pffa.h"
#include "gaffa/periodic_peak.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

class CudaDeviceScope {
 public:
  explicit CudaDeviceScope(int device_id) {
    check_cuda(cudaGetDevice(&previous_device_), "cudaGetDevice");
    if (previous_device_ != device_id) {
      check_cuda(cudaSetDevice(device_id), "cudaSetDevice");
      restore_device_ = true;
    }
  }

  ~CudaDeviceScope() {
    if (restore_device_) {
      static_cast<void>(cudaSetDevice(previous_device_));
    }
  }

  CudaDeviceScope(const CudaDeviceScope&) = delete;
  CudaDeviceScope& operator=(const CudaDeviceScope&) = delete;

 private:
  int previous_device_ = 0;
  bool restore_device_ = false;
};

struct HostBatchView {
  const float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
};

HostBatchView validate_host_batch_array(const py::array& array,
                                        bool allow_batch) {
  const py::buffer_info info = array.request();
  if (info.ndim != 1 && (!allow_batch || info.ndim != 2)) {
    throw py::value_error("Loki input must be a 1D series or 2D batch");
  }
  if (info.itemsize != static_cast<py::ssize_t>(sizeof(float)) ||
      info.format != py::format_descriptor<float>::format()) {
    throw py::type_error("Loki input must have dtype float32");
  }
  if (info.ndim == 1) {
    if (info.shape[0] <= 0 ||
        info.strides[0] != static_cast<py::ssize_t>(sizeof(float))) {
      throw py::value_error(
          "Loki time_series must be non-empty and C-contiguous; use "
          "numpy.ascontiguousarray explicitly if a copy is intended");
    }
    return HostBatchView{
        .data = static_cast<const float*>(info.ptr),
        .nseries = 1,
        .nsamples = static_cast<std::size_t>(info.shape[0]),
    };
  }

  if (info.shape[0] <= 0 || info.shape[1] <= 0 ||
      info.strides[1] != static_cast<py::ssize_t>(sizeof(float)) ||
      info.strides[0] != info.shape[1] * info.itemsize) {
    throw py::value_error(
        "Loki batch must be non-empty and C-contiguous; use "
        "numpy.ascontiguousarray explicitly if a copy is intended");
  }
  return HostBatchView{
      .data = static_cast<const float*>(info.ptr),
      .nseries = static_cast<std::size_t>(info.shape[0]),
      .nsamples = static_cast<std::size_t>(info.shape[1]),
  };
}

gaffa::ValueRange parse_range(const py::object& object, const char* name) {
  if (!py::isinstance<py::sequence>(object) ||
      py::len(object) != static_cast<py::ssize_t>(2)) {
    throw py::value_error(std::string(name) + " must be a pair (minimum, maximum)");
  }
  return gaffa::ValueRange{
      .minimum = object[py::int_(0)].cast<double>(),
      .maximum = object[py::int_(1)].cast<double>(),
  };
}

std::optional<gaffa::ValueRange> optional_range(const py::object& object,
                                                 const char* name) {
  if (object.is_none()) {
    return std::nullopt;
  }
  return parse_range(object, name);
}

gaffa::LokiPffaPlan make_pffa_plan_for_python(
    std::size_t nsamples, double tsamp, const py::object& frequency,
    const py::object& acceleration, const py::object& jerk,
    const py::object& snap, std::size_t phase_bins_min,
    std::size_t phase_bins_max, double eta, double duty_cycle_max,
    double width_spacing, float snr_threshold) {
  return gaffa::make_loki_pffa_plan(
      nsamples, tsamp,
      gaffa::LokiTaylorSearchSpace{
          .frequency_hz = parse_range(frequency, "frequency"),
          .acceleration_m_per_s2 =
              optional_range(acceleration, "acceleration"),
          .jerk_m_per_s3 = optional_range(jerk, "jerk"),
          .snap_m_per_s4 = optional_range(snap, "snap"),
      },
      gaffa::LokiPffaPlanOptions{
          .phase_bins_min = phase_bins_min,
          .phase_bins_max = phase_bins_max,
          .eta = eta,
          .duty_cycle_max = duty_cycle_max,
          .width_spacing = width_spacing,
          .snr_threshold = snr_threshold,
      });
}

class HostLokiPffaProgram {
 public:
  HostLokiPffaProgram(const gaffa::LokiPffaPlan& plan, int device_id,
                      std::size_t max_peaks_per_series)
      : program_(plan,
                 gaffa::LokiPffaProgramOptions{.device_id = device_id}),
        max_peaks_per_series_(max_peaks_per_series) {
    if (max_peaks_per_series == 0) {
      throw py::value_error("max_peaks_per_series must be positive");
    }
  }

  std::vector<gaffa::PeriodicPeak> search(const py::array& array,
                                           std::size_t max_peaks) {
    const HostBatchView input = validate_host_batch_array(array, false);
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      throw std::runtime_error(
          "PffaProgram does not permit concurrent search() calls");
    }
    validate_input_contract(input);
    const std::size_t element_count = checked_element_count(input);
    py::gil_scoped_release release;
    const auto batch = search_batch_impl(input, element_count, max_peaks);
    return batch.front();
  }

  std::vector<std::vector<gaffa::PeriodicPeak>> search_batch(
      const py::array& array, std::size_t max_peaks) {
    const HostBatchView input = validate_host_batch_array(array, true);
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      throw std::runtime_error(
          "PffaProgram does not permit concurrent search() calls");
    }
    validate_input_contract(input);
    const std::size_t element_count = checked_element_count(input);
    py::gil_scoped_release release;
    return search_batch_impl(input, element_count, max_peaks);
  }

  int device_id() const noexcept {
    return program_.device_id();
  }

  std::size_t nsamples() const noexcept {
    return program_.plan().input_nsamples();
  }

 private:
  void validate_input_contract(HostBatchView input) const {
    if (input.nsamples != nsamples()) {
      throw py::value_error(
          "Loki input_nsamples must match PffaProgram plan nsamples");
    }
  }

  static std::size_t checked_element_count(HostBatchView input) {
    if (input.nseries != 0 &&
        input.nsamples > std::numeric_limits<std::size_t>::max() /
                            input.nseries) {
      throw py::value_error("Loki batch element count overflow");
    }
    return input.nseries * input.nsamples;
  }

  void ensure_input_capacity(std::size_t element_count) {
    if (device_input_.size() >= element_count) {
      return;
    }
    gaffa::CudaDeviceBuffer<float> replacement(element_count);
    device_input_ = std::move(replacement);
  }

  std::vector<std::vector<gaffa::PeriodicPeak>> search_batch_impl(
      HostBatchView input, std::size_t element_count,
      std::size_t max_peaks) {
    CudaDeviceScope device_scope(device_id());
    ensure_input_capacity(element_count);
    check_cuda(cudaMemcpy(device_input_.data(), input.data,
                          element_count * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy host-to-device Loki input");

    const gaffa::SeriesPeaks flat = program_.search_batch(
        gaffa::CudaTimeSeriesBatchView{
            .data = device_input_.data(),
            .nseries = input.nseries,
            .nsamples = input.nsamples,
            .device_id = device_id(),
        },
        gaffa::LokiPffaExecutionOptions{
            .max_peaks_per_series = max_peaks,
        });

    std::vector<std::vector<gaffa::PeriodicPeak>> result(input.nseries);
    for (const gaffa::SeriesPeak& peak : flat) {
      if (peak.series_index >= result.size()) {
        throw std::logic_error("Loki returned an invalid series index");
      }
      result[peak.series_index].push_back(peak.peak);
    }
    return result;
  }

  gaffa::LokiPffaProgram program_;
  gaffa::CudaDeviceBuffer<float> device_input_;
  std::size_t max_peaks_per_series_ = 0;
  std::mutex mutex_;
};

}  // namespace

PYBIND11_MODULE(_loki, module) {
  module.doc() = "Optional Loki P-FFA Python extension";

  py::class_<gaffa::LokiPffaPlan>(module, "PffaPlan")
      .def_property_readonly("nsamples", &gaffa::LokiPffaPlan::input_nsamples)
      .def_property_readonly("tsamp", &gaffa::LokiPffaPlan::tsamp_seconds)
      .def("__repr__", [](const gaffa::LokiPffaPlan& plan) {
        return "<PffaPlan nsamples=" + std::to_string(plan.input_nsamples()) +
               ">";
      });

  module.def(
      "_make_pffa_plan", &make_pffa_plan_for_python, py::kw_only(),
      py::arg("nsamples"), py::arg("tsamp"), py::arg("frequency"),
      py::arg("acceleration") = py::none(), py::arg("jerk") = py::none(),
      py::arg("snap") = py::none(), py::arg("phase_bins_min") = 256,
      py::arg("phase_bins_max") = 256, py::arg("eta") = 1.0,
      py::arg("duty_cycle_max") = 0.20, py::arg("width_spacing") = 1.5,
      py::arg("snr_threshold") = 6.0F);

  py::class_<HostLokiPffaProgram>(module, "_PffaProgram")
      .def(py::init<const gaffa::LokiPffaPlan&, int, std::size_t>(),
           py::arg("plan"), py::arg("device_id"),
           py::arg("max_peaks_per_series") = 1'000'000)
      .def("search", &HostLokiPffaProgram::search, py::arg("data"),
           py::kw_only(), py::arg("max_peaks") = 1'000'000)
      .def("search_batch", &HostLokiPffaProgram::search_batch,
           py::arg("data"), py::kw_only(),
           py::arg("max_peaks") = 1'000'000)
      .def_property_readonly("device_id", &HostLokiPffaProgram::device_id)
      .def_property_readonly("nsamples", &HostLokiPffaProgram::nsamples);
}
