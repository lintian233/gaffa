#include "ffa_bindings.h"

#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_cuda.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"
#include "gaffa/dm_search.h"
#include "gaffa/preprocessing.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <mutex>
#include <span>
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

void validate_time_series_array(const py::buffer_info& info) {
  if (info.ndim != 1) {
    throw py::value_error("FFA time_series must be a 1D array");
  }
  if (info.shape[0] <= 0) {
    throw py::value_error("FFA time_series must not be empty");
  }
  if (info.itemsize != static_cast<py::ssize_t>(sizeof(float)) ||
      info.format != py::format_descriptor<float>::format()) {
    throw py::type_error("FFA time_series must have dtype float32");
  }
  if (info.strides[0] != static_cast<py::ssize_t>(sizeof(float))) {
    throw py::value_error(
        "FFA time_series must be C-contiguous; use "
        "numpy.ascontiguousarray explicitly if a copy is intended");
  }
}

struct HostBatchView {
  const float* data = nullptr;
  std::size_t nseries = 0;
  std::size_t nsamples = 0;
};

HostBatchView validate_host_batch_array(const py::array& array,
                                        bool allow_batch) {
  const py::buffer_info info = array.request();
  if (info.ndim != 1 && (!allow_batch || info.ndim != 2)) {
    throw py::value_error("FFA input must be a 1D series or 2D batch");
  }
  if (info.itemsize != static_cast<py::ssize_t>(sizeof(float)) ||
      info.format != py::format_descriptor<float>::format()) {
    throw py::type_error("FFA input must have dtype float32");
  }
  if (info.ndim == 1) {
    if (info.shape[0] <= 0 ||
        info.strides[0] != static_cast<py::ssize_t>(sizeof(float))) {
      throw py::value_error(
          "FFA time_series must be non-empty and C-contiguous; use "
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
        "FFA batch must be non-empty and C-contiguous; use "
        "numpy.ascontiguousarray explicitly if a copy is intended");
  }
  return HostBatchView{
      .data = static_cast<const float*>(info.ptr),
      .nseries = static_cast<std::size_t>(info.shape[0]),
      .nsamples = static_cast<std::size_t>(info.shape[1]),
  };
}

std::size_t parse_peak_limit(const py::object& max_peaks) {
  if (max_peaks.is_none()) {
    return 0;
  }
  const std::size_t peak_limit = max_peaks.cast<std::size_t>();
  if (peak_limit == 0) {
    throw py::value_error("max_peaks must be positive or None");
  }
  return peak_limit;
}

class HostCudaFfaProgram {
 public:
  HostCudaFfaProgram(const gaffa::FfaSearchPlan& plan, int device_id,
                     std::size_t series_tile_size)
      : program_(plan,
                 gaffa::CudaFfaProgramOptions{.device_id = device_id},
                 gaffa::CudaFfaExecutionOptions{
                     .series_tile_size = series_tile_size,
                 }) {}

  std::vector<gaffa::PeriodicPeak> search(const py::array& array,
                                           float snr_threshold,
                                           std::size_t max_peaks) {
    const HostBatchView input = validate_host_batch_array(array, false);
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      throw std::runtime_error(
          "CudaProgram does not permit concurrent search() calls");
    }
    validate_input_contract(input);
    const std::size_t element_count = checked_element_count(input);
    py::gil_scoped_release release;
    const std::vector<std::vector<gaffa::PeriodicPeak>> batch =
        search_batch_impl(input, element_count, snr_threshold, max_peaks);
    return batch.front();
  }

  std::vector<std::vector<gaffa::PeriodicPeak>> search_batch(
      const py::array& array, float snr_threshold, std::size_t max_peaks) {
    const HostBatchView input = validate_host_batch_array(array, true);
    std::unique_lock lock(mutex_, std::try_to_lock);
    if (!lock.owns_lock()) {
      throw std::runtime_error(
          "CudaProgram does not permit concurrent search() calls");
    }
    validate_input_contract(input);
    const std::size_t element_count = checked_element_count(input);
    py::gil_scoped_release release;
    return search_batch_impl(input, element_count, snr_threshold, max_peaks);
  }

  int device_id() const noexcept {
    return program_.device_id();
  }

  std::size_t tile_capacity() const {
    return program_.tile_capacity();
  }

  std::size_t nsamples() const {
    return program_.execution_plan().observation().nsamples;
  }

 private:
  std::vector<std::vector<gaffa::PeriodicPeak>> search_batch_impl(
      HostBatchView input, std::size_t element_count, float snr_threshold,
      std::size_t max_peaks) {
    CudaDeviceScope device_scope(device_id());
    ensure_input_capacity(element_count);
    check_cuda(cudaMemcpy(device_input_.data(), input.data,
                          element_count * sizeof(float),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy host-to-device FFA input");

    const gaffa::SeriesPeaks flat = gaffa::search_ffa_batch_cuda(
        program_,
        gaffa::CudaTimeSeriesBatchView{
            .data = device_input_.data(),
            .nseries = input.nseries,
            .nsamples = input.nsamples,
            .device_id = device_id(),
        },
        gaffa::FfaSearchOptions{
            .snr_threshold = snr_threshold,
            .max_peaks = max_peaks,
        });

    std::vector<std::vector<gaffa::PeriodicPeak>> result(input.nseries);
    for (const gaffa::SeriesPeak& peak : flat) {
      if (peak.series_index >= result.size()) {
        throw std::logic_error("CUDA FFA returned an invalid series index");
      }
      result[peak.series_index].push_back(peak.peak);
    }
    return result;
  }

  void validate_input_contract(HostBatchView input) const {
    if (input.nseries > program_.tile_capacity()) {
      throw py::value_error("FFA batch exceeds CudaProgram tile capacity");
    }
    if (input.nsamples != nsamples()) {
      throw py::value_error(
          "FFA input_nsamples must match CudaProgram plan nsamples");
    }
  }

  static std::size_t checked_element_count(HostBatchView input) {
    if (input.nseries != 0 &&
        input.nsamples > std::numeric_limits<std::size_t>::max() /
                            input.nseries) {
      throw py::value_error("FFA batch element count overflow");
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

  gaffa::CudaFfaProgram program_;
  gaffa::CudaDeviceBuffer<float> device_input_;
  std::mutex mutex_;
};

std::vector<gaffa::PeriodicPeak> periodic_search_cpu_for_python(
    const py::array& array, const gaffa::FfaSearchPlan& plan,
    float snr_threshold, std::size_t max_peaks) {
  const HostBatchView input = validate_host_batch_array(array, false);
  if (input.nsamples != plan.observation.nsamples) {
    throw py::value_error(
        "FFA input_nsamples must match plan observation nsamples");
  }
  py::gil_scoped_release release;
  return gaffa::search_ffa_cpu(
      std::span<const float>(input.data, input.nsamples), plan,
      gaffa::FfaSearchOptions{
          .snr_threshold = snr_threshold,
          .max_peaks = max_peaks,
      });
}

std::vector<std::vector<gaffa::PeriodicPeak>> periodic_search_batch_cpu_for_python(
    const py::array& array, const gaffa::FfaSearchPlan& plan,
    float snr_threshold, std::size_t max_peaks) {
  const HostBatchView input = validate_host_batch_array(array, true);
  if (input.nsamples != plan.observation.nsamples) {
    throw py::value_error(
        "FFA input_nsamples must match plan observation nsamples");
  }
  py::gil_scoped_release release;
  std::vector<std::vector<gaffa::PeriodicPeak>> result;
  result.reserve(input.nseries);
  for (std::size_t series_index = 0; series_index < input.nseries;
       ++series_index) {
    const std::span<const float> series(
        input.data + series_index * input.nsamples, input.nsamples);
    result.push_back(gaffa::search_ffa_cpu(
        series, plan,
        gaffa::FfaSearchOptions{
            .snr_threshold = snr_threshold,
            .max_peaks = max_peaks,
        }));
  }
  return result;
}

void validate_dm_array(const py::buffer_info& info) {
  if (info.ndim != 2) {
    throw py::value_error("DM series data must be a 2D array");
  }
  if (info.shape[0] <= 0 || info.shape[1] <= 0) {
    throw py::value_error("DM series data must not be empty");
  }
  if (info.strides[1] != info.itemsize ||
      info.strides[0] != info.shape[1] * info.itemsize) {
    throw py::value_error(
        "DM series data must be C-contiguous; use "
        "numpy.ascontiguousarray explicitly if a copy is intended");
  }
  const bool is_uint32 =
      info.itemsize == static_cast<py::ssize_t>(sizeof(std::uint32_t)) &&
      info.format == py::format_descriptor<std::uint32_t>::format();
  const bool is_float =
      info.itemsize == static_cast<py::ssize_t>(sizeof(float)) &&
      info.format == py::format_descriptor<float>::format();
  if (!is_uint32 && !is_float) {
    throw py::type_error("DM series data must have dtype uint32 or float32");
  }
}

gaffa::FfaSearchPlan make_riptide_plan_for_python(
    std::size_t nsamples, double tsamp, double period_min, double period_max,
    std::size_t bins_min, std::size_t bins_max, std::size_t min_periods,
    double duty_cycle_max, double width_trial_spacing, std::size_t max_tasks) {
  return gaffa::make_riptide_ffa_plan(
      nsamples, tsamp,
      gaffa::RiptideFfaPlanOptions{
          .period_min = period_min,
          .period_max = period_max,
          .bins_min = bins_min,
          .bins_max = bins_max,
          .min_periods = min_periods,
          .duty_cycle_max = duty_cycle_max,
          .width_trial_spacing = width_trial_spacing,
          .max_tasks = max_tasks,
      });
}

std::vector<gaffa::FfaPeak> ffa_search_cpu_for_python(
    const py::object& time_series_object, const gaffa::FfaSearchPlan& plan,
    float snr_threshold, const py::object& max_peaks) {
  if (!py::isinstance<py::array>(time_series_object)) {
    throw py::type_error("FFA time_series must be a numpy.ndarray");
  }
  const py::array time_series = py::reinterpret_borrow<py::array>(
      time_series_object);
  const py::buffer_info info = time_series.request();
  validate_time_series_array(info);

  const std::size_t peak_limit = parse_peak_limit(max_peaks);

  const auto nsamples = static_cast<std::size_t>(info.shape[0]);
  if (nsamples != plan.observation.nsamples) {
    throw py::value_error(
        "FFA input_nsamples must match plan observation nsamples");
  }
  const auto samples = std::span<const float>(
      static_cast<const float*>(info.ptr), nsamples);

  py::gil_scoped_release release;
  return gaffa::search_ffa_raw_cpu(
             samples, plan,
             gaffa::FfaSearchOptions{
                 .snr_threshold = snr_threshold,
                 .max_peaks = peak_limit,
             })
      .peaks;
}

std::vector<gaffa::FfaPeak> ffa_search_cuda_host_for_python(
    const py::object& time_series_object, const gaffa::FfaSearchPlan& plan,
    int device_id, float snr_threshold, const py::object& max_peaks) {
  if (!py::isinstance<py::array>(time_series_object)) {
    throw py::type_error("FFA time_series must be a numpy.ndarray");
  }
  const py::array time_series = py::reinterpret_borrow<py::array>(
      time_series_object);
  const py::buffer_info info = time_series.request();
  validate_time_series_array(info);

  const std::size_t peak_limit = parse_peak_limit(max_peaks);

  const auto nsamples = static_cast<std::size_t>(info.shape[0]);
  if (nsamples != plan.observation.nsamples) {
    throw py::value_error(
        "FFA input_nsamples must match plan observation nsamples");
  }
  const auto samples = static_cast<const float*>(info.ptr);

  py::gil_scoped_release release;
  CudaDeviceScope device_scope(device_id);
  gaffa::CudaDeviceBuffer<float> device_samples(nsamples);
  check_cuda(cudaMemcpy(device_samples.data(), samples, device_samples.bytes(),
                        cudaMemcpyHostToDevice),
             "cudaMemcpy host-to-device FFA input");

  return gaffa::search_ffa_raw_cuda(
             static_cast<const gaffa::CudaDeviceBuffer<float>&>(device_samples)
                 .as_span(device_id),
             plan,
             gaffa::FfaSearchOptions{
                 .snr_threshold = snr_threshold,
                 .max_peaks = peak_limit,
             },
             gaffa::CudaFfaProgramOptions{.device_id = device_id})
      .peaks;
}

std::vector<gaffa::DmPeak> search_dms_cpu_for_python(
    const py::object& data_object, double tsamp, double dm_low, double dm_step,
    std::size_t dm_index_offset, const gaffa::FfaSearchPlan& plan,
    const gaffa::PreprocessPlan& preprocess, float snr_threshold,
    const py::object& max_peaks) {
  if (!py::isinstance<py::array>(data_object)) {
    throw py::type_error("DM series data must be a numpy.ndarray");
  }
  const py::array data = py::reinterpret_borrow<py::array>(data_object);
  const py::buffer_info info = data.request();
  validate_dm_array(info);
  if (!std::isfinite(tsamp) || !(tsamp > 0.0)) {
    throw py::value_error("DM series tsamp must be finite and positive");
  }
  if (tsamp != plan.observation.tsamp_seconds) {
    throw py::value_error("DM series tsamp must match FFA plan tsamp");
  }

  const auto ndm = static_cast<std::size_t>(info.shape[0]);
  const auto nsamples = static_cast<std::size_t>(info.shape[1]);
  std::vector<double> dm_values(ndm);
  for (std::size_t index = 0; index < ndm; ++index) {
    dm_values[index] = dm_low + static_cast<double>(index) * dm_step;
  }

  const gaffa::DmTrialView trials{
      .values = dm_values,
      .index_offset = dm_index_offset,
  };
  const gaffa::DmFfaOptions options{
      .preprocess = preprocess,
      .search = {.snr_threshold = snr_threshold,
                 .max_peaks = parse_peak_limit(max_peaks)},
  };
  const gaffa::DedispersedShape shape{.ndm = ndm, .nsamples = nsamples};
  const std::size_t element_count = gaffa::dedispersed_element_count(shape);

  py::gil_scoped_release release;
  if (info.format == py::format_descriptor<std::uint32_t>::format()) {
    return gaffa::search_dm_ffa_cpu(
        gaffa::DedispersedResultView<std::uint32_t>{
            .data = std::span<const std::uint32_t>(
                static_cast<const std::uint32_t*>(info.ptr), element_count),
            .shape = shape,
        },
        trials, plan, options);
  }
  return gaffa::search_dm_ffa_cpu(
      gaffa::DedispersedResultView<float>{
          .data = std::span<const float>(static_cast<const float*>(info.ptr),
                                         element_count),
          .shape = shape,
      },
      trials, plan, options);
}

}  // namespace

namespace gaffa::python {

void bind_ffa(py::module_& module) {
  py::class_<gaffa::FfaSearchPlan>(module, "FfaPlan")
      .def_property_readonly("nsamples", [](const gaffa::FfaSearchPlan& plan) {
        return plan.observation.nsamples;
      })
      .def_property_readonly("tsamp", [](const gaffa::FfaSearchPlan& plan) {
        return plan.observation.tsamp_seconds;
      })
      .def_property_readonly("task_count", [](const gaffa::FfaSearchPlan& plan) {
        return plan.tasks.size();
      })
      .def_property_readonly(
          "width_trials", [](const gaffa::FfaSearchPlan& plan) {
            return py::tuple(py::cast(plan.width_trials));
          })
      .def("__repr__", [](const gaffa::FfaSearchPlan& plan) {
        return "<FfaPlan task_count=" + std::to_string(plan.tasks.size()) +
               " width_trials=" +
               std::to_string(plan.width_trials.size()) + ">";
      });

  py::class_<gaffa::FfaPeak>(module, "FfaPeak")
      .def_readonly("period", &gaffa::FfaPeak::period)
      .def_readonly("frequency", &gaffa::FfaPeak::frequency)
      .def_readonly("snr", &gaffa::FfaPeak::snr)
      .def_readonly("width", &gaffa::FfaPeak::width)
      .def_readonly("duty_cycle", &gaffa::FfaPeak::duty_cycle)
      .def_readonly("phase", &gaffa::FfaPeak::phase)
      .def_readonly("shift", &gaffa::FfaPeak::shift)
      .def_readonly("bins", &gaffa::FfaPeak::bins)
      .def_readonly("width_index", &gaffa::FfaPeak::width_index)
      .def_readonly("period_index", &gaffa::FfaPeak::period_index)
      .def("__repr__", [](const gaffa::FfaPeak& peak) {
        return "<FfaPeak period=" + std::to_string(peak.period) +
               " snr=" + std::to_string(peak.snr) + ">";
      });

  module.def("_make_riptide_ffa_plan", &make_riptide_plan_for_python,
             py::kw_only(), py::arg("nsamples"), py::arg("tsamp"),
             py::arg("period_min"), py::arg("period_max"),
             py::arg("bins_min") = 180, py::arg("bins_max") = 256,
             py::arg("min_periods") = 1,
             py::arg("duty_cycle_max") = 0.20,
             py::arg("width_trial_spacing") = 1.5,
             py::arg("max_tasks") = 1'000'000);

  module.def("_search_raw_cpu", &ffa_search_cpu_for_python,
             py::arg("time_series"), py::arg("plan"), py::kw_only(),
             py::arg("snr_threshold") = 6.0F,
             py::arg("max_peaks") = py::none());

  module.def("_search_raw_cuda_host", &ffa_search_cuda_host_for_python,
             py::arg("time_series"), py::arg("plan"), py::kw_only(),
             py::arg("device_id") = 0, py::arg("snr_threshold") = 6.0F,
             py::arg("max_peaks") = py::none());

  py::class_<HostCudaFfaProgram>(module, "_CudaProgram")
      .def(py::init<const gaffa::FfaSearchPlan&, int, std::size_t>(),
           py::arg("plan"), py::arg("device_id"),
           py::arg("series_tile_size") = 16)
      .def("search", &HostCudaFfaProgram::search, py::arg("data"),
           py::kw_only(), py::arg("snr_threshold") = 6.0F,
           py::arg("max_peaks") = 0)
      .def("search_batch", &HostCudaFfaProgram::search_batch,
           py::arg("data"), py::kw_only(),
           py::arg("snr_threshold") = 6.0F, py::arg("max_peaks") = 0)
      .def_property_readonly("device_id", &HostCudaFfaProgram::device_id)
      .def_property_readonly("tile_capacity",
                             &HostCudaFfaProgram::tile_capacity)
      .def_property_readonly("nsamples", &HostCudaFfaProgram::nsamples);

  module.def("_search_periodic_cpu", &periodic_search_cpu_for_python,
             py::arg("time_series"), py::arg("plan"), py::kw_only(),
             py::arg("snr_threshold") = 6.0F, py::arg("max_peaks") = 0);
  module.def("_search_periodic_batch_cpu",
             &periodic_search_batch_cpu_for_python, py::arg("data"),
             py::arg("plan"), py::kw_only(),
             py::arg("snr_threshold") = 6.0F, py::arg("max_peaks") = 0);

  module.def("_search_dms_cpu", &search_dms_cpu_for_python,
             py::arg("data"), py::kw_only(), py::arg("tsamp"),
             py::arg("dm_low"), py::arg("dm_step"),
             py::arg("dm_index_offset"), py::arg("plan"),
             py::arg("preprocess"), py::arg("snr_threshold") = 6.0F,
             py::arg("max_peaks") = py::none());
}

}  // namespace gaffa::python
