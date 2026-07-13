#include "ffa_bindings.h"

#include "gaffa/ffa_peak.h"
#include "gaffa/ffa_plan.h"
#include "gaffa/ffa_search.h"

#include <cstddef>
#include <span>
#include <string>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

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

  std::size_t peak_limit = 0;
  if (!max_peaks.is_none()) {
    peak_limit = max_peaks.cast<std::size_t>();
    if (peak_limit == 0) {
      throw py::value_error("max_peaks must be positive or None");
    }
  }

  const auto nsamples = static_cast<std::size_t>(info.shape[0]);
  const auto samples = std::span<const float>(
      static_cast<const float*>(info.ptr), nsamples);

  py::gil_scoped_release release;
  return gaffa::search_ffa_cpu(
             samples, plan,
             gaffa::FfaSearchOptions{
                 .snr_threshold = snr_threshold,
                 .max_peaks = peak_limit,
             })
      .peaks;
}

}  // namespace

namespace gaffa::python {

void bind_ffa(py::module_& module) {
  py::class_<gaffa::FfaSearchPlan>(module, "FfaPlan")
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

  module.def("_ffa_search_cpu", &ffa_search_cpu_for_python,
             py::arg("time_series"), py::arg("plan"), py::kw_only(),
             py::arg("snr_threshold") = 6.0F,
             py::arg("max_peaks") = py::none());
}

}  // namespace gaffa::python
