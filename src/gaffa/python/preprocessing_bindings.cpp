#include "preprocessing_bindings.h"

#include "gaffa/preprocessing.h"

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace {

gaffa::PreprocessPlan make_riptide_preprocess_plan_for_python(
    double tsamp, double running_median_width_seconds,
    std::size_t running_median_min_points, bool normalise,
    bool reject_constant) {
  return gaffa::make_riptide_preprocess_plan(
      tsamp,
      gaffa::RiptidePreprocessOptions{
          .running_median_width_seconds = running_median_width_seconds,
          .running_median_min_points = running_median_min_points,
          .normalise = normalise,
          .normalise_options = {.reject_constant = reject_constant},
      });
}

}  // namespace

namespace gaffa::python {

void bind_preprocessing(py::module_& module) {
  py::class_<PreprocessPlan>(module, "PreprocessPlan")
      .def_property_readonly("step_count", [](const PreprocessPlan& plan) {
        return plan.steps.size();
      })
      .def("__repr__", [](const PreprocessPlan& plan) {
        return "<PreprocessPlan step_count=" +
               std::to_string(plan.steps.size()) + ">";
      });

  module.def("_make_riptide_preprocess_plan",
             &make_riptide_preprocess_plan_for_python, py::kw_only(),
             py::arg("tsamp"),
             py::arg("running_median_width_seconds") = 5.0,
             py::arg("running_median_min_points") = 101,
             py::arg("normalise") = true,
             py::arg("reject_constant") = true);
}

}  // namespace gaffa::python
