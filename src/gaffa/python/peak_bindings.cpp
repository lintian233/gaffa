#include "peak_bindings.h"

#include "gaffa/periodic_peak.h"

#include <pybind11/pybind11.h>

#include <string>

namespace py = pybind11;

namespace gaffa::python {

void bind_periodic_peaks(py::module_& module) {
  py::enum_<MotionOrder>(module, "MotionOrder")
      .value("FREQUENCY", MotionOrder::Frequency)
      .value("ACCELERATION", MotionOrder::Acceleration)
      .value("JERK", MotionOrder::Jerk)
      .value("SNAP", MotionOrder::Snap);

  py::class_<PeriodicMotion>(module, "PeriodicMotion")
      .def_readonly("order", &PeriodicMotion::order)
      .def_readonly("reference_time_seconds",
                    &PeriodicMotion::reference_time_seconds)
      .def_readonly("frequency_hz", &PeriodicMotion::frequency_hz)
      .def_readonly("acceleration_m_per_s2",
                    &PeriodicMotion::acceleration_m_per_s2)
      .def_readonly("jerk_m_per_s3", &PeriodicMotion::jerk_m_per_s3)
      .def_readonly("snap_m_per_s4", &PeriodicMotion::snap_m_per_s4)
      .def("__repr__", [](const PeriodicMotion& motion) {
        return "<PeriodicMotion frequency_hz=" +
               std::to_string(motion.frequency_hz) + ">";
      });

  py::class_<PeriodicPeak>(module, "PeriodicPeak")
      .def_readonly("motion", &PeriodicPeak::motion)
      .def_readonly("phase_bin", &PeriodicPeak::phase_bin)
      .def_readonly("phase_bins", &PeriodicPeak::phase_bins)
      .def_readonly("boxcar_width_bins", &PeriodicPeak::boxcar_width_bins)
      .def_readonly("duty_cycle", &PeriodicPeak::duty_cycle)
      .def_readonly("snr", &PeriodicPeak::snr)
      .def_property_readonly("period_seconds", &PeriodicPeak::period_seconds)
      .def("__repr__", [](const PeriodicPeak& peak) {
        return "<PeriodicPeak period_seconds=" +
               std::to_string(peak.period_seconds()) + " snr=" +
               std::to_string(peak.snr) + ">";
      });

  py::class_<DmPeak>(module, "DmPeak")
      .def_readonly("dm", &DmPeak::dm)
      .def_readonly("dm_index", &DmPeak::dm_index)
      .def_readonly("peak", &DmPeak::peak)
      .def("__repr__", [](const DmPeak& peak) {
        return "<DmPeak dm=" + std::to_string(peak.dm) +
               " dm_index=" + std::to_string(peak.dm_index) +
               " snr=" + std::to_string(peak.peak.snr) + ">";
      });
}

}  // namespace gaffa::python
