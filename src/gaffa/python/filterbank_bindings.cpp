#include "filterbank_bindings.h"

#include "gaffa/filterbank.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

namespace py = pybind11;

namespace {

struct PyFilterbank {
  gaffa::FilterbankHeader header;
  py::array data;
};

gaffa::ChannelOrderPolicy parse_channel_order(const std::string& value) {
  if (value == "frequency_ascending") {
    return gaffa::ChannelOrderPolicy::FrequencyAscending;
  }
  if (value == "preserve_file_order") {
    return gaffa::ChannelOrderPolicy::PreserveFileOrder;
  }
  throw py::value_error("unknown channel_order: " + value);
}

gaffa::ReverseBackend parse_reverse_backend(const std::string& value) {
  if (value == "auto") {
    return gaffa::ReverseBackend::Auto;
  }
  if (value == "cpu_scalar") {
    return gaffa::ReverseBackend::CpuScalar;
  }
  if (value == "cpu_openmp") {
    return gaffa::ReverseBackend::CpuOpenmp;
  }
  throw py::value_error("unknown reverse_backend: " + value);
}

template <typename T>
py::array make_numpy_array(std::vector<T>&& samples,
                           const gaffa::FilterbankHeader& header) {
  if (header.nsamples < 0 || header.nifs < 0 || header.nchans < 0) {
    throw std::runtime_error("filterbank shape contains a negative dimension");
  }

  auto owner = std::make_unique<std::vector<T>>(std::move(samples));
  T* data = owner->data();
  py::capsule capsule(owner.release(), [](void* pointer) {
    delete static_cast<std::vector<T>*>(pointer);
  });

  const auto nsamples = static_cast<py::ssize_t>(header.nsamples);
  const auto nifs = static_cast<py::ssize_t>(header.nifs);
  const auto nchans = static_cast<py::ssize_t>(header.nchans);
  const auto item_size = static_cast<py::ssize_t>(sizeof(T));

  return py::array_t<T>(
      {nsamples, nifs, nchans},
      {nifs * nchans * item_size, nchans * item_size, item_size},
      data,
      capsule);
}

py::array make_numpy_array(gaffa::FilterbankData&& filterbank) {
  const gaffa::FilterbankHeader& header = filterbank.header;
  return std::visit(
      [&](auto&& samples) -> py::array {
        using Vector = std::decay_t<decltype(samples)>;
        using T = typename Vector::value_type;
        return make_numpy_array<T>(std::move(samples), header);
      },
      std::move(filterbank.samples));
}

PyFilterbank read_filterbank_for_python(
    const std::filesystem::path& path,
    const std::string& channel_order,
    const std::string& reverse_backend,
    std::size_t io_buffer_bytes,
    std::size_t openmp_min_rows) {
  gaffa::FilterbankReadOptions options;
  options.channel_order = parse_channel_order(channel_order);
  options.reverse_backend = parse_reverse_backend(reverse_backend);
  options.io_buffer_bytes = io_buffer_bytes;
  options.openmp_min_rows = openmp_min_rows;

  gaffa::FilterbankData data = gaffa::read_filterbank(path, options);

  PyFilterbank result;
  result.header = data.header;
  result.data = make_numpy_array(std::move(data));
  return result;
}

std::string filterbank_header_repr(const gaffa::FilterbankHeader& header) {
  return "<FilterbankHeader nsamples=" + std::to_string(header.nsamples) +
         " nifs=" + std::to_string(header.nifs) +
         " nchans=" + std::to_string(header.nchans) +
         " nbits=" + std::to_string(header.nbits) +
         " foff=" + std::to_string(header.foff) + ">";
}

std::string filterbank_repr(const PyFilterbank& filterbank) {
  const gaffa::FilterbankHeader& header = filterbank.header;
  return "<Filterbank shape=(" + std::to_string(header.nsamples) + ", " +
         std::to_string(header.nifs) + ", " +
         std::to_string(header.nchans) + ") nbits=" +
         std::to_string(header.nbits) + ">";
}

}  // namespace

namespace gaffa::python {

void bind_filterbank(py::module_& module) {
  py::enum_<gaffa::ChannelOrderPolicy>(module, "ChannelOrderPolicy")
      .value("FrequencyAscending", gaffa::ChannelOrderPolicy::FrequencyAscending)
      .value("PreserveFileOrder", gaffa::ChannelOrderPolicy::PreserveFileOrder);

  py::enum_<gaffa::ReverseBackend>(module, "ReverseBackend")
      .value("Auto", gaffa::ReverseBackend::Auto)
      .value("CpuScalar", gaffa::ReverseBackend::CpuScalar)
      .value("CpuOpenmp", gaffa::ReverseBackend::CpuOpenmp);

  py::class_<gaffa::FilterbankHeader>(module, "FilterbankHeader")
      .def_readonly("header_size", &gaffa::FilterbankHeader::header_size)
      .def_readonly("nsamples", &gaffa::FilterbankHeader::nsamples)
      .def_readonly("telescope_id", &gaffa::FilterbankHeader::telescope_id)
      .def_readonly("machine_id", &gaffa::FilterbankHeader::machine_id)
      .def_readonly("data_type", &gaffa::FilterbankHeader::data_type)
      .def_readonly("barycentric", &gaffa::FilterbankHeader::barycentric)
      .def_readonly("pulsarcentric", &gaffa::FilterbankHeader::pulsarcentric)
      .def_readonly("ibeam", &gaffa::FilterbankHeader::ibeam)
      .def_readonly("nbeams", &gaffa::FilterbankHeader::nbeams)
      .def_readonly("npuls", &gaffa::FilterbankHeader::npuls)
      .def_readonly("nbins", &gaffa::FilterbankHeader::nbins)
      .def_readonly("nbits", &gaffa::FilterbankHeader::nbits)
      .def_readonly("nifs", &gaffa::FilterbankHeader::nifs)
      .def_readonly("nchans", &gaffa::FilterbankHeader::nchans)
      .def_readonly("az_start", &gaffa::FilterbankHeader::az_start)
      .def_readonly("za_start", &gaffa::FilterbankHeader::za_start)
      .def_readonly("src_raj", &gaffa::FilterbankHeader::src_raj)
      .def_readonly("src_dej", &gaffa::FilterbankHeader::src_dej)
      .def_readonly("tstart", &gaffa::FilterbankHeader::tstart)
      .def_readonly("tsamp", &gaffa::FilterbankHeader::tsamp)
      .def_readonly("fch1", &gaffa::FilterbankHeader::fch1)
      .def_readonly("foff", &gaffa::FilterbankHeader::foff)
      .def_readonly("refdm", &gaffa::FilterbankHeader::refdm)
      .def_readonly("period", &gaffa::FilterbankHeader::period)
      .def_readonly("rawdatafile", &gaffa::FilterbankHeader::rawdatafile)
      .def_readonly("source_name", &gaffa::FilterbankHeader::source_name)
      .def_readonly("frequency_table", &gaffa::FilterbankHeader::frequency_table)
      .def_readonly("uses_frequency_table",
                    &gaffa::FilterbankHeader::uses_frequency_table)
      .def("__repr__", &filterbank_header_repr);

  py::class_<PyFilterbank>(module, "Filterbank")
      .def(py::init(&read_filterbank_for_python),
           "Read a SIGPROC filterbank file into a Filterbank object.",
           py::arg("path"),
           py::kw_only(),
           py::arg("channel_order") = "frequency_ascending",
           py::arg("reverse_backend") = "auto",
           py::arg("io_buffer_bytes") = gaffa::default_filterbank_io_buffer_bytes,
           py::arg("openmp_min_rows") = 4096)
      .def_readonly("header", &PyFilterbank::header)
      .def_readonly("data", &PyFilterbank::data)
      .def_property_readonly("shape",
                             [](const PyFilterbank& filterbank) {
                               return filterbank.data.attr("shape");
                             })
      .def_property_readonly("dtype",
                             [](const PyFilterbank& filterbank) {
                               return filterbank.data.attr("dtype");
                             })
      .def_property_readonly("nbytes",
                             [](const PyFilterbank& filterbank) {
                               return filterbank.data.attr("nbytes");
                             })
      .def("__repr__", &filterbank_repr);
}

}  // namespace gaffa::python
