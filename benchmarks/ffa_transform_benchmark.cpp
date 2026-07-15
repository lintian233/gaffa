#include "gaffa/cuda_memory.h"
#include "gaffa/ffa.h"
#include "gaffa/ffa_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

enum class Mode {
  CpuSerial,
  CpuOpenmp,
  Cuda,
  CudaProgram,
};

struct Args {
  std::string mode = "all";
  std::size_t rows = 10172;
  std::size_t bins = 256;
  std::size_t nseries = 32;
  int iterations = 5;
  int warmup = 1;
  bool validate = false;
};

struct IterationResult {
  int iteration = 0;
  double seconds = 0.0;
};

struct BenchmarkResult {
  Mode mode = Mode::CpuSerial;
  std::vector<IterationResult> runs;
};

struct SummaryStats {
  double mean_seconds = 0.0;
  double median_seconds = 0.0;
  double min_seconds = 0.0;
  double max_seconds = 0.0;
  double stddev_seconds = 0.0;
};

void usage(const char* program) {
  std::cerr << "Usage: " << program
            << " <cpu-serial|cpu-openmp|cuda|cuda-program|all>"
            << " <rows> <bins> <nseries>"
            << " [iterations] [warmup] [validate|no-validate]\n";
}

std::size_t parse_size(const char* value) {
  return static_cast<std::size_t>(std::stoull(value));
}

Args parse_args(int argc, char** argv) {
  if (argc < 5 || argc > 8) {
    usage(argv[0]);
    throw std::invalid_argument("invalid argument count");
  }

  Args args;
  args.mode = argv[1];
  args.rows = parse_size(argv[2]);
  args.bins = parse_size(argv[3]);
  args.nseries = parse_size(argv[4]);
  if (argc >= 6) {
    args.iterations = std::stoi(argv[5]);
  }
  if (argc >= 7) {
    args.warmup = std::stoi(argv[6]);
  }
  if (argc >= 8) {
    const std::string validate_arg = argv[7];
    if (validate_arg != "validate" && validate_arg != "no-validate") {
      usage(argv[0]);
      throw std::invalid_argument(
          "validate option must be validate or no-validate");
    }
    args.validate = validate_arg == "validate";
  }

  if (args.mode != "cpu-serial" && args.mode != "cpu-openmp" &&
      args.mode != "cuda" && args.mode != "cuda-program" &&
      args.mode != "all") {
    usage(argv[0]);
    throw std::invalid_argument("unknown benchmark mode");
  }
  if (args.rows == 0) {
    throw std::invalid_argument("rows must be > 0");
  }
  if (args.bins <= 1) {
    throw std::invalid_argument("bins must be > 1");
  }
  if (args.nseries == 0) {
    throw std::invalid_argument("nseries must be > 0");
  }
  if (args.iterations <= 0) {
    throw std::invalid_argument("iterations must be > 0");
  }
  if (args.warmup < 0) {
    throw std::invalid_argument("warmup must be >= 0");
  }
  return args;
}

std::vector<Mode> selected_modes(const std::string& mode) {
  if (mode == "cpu-serial") {
    return {Mode::CpuSerial};
  }
  if (mode == "cpu-openmp") {
    return {Mode::CpuOpenmp};
  }
  if (mode == "cuda") {
    return {Mode::Cuda};
  }
  if (mode == "cuda-program") {
    return {Mode::CudaProgram};
  }
  return {Mode::CpuSerial, Mode::CpuOpenmp, Mode::Cuda, Mode::CudaProgram};
}

std::string_view mode_name(Mode mode) {
  switch (mode) {
    case Mode::CpuSerial:
      return "cpu-serial";
    case Mode::CpuOpenmp:
      return "cpu-openmp";
    case Mode::Cuda:
      return "cuda";
    case Mode::CudaProgram:
      return "cuda-program";
  }
  return "unknown";
}

template <typename Func>
double time_once(Func&& func) {
  const auto start = std::chrono::steady_clock::now();
  func();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

void check_cuda(cudaError_t status, const char* operation) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

std::size_t checked_multiply(std::size_t lhs,
                             std::size_t rhs,
                             const char* message) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    throw std::overflow_error(message);
  }
  return lhs * rhs;
}

double bytes_to_gib(std::size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

std::vector<float> make_input(std::size_t total_elements) {
  std::vector<float> input(total_elements);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>((index * 17 + 3) % 257) -
                                      128) *
                   0.125F;
  }
  return input;
}

gaffa::FfaSearchPlan make_single_task_plan(std::size_t rows,
                                           std::size_t bins) {
  const std::size_t elements =
      checked_multiply(rows, bins, "single task element count overflow");
  return gaffa::FfaSearchPlan{
      .tasks = {gaffa::FfaSearchTask{
          .downsample_factor = 1.0,
          .effective_tsamp = 1.0,
          .input_nsamples = elements,
          .prepared_nsamples = elements,
          .bins = bins,
          .rows = rows,
          .rows_eval = rows,
          .period_begin = static_cast<double>(bins),
          .period_end = static_cast<double>(bins + 1),
      }},
      .width_trials = {1},
  };
}

void run_cpu_serial_once(const std::vector<float>& input,
                         std::size_t nseries,
                         gaffa::FfaTransformShape shape,
                         std::vector<float>& scratch,
                         std::vector<float>& output) {
  const std::size_t series_elements = shape.rows * shape.bins;
  for (std::size_t series = 0; series < nseries; ++series) {
    const auto input_series = std::span<const float>(input).subspan(
        series * series_elements, series_elements);
    gaffa::ffa_transform_block_cpu(input_series, shape, scratch, output);
  }
}

void run_cpu_openmp_once(const std::vector<float>& input,
                         std::size_t nseries,
                         gaffa::FfaTransformShape shape,
                         std::vector<float>& scratch,
                         std::vector<float>& output) {
  const std::size_t series_elements = shape.rows * shape.bins;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (std::ptrdiff_t series_index = 0;
       series_index < static_cast<std::ptrdiff_t>(nseries); ++series_index) {
    const auto series = static_cast<std::size_t>(series_index);
    const auto input_series = std::span<const float>(input).subspan(
        series * series_elements, series_elements);
    auto scratch_series =
        std::span<float>(scratch).subspan(series * series_elements,
                                          series_elements);
    auto output_series =
        std::span<float>(output).subspan(series * series_elements,
                                         series_elements);
    gaffa::ffa_transform_block_cpu(input_series, shape, scratch_series,
                                   output_series);
  }
}

BenchmarkResult run_cpu_serial(const std::vector<float>& input,
                               const Args& args,
                               gaffa::FfaTransformShape shape) {
  const std::size_t series_elements = shape.rows * shape.bins;
  std::vector<float> scratch(series_elements);
  std::vector<float> output(series_elements);

  for (int index = 0; index < args.warmup; ++index) {
    run_cpu_serial_once(input, args.nseries, shape, scratch, output);
  }

  BenchmarkResult result{.mode = Mode::CpuSerial};
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    const double seconds = time_once([&] {
      run_cpu_serial_once(input, args.nseries, shape, scratch, output);
    });
    result.runs.push_back(IterationResult{
        .iteration = iteration,
        .seconds = seconds,
    });
    std::cout << "mode=" << mode_name(result.mode) << " iteration="
              << iteration << " seconds=" << seconds << '\n';
  }
  return result;
}

BenchmarkResult run_cpu_openmp(const std::vector<float>& input,
                               const Args& args,
                               gaffa::FfaTransformShape shape) {
  std::vector<float> scratch(input.size());
  std::vector<float> output(input.size());

  for (int index = 0; index < args.warmup; ++index) {
    run_cpu_openmp_once(input, args.nseries, shape, scratch, output);
  }

  BenchmarkResult result{.mode = Mode::CpuOpenmp};
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    const double seconds = time_once([&] {
      run_cpu_openmp_once(input, args.nseries, shape, scratch, output);
    });
    result.runs.push_back(IterationResult{
        .iteration = iteration,
        .seconds = seconds,
    });
    std::cout << "mode=" << mode_name(result.mode) << " iteration="
              << iteration << " seconds=" << seconds << '\n';
  }
  return result;
}

double time_cuda_transform(gaffa::CudaFfaInput input,
                           gaffa::CudaFfaBuffer scratch,
                           gaffa::CudaFfaBuffer output) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  gaffa::ffa_transform_block_cuda(input, scratch, output);
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
             "cudaEventElapsedTime");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  return static_cast<double>(milliseconds) / 1000.0;
}

double time_cuda_program_transform(const gaffa::CudaFfaProgram& program,
                                   gaffa::CudaFfaInput input,
                                   gaffa::CudaFfaBuffer scratch,
                                   gaffa::CudaFfaBuffer output) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  check_cuda(cudaEventCreate(&start), "cudaEventCreate start");
  check_cuda(cudaEventCreate(&stop), "cudaEventCreate stop");

  check_cuda(cudaEventRecord(start), "cudaEventRecord start");
  gaffa::ffa_transform_block_cuda(program, 0, 0, input, scratch, output);
  check_cuda(cudaEventRecord(stop), "cudaEventRecord stop");
  check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

  float milliseconds = 0.0F;
  check_cuda(cudaEventElapsedTime(&milliseconds, start, stop),
             "cudaEventElapsedTime");
  check_cuda(cudaEventDestroy(stop), "cudaEventDestroy stop");
  check_cuda(cudaEventDestroy(start), "cudaEventDestroy start");
  return static_cast<double>(milliseconds) / 1000.0;
}

BenchmarkResult run_cuda(const std::vector<float>& input,
                         const Args& args,
                         gaffa::FfaTransformShape shape) {
  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    throw std::runtime_error("CUDA device is not visible");
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  gaffa::CudaDeviceBuffer<float> device_scratch(input.size());
  gaffa::CudaDeviceBuffer<float> device_output(input.size());
  check_cuda(cudaMemcpy(device_input.data(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice),
             "benchmark input H2D");

  const std::size_t series_elements = shape.rows * shape.bins;
  const auto transform_input = gaffa::CudaFfaInput{
      .data = device_input.data(),
      .nseries = args.nseries,
      .nsamples = series_elements,
      .stride = series_elements,
      .shape = shape,
      .device_id = 0,
  };
  const auto transform_scratch = gaffa::CudaFfaBuffer{
      .data = device_scratch.data(),
      .nseries = args.nseries,
      .stride = series_elements,
      .shape = shape,
      .device_id = 0,
  };
  const auto transform_output = gaffa::CudaFfaBuffer{
      .data = device_output.data(),
      .nseries = args.nseries,
      .stride = series_elements,
      .shape = shape,
      .device_id = 0,
  };

  for (int index = 0; index < args.warmup; ++index) {
    (void)time_cuda_transform(transform_input, transform_scratch,
                              transform_output);
  }

  BenchmarkResult result{.mode = Mode::Cuda};
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    const double seconds = time_cuda_transform(
        transform_input, transform_scratch, transform_output);
    result.runs.push_back(IterationResult{
        .iteration = iteration,
        .seconds = seconds,
    });
    std::cout << "mode=" << mode_name(result.mode) << " iteration="
              << iteration << " seconds=" << seconds << '\n';
  }

  if (args.validate) {
    std::vector<float> cuda_output(input.size());
    check_cuda(cudaMemcpy(cuda_output.data(), device_output.data(),
                          cuda_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "benchmark output D2H");

    std::vector<float> expected(input.size());
    std::vector<float> scratch(shape.rows * shape.bins);
    const std::size_t series_elements = shape.rows * shape.bins;
    for (std::size_t series = 0; series < args.nseries; ++series) {
      const auto input_series = std::span<const float>(input).subspan(
          series * series_elements, series_elements);
      auto output_series =
          std::span<float>(expected).subspan(series * series_elements,
                                             series_elements);
      gaffa::ffa_transform_block_cpu(input_series, shape, scratch,
                                     output_series);
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (cuda_output[index] != expected[index]) {
        throw std::runtime_error("CUDA transform validation failed");
      }
    }
  }

  return result;
}

BenchmarkResult run_cuda_program(const std::vector<float>& input,
                                 const Args& args,
                                 gaffa::FfaTransformShape shape) {
  int device_count = 0;
  check_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    throw std::runtime_error("CUDA device is not visible");
  }
  check_cuda(cudaSetDevice(0), "cudaSetDevice");

  const auto program_build_start = std::chrono::steady_clock::now();
  const gaffa::CudaFfaProgram program(
      make_single_task_plan(shape.rows, shape.bins));
  const double program_build_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    program_build_start)
          .count();
  std::cout << "cuda_program_build_seconds=" << program_build_seconds
            << " cuda_program_device_metadata_bytes="
            << program.device_metadata_bytes()
            << '\n';

  gaffa::CudaDeviceBuffer<float> device_input(input.size());
  gaffa::CudaDeviceBuffer<float> device_scratch(input.size());
  gaffa::CudaDeviceBuffer<float> device_output(input.size());
  check_cuda(cudaMemcpy(device_input.data(), input.data(),
                        input.size() * sizeof(float), cudaMemcpyHostToDevice),
             "benchmark input H2D");

  const std::size_t series_elements = shape.rows * shape.bins;
  const auto transform_input = gaffa::CudaFfaInput{
      .data = device_input.data(),
      .nseries = args.nseries,
      .nsamples = series_elements,
      .stride = series_elements,
      .shape = shape,
      .device_id = 0,
  };
  const auto transform_scratch = gaffa::CudaFfaBuffer{
      .data = device_scratch.data(),
      .nseries = args.nseries,
      .stride = series_elements,
      .shape = shape,
      .device_id = 0,
  };
  const auto transform_output = gaffa::CudaFfaBuffer{
      .data = device_output.data(),
      .nseries = args.nseries,
      .stride = series_elements,
      .shape = shape,
      .device_id = 0,
  };

  for (int index = 0; index < args.warmup; ++index) {
    (void)time_cuda_program_transform(program, transform_input,
                                      transform_scratch, transform_output);
  }

  BenchmarkResult result{.mode = Mode::CudaProgram};
  for (int iteration = 0; iteration < args.iterations; ++iteration) {
    const double seconds = time_cuda_program_transform(
        program, transform_input, transform_scratch, transform_output);
    result.runs.push_back(IterationResult{
        .iteration = iteration,
        .seconds = seconds,
    });
    std::cout << "mode=" << mode_name(result.mode) << " iteration="
              << iteration << " seconds=" << seconds << '\n';
  }

  if (args.validate) {
    std::vector<float> cuda_output(input.size());
    check_cuda(cudaMemcpy(cuda_output.data(), device_output.data(),
                          cuda_output.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "benchmark output D2H");

    std::vector<float> expected(input.size());
    std::vector<float> scratch(shape.rows * shape.bins);
    const std::size_t series_elements = shape.rows * shape.bins;
    for (std::size_t series = 0; series < args.nseries; ++series) {
      const auto input_series = std::span<const float>(input).subspan(
          series * series_elements, series_elements);
      auto output_series =
          std::span<float>(expected).subspan(series * series_elements,
                                             series_elements);
      gaffa::ffa_transform_block_cpu(input_series, shape, scratch,
                                     output_series);
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
      if (cuda_output[index] != expected[index]) {
        throw std::runtime_error("CUDA program transform validation failed");
      }
    }
  }

  return result;
}

SummaryStats summarize(const std::vector<IterationResult>& runs) {
  if (runs.empty()) {
    return {};
  }

  std::vector<double> values;
  values.reserve(runs.size());
  for (const auto& run : runs) {
    values.push_back(run.seconds);
  }
  std::sort(values.begin(), values.end());

  const double sum = std::accumulate(values.begin(), values.end(), 0.0);
  const double mean = sum / static_cast<double>(values.size());
  double variance = 0.0;
  for (const double value : values) {
    const double diff = value - mean;
    variance += diff * diff;
  }
  variance /= static_cast<double>(values.size());

  return SummaryStats{
      .mean_seconds = mean,
      .median_seconds = values[values.size() / 2],
      .min_seconds = values.front(),
      .max_seconds = values.back(),
      .stddev_seconds = std::sqrt(variance),
  };
}

int omp_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

std::string cuda_device_name() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return "none";
  }
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, 0) != cudaSuccess) {
    return "unknown";
  }
  return props.name;
}

void print_final_report(const Args& args,
                        std::size_t total_elements,
                        const std::vector<BenchmarkResult>& results) {
  const std::size_t output_bytes = total_elements * sizeof(float);
  const std::size_t three_buffer_bytes = 3 * output_bytes;
  const double output_gib = bytes_to_gib(output_bytes);

  std::cout << "final_report_begin rows=" << args.rows
            << " bins=" << args.bins
            << " nseries=" << args.nseries
            << " iterations=" << args.iterations
            << " warmup=" << args.warmup
            << " validate=" << (args.validate ? 1 : 0)
            << " elements=" << total_elements
            << " output_bytes=" << output_bytes
            << " three_buffer_bytes=" << three_buffer_bytes
            << " output_gib=" << output_gib
            << " omp_threads=" << omp_threads()
            << " cuda_device=\"" << cuda_device_name() << "\"\n";

  for (const auto& result : results) {
    const SummaryStats stats = summarize(result.runs);
    std::cout << "final_report mode=" << mode_name(result.mode)
              << " mean_seconds=" << stats.mean_seconds
              << " median_seconds=" << stats.median_seconds
              << " min_seconds=" << stats.min_seconds
              << " max_seconds=" << stats.max_seconds
              << " stddev_seconds=" << stats.stddev_seconds
              << " output_gib_per_second="
              << (stats.mean_seconds > 0.0 ? output_gib / stats.mean_seconds
                                            : 0.0)
              << '\n';
  }
  std::cout << "final_report_end\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    const Args args = parse_args(argc, argv);
    const auto shape = gaffa::FfaTransformShape{
        .rows = args.rows,
        .bins = args.bins,
    };
    const std::size_t series_elements =
        checked_multiply(args.rows, args.bins, "series element count overflow");
    const std::size_t total_elements = checked_multiply(
        args.nseries, series_elements, "total element count overflow");
    const std::vector<float> input = make_input(total_elements);

    std::vector<BenchmarkResult> results;
    for (const Mode mode : selected_modes(args.mode)) {
      switch (mode) {
        case Mode::CpuSerial:
          results.push_back(run_cpu_serial(input, args, shape));
          break;
        case Mode::CpuOpenmp:
          results.push_back(run_cpu_openmp(input, args, shape));
          break;
        case Mode::Cuda:
          results.push_back(run_cuda(input, args, shape));
          break;
        case Mode::CudaProgram:
          results.push_back(run_cuda_program(input, args, shape));
          break;
      }
    }

    print_final_report(args, total_elements, results);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
