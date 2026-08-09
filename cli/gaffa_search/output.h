#pragma once

#include "config.h"
#include "report.h"

#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <span>
#include <vector>

namespace gaffa_search {

struct OutputPaths {
  std::filesystem::path candidates;
  std::filesystem::path report;
};

OutputPaths make_output_paths(const Config& config,
                              const std::filesystem::path& input);
std::vector<OutputPaths> plan_output_paths(
    const Config& config,
    std::span<const std::filesystem::path> inputs);
void validate_output_plan(std::span<const OutputPaths> outputs,
                          bool overwrite);

// limit == 0 means render every final candidate.
void render_human_report(std::ostream& output, const Config& config,
                         const Report& report, std::size_t limit);

void write_candidate_file(const std::filesystem::path& path,
                          const Report& report, bool overwrite);
void write_report_file(const std::filesystem::path& path, const Config& config,
                       const Report& report, bool overwrite);

}  // namespace gaffa_search
