#include "config.h"
#include "output.h"
#include "progress_renderer.h"
#include "report.h"
#include "search.h"

#ifdef GAFFA_SEARCH_ENABLE_LOKI
#include "gaffa/loki_logging.h"
#endif

#include <cstdlib>
#include <exception>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
  try {
    const gaffa_search::Config config =
        gaffa_search::parse_arguments(argc, argv);
#ifdef GAFFA_SEARCH_ENABLE_LOKI
    gaffa::suppress_loki_info();
#endif
    const std::vector<std::filesystem::path> inputs =
        gaffa_search::discover_inputs(config.input);
    const std::vector<gaffa_search::OutputPaths> output_plan =
        gaffa_search::plan_output_paths(config, inputs);
    gaffa_search::validate_output_plan(output_plan,
                                       config.overwrite_output);

    {
      gaffa_search::ProgressTracker progress;
      gaffa_search::ProgressRenderer renderer(std::cerr);
      gaffa_search::ProgressSession session(renderer, progress);
      std::size_t file_index = 0;
      gaffa_search::search_each_file(
          config, inputs, progress, [&](gaffa_search::FileResult file) {
            const gaffa_search::Report report =
                gaffa_search::make_report(file);
            if (config.candidate_output) {
              const gaffa_search::OutputPaths& paths =
                  output_plan.at(file_index);
              gaffa_search::write_candidate_file(
                  paths.candidates, report, config.overwrite_output);
              gaffa_search::write_report_file(
                  paths.report, config, report, config.overwrite_output);
            }
            gaffa_search::render_human_report(std::cout, config, report,
                                              config.print_candidates);
            ++file_index;
          });
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
