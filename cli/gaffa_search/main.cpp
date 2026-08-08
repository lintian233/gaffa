#include "config.h"
#include "output.h"
#include "report.h"
#include "search.h"

#include <cstdlib>
#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  try {
    const gaffa_search::Config config =
        gaffa_search::parse_arguments(argc, argv);
    const gaffa_search::RunResult result = gaffa_search::execute(config);
    for (const gaffa_search::FileResult& file : result.files) {
      const gaffa_search::Report report = gaffa_search::make_report(file);
      if (config.candidate_output) {
        const gaffa_search::OutputPaths paths =
            gaffa_search::make_output_paths(config, file.input);
        gaffa_search::validate_output_paths(paths, config.overwrite_output);
        gaffa_search::write_candidate_file(paths.candidates, report,
                                           config.overwrite_output);
        gaffa_search::write_report_file(paths.report, config, report,
                                         config.overwrite_output);
      }
      gaffa_search::render_human_report(std::cout, config, report,
                                        config.print_candidates);
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
