#include "gaffa/loki_logging.h"

#include <spdlog/spdlog.h>

namespace gaffa {

void suppress_loki_info() {
  spdlog::set_level(spdlog::level::warn);
}

}  // namespace gaffa
