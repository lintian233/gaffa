#pragma once

namespace gaffa {

// Loki uses a process-wide spdlog registry. Call this once at the application
// boundary, before starting any Loki search, to retain only warnings/errors.
void suppress_loki_info();

}  // namespace gaffa
