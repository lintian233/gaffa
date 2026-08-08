#pragma once

#include "config.h"

#include <filesystem>

namespace gaffa_search {

// Loads and validates one complete CLI configuration. Paths relative to the
// YAML file are resolved against that file's parent directory.
Config load_yaml_config(const std::filesystem::path& path);

}  // namespace gaffa_search
