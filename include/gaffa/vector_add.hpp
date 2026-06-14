#pragma once

#include <vector>

namespace gaffa {

std::vector<float> vector_add(const std::vector<float>& lhs, const std::vector<float>& rhs);
int cuda_device_count();
int cuda_runtime_version();

}  // namespace gaffa
