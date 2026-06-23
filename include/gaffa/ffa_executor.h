#pragma once

#include "gaffa/ffa.h"
#include "gaffa/ffa_plan.h"

#include <functional>
#include <span>

namespace gaffa {

struct FfaBlockView {
  // Points into the FfaSearchPlan passed to for_each_ffa_block_cpu().
  const FfaSearchTask* task = nullptr;

  // Shape of the exposed transform block. This is rows_eval x bins, not the
  // full rows x bins transform used internally to preserve FFA period mapping.
  FfaTransformShape shape{};

  // Valid only for the duration of the consumer callback. The executor reuses
  // this backing buffer for the next task.
  std::span<const float> transform;

  // Noise scale for Riptide-compatible FFA S/N normalization.
  float stdnoise = 0.0F;
};

using FfaBlockConsumer = std::function<void(const FfaBlockView&)>;

// Streams one FFA transform block per plan task to consumer. This function owns
// and reuses all temporary buffers; it does not store blocks or run detection.
void for_each_ffa_block_cpu(std::span<const float> time_series,
                            const FfaSearchPlan& plan,
                            const FfaBlockConsumer& consumer);

}  // namespace gaffa
