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

struct FfaRowView {
  // Points into the FfaSearchPlan passed to for_each_ffa_row_cpu().
  const FfaSearchTask* task = nullptr;

  // Row index within this task's full FFA transform result.
  std::size_t shift = 0;

  // One FFA transform result row. Equivalent to the corresponding row in a
  // materialized FFA block and valid only for the callback duration.
  std::span<const float> profile;

  // Noise scale for Riptide-compatible FFA S/N normalization.
  float stdnoise = 0.0F;
};

using FfaBlockConsumer = std::function<void(const FfaBlockView&)>;
using FfaRowConsumer = std::function<void(const FfaRowView&)>;

// Streams one FFA transform block per plan task to consumer. This function owns
// and reuses all temporary buffers; it does not store blocks or run detection.
void for_each_ffa_block_cpu(std::span<const float> time_series,
                            const FfaSearchPlan& plan,
                            const FfaBlockConsumer& consumer);

// Streams FFA transform result rows to consumer without materializing the final
// rows_eval x bins block. This preserves FFA transform semantics: each emitted
// profile is equivalent to the matching row in the materialized block API.
void for_each_ffa_row_cpu(std::span<const float> time_series,
                          const FfaSearchPlan& plan,
                          const FfaRowConsumer& consumer);

}  // namespace gaffa
