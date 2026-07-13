#pragma once
#include "flashattention.h"
#include <cstddef>
namespace transformer {
struct FaCandidate {
  int bm, bn, nw;
  void (*launch)(const FlashAttentionParams &);
  size_t smem;
};
int fa_autotune_pick(const FaCandidate *cands, int n,
                     const FlashAttentionParams &params);

} // namespace transformer