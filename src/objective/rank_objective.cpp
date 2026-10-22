#include "rank_objective.hpp"

namespace LightGBM {

void UpdatePointwiseScoresForOneQuery(double* score_pointwise, const double* score_pairwise, data_size_t cnt_pointwise,
  const std::pair<data_size_t, data_size_t>* paired_index_map,
  const std::vector<std::vector<std::pair<short, data_size_t>>>& right2left2pair_map,
  const std::vector<std::vector<std::pair<short, data_size_t>>>& left2right2pair_map,
  int truncation_level, double sigma, const CommonC::SigmoidCache& sigmoid_cache, bool model_indirect_comparison, bool model_conditional_rel,
  bool indirect_comparison_above_only, bool logarithmic_discounts, bool hard_pairwise_preference, int indirect_comparison_max_rank,
  double indirect_comparison_weight, double l2_pairwise_diff_weight) {

  // get sorted indices for scores
  global_timer.Start("pairwise_lambdarank::UpdatePointwiseScoresForOneQuery part 0");
  std::vector<data_size_t> sorted_idx(cnt_pointwise);
  for (data_size_t i = 0; i < cnt_pointwise; ++i) {
    sorted_idx[i] = i;
  }
  std::stable_sort(
    sorted_idx.begin(), sorted_idx.end(),
    [score_pointwise](data_size_t a, data_size_t b) { return score_pointwise[a] > score_pointwise[b]; });
  // get ranks when sorted by scores
  std::vector<int> ranks(cnt_pointwise);
  for (int i = 0; i < cnt_pointwise; i++) {
    ranks[sorted_idx.at(i)] = i;
  }
  global_timer.Stop("pairwise_lambdarank::UpdatePointwiseScoresForOneQuery part 0");
  global_timer.Start("pairwise_lambdarank::UpdatePointwiseScoresForOneQuery part 1");
  std::vector<double> gradients(cnt_pointwise);
  std::vector<double> hessians(cnt_pointwise);

  for (data_size_t i = 0; i < cnt_pointwise - 1 && i < truncation_level; ++i) {
    if (score_pointwise[sorted_idx[i]] == kMinScore) { continue; }
    for (data_size_t j = i + 1; j < cnt_pointwise; ++j) {
      if (score_pointwise[sorted_idx[j]] == kMinScore) { continue; }

      const data_size_t indexLeft = sorted_idx[i];
      const data_size_t indexRight = sorted_idx[j];

      double delta_score = 0.0;
      double delta_score_total_weight = 0.0;
      int comparisons_direct = 0;

      data_size_t pair = get_pair_index(left2right2pair_map[indexLeft], indexRight);
      if (pair != static_cast<data_size_t>(-1)) {
        delta_score += score_pairwise[pair];
        ++comparisons_direct;
      }

      data_size_t pair_inverse = get_pair_index(left2right2pair_map[indexRight], indexLeft);
      if (pair_inverse != static_cast<data_size_t>(-1)) {
        delta_score -= score_pairwise[pair_inverse];
        ++comparisons_direct;
      }

      if (comparisons_direct > 0) {
        delta_score /= comparisons_direct;
        delta_score *= (1.0 - indirect_comparison_weight);
        delta_score_total_weight += (1.0 - indirect_comparison_weight);
      }

      if (model_indirect_comparison) {
        double delta_score_indirect = 0.0;
        int comparisons_indirect = 0;

        auto apply_score = [&](data_size_t a, data_size_t b, score_t signA, score_t signB) noexcept {
          delta_score_indirect += signA * score_pairwise[a] + signB * score_pairwise[b];
          ++comparisons_indirect;
          };

        process_indirect_comparisons_optimized(
          indexLeft, indexRight, ranks, left2right2pair_map, right2left2pair_map,
          indirect_comparison_max_rank, indirect_comparison_above_only, model_conditional_rel,
          // Case 1: (+, −)
          [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, +1, -1); },
          // Case 2: (−, −)
          [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, -1, -1); },
          // Case 3: (+, +)
          [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, +1, +1); },
          // Case 4: (−, +)
          [&](data_size_t idxA, data_size_t idxB) noexcept { apply_score(idxA, idxB, -1, +1); }
        );

        if (comparisons_indirect > 0) {
          delta_score_indirect /= comparisons_indirect;
          delta_score += delta_score_indirect * indirect_comparison_weight;
          delta_score_total_weight += indirect_comparison_weight;
        }
      }

      if (delta_score_total_weight <= 0.0) { continue;  }
      delta_score /= delta_score_total_weight;

      double delta_score_pointwise = score_pointwise[indexLeft] - score_pointwise[indexRight];
      if (delta_score_pointwise == kMinScore || -delta_score_pointwise == kMinScore || delta_score == kMinScore || -delta_score == kMinScore) { continue; }
      // get discount of this pair	
      double paired_discount = logarithmic_discounts ? fabs(DCGCalculator::GetDiscount(ranks[indexRight]) - DCGCalculator::GetDiscount(ranks[indexLeft])) : 1.0;
      double p_lr_pairwise = sigmoid_cache.compute(-delta_score);
      double p_rl_pairwise = 1.0 - p_lr_pairwise;
      double p_lr_pointwise = sigmoid_cache.compute(-delta_score_pointwise);
      double p_rl_pointwise = 1.0 - p_lr_pointwise;

      if (hard_pairwise_preference) {
        paired_discount *= std::abs(0.5 - p_lr_pairwise);
        p_lr_pairwise = p_lr_pairwise >= 0.5 ? 1.0 : 0.0;
        p_rl_pairwise = 1.0 - p_lr_pairwise;
      }

      gradients[indexLeft] += sigma * paired_discount * (p_rl_pointwise - p_rl_pairwise);
      hessians[indexLeft] += sigma * sigma * paired_discount * p_rl_pointwise * p_lr_pointwise;
      gradients[indexRight] -= sigma * paired_discount * (p_rl_pointwise - p_rl_pairwise);
      hessians[indexRight] += sigma * sigma * paired_discount * p_rl_pointwise * p_lr_pointwise;


      // === Pairwise-difference L2 regularization (observed pairs only)
      // Apply iff there is at least one direct observation between (indexLeft,indexRight)
      if (l2_pairwise_diff_weight > 0.0) {
        const double w = paired_discount;
        const double d = delta_score_pointwise;                             // (s_i - s_j)
        // Gradient contributions: -lambda*w·(s_i-s_j)  and  +2lambda*w·(s_i-s_j)
        gradients[indexLeft] -= 2.0 * l2_pairwise_diff_weight * w * d;
        gradients[indexRight] += 2.0 * l2_pairwise_diff_weight * w * d;
        // Diagonal Hessian contributions: +2lambda*w for both sides (ignore off-diagonals for speed)
        hessians[indexLeft] += 2.0 * l2_pairwise_diff_weight * w;
        hessians[indexRight] += 2.0 * l2_pairwise_diff_weight * w;
      }
    }
  }
  global_timer.Stop("pairwise_lambdarank::UpdatePointwiseScoresForOneQuery part 1");
  global_timer.Start("pairwise_lambdarank::UpdatePointwiseScoresForOneQuery part 2");
  for (data_size_t i = 0; i < cnt_pointwise; i++) {
    double delta = 0.3 * gradients[i] / (std::abs(hessians[i]) + 0.001);
    delta = std::min(delta, 0.3);
    delta = std::max(delta, -0.3);
    score_pointwise[i] += delta;
  }
  global_timer.Stop("pairwise_lambdarank::UpdatePointwiseScoresForOneQuery part 2");
}

}  // namespace LightGBM
