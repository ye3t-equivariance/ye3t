#include "ye3t_covariant_cauchy_core.h"

#include <vector>

namespace ye3t {
namespace covariant_cauchy {

void forward_vjp(std::int64_t site_count, std::int64_t input_count,
                 std::int64_t row_count, std::int64_t term_count,
                 const std::int64_t* term_row, const double* term_coefficient,
                 const std::int64_t* factor_offsets, const std::int64_t* factors,
                 const double* inputs, const double* cotangent, double* outputs,
                 double* input_gradient) {
  std::int64_t maximum_degree = 0;
  for (std::int64_t term = 0; term < term_count; ++term) {
    const std::int64_t degree = factor_offsets[term + 1] - factor_offsets[term];
    if (degree > maximum_degree) maximum_degree = degree;
  }
  std::vector<double> prefix(static_cast<std::size_t>(maximum_degree) + 1);
  for (std::int64_t site = 0; site < site_count; ++site) {
    const double* x = inputs + site * input_count;
    double* out = outputs + site * row_count;
    for (std::int64_t row = 0; row < row_count; ++row) out[row] = 0.0;
    const double* seed = cotangent ? cotangent + site * row_count : nullptr;
    double* gradient = seed ? input_gradient + site * input_count : nullptr;
    if (gradient) {
      for (std::int64_t index = 0; index < input_count; ++index) {
        gradient[index] = 0.0;
      }
    }
    for (std::int64_t term = 0; term < term_count; ++term) {
      const std::int64_t begin = factor_offsets[term];
      const std::int64_t degree = factor_offsets[term + 1] - begin;
      prefix[0] = term_coefficient[term];
      for (std::int64_t k = 0; k < degree; ++k) {
        prefix[static_cast<std::size_t>(k) + 1] =
            prefix[static_cast<std::size_t>(k)] * x[factors[begin + k]];
      }
      out[term_row[term]] += prefix[static_cast<std::size_t>(degree)];
      if (!gradient) continue;
      const double weight = seed[term_row[term]];
      if (weight == 0.0) continue;
      // Leave-one-out products from the stored prefixes and a running suffix.
      double suffix = weight;
      for (std::int64_t k = degree - 1; k >= 0; --k) {
        const std::int64_t index = factors[begin + k];
        gradient[index] += prefix[static_cast<std::size_t>(k)] * suffix;
        suffix *= x[index];
      }
    }
  }
}

void accumulate_normal_equations(std::int64_t site_count,
                                 std::int64_t multiplet_count,
                                 std::int64_t multiplet_width,
                                 const double* features, const double* targets,
                                 const double* site_weights, double* gram,
                                 double* rhs) {
  const std::int64_t stride = multiplet_count * multiplet_width;
  for (std::int64_t site = 0; site < site_count; ++site) {
    const double weight = site_weights ? site_weights[site] : 1.0;
    if (weight == 0.0) continue;
    const double* f = features + site * stride;
    const double* y = targets + site * multiplet_width;
    for (std::int64_t d = 0; d < multiplet_count; ++d) {
      const double* fd = f + d * multiplet_width;
      double projection = 0.0;
      for (std::int64_t a = 0; a < multiplet_width; ++a) {
        projection += fd[a] * y[a];
      }
      rhs[d] += weight * projection;
      for (std::int64_t e = d; e < multiplet_count; ++e) {
        const double* fe = f + e * multiplet_width;
        double inner = 0.0;
        for (std::int64_t a = 0; a < multiplet_width; ++a) {
          inner += fd[a] * fe[a];
        }
        gram[d * multiplet_count + e] += weight * inner;
      }
    }
  }
  for (std::int64_t d = 0; d < multiplet_count; ++d) {
    for (std::int64_t e = d + 1; e < multiplet_count; ++e) {
      gram[e * multiplet_count + d] = gram[d * multiplet_count + e];
    }
  }
}

}  // namespace covariant_cauchy
}  // namespace ye3t
