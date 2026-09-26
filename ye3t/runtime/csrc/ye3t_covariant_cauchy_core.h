// Torch-free kernels for covariant lifted-Cauchy multiplets.
//
// A compiled model is an exact real sparse polynomial schedule:
//
//   out[row(t)] += c(t) * prod_k x[factor(t, k)],
//
// where every output row is one real component of a complete O(3) multiplet
// and x holds the role-resolved real-tesseral densities of one site. The
// reverse kernel seeds an arbitrary output cotangent, so a scalar energy with
// folded readout weights is the one-row special case of the same arithmetic.
// The product-rule derivative is explicit and division free.

#ifndef YE3T_COVARIANT_CAUCHY_CORE_H
#define YE3T_COVARIANT_CAUCHY_CORE_H

#include <cstdint>

namespace ye3t {
namespace covariant_cauchy {

// inputs:          [site_count, input_count]
// outputs:         [site_count, row_count], overwritten
// cotangent:       [site_count, row_count] or nullptr
// input_gradient:  [site_count, input_count], overwritten when cotangent is set
void forward_vjp(std::int64_t site_count, std::int64_t input_count,
                 std::int64_t row_count, std::int64_t term_count,
                 const std::int64_t* term_row, const double* term_coefficient,
                 const std::int64_t* factor_offsets, const std::int64_t* factors,
                 const double* inputs, const double* cotangent, double* outputs,
                 double* input_gradient);

// Streamed normal equations of the covariant linear model
//
//   y[site, a] = sum_d theta[d] * features[site, d, a],
//
// where one weight per multiplet is shared by all components a. Accumulates
// gram[d, e] += sum_{site, a} w * f[site, d, a] * f[site, e, a] and
// rhs[d] += sum_{site, a} w * f[site, d, a] * y[site, a], with one sample
// weight per site so that the weighting never distinguishes components.
void accumulate_normal_equations(std::int64_t site_count,
                                 std::int64_t multiplet_count,
                                 std::int64_t multiplet_width,
                                 const double* features, const double* targets,
                                 const double* site_weights, double* gram,
                                 double* rhs);

}  // namespace covariant_cauchy
}  // namespace ye3t

#endif
