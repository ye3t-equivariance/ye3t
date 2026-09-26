#include "ye3t_runtime_core.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

// Lane loops of the split-real tiled kernels address disjoint entries per
// lane (every store is indexed by a loop-invariant offset plus the lane), so
// no iteration depends on another. GCC/Clang cannot prove that from the
// strided pointers and emit scalar code with runtime alias checks; the
// annotation states the independence and lets them vectorize the lanes.
// Measured 1.19-1.24x on the wide dummy catalogues at the x86-64 baseline
// and 1.35x together with -march=native. Define YE3T_RUNTIME_NO_LANE_IVDEP
// to compile the annotated loops without the assertion.
#if defined(YE3T_RUNTIME_NO_LANE_IVDEP)
#define YE3T_LANE_LOOP
#elif defined(__GNUC__) && !defined(__clang__)
#define YE3T_LANE_LOOP _Pragma("GCC ivdep")
#elif defined(__clang__)
#define YE3T_LANE_LOOP _Pragma("clang loop vectorize(assume_safety)")
#else
#define YE3T_LANE_LOOP
#endif

namespace ye3t {
namespace runtime {
namespace {

template <typename Real>
std::vector<Real> legendre_coefficients(std::int64_t degree) {
  if (degree == 0) {
    return {Real(1)};
  }
  std::vector<Real> previous{Real(1)};
  std::vector<Real> current{Real(0), Real(1)};
  for (std::int64_t n = 1; n < degree; ++n) {
    std::vector<Real> next(static_cast<std::size_t>(n + 2), Real(0));
    const Real forward = Real(2 * n + 1) / Real(n + 1);
    const Real backward = Real(n) / Real(n + 1);
    for (std::size_t power = 0; power < current.size(); ++power) {
      next[power + 1] += forward * current[power];
    }
    for (std::size_t power = 0; power < previous.size(); ++power) {
      next[power] -= backward * previous[power];
    }
    previous = std::move(current);
    current = std::move(next);
  }
  return current;
}

template <typename Real>
std::vector<Real> derivative_coefficients(
    std::vector<Real> coefficients,
    std::int64_t order) {
  for (std::int64_t derivative = 0; derivative < order; ++derivative) {
    if (coefficients.size() <= 1) {
      return {Real(0)};
    }
    std::vector<Real> next(coefficients.size() - 1, Real(0));
    for (std::size_t power = 1; power < coefficients.size(); ++power) {
      next[power - 1] = Real(power) * coefficients[power];
    }
    coefficients = std::move(next);
  }
  return coefficients;
}

template <typename Real>
Real polynomial_value(const std::vector<Real>& coefficients, Real x) {
  Real value = Real(0);
  for (auto it = coefficients.rbegin(); it != coefficients.rend(); ++it) {
    value = value * x + *it;
  }
  return value;
}

template <typename Real>
Real polynomial_value(
    const Real* coefficients,
    std::int64_t coefficient_count,
    Real x) {
  Real value = Real(0);
  for (std::int64_t coefficient = coefficient_count;
       coefficient > 0;
       --coefficient) {
    value = value * x + coefficients[coefficient - 1];
  }
  return value;
}

template <typename Scalar>
Scalar integer_power(Scalar base, std::int64_t exponent) {
  Scalar value = Scalar(1);
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      value *= base;
    }
    exponent >>= 1;
    if (exponent > 0) {
      base *= base;
    }
  }
  return value;
}

std::int64_t binomial_coefficient(
    std::int64_t dimension,
    std::int64_t order) {
  if (order < 0 || order > dimension) {
    return 0;
  }
  order = std::min(order, dimension - order);
  std::int64_t value = 1;
  for (std::int64_t index = 1; index <= order; ++index) {
    value = value * (dimension - order + index) / index;
  }
  return value;
}

std::int64_t population_count(std::size_t value) {
  std::int64_t count = 0;
  while (value != 0) {
    value &= value - std::size_t(1);
    ++count;
  }
  return count;
}

bool next_combination(
    std::vector<std::int64_t>* combination,
    std::int64_t dimension) {
  const std::int64_t order =
      static_cast<std::int64_t>(combination->size());
  for (std::int64_t index = order - 1; index >= 0; --index) {
    const std::int64_t maximum = dimension - order + index;
    if ((*combination)[static_cast<std::size_t>(index)] >= maximum) {
      continue;
    }
    ++(*combination)[static_cast<std::size_t>(index)];
    for (std::int64_t next = index + 1; next < order; ++next) {
      (*combination)[static_cast<std::size_t>(next)] =
          (*combination)[static_cast<std::size_t>(next - 1)] + 1;
    }
    return true;
  }
  return false;
}

template <typename Scalar>
Scalar determinant_subset_dp(
    const std::vector<Scalar>& matrix,
    std::int64_t order) {
  if (order == 0) {
    return Scalar(1);
  }
  const std::size_t state_count =
      std::size_t(1) << static_cast<std::size_t>(order);
  std::vector<Scalar> states(state_count, Scalar(0));
  states[0] = Scalar(1);
  for (std::size_t mask = 0; mask < state_count; ++mask) {
    const std::int64_t row = population_count(mask);
    if (row >= order) {
      continue;
    }
    for (std::int64_t column = 0; column < order; ++column) {
      const std::size_t bit =
          std::size_t(1) << static_cast<std::size_t>(column);
      if ((mask & bit) != 0) {
        continue;
      }
      const std::size_t larger_mask =
          mask & ~((bit << std::size_t(1)) - std::size_t(1));
      const bool negative = (population_count(larger_mask) & 1) != 0;
      const Scalar term =
          states[mask] *
          matrix[static_cast<std::size_t>(row * order + column)];
      states[mask | bit] += negative ? -term : term;
    }
  }
  return states.back();
}

template <typename Scalar>
Scalar determinant_minor(
    const std::vector<Scalar>& matrix,
    std::int64_t order,
    std::int64_t removed_row,
    std::int64_t removed_column) {
  std::vector<Scalar> minor;
  minor.reserve(
      static_cast<std::size_t>((order - 1) * (order - 1)));
  for (std::int64_t row = 0; row < order; ++row) {
    if (row == removed_row) {
      continue;
    }
    for (std::int64_t column = 0; column < order; ++column) {
      if (column == removed_column) {
        continue;
      }
      minor.push_back(
          matrix[static_cast<std::size_t>(row * order + column)]);
    }
  }
  return determinant_subset_dp(minor, order - 1);
}

template <typename Real>
Real spherical_normalization(
    std::int64_t angular_momentum,
    std::int64_t magnetic_abs) {
  const Real log_norm = Real(0.5) * (
      std::log(Real(2 * angular_momentum + 1)) -
      std::log(Real(4) * std::acos(Real(-1))) +
      std::lgamma(Real(angular_momentum - magnetic_abs + 1)) -
      std::lgamma(Real(angular_momentum + magnetic_abs + 1)));
  return std::exp(log_norm);
}

template <typename Real>
void real_spherical_unit_value_and_gradient(
    std::int64_t angular_momentum,
    Real x,
    Real y,
    Real z,
    Real* values,
    Real* unit_gradients) {
  const std::int64_t width = 2 * angular_momentum + 1;
  std::fill(values, values + width, Real(0));
  std::fill(unit_gradients, unit_gradients + width * 3, Real(0));
  const auto legendre = legendre_coefficients<Real>(angular_momentum);
  const auto legendre_z = derivative_coefficients<Real>(legendre, 1);
  const Real norm_zero =
      spherical_normalization<Real>(angular_momentum, 0);
  values[angular_momentum] =
      norm_zero * polynomial_value(legendre, z);
  unit_gradients[(angular_momentum * 3) + 2] =
      norm_zero * polynomial_value(legendre_z, z);

  const Real sqrt_two = std::sqrt(Real(2));
  const std::complex<Real> xy(x, y);
  std::complex<Real> power(Real(1), Real(0));
  std::complex<Real> previous_power(Real(1), Real(0));
  for (std::int64_t magnetic = 1;
       magnetic <= angular_momentum;
       ++magnetic) {
    previous_power = power;
    power *= xy;
    const auto derivative = derivative_coefficients<Real>(
        legendre,
        magnetic);
    const auto derivative_z = derivative_coefficients<Real>(
        legendre,
        magnetic + 1);
    const Real polynomial = polynomial_value(derivative, z);
    const Real polynomial_z = polynomial_value(derivative_z, z);
    const Real scale = sqrt_two * spherical_normalization<Real>(
        angular_momentum,
        magnetic);
    const std::int64_t positive = angular_momentum + magnetic;
    const std::int64_t negative = angular_momentum - magnetic;
    const Real power_dx = Real(magnetic) * previous_power.real();
    const Real power_dy_real =
        -Real(magnetic) * previous_power.imag();
    const Real power_dy_imag =
        Real(magnetic) * previous_power.real();
    const Real power_dx_imag =
        Real(magnetic) * previous_power.imag();

    values[positive] = scale * polynomial * power.real();
    unit_gradients[positive * 3] =
        scale * polynomial * power_dx;
    unit_gradients[positive * 3 + 1] =
        scale * polynomial * power_dy_real;
    unit_gradients[positive * 3 + 2] =
        scale * polynomial_z * power.real();

    values[negative] = -scale * polynomial * power.imag();
    unit_gradients[negative * 3] =
        -scale * polynomial * power_dx_imag;
    unit_gradients[negative * 3 + 1] =
        -scale * polynomial * power_dy_imag;
    unit_gradients[negative * 3 + 2] =
        -scale * polynomial_z * power.imag();
  }
}

template <typename Scalar>
Scalar conjugate(const Scalar& value) {
  return value;
}

template <typename Real>
std::complex<Real> conjugate(const std::complex<Real>& value) {
  return std::conj(value);
}

template <typename Real>
Real real_component(const std::complex<Real>& value) {
  return value.real();
}

template <typename Scalar>
Scalar project_control_gradient(const Scalar& value) {
  return value;
}

template <typename Real>
Real project_control_gradient(const std::complex<Real>& value) {
  return value.real();
}

}  // namespace

template <typename Real>
void cheb_exp_cos_radial_with_derivative(
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t radial_index,
    Real* values,
    Real* derivatives) {
  const Real pi = std::acos(Real(-1));
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Real radius = radii[edge];
    const Real cutoff = std::max(cutoffs[edge], epsilon);
    const Real lambda = lambdas[edge];
    const Real scaled = radius / cutoff;
    if (scaled > Real(1)) {
      values[edge] = Real(0);
      derivatives[edge] = Real(0);
      continue;
    }
    if (radial_index == 0) {
      values[edge] = Real(1);
      derivatives[edge] = Real(0);
      continue;
    }
    if (radial_index == 1) {
      values[edge] = Real(0.5) * (Real(1) + std::cos(pi * scaled));
      derivatives[edge] =
          -Real(0.5) * pi * std::sin(pi * scaled) / cutoff;
      continue;
    }

    const Real numerator_exp =
        std::exp(-lambda * (scaled - Real(1)));
    const Real denominator = std::expm1(lambda);
    const Real warped =
        Real(1) - Real(2) * std::expm1(
            -lambda * (scaled - Real(1))) / denominator;
    Real t_previous = Real(1);
    Real t_current = warped;
    Real u_previous = Real(1);
    Real u_current = Real(2) * warped;
    for (std::int64_t order = 2; order <= radial_index; ++order) {
      const Real t_next =
          Real(2) * warped * t_current - t_previous;
      t_previous = t_current;
      t_current = t_next;
      if (order < radial_index) {
        const Real u_next =
            Real(2) * warped * u_current - u_previous;
        u_previous = u_current;
        u_current = u_next;
      }
    }
    const Real chebyshev = t_current;
    const Real chebyshev_derivative =
        Real(radial_index) * u_current;
    const Real warped_derivative =
        Real(2) * lambda * numerator_exp / (denominator * cutoff);
    const Real envelope = Real(1) + std::cos(pi * scaled);
    const Real envelope_derivative =
        -pi * std::sin(pi * scaled) / cutoff;
    values[edge] =
        Real(0.25) * (Real(1) - chebyshev) * envelope;
    derivatives[edge] = Real(0.25) * (
        -chebyshev_derivative * warped_derivative * envelope +
        (Real(1) - chebyshev) * envelope_derivative);
  }
}

template <typename Real>
void cheb_exp_cos_radial_table_with_derivative(
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t maximum_radial_index,
    Real* values,
    Real* derivatives) {
  const std::int64_t width = maximum_radial_index + 1;
  std::vector<Real> column_values(
      static_cast<std::size_t>(edge_count));
  std::vector<Real> column_derivatives(
      static_cast<std::size_t>(edge_count));
  for (std::int64_t radial_index = 0;
       radial_index <= maximum_radial_index;
       ++radial_index) {
    cheb_exp_cos_radial_with_derivative<Real>(
        radii,
        cutoffs,
        lambdas,
        edge_count,
        radial_index,
        column_values.data(),
        column_derivatives.data());
    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
      const std::int64_t output = edge * width + radial_index;
      values[output] = column_values[edge];
      derivatives[output] = column_derivatives[edge];
    }
  }
}

template <typename Real>
void pace_cheb_exp_cos_radial_table_with_derivative(
    const Real* radii,
    const Real* cutoffs,
    const Real* cutoff_widths,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t radial_count,
    Real* values,
    Real* derivatives) {
  if (edge_count <= 0 || radial_count <= 0) {
    return;
  }
  const Real pi = std::acos(Real(-1));
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Real radius = radii[edge];
    const Real cutoff = std::max(cutoffs[edge], epsilon);
    const Real cutoff_width = cutoff_widths[edge];
    const std::int64_t offset = edge * radial_count;
    if (radius >= cutoff) {
      std::fill(values + offset, values + offset + radial_count, Real(0));
      std::fill(
          derivatives + offset,
          derivatives + offset + radial_count,
          Real(0));
      continue;
    }

    const Real scaled = radius / cutoff;
    const Real envelope = Real(0.5) * (
        Real(1) + std::cos(pi * scaled));
    const Real envelope_derivative =
        -Real(0.5) * pi * std::sin(pi * scaled) / cutoff;
    Real outer_switch = Real(1);
    Real outer_switch_derivative = Real(0);
    if (cutoff_width > Real(0) &&
        radius > cutoff - cutoff_width) {
      const Real phase =
          pi * (radius - (cutoff - cutoff_width)) / cutoff_width;
      outer_switch = Real(0.5) * (Real(1) + std::cos(phase));
      outer_switch_derivative =
          -Real(0.5) * pi * std::sin(phase) / cutoff_width;
    }
    const Real combined_envelope = envelope * outer_switch;
    const Real combined_envelope_derivative =
        envelope_derivative * outer_switch +
        envelope * outer_switch_derivative;

    values[offset] = combined_envelope;
    derivatives[offset] = combined_envelope_derivative;
    if (radial_count == 1) {
      continue;
    }

    const Real lambda = lambdas[edge];
    const Real exponential = std::exp(-lambda * (scaled - Real(1)));
    const Real denominator = std::expm1(lambda);
    const Real warped =
        Real(1) - Real(2) * (exponential - Real(1)) / denominator;
    const Real warped_derivative =
        Real(2) * lambda * exponential / (denominator * cutoff);
    Real chebyshev_previous = Real(1);
    Real chebyshev = warped;
    Real chebyshev_derivative_previous = Real(0);
    Real chebyshev_derivative = warped_derivative;
    for (std::int64_t radial = 1; radial < radial_count; ++radial) {
      if (radial > 1) {
        const Real next =
            Real(2) * warped * chebyshev - chebyshev_previous;
        const Real next_derivative =
            Real(2) * (
                warped_derivative * chebyshev +
                warped * chebyshev_derivative) -
            chebyshev_derivative_previous;
        chebyshev_previous = chebyshev;
        chebyshev = next;
        chebyshev_derivative_previous = chebyshev_derivative;
        chebyshev_derivative = next_derivative;
      }
      const Real base = Real(0.5) * (Real(1) - chebyshev);
      const Real base_derivative =
          -Real(0.5) * chebyshev_derivative;
      values[offset + radial] = base * combined_envelope;
      derivatives[offset + radial] =
          base_derivative * combined_envelope +
          base * combined_envelope_derivative;
    }
  }
}

template <typename Real>
std::int64_t pace_uniform_spline_interval_count(
    Real requested_spacing,
    Real cutoff) {
  if (!std::isfinite(requested_spacing) ||
      !std::isfinite(cutoff) ||
      requested_spacing <= Real(0) ||
      cutoff <= Real(0)) {
    throw std::invalid_argument(
        "PACE spline spacing and cutoff must be finite and positive");
  }
  const Real ratio = cutoff / requested_spacing;
  if (ratio < Real(1) ||
      ratio >= Real(std::numeric_limits<std::int64_t>::max())) {
    throw std::invalid_argument("PACE spline interval count is out of range");
  }
  return static_cast<std::int64_t>(ratio);
}

template <typename Real>
void pace_uniform_cubic_spline_build(
    const Real* node_values,
    const Real* node_derivatives,
    std::int64_t interval_count,
    std::int64_t function_count,
    Real cutoff,
    Real* coefficients) {
  if (interval_count <= 0 || function_count <= 0 ||
      !std::isfinite(cutoff) || cutoff <= Real(0)) {
    throw std::invalid_argument("Invalid PACE spline table dimensions");
  }
  if (node_values == nullptr || node_derivatives == nullptr ||
      coefficients == nullptr) {
    throw std::invalid_argument("PACE spline table received a null array");
  }
  const Real scale = Real(interval_count) / cutoff;
  const Real spacing = Real(1) / scale;
  const std::int64_t coefficient_count =
      (interval_count + 1) * function_count * 4;
  std::fill(
      coefficients,
      coefficients + coefficient_count,
      Real(0));
  for (std::int64_t interval = 1;
       interval <= interval_count;
       ++interval) {
    const std::int64_t left_node = interval - 1;
    const bool has_right_node = interval < interval_count;
    for (std::int64_t function = 0;
         function < function_count;
         ++function) {
      const std::int64_t left =
          left_node * function_count + function;
      const std::int64_t right =
          interval * function_count + function;
      const Real f0 = node_values[left];
      const Real f1 = has_right_node ? node_values[right] : Real(0);
      const Real d0 = node_derivatives[left] * spacing;
      const Real d1 = has_right_node
          ? node_derivatives[right] * spacing
          : Real(0);
      const std::int64_t output =
          (interval * function_count + function) * 4;
      coefficients[output] = f0;
      coefficients[output + 1] = d0;
      coefficients[output + 2] =
          Real(3) * (f1 - f0) - d1 - Real(2) * d0;
      coefficients[output + 3] =
          -Real(2) * (f1 - f0) + d1 + d0;
    }
  }
}

template <typename Real>
void pace_uniform_cubic_spline_evaluate_with_derivative(
    const Real* radii,
    const Real* coefficients,
    std::int64_t point_count,
    std::int64_t function_count,
    std::int64_t interval_count,
    Real cutoff,
    Real* values,
    Real* derivatives) {
  if (point_count < 0 || interval_count <= 0 || function_count <= 0 ||
      !std::isfinite(cutoff) || cutoff <= Real(0)) {
    throw std::invalid_argument("Invalid PACE spline evaluation dimensions");
  }
  if (point_count == 0) {
    return;
  }
  if (radii == nullptr || coefficients == nullptr || values == nullptr ||
      derivatives == nullptr) {
    throw std::invalid_argument("PACE spline evaluation received a null array");
  }
  const Real scale = Real(interval_count) / cutoff;
  for (std::int64_t point = 0; point < point_count; ++point) {
    const Real radius = radii[point];
    if (!std::isfinite(radius)) {
      throw std::invalid_argument("PACE spline radius must be finite");
    }
    const std::int64_t output = point * function_count;
    if (radius >= cutoff) {
      std::fill(values + output, values + output + function_count, Real(0));
      std::fill(
          derivatives + output,
          derivatives + output + function_count,
          Real(0));
      continue;
    }
    const Real scaled = radius * scale;
    const std::int64_t interval =
        static_cast<std::int64_t>(std::floor(scaled));
    if (interval <= 0) {
      throw std::invalid_argument("PACE spline radius is below its first interval");
    }
    const Real local = scaled - Real(interval);
    const Real local2 = local * local;
    const Real local3 = local2 * local;
    for (std::int64_t function = 0;
         function < function_count;
         ++function) {
      const std::int64_t coefficient =
          (interval * function_count + function) * 4;
      const Real c0 = coefficients[coefficient];
      const Real c1 = coefficients[coefficient + 1];
      const Real c2 = coefficients[coefficient + 2];
      const Real c3 = coefficients[coefficient + 3];
      values[output + function] =
          c0 + c1 * local + c2 * local2 + c3 * local3;
      derivatives[output + function] =
          (c1 + Real(2) * c2 * local + Real(3) * c3 * local2) * scale;
    }
  }
}

template <typename Real>
void pace_radial_channel_contraction_with_derivative(
    const Real* input_values,
    const Real* input_derivatives,
    const Real* coefficients,
    std::int64_t point_count,
    std::int64_t input_count,
    std::int64_t output_count,
    Real* output_values,
    Real* output_derivatives) {
  if (point_count < 0 || input_count <= 0 || output_count <= 0) {
    throw std::invalid_argument("Invalid PACE radial contraction dimensions");
  }
  if (point_count == 0) {
    return;
  }
  if (input_values == nullptr || input_derivatives == nullptr ||
      coefficients == nullptr || output_values == nullptr ||
      output_derivatives == nullptr) {
    throw std::invalid_argument("PACE radial contraction received a null array");
  }
  for (std::int64_t point = 0; point < point_count; ++point) {
    for (std::int64_t output = 0; output < output_count; ++output) {
      Real value = Real(0);
      Real derivative = Real(0);
      for (std::int64_t input = 0; input < input_count; ++input) {
        const Real coefficient = coefficients[output * input_count + input];
        value += coefficient * input_values[point * input_count + input];
        derivative +=
            coefficient * input_derivatives[point * input_count + input];
      }
      output_values[point * output_count + output] = value;
      output_derivatives[point * output_count + output] = derivative;
    }
  }
}

template <typename Real>
void cheb_exp_cos_radial_table_double_backward(
    const Real* values_adjoint,
    const Real* radial_derivatives,
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    const Real* grad_grad_radii,
    std::int64_t edge_count,
    std::int64_t maximum_radial_index,
    Real* values_adjoint_gradient,
    Real* radii_gradient) {
  const Real pi = std::acos(Real(-1));
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const std::int64_t width = maximum_radial_index + 1;
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Real tangent = grad_grad_radii[edge];
    Real second_contraction = Real(0);
    for (std::int64_t radial_index = 0;
         radial_index <= maximum_radial_index;
         ++radial_index) {
      const std::int64_t index = edge * width + radial_index;
      values_adjoint_gradient[index] =
          tangent * radial_derivatives[index];
    }
    const Real cutoff = std::max(cutoffs[edge], epsilon);
    const Real scaled = radii[edge] / cutoff;
    if (scaled > Real(1) || maximum_radial_index == 0) {
      radii_gradient[edge] = Real(0);
      continue;
    }
    const Real cosine = std::cos(pi * scaled);
    const Real sine = std::sin(pi * scaled);
    const Real inverse_cutoff = Real(1) / cutoff;
    const Real envelope = Real(1) + cosine;
    const Real envelope_derivative =
        -pi * sine * inverse_cutoff;
    const Real envelope_second =
        -pi * pi * cosine * inverse_cutoff * inverse_cutoff;
    second_contraction +=
        values_adjoint[edge * width + 1] *
        Real(0.5) * envelope_second;
    if (maximum_radial_index >= 2) {
      const Real lambda = lambdas[edge];
      const Real numerator_exp =
          std::exp(-lambda * (scaled - Real(1)));
      const Real denominator = std::expm1(lambda);
      const Real warped =
          Real(1) - Real(2) * std::expm1(
              -lambda * (scaled - Real(1))) / denominator;
      const Real warped_derivative =
          Real(2) * lambda * numerator_exp /
          (denominator * cutoff);
      const Real warped_second =
          -Real(2) * lambda * lambda * numerator_exp /
          (denominator * cutoff * cutoff);
      Real t_previous = Real(1);
      Real t_current = warped;
      Real dt_previous = Real(0);
      Real dt_current = Real(1);
      Real d2t_previous = Real(0);
      Real d2t_current = Real(0);
      for (std::int64_t radial_index = 2;
           radial_index <= maximum_radial_index;
           ++radial_index) {
        const Real t_next =
            Real(2) * warped * t_current - t_previous;
        const Real dt_next =
            Real(2) * t_current +
            Real(2) * warped * dt_current -
            dt_previous;
        const Real d2t_next =
            Real(4) * dt_current +
            Real(2) * warped * d2t_current -
            d2t_previous;
        const Real radial_second = Real(0.25) * (
            -(
                d2t_next * warped_derivative *
                    warped_derivative +
                dt_next * warped_second) *
                envelope -
            Real(2) * dt_next * warped_derivative *
                envelope_derivative +
            (Real(1) - t_next) * envelope_second);
        second_contraction +=
            values_adjoint[edge * width + radial_index] *
            radial_second;
        t_previous = t_current;
        t_current = t_next;
        dt_previous = dt_current;
        dt_current = dt_next;
        d2t_previous = d2t_current;
        d2t_current = d2t_next;
      }
    }
    radii_gradient[edge] = tangent * second_contraction;
  }
}

template <typename Real>
void real_spherical_harmonics_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t angular_momentum,
    Real epsilon,
    Real* values,
    Real* derivatives) {
  const std::int64_t width = 2 * angular_momentum + 1;
  std::vector<Real> unit_values(static_cast<std::size_t>(width));
  std::vector<Real> unit_gradients(
      static_cast<std::size_t>(width * 3));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Real x = edge_vectors[edge * 3];
    const Real y = edge_vectors[edge * 3 + 1];
    const Real z = edge_vectors[edge * 3 + 2];
    const Real radius = std::sqrt(x * x + y * y + z * z);
    const Real safe_radius = std::max(radius, epsilon);
    const Real unit[3] = {
        x / safe_radius,
        y / safe_radius,
        z / safe_radius};
    real_spherical_unit_value_and_gradient<Real>(
        angular_momentum,
        unit[0],
        unit[1],
        unit[2],
        unit_values.data(),
        unit_gradients.data());
    for (std::int64_t magnetic = 0; magnetic < width; ++magnetic) {
      values[edge * width + magnetic] = unit_values[magnetic];
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        Real gradient = Real(0);
        for (std::int64_t unit_axis = 0;
             unit_axis < 3;
             ++unit_axis) {
          Real jacobian = Real(0);
          if (radius > epsilon) {
            jacobian =
                ((axis == unit_axis ? Real(1) : Real(0)) -
                 unit[axis] * unit[unit_axis]) /
                safe_radius;
          } else if (axis == unit_axis) {
            jacobian = Real(1) / safe_radius;
          }
          gradient +=
              unit_gradients[magnetic * 3 + unit_axis] *
              jacobian;
        }
        derivatives[(edge * width + magnetic) * 3 + axis] =
            gradient;
      }
    }
  }
}

template <typename Real>
void real_spherical_harmonics_table_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    Real* values,
    Real* derivatives) {
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  for (std::int64_t angular_momentum = 0;
       angular_momentum <= maximum_angular_momentum;
       ++angular_momentum) {
    const std::int64_t degree_width =
        2 * angular_momentum + 1;
    std::vector<Real> degree_values(
        static_cast<std::size_t>(edge_count * degree_width));
    std::vector<Real> degree_derivatives(
        static_cast<std::size_t>(
            edge_count * degree_width * 3));
    real_spherical_harmonics_with_derivative<Real>(
        edge_vectors,
        edge_count,
        angular_momentum,
        epsilon,
        degree_values.data(),
        degree_derivatives.data());
    const std::int64_t offset =
        angular_momentum * angular_momentum;
    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
      for (std::int64_t component = 0;
           component < degree_width;
           ++component) {
        const std::int64_t source =
            edge * degree_width + component;
        const std::int64_t output =
            edge * width + offset + component;
        values[output] = degree_values[source];
        for (std::int64_t axis = 0; axis < 3; ++axis) {
          derivatives[output * 3 + axis] =
              degree_derivatives[source * 3 + axis];
        }
      }
    }
  }
}

template <typename Real>
void complex_spherical_harmonics_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t angular_momentum,
    Real epsilon,
    std::complex<Real>* values,
    std::complex<Real>* derivatives) {
  const std::int64_t width = 2 * angular_momentum + 1;
  std::vector<Real> real_values(
      static_cast<std::size_t>(edge_count * width));
  std::vector<Real> real_derivatives(
      static_cast<std::size_t>(edge_count * width * 3));
  real_spherical_harmonics_with_derivative<Real>(
      edge_vectors,
      edge_count,
      angular_momentum,
      epsilon,
      real_values.data(),
      real_derivatives.data());
  const Real inverse_sqrt_two = Real(1) / std::sqrt(Real(2));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t center = edge * width + angular_momentum;
    values[center] =
        std::complex<Real>(real_values[center], Real(0));
    for (std::int64_t axis = 0; axis < 3; ++axis) {
      derivatives[center * 3 + axis] =
          std::complex<Real>(
              real_derivatives[center * 3 + axis],
              Real(0));
    }
    for (std::int64_t magnetic = 1;
         magnetic <= angular_momentum;
         ++magnetic) {
      const std::int64_t negative =
          edge * width + angular_momentum - magnetic;
      const std::int64_t positive =
          edge * width + angular_momentum + magnetic;
      const Real sign = magnetic % 2 == 0 ? Real(1) : Real(-1);
      values[negative] = inverse_sqrt_two * std::complex<Real>(
          real_values[positive],
          real_values[negative]);
      values[positive] = sign * inverse_sqrt_two *
          std::complex<Real>(
              real_values[positive],
              -real_values[negative]);
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[negative * 3 + axis] =
            inverse_sqrt_two * std::complex<Real>(
                real_derivatives[positive * 3 + axis],
                real_derivatives[negative * 3 + axis]);
        derivatives[positive * 3 + axis] =
            sign * inverse_sqrt_two * std::complex<Real>(
                real_derivatives[positive * 3 + axis],
                -real_derivatives[negative * 3 + axis]);
      }
    }
  }
}

std::int64_t complex_spherical_harmonics_table_plan_size(
    std::int64_t maximum_angular_momentum) {
  const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
  if (maximum_angular_momentum < 0 || maximum_angular_momentum == maximum) {
    throw std::invalid_argument(
        "Invalid complex spherical-harmonic plan dimensions");
  }
  const std::int64_t degree_count = maximum_angular_momentum + 1;
  if (degree_count > maximum / (degree_count + 1)) {
    throw std::overflow_error("Complex spherical-harmonic plan is too large");
  }
  const std::int64_t order_rows = degree_count * (degree_count + 1);
  if (order_rows > maximum / degree_count ||
      degree_count > maximum / degree_count) {
    throw std::overflow_error("Complex spherical-harmonic plan is too large");
  }
  const std::int64_t coefficient_count = order_rows * degree_count;
  const std::int64_t normalization_count = degree_count * degree_count;
  if (coefficient_count > maximum - normalization_count) {
    throw std::overflow_error("Complex spherical-harmonic plan is too large");
  }
  return coefficient_count + normalization_count;
}

std::int64_t complex_spherical_harmonics_nonnegative_table_width(
    std::int64_t maximum_angular_momentum) {
  const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
  if (maximum_angular_momentum < 0 ||
      maximum_angular_momentum > maximum - 2) {
    throw std::invalid_argument(
        "Invalid complex spherical-harmonic table dimensions");
  }
  const std::int64_t degree_count = maximum_angular_momentum + 1;
  const std::int64_t next_degree_count = degree_count + 1;
  if (degree_count > maximum / next_degree_count) {
    throw std::overflow_error("Complex spherical-harmonic table is too large");
  }
  return degree_count * next_degree_count / 2;
}

template <typename Real>
void build_complex_spherical_harmonics_table_plan(
    std::int64_t maximum_angular_momentum,
    Real* plan,
    std::int64_t plan_size,
    Real normalization_scale) {
  const std::int64_t required =
      complex_spherical_harmonics_table_plan_size(maximum_angular_momentum);
  if (plan == nullptr || plan_size < required) {
    throw std::invalid_argument(
        "Complex spherical-harmonic plan storage is too small");
  }
  std::fill(plan, plan + required, Real(0));
  const std::int64_t degree_count = maximum_angular_momentum + 1;
  const std::int64_t order_count = degree_count + 1;
  const std::int64_t normalization_offset =
      degree_count * order_count * degree_count;
  for (std::int64_t angular_momentum = 0;
       angular_momentum <= maximum_angular_momentum;
       ++angular_momentum) {
    const auto legendre = legendre_coefficients<Real>(angular_momentum);
    for (std::int64_t order = 0;
         order <= angular_momentum + 1;
         ++order) {
      const auto derivative =
          derivative_coefficients<Real>(legendre, order);
      const std::int64_t offset =
          (angular_momentum * order_count + order) * degree_count;
      std::copy(derivative.begin(), derivative.end(), plan + offset);
    }
    for (std::int64_t magnetic = 0;
         magnetic <= angular_momentum;
         ++magnetic) {
      plan[normalization_offset + angular_momentum * degree_count + magnetic] =
          normalization_scale *
          spherical_normalization<Real>(angular_momentum, magnetic);
    }
  }
}

namespace {

template <typename Real, bool NonnegativeOnly, bool PrecomputedGeometry>
void complex_spherical_harmonics_table_with_derivative_prevalidated_impl(
    const Real* edge_vectors,
    const Real* precomputed_radii,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    const Real* plan,
    std::int64_t plan_size,
    std::complex<Real>* workspace,
    std::int64_t workspace_size,
    std::complex<Real>* values,
    std::complex<Real>* derivatives,
    Real output_scale) {
  const std::int64_t required =
      complex_spherical_harmonics_table_plan_size(maximum_angular_momentum);
  const std::int64_t degree_count = maximum_angular_momentum + 1;
  const std::int64_t width = NonnegativeOnly
      ? complex_spherical_harmonics_nonnegative_table_width(
            maximum_angular_momentum)
      : degree_count * degree_count;
  if (edge_count < 0 || !std::isfinite(epsilon) || epsilon <= Real(0) ||
      plan_size < required || workspace_size < degree_count) {
    throw std::invalid_argument(
        "Invalid prevalidated complex spherical-harmonic table dimensions");
  }
  if (edge_count == 0) {
    return;
  }
  if (edge_vectors == nullptr ||
      (PrecomputedGeometry && precomputed_radii == nullptr) ||
      plan == nullptr || workspace == nullptr || values == nullptr ||
      derivatives == nullptr) {
    throw std::invalid_argument(
        "Prevalidated complex spherical-harmonic table received a null array");
  }
  const std::int64_t order_count = degree_count + 1;
  const std::int64_t normalization_offset =
      degree_count * order_count * degree_count;
  std::complex<Real>* xy_powers = workspace;
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    Real radius = Real(0);
    Real unit[3];
    if constexpr (PrecomputedGeometry) {
      radius = precomputed_radii[edge];
      unit[0] = edge_vectors[edge * 3];
      unit[1] = edge_vectors[edge * 3 + 1];
      unit[2] = edge_vectors[edge * 3 + 2];
    } else {
      const Real x = edge_vectors[edge * 3];
      const Real y = edge_vectors[edge * 3 + 1];
      const Real z = edge_vectors[edge * 3 + 2];
      radius = std::sqrt(x * x + y * y + z * z);
      if (radius <= epsilon) {
        for (std::int64_t angular_momentum = 0;
             angular_momentum <= maximum_angular_momentum;
             ++angular_momentum) {
          const std::int64_t degree_width = 2 * angular_momentum + 1;
          std::vector<std::complex<Real>> degree_values(
              static_cast<std::size_t>(degree_width));
          std::vector<std::complex<Real>> degree_derivatives(
              static_cast<std::size_t>(degree_width * 3));
          complex_spherical_harmonics_with_derivative<Real>(
              edge_vectors + edge * 3, 1, angular_momentum, epsilon,
              degree_values.data(), degree_derivatives.data());
          const std::int64_t offset = NonnegativeOnly
              ? angular_momentum * (angular_momentum + 1) / 2
              : angular_momentum * angular_momentum;
          const std::int64_t first_component =
              NonnegativeOnly ? angular_momentum : 0;
          const std::int64_t stored_width =
              NonnegativeOnly ? angular_momentum + 1 : degree_width;
          for (std::int64_t component = 0;
               component < stored_width;
               ++component) {
            const std::int64_t source = first_component + component;
            const std::int64_t destination = edge * width + offset + component;
            values[destination] = output_scale * degree_values[source];
            for (std::int64_t axis = 0; axis < 3; ++axis) {
              derivatives[destination * 3 + axis] = output_scale *
                  degree_derivatives[source * 3 + axis];
            }
          }
        }
        continue;
      }
      unit[0] = x / radius;
      unit[1] = y / radius;
      unit[2] = z / radius;
    }
    xy_powers[0] = std::complex<Real>(Real(1), Real(0));
    for (std::int64_t magnetic = 1;
         magnetic <= maximum_angular_momentum;
         ++magnetic) {
      const auto previous =
          xy_powers[static_cast<std::size_t>(magnetic - 1)];
      xy_powers[static_cast<std::size_t>(magnetic)] = {
          previous.real() * unit[0] - previous.imag() * unit[1],
          previous.real() * unit[1] + previous.imag() * unit[0]};
    }
    const Real derivative_scale = output_scale / radius;

    for (std::int64_t angular_momentum = 0;
         angular_momentum <= maximum_angular_momentum;
         ++angular_momentum) {
      const std::int64_t offset = NonnegativeOnly
          ? angular_momentum * (angular_momentum + 1) / 2
          : angular_momentum * angular_momentum;
      const Real* degree_coefficients = plan +
          angular_momentum * order_count * degree_count;
      const Real* degree_normalizations = plan + normalization_offset +
          angular_momentum * degree_count;

      auto write_component = [&](std::int64_t component,
                                 Real value_real,
                                 Real value_imaginary,
                                 const Real* unit_gradient_real,
                                 const Real* unit_gradient_imaginary) {
        const std::int64_t output = edge * width + offset + component;
        values[output] = std::complex<Real>(
            output_scale * value_real, output_scale * value_imaginary);
        const Real radial_gradient_real =
            unit_gradient_real[0] * unit[0] +
            unit_gradient_real[1] * unit[1] +
            unit_gradient_real[2] * unit[2];
        const Real radial_gradient_imaginary =
            unit_gradient_imaginary[0] * unit[0] +
            unit_gradient_imaginary[1] * unit[1] +
            unit_gradient_imaginary[2] * unit[2];
        for (std::int64_t axis = 0; axis < 3; ++axis) {
          derivatives[output * 3 + axis] = std::complex<Real>(
              derivative_scale *
                  (unit_gradient_real[axis] -
                   unit[axis] * radial_gradient_real),
              derivative_scale *
                  (unit_gradient_imaginary[axis] -
                   unit[axis] * radial_gradient_imaginary));
        }
      };

      const Real norm_zero = degree_normalizations[0];
      const Real zero_gradient_real[3] = {
          Real(0), Real(0),
          norm_zero * polynomial_value(
              degree_coefficients + degree_count,
              std::max<std::int64_t>(1, angular_momentum), unit[2])};
      const Real zero_gradient_imaginary[3] = {Real(0), Real(0), Real(0)};
      write_component(
          NonnegativeOnly ? 0 : angular_momentum,
          norm_zero * polynomial_value(
              degree_coefficients, angular_momentum + 1, unit[2]),
          Real(0), zero_gradient_real, zero_gradient_imaginary);

      for (std::int64_t magnetic = 1;
           magnetic <= angular_momentum;
           ++magnetic) {
        const Real norm = degree_normalizations[magnetic];
        const Real* magnetic_coefficients =
            degree_coefficients + magnetic * degree_count;
        const Real* magnetic_z_coefficients =
            magnetic_coefficients + degree_count;
        const Real polynomial = polynomial_value(
            magnetic_coefficients,
            angular_momentum - magnetic + 1, unit[2]);
        const Real polynomial_z = polynomial_value(
            magnetic_z_coefficients,
            std::max<std::int64_t>(1, angular_momentum - magnetic), unit[2]);
        const auto power =
            xy_powers[static_cast<std::size_t>(magnetic)];
        const auto previous_power =
            xy_powers[static_cast<std::size_t>(magnetic - 1)];
        const Real sign = magnetic % 2 == 0 ? Real(1) : Real(-1);
        const Real value_scale = sign * norm * polynomial;
        const Real xy_gradient_scale = value_scale * Real(magnetic);
        const Real z_gradient_scale = sign * norm * polynomial_z;
        const Real positive_gradient_real[3] = {
            xy_gradient_scale * previous_power.real(),
            -xy_gradient_scale * previous_power.imag(),
            z_gradient_scale * power.real()};
        const Real positive_gradient_imaginary[3] = {
            xy_gradient_scale * previous_power.imag(),
            xy_gradient_scale * previous_power.real(),
            z_gradient_scale * power.imag()};
        write_component(
            NonnegativeOnly ? magnetic : angular_momentum + magnetic,
            value_scale * power.real(), value_scale * power.imag(),
            positive_gradient_real, positive_gradient_imaginary);
        if constexpr (!NonnegativeOnly) {
          const std::int64_t positive =
              edge * width + offset + angular_momentum + magnetic;
          const std::int64_t negative =
              edge * width + offset + angular_momentum - magnetic;
          values[negative] = sign * std::conj(values[positive]);
          for (std::int64_t axis = 0; axis < 3; ++axis) {
            derivatives[negative * 3 + axis] =
                sign * std::conj(derivatives[positive * 3 + axis]);
          }
        }
      }
    }
  }
}

}  // namespace

template <typename Real>
void complex_spherical_harmonics_table_with_derivative_prevalidated(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    const Real* plan,
    std::int64_t plan_size,
    std::complex<Real>* workspace,
    std::int64_t workspace_size,
    std::complex<Real>* values,
    std::complex<Real>* derivatives,
    Real output_scale) {
  complex_spherical_harmonics_table_with_derivative_prevalidated_impl<
      Real, false, false>(
      edge_vectors, nullptr, edge_count, maximum_angular_momentum, epsilon, plan,
      plan_size, workspace, workspace_size, values, derivatives, output_scale);
}

template <typename Real>
void complex_spherical_harmonics_nonnegative_table_with_derivative_prevalidated(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    const Real* plan,
    std::int64_t plan_size,
    std::complex<Real>* workspace,
    std::int64_t workspace_size,
    std::complex<Real>* values,
    std::complex<Real>* derivatives,
    Real output_scale) {
  complex_spherical_harmonics_table_with_derivative_prevalidated_impl<
      Real, true, false>(
      edge_vectors, nullptr, edge_count, maximum_angular_momentum, epsilon, plan,
      plan_size, workspace, workspace_size, values, derivatives, output_scale);
}

template <typename Real>
void complex_spherical_harmonics_nonnegative_unit_table_with_derivative_prevalidated(
    const Real* unit_vectors,
    const Real* radii,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    const Real* plan,
    std::int64_t plan_size,
    std::complex<Real>* workspace,
    std::int64_t workspace_size,
    std::complex<Real>* values,
    std::complex<Real>* derivatives) {
  complex_spherical_harmonics_table_with_derivative_prevalidated_impl<
      Real, true, true>(
      unit_vectors, radii, edge_count, maximum_angular_momentum, epsilon, plan,
      plan_size, workspace, workspace_size, values, derivatives, Real(1));
}

std::int64_t complex_spherical_harmonics_recurrence_plan_size(
    std::int64_t maximum_angular_momentum) {
  const std::int64_t width =
      complex_spherical_harmonics_nonnegative_table_width(
          maximum_angular_momentum);
  const std::int64_t maximum = std::numeric_limits<std::int64_t>::max();
  if (width > (maximum - 1) / 2) {
    throw std::overflow_error(
        "Complex spherical-harmonic recurrence plan is too large");
  }
  return 1 + 2 * width;
}

template <typename Real>
void build_complex_spherical_harmonics_recurrence_plan(
    std::int64_t maximum_angular_momentum,
    Real* plan,
    std::int64_t plan_size,
    Real normalization_scale) {
  const std::int64_t required =
      complex_spherical_harmonics_recurrence_plan_size(
          maximum_angular_momentum);
  if (plan == nullptr || plan_size < required ||
      !std::isfinite(normalization_scale)) {
    throw std::invalid_argument(
        "Complex spherical-harmonic recurrence plan storage is invalid");
  }
  std::fill(plan, plan + required, Real(0));
  // Y_0^0 with the caller's scale; every later entry is a scale-free ratio of
  // fully normalized harmonics, so the scale propagates through the recurrence.
  plan[0] = normalization_scale * spherical_normalization<Real>(0, 0);
  for (std::int64_t degree = 1; degree <= maximum_angular_momentum; ++degree) {
    const double l = static_cast<double>(degree);
    for (std::int64_t order = 0; order <= degree; ++order) {
      const double m = static_cast<double>(order);
      const std::int64_t index = degree * (degree + 1) / 2 + order;
      if (order == degree) {
        // Condon-Shortley phase: Y_mm = -sqrt((2m+1)/(2m)) sin(theta) e^{i phi}
        // Y_{m-1,m-1}, with sin(theta) e^{i phi} = u_x + i u_y.
        plan[1 + 2 * index] =
            static_cast<Real>(-std::sqrt((2.0 * l + 1.0) / (2.0 * l)));
        plan[2 + 2 * index] = Real(0);
        continue;
      }
      // Fully normalized associated-Legendre degree recurrence in cos(theta).
      plan[1 + 2 * index] = static_cast<Real>(std::sqrt(
          (2.0 * l - 1.0) * (2.0 * l + 1.0) / ((l - m) * (l + m))));
      plan[2 + 2 * index] = static_cast<Real>(std::sqrt(
          (2.0 * l + 1.0) * (l + m - 1.0) * (l - m - 1.0) /
          ((2.0 * l - 3.0) * (l - m) * (l + m))));
    }
  }
}

template <typename Real>
void complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated(
    const Real* unit_vectors,
    const Real* radii,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    const Real* plan,
    std::int64_t plan_size,
    std::complex<Real>* values,
    std::complex<Real>* derivatives) {
  const std::int64_t required =
      complex_spherical_harmonics_recurrence_plan_size(
          maximum_angular_momentum);
  const std::int64_t width =
      complex_spherical_harmonics_nonnegative_table_width(
          maximum_angular_momentum);
  if (edge_count < 0 || plan_size < required) {
    throw std::invalid_argument(
        "Invalid complex spherical-harmonic recurrence dimensions");
  }
  if (edge_count == 0) {
    return;
  }
  if (unit_vectors == nullptr || radii == nullptr || plan == nullptr ||
      values == nullptr || derivatives == nullptr) {
    throw std::invalid_argument(
        "Complex spherical-harmonic recurrence received a null array");
  }
  struct State {
    Real real = Real(0);
    Real imaginary = Real(0);
    Real gradient_real[3] = {Real(0), Real(0), Real(0)};
    Real gradient_imaginary[3] = {Real(0), Real(0), Real(0)};
  };
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Real* unit = unit_vectors + edge * 3;
    const Real inverse_radius = Real(1) / radii[edge];
    // d(unit_k)/d(raw_j) = (delta_kj - unit_k unit_j) / radius.
    Real du[3][3];
    for (int k = 0; k < 3; ++k) {
      for (int j = 0; j < 3; ++j) {
        du[k][j] = ((k == j ? Real(1) : Real(0)) - unit[k] * unit[j]) *
            inverse_radius;
      }
    }
    std::complex<Real>* edge_values = values + edge * width;
    std::complex<Real>* edge_derivatives = derivatives + edge * width * 3;
    auto store = [&](std::int64_t index, const State& state) {
      edge_values[index] = std::complex<Real>(state.real, state.imaginary);
      for (int axis = 0; axis < 3; ++axis) {
        edge_derivatives[index * 3 + axis] = std::complex<Real>(
            state.gradient_real[axis], state.gradient_imaginary[axis]);
      }
    };
    State diagonal;
    diagonal.real = plan[0];
    for (std::int64_t order = 0; order <= maximum_angular_momentum; ++order) {
      if (order > 0) {
        const std::int64_t index = order * (order + 1) / 2 + order;
        const Real a = plan[1 + 2 * index];
        State next;
        next.real = a * (unit[0] * diagonal.real - unit[1] * diagonal.imaginary);
        next.imaginary =
            a * (unit[0] * diagonal.imaginary + unit[1] * diagonal.real);
        for (int axis = 0; axis < 3; ++axis) {
          next.gradient_real[axis] = a *
              (unit[0] * diagonal.gradient_real[axis] -
               unit[1] * diagonal.gradient_imaginary[axis] +
               du[0][axis] * diagonal.real - du[1][axis] * diagonal.imaginary);
          next.gradient_imaginary[axis] = a *
              (unit[0] * diagonal.gradient_imaginary[axis] +
               unit[1] * diagonal.gradient_real[axis] +
               du[0][axis] * diagonal.imaginary + du[1][axis] * diagonal.real);
        }
        diagonal = next;
      }
      store(order * (order + 1) / 2 + order, diagonal);
      State previous = diagonal;
      State previous2;
      for (std::int64_t degree = order + 1; degree <= maximum_angular_momentum;
           ++degree) {
        const std::int64_t index = degree * (degree + 1) / 2 + order;
        const Real a = plan[1 + 2 * index];
        const Real b = plan[2 + 2 * index];
        State next;
        next.real = a * unit[2] * previous.real - b * previous2.real;
        next.imaginary = a * unit[2] * previous.imaginary - b * previous2.imaginary;
        for (int axis = 0; axis < 3; ++axis) {
          next.gradient_real[axis] = a *
                  (unit[2] * previous.gradient_real[axis] +
                   du[2][axis] * previous.real) -
              b * previous2.gradient_real[axis];
          next.gradient_imaginary[axis] = a *
                  (unit[2] * previous.gradient_imaginary[axis] +
                   du[2][axis] * previous.imaginary) -
              b * previous2.gradient_imaginary[axis];
        }
        store(index, next);
        previous2 = previous;
        previous = next;
      }
    }
  }
}

template <typename Real>
void complex_spherical_harmonics_table_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    std::complex<Real>* values,
    std::complex<Real>* derivatives,
    Real output_scale) {
  const std::int64_t plan_size =
      complex_spherical_harmonics_table_plan_size(maximum_angular_momentum);
  std::vector<Real> plan(static_cast<std::size_t>(plan_size));
  build_complex_spherical_harmonics_table_plan<Real>(
      maximum_angular_momentum, plan.data(), plan_size);
  std::vector<std::complex<Real>> workspace(
      static_cast<std::size_t>(maximum_angular_momentum + 1));
  complex_spherical_harmonics_table_with_derivative_prevalidated<Real>(
      edge_vectors, edge_count, maximum_angular_momentum, epsilon,
      plan.data(), plan_size, workspace.data(),
      static_cast<std::int64_t>(workspace.size()), values, derivatives,
      output_scale);
}

template <typename Scalar>
void plain_site_basis_product_with_derivative(
    const Scalar* radial_values,
    const Scalar* radial_derivatives,
    const Scalar* angular_values,
    const Scalar* angular_derivatives,
    const Scalar* prefactors,
    const Scalar* prefactor_derivatives_center,
    const Scalar* prefactor_derivatives_neighbor,
    const Scalar* radial_directions,
    const std::int64_t* term_groups,
    const std::int64_t* term_channels,
    std::int64_t edge_count,
    std::int64_t group_count,
    std::int64_t term_count,
    std::int64_t channel_count,
    Scalar* edge_values,
    Scalar* edge_derivatives,
    Scalar* edge_charge_derivatives_center,
    Scalar* edge_charge_derivatives_neighbor) {
  (void)group_count;
  std::fill(
      edge_values,
      edge_values + edge_count * channel_count,
      Scalar(0));
  std::fill(
      edge_derivatives,
      edge_derivatives + edge_count * channel_count * 3,
      Scalar(0));
  std::fill(
      edge_charge_derivatives_center,
      edge_charge_derivatives_center +
          edge_count * channel_count,
      Scalar(0));
  std::fill(
      edge_charge_derivatives_neighbor,
      edge_charge_derivatives_neighbor +
          edge_count * channel_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    for (std::int64_t term = 0; term < term_count; ++term) {
      const std::int64_t group = term_groups[term];
      const std::int64_t channel = term_channels[term];
      const std::int64_t group_entry =
          group * edge_count + edge;
      const std::int64_t term_entry =
          term * edge_count + edge;
      const std::int64_t output =
          edge * channel_count + channel;
      const Scalar radial = radial_values[group_entry];
      const Scalar angular = angular_values[term_entry];
      const Scalar prefactor = prefactors[group_entry];
      edge_values[output] = prefactor * radial * angular;
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        edge_derivatives[output * 3 + axis] =
            prefactor * (
                radial_derivatives[group_entry] *
                    radial_directions[edge * 3 + axis] *
                    angular +
                radial *
                    angular_derivatives[term_entry * 3 + axis]);
      }
      const Scalar charge_common = radial * angular;
      edge_charge_derivatives_center[output] =
          prefactor_derivatives_center[group_entry] *
          charge_common;
      edge_charge_derivatives_neighbor[output] =
          prefactor_derivatives_neighbor[group_entry] *
          charge_common;
    }
  }
}

template <typename Real>
void scheduled_radial_angular_channels_with_derivative(
    const Real* radial_values,
    const Real* radial_derivatives,
    const Real* angular_values,
    const Real* angular_derivatives,
    const Real* radial_directions,
    const std::int64_t* edge_types,
    const std::int64_t* channel_radial_indices,
    const std::int64_t* channel_angular_indices,
    const std::int64_t* channel_types,
    const Real* channel_scales,
    std::int64_t edge_count,
    std::int64_t radial_width,
    std::int64_t angular_width,
    std::int64_t channel_count,
    Real* edge_values,
    Real* edge_derivatives) {
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    for (std::int64_t channel = 0;
         channel < channel_count;
         ++channel) {
      const std::int64_t output =
          edge * channel_count + channel;
      const std::int64_t required_type = channel_types[channel];
      if (required_type >= 0 && edge_types[edge] != required_type) {
        edge_values[output] = Real(0);
        for (std::int64_t axis = 0; axis < 3; ++axis) {
          edge_derivatives[output * 3 + axis] = Real(0);
        }
        continue;
      }
      const std::int64_t radial_index =
          channel_radial_indices[channel];
      const std::int64_t angular_index =
          channel_angular_indices[channel];
      const std::int64_t radial_entry =
          edge * radial_width + radial_index;
      const std::int64_t angular_entry =
          edge * angular_width + angular_index;
      const Real radial = radial_values[radial_entry];
      const Real angular = angular_values[angular_entry];
      const Real scale = channel_scales[channel];
      edge_values[output] = scale * radial * angular;
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        edge_derivatives[output * 3 + axis] = scale * (
            radial_derivatives[radial_entry] *
                radial_directions[edge * 3 + axis] *
                angular +
            radial *
                angular_derivatives[angular_entry * 3 + axis]);
      }
    }
  }
}

template <typename Scalar>
void plain_site_basis_product_adjoint(
    const Scalar* radial_values,
    const Scalar* radial_derivatives,
    const Scalar* angular_values,
    const Scalar* angular_derivatives,
    const Scalar* prefactors,
    const Scalar* prefactor_derivatives_center,
    const Scalar* prefactor_derivatives_neighbor,
    const Scalar* radial_directions,
    const Scalar* edge_weights,
    const Scalar* edge_weight_derivatives,
    const std::int64_t* term_groups,
    const std::int64_t* term_channels,
    const Scalar* edge_adjoint,
    std::int64_t edge_count,
    std::int64_t group_count,
    std::int64_t term_count,
    std::int64_t channel_count,
    Scalar* edge_position_adjoint,
    Scalar* edge_charge_adjoint_center,
    Scalar* edge_charge_adjoint_neighbor) {
  (void)group_count;
  std::fill(
      edge_position_adjoint,
      edge_position_adjoint + edge_count * 3,
      Scalar(0));
  std::fill(
      edge_charge_adjoint_center,
      edge_charge_adjoint_center + edge_count,
      Scalar(0));
  std::fill(
      edge_charge_adjoint_neighbor,
      edge_charge_adjoint_neighbor + edge_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    for (std::int64_t term = 0; term < term_count; ++term) {
      const std::int64_t group = term_groups[term];
      const std::int64_t channel = term_channels[term];
      const std::int64_t group_entry =
          group * edge_count + edge;
      const std::int64_t term_entry =
          term * edge_count + edge;
      const Scalar radial = radial_values[group_entry];
      const Scalar angular = angular_values[term_entry];
      const Scalar prefactor = prefactors[group_entry];
      const Scalar adjoint =
          edge_adjoint[edge * channel_count + channel];
      const Scalar unweighted_value =
          prefactor * radial * angular;
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        const Scalar unweighted_derivative =
            prefactor * (
                radial_derivatives[group_entry] *
                    radial_directions[edge * 3 + axis] *
                    angular +
                radial *
                    angular_derivatives[term_entry * 3 + axis]);
        edge_position_adjoint[edge * 3 + axis] +=
            adjoint * (
                unweighted_derivative * edge_weights[edge] +
                unweighted_value *
                    edge_weight_derivatives[edge * 3 + axis]);
      }
      const Scalar charge_common =
          radial * angular * edge_weights[edge];
      edge_charge_adjoint_center[edge] +=
          adjoint *
          prefactor_derivatives_center[group_entry] *
          charge_common;
      edge_charge_adjoint_neighbor[edge] +=
          adjoint *
          prefactor_derivatives_neighbor[group_entry] *
          charge_common;
    }
  }
}

template <typename Scalar>
void density_accumulate_forward(
    const Scalar* edge_values,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_values) {
  std::fill(
      atomic_values,
      atomic_values + atom_count * channel_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* edge_row = edge_values + edge * channel_count;
    Scalar* atom_row = atomic_values + centers[edge] * channel_count;
    for (std::int64_t channel = 0; channel < channel_count; ++channel) {
      atom_row[channel] += edge_row[channel];
    }
  }
}

template <typename Scalar>
void density_accumulate_adjoint(
    const Scalar* atomic_adjoint,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t channel_count,
    Scalar* edge_adjoint) {
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* atom_row =
        atomic_adjoint + centers[edge] * channel_count;
    Scalar* edge_row = edge_adjoint + edge * channel_count;
    std::copy(atom_row, atom_row + channel_count, edge_row);
  }
}

template <typename Scalar>
void edge_outer_accumulate_forward(
    const Scalar* left,
    const Scalar* right,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t left_dimension,
    std::int64_t right_dimension,
    std::int64_t atom_count,
    Scalar* atomic_values) {
  const std::int64_t atomic_width =
      left_dimension * right_dimension;
  std::fill(
      atomic_values,
      atomic_values + atom_count * atomic_width,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* left_row = left + edge * left_dimension;
    const Scalar* right_row = right + edge * right_dimension;
    Scalar* atom_row =
        atomic_values + centers[edge] * atomic_width;
    for (std::int64_t left_index = 0;
         left_index < left_dimension;
         ++left_index) {
      Scalar* output_row =
          atom_row + left_index * right_dimension;
      for (std::int64_t right_index = 0;
           right_index < right_dimension;
           ++right_index) {
        output_row[right_index] +=
            left_row[left_index] * right_row[right_index];
      }
    }
  }
}

template <typename Scalar>
void edge_outer_accumulate_adjoint(
    const Scalar* atomic_adjoint,
    const Scalar* left,
    const Scalar* right,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t left_dimension,
    std::int64_t right_dimension,
    Scalar* left_adjoint,
    Scalar* right_adjoint) {
  const std::int64_t atomic_width =
      left_dimension * right_dimension;
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* atom_row =
        atomic_adjoint + centers[edge] * atomic_width;
    const Scalar* left_row = left + edge * left_dimension;
    const Scalar* right_row = right + edge * right_dimension;
    Scalar* left_adjoint_row =
        left_adjoint + edge * left_dimension;
    Scalar* right_adjoint_row =
        right_adjoint + edge * right_dimension;
    std::fill(
        left_adjoint_row,
        left_adjoint_row + left_dimension,
        Scalar(0));
    std::fill(
        right_adjoint_row,
        right_adjoint_row + right_dimension,
        Scalar(0));
    for (std::int64_t left_index = 0;
         left_index < left_dimension;
         ++left_index) {
      const Scalar* adjoint_row =
          atom_row + left_index * right_dimension;
      for (std::int64_t right_index = 0;
           right_index < right_dimension;
           ++right_index) {
        const Scalar adjoint = adjoint_row[right_index];
        left_adjoint_row[left_index] +=
            adjoint * right_row[right_index];
        right_adjoint_row[right_index] +=
            adjoint * left_row[left_index];
      }
    }
  }
}

template <typename Scalar>
void edge_outer_accumulate_double_backward(
    const Scalar* atomic_adjoint,
    const Scalar* left,
    const Scalar* right,
    const Scalar* left_adjoint_tangent,
    const Scalar* right_adjoint_tangent,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t left_dimension,
    std::int64_t right_dimension,
    std::int64_t atom_count,
    Scalar* atomic_adjoint_tangent,
    Scalar* left_second_adjoint,
    Scalar* right_second_adjoint) {
  const std::int64_t atomic_width =
      left_dimension * right_dimension;
  std::fill(
      atomic_adjoint_tangent,
      atomic_adjoint_tangent + atom_count * atomic_width,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* atom_row =
        atomic_adjoint + centers[edge] * atomic_width;
    const Scalar* left_row = left + edge * left_dimension;
    const Scalar* right_row = right + edge * right_dimension;
    const Scalar* left_tangent_row =
        left_adjoint_tangent + edge * left_dimension;
    const Scalar* right_tangent_row =
        right_adjoint_tangent + edge * right_dimension;
    Scalar* atomic_tangent_row =
        atomic_adjoint_tangent + centers[edge] * atomic_width;
    Scalar* left_second_row =
        left_second_adjoint + edge * left_dimension;
    Scalar* right_second_row =
        right_second_adjoint + edge * right_dimension;
    std::fill(
        left_second_row,
        left_second_row + left_dimension,
        Scalar(0));
    std::fill(
        right_second_row,
        right_second_row + right_dimension,
        Scalar(0));
    for (std::int64_t left_index = 0;
         left_index < left_dimension;
         ++left_index) {
      const Scalar* adjoint_row =
          atom_row + left_index * right_dimension;
      Scalar* atomic_tangent_output =
          atomic_tangent_row + left_index * right_dimension;
      for (std::int64_t right_index = 0;
           right_index < right_dimension;
           ++right_index) {
        atomic_tangent_output[right_index] +=
            left_tangent_row[left_index] *
                right_row[right_index] +
            left_row[left_index] *
                right_tangent_row[right_index];
        left_second_row[left_index] +=
            adjoint_row[right_index] *
                right_tangent_row[right_index];
        right_second_row[right_index] +=
            adjoint_row[right_index] *
                left_tangent_row[left_index];
      }
    }
  }
}

template <typename Scalar>
void softmax_gaussian_role_density_forward(
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* edge_values,
    const std::int64_t* atom_centers,
    std::int64_t edge_count,
    std::int64_t role_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_values) {
  std::fill(
      atomic_values,
      atomic_values + atom_count * role_count * channel_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar scaled = distances[edge] / cutoffs[edge];
    Scalar maximum = -std::numeric_limits<Scalar>::infinity();
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      maximum = std::max(maximum, Scalar(-0.5) * delta * delta);
    }
    Scalar denominator = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      denominator += std::exp(Scalar(-0.5) * delta * delta - maximum);
    }
    const Scalar* edge_row = edge_values + edge * channel_count;
    Scalar* atom_row = atomic_values +
        atom_centers[edge] * role_count * channel_count;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      Scalar* output_row = atom_row + role * channel_count;
      for (std::int64_t channel = 0; channel < channel_count; ++channel) {
        output_row[channel] += weight * edge_row[channel];
      }
    }
  }
}

template <typename Scalar>
void softmax_gaussian_role_density_adjoint(
    const Scalar* atomic_adjoint,
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* edge_values,
    const std::int64_t* atom_centers,
    std::int64_t edge_count,
    std::int64_t role_count,
    std::int64_t channel_count,
    Scalar* distance_adjoint,
    Scalar* edge_adjoint) {
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar cutoff = cutoffs[edge];
    const Scalar scaled = distances[edge] / cutoff;
    Scalar maximum = -std::numeric_limits<Scalar>::infinity();
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      maximum = std::max(maximum, Scalar(-0.5) * delta * delta);
    }
    Scalar denominator = Scalar(0);
    Scalar mean_log_derivative = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      denominator += std::exp(Scalar(-0.5) * delta * delta - maximum);
    }
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      mean_log_derivative += weight * log_derivative;
    }
    const Scalar* atom_row = atomic_adjoint +
        atom_centers[edge] * role_count * channel_count;
    const Scalar* edge_row = edge_values + edge * channel_count;
    Scalar* edge_output = edge_adjoint + edge * channel_count;
    std::fill(edge_output, edge_output + channel_count, Scalar(0));
    Scalar distance_output = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      const Scalar derivative =
          weight * (log_derivative - mean_log_derivative);
      const Scalar* adjoint_row = atom_row + role * channel_count;
      Scalar role_adjoint = Scalar(0);
      for (std::int64_t channel = 0; channel < channel_count; ++channel) {
        edge_output[channel] += weight * adjoint_row[channel];
        role_adjoint += adjoint_row[channel] * edge_row[channel];
      }
      distance_output += derivative * role_adjoint;
    }
    distance_adjoint[edge] = distance_output;
  }
}

template <typename Scalar>
void softmax_gaussian_role_density_double_backward(
    const Scalar* atomic_adjoint,
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* edge_values,
    const Scalar* distance_adjoint_tangent,
    const Scalar* edge_adjoint_tangent,
    const std::int64_t* atom_centers,
    std::int64_t edge_count,
    std::int64_t role_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_adjoint_tangent,
    Scalar* distance_second_adjoint,
    Scalar* edge_second_adjoint) {
  std::fill(
      atomic_adjoint_tangent,
      atomic_adjoint_tangent + atom_count * role_count * channel_count,
      Scalar(0));
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar cutoff = cutoffs[edge];
    const Scalar scaled = distances[edge] / cutoff;
    Scalar maximum = -std::numeric_limits<Scalar>::infinity();
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      maximum = std::max(maximum, Scalar(-0.5) * delta * delta);
    }
    Scalar denominator = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      denominator += std::exp(Scalar(-0.5) * delta * delta - maximum);
    }
    Scalar mean_log_derivative = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      mean_log_derivative += weight * log_derivative;
    }
    Scalar derivative_log_moment = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      derivative_log_moment +=
          weight * (log_derivative - mean_log_derivative) *
          log_derivative;
    }
    const Scalar* atom_row = atomic_adjoint +
        atom_centers[edge] * role_count * channel_count;
    Scalar* atom_tangent_row = atomic_adjoint_tangent +
        atom_centers[edge] * role_count * channel_count;
    const Scalar* edge_row = edge_values + edge * channel_count;
    const Scalar* edge_tangent =
        edge_adjoint_tangent + edge * channel_count;
    Scalar* edge_second = edge_second_adjoint + edge * channel_count;
    std::fill(edge_second, edge_second + channel_count, Scalar(0));
    const Scalar distance_tangent = distance_adjoint_tangent[edge];
    Scalar distance_second = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      const Scalar derivative =
          weight * (log_derivative - mean_log_derivative);
      const Scalar second_derivative = weight *
          ((log_derivative - mean_log_derivative) *
               (log_derivative - mean_log_derivative) -
           derivative_log_moment);
      const Scalar* adjoint_row = atom_row + role * channel_count;
      Scalar* tangent_row = atom_tangent_row + role * channel_count;
      Scalar role_adjoint = Scalar(0);
      Scalar role_edge_tangent = Scalar(0);
      for (std::int64_t channel = 0; channel < channel_count; ++channel) {
        tangent_row[channel] +=
            distance_tangent * derivative * edge_row[channel] +
            weight * edge_tangent[channel];
        edge_second[channel] +=
            distance_tangent * derivative * adjoint_row[channel];
        role_adjoint += adjoint_row[channel] * edge_row[channel];
        role_edge_tangent += adjoint_row[channel] * edge_tangent[channel];
      }
      distance_second +=
          distance_tangent * second_derivative * role_adjoint +
          derivative * role_edge_tangent;
    }
    distance_second_adjoint[edge] = distance_second;
  }
}

template <typename Scalar>
void scheduled_softmax_gaussian_role_density_forward(
    const Scalar* radial_values,
    const Scalar* angular_values,
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* soft_weights,
    const std::int64_t* edge_types,
    const std::int64_t* channel_radial_indices,
    const std::int64_t* channel_angular_indices,
    const std::int64_t* channel_types,
    const Scalar* channel_scales,
    const std::int64_t* atom_centers,
    std::int64_t edge_count,
    std::int64_t radial_width,
    std::int64_t angular_width,
    std::int64_t role_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_values) {
  const std::int64_t output_width = channel_count + 1;
  std::fill(
      atomic_values,
      atomic_values + atom_count * role_count * output_width,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar cutoff = cutoffs[edge];
    const Scalar scaled = distances[edge] / cutoff;
    Scalar maximum = -std::numeric_limits<Scalar>::infinity();
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      maximum = std::max(maximum, Scalar(-0.5) * delta * delta);
    }
    Scalar denominator = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      denominator +=
          std::exp(Scalar(-0.5) * delta * delta - maximum);
    }
    Scalar* atom_row = atomic_values +
        atom_centers[edge] * role_count * output_width;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      Scalar* output_row = atom_row + role * output_width;
      for (std::int64_t channel = 0;
           channel < channel_count;
           ++channel) {
        if (channel_types[channel] >= 0 &&
            channel_types[channel] != edge_types[edge]) {
          continue;
        }
        const std::int64_t radial_index =
            channel_radial_indices[channel];
        const std::int64_t angular_index =
            channel_angular_indices[channel];
        output_row[channel] +=
            weight * channel_scales[channel] *
            radial_values[edge * radial_width + radial_index] *
            angular_values[edge * angular_width + angular_index];
      }
      output_row[channel_count] += weight * soft_weights[edge];
    }
  }
}

template <typename Scalar>
void scheduled_softmax_gaussian_role_density_adjoint(
    const Scalar* atomic_adjoint,
    const Scalar* radial_values,
    const Scalar* angular_values,
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* soft_weights,
    const std::int64_t* edge_types,
    const std::int64_t* channel_radial_indices,
    const std::int64_t* channel_angular_indices,
    const std::int64_t* channel_types,
    const Scalar* channel_scales,
    const std::int64_t* atom_centers,
    std::int64_t edge_count,
    std::int64_t radial_width,
    std::int64_t angular_width,
    std::int64_t role_count,
    std::int64_t channel_count,
    Scalar* radial_adjoint,
    Scalar* angular_adjoint,
    Scalar* distance_adjoint,
    Scalar* soft_weight_adjoint) {
  std::fill(
      radial_adjoint,
      radial_adjoint + edge_count * radial_width,
      Scalar(0));
  std::fill(
      angular_adjoint,
      angular_adjoint + edge_count * angular_width,
      Scalar(0));
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  const std::int64_t output_width = channel_count + 1;
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar cutoff = cutoffs[edge];
    const Scalar scaled = distances[edge] / cutoff;
    Scalar maximum = -std::numeric_limits<Scalar>::infinity();
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      maximum = std::max(maximum, Scalar(-0.5) * delta * delta);
    }
    Scalar denominator = Scalar(0);
    Scalar mean_log_derivative = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      denominator +=
          std::exp(Scalar(-0.5) * delta * delta - maximum);
    }
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      mean_log_derivative += weight * log_derivative;
    }
    const Scalar* atom_row = atomic_adjoint +
        atom_centers[edge] * role_count * output_width;
    Scalar distance_output = Scalar(0);
    Scalar soft_output = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      const Scalar derivative =
          weight * (log_derivative - mean_log_derivative);
      const Scalar* adjoint_row = atom_row + role * output_width;
      Scalar role_adjoint =
          adjoint_row[channel_count] * soft_weights[edge];
      soft_output += weight * adjoint_row[channel_count];
      for (std::int64_t channel = 0;
           channel < channel_count;
           ++channel) {
        if (channel_types[channel] >= 0 &&
            channel_types[channel] != edge_types[edge]) {
          continue;
        }
        const std::int64_t radial_index =
            channel_radial_indices[channel];
        const std::int64_t angular_index =
            channel_angular_indices[channel];
        const Scalar radial =
            radial_values[edge * radial_width + radial_index];
        const Scalar angular =
            angular_values[edge * angular_width + angular_index];
        const Scalar scale = channel_scales[channel];
        const Scalar adjoint = adjoint_row[channel];
        radial_adjoint[edge * radial_width + radial_index] +=
            weight * adjoint * scale * angular;
        angular_adjoint[edge * angular_width + angular_index] +=
            weight * adjoint * scale * radial;
        role_adjoint += adjoint * scale * radial * angular;
      }
      distance_output += derivative * role_adjoint;
    }
    distance_adjoint[edge] = distance_output;
    soft_weight_adjoint[edge] = soft_output;
  }
}

template <typename Scalar>
void scheduled_softmax_gaussian_role_density_double_backward(
    const Scalar* atomic_adjoint,
    const Scalar* radial_values,
    const Scalar* angular_values,
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* soft_weights,
    const Scalar* radial_adjoint_tangent,
    const Scalar* angular_adjoint_tangent,
    const Scalar* distance_adjoint_tangent,
    const Scalar* soft_weight_adjoint_tangent,
    const std::int64_t* edge_types,
    const std::int64_t* channel_radial_indices,
    const std::int64_t* channel_angular_indices,
    const std::int64_t* channel_types,
    const Scalar* channel_scales,
    const std::int64_t* atom_centers,
    std::int64_t edge_count,
    std::int64_t radial_width,
    std::int64_t angular_width,
    std::int64_t role_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_adjoint_tangent,
    Scalar* radial_second_adjoint,
    Scalar* angular_second_adjoint,
    Scalar* distance_second_adjoint,
    Scalar* soft_weight_second_adjoint) {
  const std::int64_t output_width = channel_count + 1;
  std::fill(
      atomic_adjoint_tangent,
      atomic_adjoint_tangent + atom_count * role_count * output_width,
      Scalar(0));
  std::fill(
      radial_second_adjoint,
      radial_second_adjoint + edge_count * radial_width,
      Scalar(0));
  std::fill(
      angular_second_adjoint,
      angular_second_adjoint + edge_count * angular_width,
      Scalar(0));
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar cutoff = cutoffs[edge];
    const Scalar scaled = distances[edge] / cutoff;
    Scalar maximum = -std::numeric_limits<Scalar>::infinity();
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      maximum = std::max(maximum, Scalar(-0.5) * delta * delta);
    }
    Scalar denominator = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      denominator +=
          std::exp(Scalar(-0.5) * delta * delta - maximum);
    }
    Scalar mean_log_derivative = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      mean_log_derivative += weight * log_derivative;
    }
    Scalar derivative_log_moment = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      derivative_log_moment +=
          weight * (log_derivative - mean_log_derivative) *
          log_derivative;
    }
    const Scalar* atom_row = atomic_adjoint +
        atom_centers[edge] * role_count * output_width;
    Scalar* atom_tangent_row = atomic_adjoint_tangent +
        atom_centers[edge] * role_count * output_width;
    const Scalar distance_tangent = distance_adjoint_tangent[edge];
    Scalar distance_second = Scalar(0);
    Scalar soft_second = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          std::exp(Scalar(-0.5) * delta * delta - maximum) /
          denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoff;
      const Scalar derivative =
          weight * (log_derivative - mean_log_derivative);
      const Scalar second_derivative = weight *
          ((log_derivative - mean_log_derivative) *
               (log_derivative - mean_log_derivative) -
           derivative_log_moment);
      const Scalar* adjoint_row = atom_row + role * output_width;
      Scalar* tangent_row = atom_tangent_row + role * output_width;
      const Scalar soft_tangent = soft_weight_adjoint_tangent[edge];
      tangent_row[channel_count] +=
          weight * soft_tangent +
          derivative * distance_tangent * soft_weights[edge];
      Scalar role_primal =
          adjoint_row[channel_count] * soft_weights[edge];
      Scalar role_tangent =
          adjoint_row[channel_count] * soft_tangent;
      soft_second +=
          derivative * distance_tangent *
          adjoint_row[channel_count];
      for (std::int64_t channel = 0;
           channel < channel_count;
           ++channel) {
        if (channel_types[channel] >= 0 &&
            channel_types[channel] != edge_types[edge]) {
          continue;
        }
        const std::int64_t radial_index =
            channel_radial_indices[channel];
        const std::int64_t angular_index =
            channel_angular_indices[channel];
        const std::int64_t radial_offset =
            edge * radial_width + radial_index;
        const std::int64_t angular_offset =
            edge * angular_width + angular_index;
        const Scalar radial = radial_values[radial_offset];
        const Scalar angular = angular_values[angular_offset];
        const Scalar radial_tangent =
            radial_adjoint_tangent[radial_offset];
        const Scalar angular_tangent =
            angular_adjoint_tangent[angular_offset];
        const Scalar scale = channel_scales[channel];
        const Scalar value = scale * radial * angular;
        const Scalar value_tangent = scale * (
            radial_tangent * angular +
            radial * angular_tangent);
        tangent_row[channel] +=
            weight * value_tangent +
            derivative * distance_tangent * value;
        const Scalar adjoint = adjoint_row[channel];
        radial_second_adjoint[radial_offset] += adjoint * scale * (
            weight * angular_tangent +
            derivative * distance_tangent * angular);
        angular_second_adjoint[angular_offset] += adjoint * scale * (
            weight * radial_tangent +
            derivative * distance_tangent * radial);
        role_primal += adjoint * value;
        role_tangent += adjoint * value_tangent;
      }
      distance_second +=
          second_derivative * distance_tangent * role_primal +
          derivative * role_tangent;
    }
    distance_second_adjoint[edge] = distance_second;
    soft_weight_second_adjoint[edge] = soft_second;
  }
}

template <typename Scalar>
void carrier_gated_scatter_forward(
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t target_count,
    Scalar* target_values) {
  std::fill(
      target_values,
      target_values + target_count * feature_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* source_row =
        node_values + edge_sources[edge] * feature_count;
    const Scalar* gate_row = edge_gates + edge * channel_count;
    Scalar* target_row =
        target_values + edge_targets[edge] * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      target_row[feature] +=
          source_row[feature] *
          gate_row[feature_channels[feature]];
    }
  }
}

template <typename Scalar>
void carrier_gated_scatter_adjoint(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Scalar* gate_adjoint) {
  std::fill(
      node_adjoint,
      node_adjoint + node_count * feature_count,
      Scalar(0));
  std::fill(
      gate_adjoint,
      gate_adjoint + edge_count * channel_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* target_row =
        target_adjoint + edge_targets[edge] * feature_count;
    const Scalar* source_row =
        node_values + edge_sources[edge] * feature_count;
    const Scalar* gate_row = edge_gates + edge * channel_count;
    Scalar* node_adjoint_row =
        node_adjoint + edge_sources[edge] * feature_count;
    Scalar* gate_adjoint_row =
        gate_adjoint + edge * channel_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      const Scalar gradient = target_row[feature];
      node_adjoint_row[feature] +=
          gradient * conjugate(gate_row[channel]);
      gate_adjoint_row[channel] +=
          gradient * conjugate(source_row[feature]);
    }
  }
}

template <typename Scalar>
void carrier_gated_scatter_double_backward(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Scalar* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t target_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent,
    Scalar* node_second_adjoint,
    Scalar* gate_second_adjoint) {
  std::fill(
      target_adjoint_tangent,
      target_adjoint_tangent + target_count * feature_count,
      Scalar(0));
  std::fill(
      node_second_adjoint,
      node_second_adjoint + node_count * feature_count,
      Scalar(0));
  std::fill(
      gate_second_adjoint,
      gate_second_adjoint + edge_count * channel_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const Scalar* target_row =
        target_adjoint + target * feature_count;
    const Scalar* source_row =
        node_values + source * feature_count;
    const Scalar* gate_row = edge_gates + edge * channel_count;
    const Scalar* node_tangent_row =
        node_adjoint_tangent + source * feature_count;
    const Scalar* gate_tangent_row =
        gate_adjoint_tangent + edge * channel_count;
    Scalar* target_tangent_row =
        target_adjoint_tangent + target * feature_count;
    Scalar* node_second_row =
        node_second_adjoint + source * feature_count;
    Scalar* gate_second_row =
        gate_second_adjoint + edge * channel_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      target_tangent_row[feature] +=
          node_tangent_row[feature] * gate_row[channel] +
          gate_tangent_row[channel] * source_row[feature];
      node_second_row[feature] +=
          conjugate(gate_tangent_row[channel]) *
          target_row[feature];
      gate_second_row[channel] +=
          conjugate(node_tangent_row[feature]) *
          target_row[feature];
    }
  }
}

template <typename Scalar>
void carrier_residual_gated_scatter_forward(
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  carrier_gated_scatter_forward(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      edge_count,
      feature_count,
      channel_count,
      node_count,
      target_values);
  for (std::int64_t index = 0;
       index < node_count * feature_count;
       ++index) {
    target_values[index] += node_values[index];
  }
}

template <typename Scalar>
void carrier_residual_gated_scatter_adjoint(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Scalar* gate_adjoint) {
  carrier_gated_scatter_adjoint(
      target_adjoint,
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      edge_count,
      node_count,
      feature_count,
      channel_count,
      node_adjoint,
      gate_adjoint);
  for (std::int64_t index = 0;
       index < node_count * feature_count;
       ++index) {
    node_adjoint[index] += target_adjoint[index];
  }
}

template <typename Scalar>
void carrier_residual_gated_scatter_double_backward(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Scalar* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent,
    Scalar* node_second_adjoint,
    Scalar* gate_second_adjoint) {
  carrier_gated_scatter_double_backward(
      target_adjoint,
      node_values,
      edge_gates,
      node_adjoint_tangent,
      gate_adjoint_tangent,
      edge_sources,
      edge_targets,
      feature_channels,
      edge_count,
      node_count,
      node_count,
      feature_count,
      channel_count,
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
  for (std::int64_t index = 0;
       index < node_count * feature_count;
       ++index) {
    target_adjoint_tangent[index] += node_adjoint_tangent[index];
  }
}

template <typename Scalar>
void carrier_segmented_residual_gated_scatter_forward(
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* target_offsets,
    const std::int64_t* feature_channels,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  for (std::int64_t target = 0; target < node_count; ++target) {
    const std::int64_t edge_start = target_offsets[target];
    const std::int64_t edge_stop = target_offsets[target + 1];
    const Scalar* residual_row =
        node_values + target * feature_count;
    Scalar* target_row = target_values + target * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = residual_row[feature];
      for (std::int64_t edge = edge_start;
           edge < edge_stop;
           ++edge) {
        value +=
            node_values[
                edge_sources[edge] * feature_count + feature] *
            edge_gates[edge * channel_count + channel];
      }
      target_row[feature] = value;
    }
  }
}

template <typename Scalar>
void carrier_segmented_residual_gated_scatter_adjoint(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* source_offsets,
    const std::int64_t* source_edges,
    const std::int64_t* feature_channels,
    const std::int64_t* channel_offsets,
    const std::int64_t* channel_features,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Scalar* gate_adjoint) {
  for (std::int64_t source = 0; source < node_count; ++source) {
    const std::int64_t edge_start = source_offsets[source];
    const std::int64_t edge_stop = source_offsets[source + 1];
    const Scalar* residual_row =
        target_adjoint + source * feature_count;
    Scalar* node_row = node_adjoint + source * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = residual_row[feature];
      for (std::int64_t source_entry = edge_start;
           source_entry < edge_stop;
           ++source_entry) {
        const std::int64_t edge = source_edges[source_entry];
        const std::int64_t target = edge_targets[edge];
        value +=
            target_adjoint[target * feature_count + feature] *
            conjugate(edge_gates[edge * channel_count + channel]);
      }
      node_row[feature] = value;
    }
  }
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    Scalar* gate_row = gate_adjoint + edge * channel_count;
    for (std::int64_t channel = 0;
         channel < channel_count;
         ++channel) {
      Scalar value = Scalar(0);
      for (std::int64_t channel_entry = channel_offsets[channel];
           channel_entry < channel_offsets[channel + 1];
           ++channel_entry) {
        const std::int64_t feature =
            channel_features[channel_entry];
        value +=
            target_adjoint[target * feature_count + feature] *
            conjugate(
                node_values[source * feature_count + feature]);
      }
      gate_row[channel] = value;
    }
  }
}

template <typename Scalar>
void carrier_segmented_residual_gated_scatter_double_backward(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Scalar* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* target_offsets,
    const std::int64_t* source_offsets,
    const std::int64_t* source_edges,
    const std::int64_t* feature_channels,
    const std::int64_t* channel_offsets,
    const std::int64_t* channel_features,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent,
    Scalar* node_second_adjoint,
    Scalar* gate_second_adjoint) {
  for (std::int64_t target = 0; target < node_count; ++target) {
    const std::int64_t edge_start = target_offsets[target];
    const std::int64_t edge_stop = target_offsets[target + 1];
    const Scalar* residual_row =
        node_adjoint_tangent + target * feature_count;
    Scalar* target_row =
        target_adjoint_tangent + target * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = residual_row[feature];
      for (std::int64_t edge = edge_start;
           edge < edge_stop;
           ++edge) {
        const std::int64_t source = edge_sources[edge];
        value +=
            node_adjoint_tangent[
                source * feature_count + feature] *
                edge_gates[edge * channel_count + channel] +
            gate_adjoint_tangent[
                edge * channel_count + channel] *
                node_values[source * feature_count + feature];
      }
      target_row[feature] = value;
    }
  }
  for (std::int64_t source = 0; source < node_count; ++source) {
    const std::int64_t edge_start = source_offsets[source];
    const std::int64_t edge_stop = source_offsets[source + 1];
    Scalar* node_row =
        node_second_adjoint + source * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = Scalar(0);
      for (std::int64_t source_entry = edge_start;
           source_entry < edge_stop;
           ++source_entry) {
        const std::int64_t edge = source_edges[source_entry];
        const std::int64_t target = edge_targets[edge];
        value +=
            conjugate(
                gate_adjoint_tangent[
                    edge * channel_count + channel]) *
            target_adjoint[target * feature_count + feature];
      }
      node_row[feature] = value;
    }
  }
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    Scalar* gate_row =
        gate_second_adjoint + edge * channel_count;
    for (std::int64_t channel = 0;
         channel < channel_count;
         ++channel) {
      Scalar value = Scalar(0);
      for (std::int64_t channel_entry = channel_offsets[channel];
           channel_entry < channel_offsets[channel + 1];
           ++channel_entry) {
        const std::int64_t feature =
            channel_features[channel_entry];
        value +=
            conjugate(
                node_adjoint_tangent[
                    source * feature_count + feature]) *
            target_adjoint[target * feature_count + feature];
      }
      gate_row[channel] = value;
    }
  }
}

template <typename Scalar, typename Real>
void carrier_gated_scatter_real_gates_forward(
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t target_count,
    Scalar* target_values) {
  std::fill(
      target_values,
      target_values + target_count * feature_count,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* source_row =
        node_values + edge_sources[edge] * feature_count;
    const Real* gate_row = edge_gates + edge * channel_count;
    Scalar* target_row =
        target_values + edge_targets[edge] * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      target_row[feature] +=
          source_row[feature] *
          gate_row[feature_channels[feature]];
    }
  }
}

template <typename Scalar, typename Real>
void carrier_gated_scatter_real_gates_adjoint(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Real* gate_adjoint) {
  std::fill(
      node_adjoint,
      node_adjoint + node_count * feature_count,
      Scalar(0));
  std::fill(
      gate_adjoint,
      gate_adjoint + edge_count * channel_count,
      Real(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Scalar* target_row =
        target_adjoint + edge_targets[edge] * feature_count;
    const Scalar* source_row =
        node_values + edge_sources[edge] * feature_count;
    const Real* gate_row = edge_gates + edge * channel_count;
    Scalar* node_adjoint_row =
        node_adjoint + edge_sources[edge] * feature_count;
    Real* gate_adjoint_row =
        gate_adjoint + edge * channel_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      const Scalar gradient = target_row[feature];
      node_adjoint_row[feature] += gradient * gate_row[channel];
      gate_adjoint_row[channel] += real_component(
          gradient * conjugate(source_row[feature]));
    }
  }
}

template <typename Scalar, typename Real>
void carrier_gated_scatter_real_gates_double_backward(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Real* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t target_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent,
    Scalar* node_second_adjoint,
    Real* gate_second_adjoint) {
  std::fill(
      target_adjoint_tangent,
      target_adjoint_tangent + target_count * feature_count,
      Scalar(0));
  std::fill(
      node_second_adjoint,
      node_second_adjoint + node_count * feature_count,
      Scalar(0));
  std::fill(
      gate_second_adjoint,
      gate_second_adjoint + edge_count * channel_count,
      Real(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const Scalar* target_row =
        target_adjoint + target * feature_count;
    const Scalar* source_row =
        node_values + source * feature_count;
    const Real* gate_row = edge_gates + edge * channel_count;
    const Scalar* node_tangent_row =
        node_adjoint_tangent + source * feature_count;
    const Real* gate_tangent_row =
        gate_adjoint_tangent + edge * channel_count;
    Scalar* target_tangent_row =
        target_adjoint_tangent + target * feature_count;
    Scalar* node_second_row =
        node_second_adjoint + source * feature_count;
    Real* gate_second_row =
        gate_second_adjoint + edge * channel_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      target_tangent_row[feature] +=
          node_tangent_row[feature] * gate_row[channel] +
          gate_tangent_row[channel] * source_row[feature];
      node_second_row[feature] +=
          gate_tangent_row[channel] * target_row[feature];
      gate_second_row[channel] += real_component(
          conjugate(node_tangent_row[feature]) *
          target_row[feature]);
    }
  }
}

template <typename Scalar, typename Real>
void carrier_residual_gated_scatter_real_gates_forward(
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  carrier_gated_scatter_real_gates_forward(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      edge_count,
      feature_count,
      channel_count,
      node_count,
      target_values);
  for (std::int64_t index = 0;
       index < node_count * feature_count;
       ++index) {
    target_values[index] += node_values[index];
  }
}

template <typename Scalar, typename Real>
void carrier_residual_gated_scatter_real_gates_adjoint(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Real* gate_adjoint) {
  carrier_gated_scatter_real_gates_adjoint(
      target_adjoint,
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      edge_count,
      node_count,
      feature_count,
      channel_count,
      node_adjoint,
      gate_adjoint);
  for (std::int64_t index = 0;
       index < node_count * feature_count;
       ++index) {
    node_adjoint[index] += target_adjoint[index];
  }
}

template <typename Scalar, typename Real>
void carrier_residual_gated_scatter_real_gates_double_backward(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Real* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent,
    Scalar* node_second_adjoint,
    Real* gate_second_adjoint) {
  carrier_gated_scatter_real_gates_double_backward(
      target_adjoint,
      node_values,
      edge_gates,
      node_adjoint_tangent,
      gate_adjoint_tangent,
      edge_sources,
      edge_targets,
      feature_channels,
      edge_count,
      node_count,
      node_count,
      feature_count,
      channel_count,
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
  for (std::int64_t index = 0;
       index < node_count * feature_count;
       ++index) {
    target_adjoint_tangent[index] += node_adjoint_tangent[index];
  }
}

template <typename Scalar, typename Real>
void carrier_segmented_residual_gated_scatter_real_gates_forward(
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* target_offsets,
    const std::int64_t* feature_channels,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  for (std::int64_t target = 0; target < node_count; ++target) {
    const std::int64_t edge_start = target_offsets[target];
    const std::int64_t edge_stop = target_offsets[target + 1];
    const Scalar* residual_row =
        node_values + target * feature_count;
    Scalar* target_row = target_values + target * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = residual_row[feature];
      for (std::int64_t edge = edge_start;
           edge < edge_stop;
           ++edge) {
        value +=
            node_values[
                edge_sources[edge] * feature_count + feature] *
            edge_gates[edge * channel_count + channel];
      }
      target_row[feature] = value;
    }
  }
}

template <typename Scalar, typename Real>
void carrier_segmented_residual_gated_scatter_real_gates_adjoint(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* source_offsets,
    const std::int64_t* source_edges,
    const std::int64_t* feature_channels,
    const std::int64_t* channel_offsets,
    const std::int64_t* channel_features,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Real* gate_adjoint) {
  for (std::int64_t source = 0; source < node_count; ++source) {
    const std::int64_t edge_start = source_offsets[source];
    const std::int64_t edge_stop = source_offsets[source + 1];
    const Scalar* residual_row =
        target_adjoint + source * feature_count;
    Scalar* node_row = node_adjoint + source * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = residual_row[feature];
      for (std::int64_t source_entry = edge_start;
           source_entry < edge_stop;
           ++source_entry) {
        const std::int64_t edge = source_edges[source_entry];
        const std::int64_t target = edge_targets[edge];
        value +=
            target_adjoint[target * feature_count + feature] *
            edge_gates[edge * channel_count + channel];
      }
      node_row[feature] = value;
    }
  }
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    Real* gate_row = gate_adjoint + edge * channel_count;
    for (std::int64_t channel = 0;
         channel < channel_count;
         ++channel) {
      Real value = Real(0);
      for (std::int64_t channel_entry = channel_offsets[channel];
           channel_entry < channel_offsets[channel + 1];
           ++channel_entry) {
        const std::int64_t feature =
            channel_features[channel_entry];
        value += real_component(
            target_adjoint[target * feature_count + feature] *
            conjugate(
                node_values[source * feature_count + feature]));
      }
      gate_row[channel] = value;
    }
  }
}

template <typename Scalar, typename Real>
void carrier_segmented_residual_gated_scatter_real_gates_double_backward(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Real* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* target_offsets,
    const std::int64_t* source_offsets,
    const std::int64_t* source_edges,
    const std::int64_t* feature_channels,
    const std::int64_t* channel_offsets,
    const std::int64_t* channel_features,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent,
    Scalar* node_second_adjoint,
    Real* gate_second_adjoint) {
  for (std::int64_t target = 0; target < node_count; ++target) {
    const std::int64_t edge_start = target_offsets[target];
    const std::int64_t edge_stop = target_offsets[target + 1];
    const Scalar* residual_row =
        node_adjoint_tangent + target * feature_count;
    Scalar* target_row =
        target_adjoint_tangent + target * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = residual_row[feature];
      for (std::int64_t edge = edge_start;
           edge < edge_stop;
           ++edge) {
        const std::int64_t source = edge_sources[edge];
        value +=
            node_adjoint_tangent[
                source * feature_count + feature] *
                edge_gates[edge * channel_count + channel] +
            gate_adjoint_tangent[
                edge * channel_count + channel] *
                node_values[source * feature_count + feature];
      }
      target_row[feature] = value;
    }
  }
  for (std::int64_t source = 0; source < node_count; ++source) {
    const std::int64_t edge_start = source_offsets[source];
    const std::int64_t edge_stop = source_offsets[source + 1];
    Scalar* node_row =
        node_second_adjoint + source * feature_count;
    for (std::int64_t feature = 0;
         feature < feature_count;
         ++feature) {
      const std::int64_t channel = feature_channels[feature];
      Scalar value = Scalar(0);
      for (std::int64_t source_entry = edge_start;
           source_entry < edge_stop;
           ++source_entry) {
        const std::int64_t edge = source_edges[source_entry];
        const std::int64_t target = edge_targets[edge];
        value +=
            gate_adjoint_tangent[
                edge * channel_count + channel] *
            target_adjoint[target * feature_count + feature];
      }
      node_row[feature] = value;
    }
  }
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    Real* gate_row =
        gate_second_adjoint + edge * channel_count;
    for (std::int64_t channel = 0;
         channel < channel_count;
         ++channel) {
      Real value = Real(0);
      for (std::int64_t channel_entry = channel_offsets[channel];
           channel_entry < channel_offsets[channel + 1];
           ++channel_entry) {
        const std::int64_t feature =
            channel_features[channel_entry];
        value += real_component(
            conjugate(
                node_adjoint_tangent[
                    source * feature_count + feature]) *
            target_adjoint[target * feature_count + feature]);
      }
      gate_row[channel] = value;
    }
  }
}

template <typename Scalar>
void source_arena_gather_forward(
    const Scalar* producer,
    const std::int64_t* gather_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t output_width,
    Scalar* output) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const std::int64_t atom_type = atom_types[batch];
    for (std::int64_t output_index = 0;
         output_index < output_width;
         ++output_index) {
      const std::int64_t center_type = center_types[output_index];
      const std::int64_t output_offset =
          batch * output_width + output_index;
      if (center_type >= 0 && center_type != atom_type) {
        output[output_offset] = Scalar{};
      } else {
        output[output_offset] = producer[
            batch * producer_width + gather_indices[output_index]];
      }
    }
  }
}

template <typename Scalar>
void source_arena_gather_adjoint(
    const Scalar* output_adjoint,
    const std::int64_t* reverse_offsets,
    const std::int64_t* reverse_output_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t output_width,
    Scalar* producer_adjoint) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const std::int64_t atom_type = atom_types[batch];
    for (std::int64_t producer_index = 0;
         producer_index < producer_width;
         ++producer_index) {
      Scalar value{};
      for (std::int64_t entry = reverse_offsets[producer_index];
           entry < reverse_offsets[producer_index + 1];
           ++entry) {
        const std::int64_t output_index = reverse_output_indices[entry];
        const std::int64_t center_type = center_types[output_index];
        if (center_type < 0 || center_type == atom_type) {
          value += output_adjoint[
              batch * output_width + output_index];
        }
      }
      producer_adjoint[batch * producer_width + producer_index] = value;
    }
  }
}

template <typename Scalar, typename Control>
void source_arena_channel_transform_forward(
    const Scalar* producer,
    const Control* channel_maps,
    const std::int64_t* gather_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t output_width = output_feature_offsets[block_count];
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const std::int64_t atom_type = atom_types[batch];
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      const std::int64_t map_start = map_offsets[block];
      const bool identity = map_offsets[block + 1] == map_start;
      const std::int64_t center_type = center_types[input_start];
      const bool active = center_type < 0 || center_type == atom_type;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t inner = 0; inner < inner_width; ++inner) {
          Scalar value{};
          if (active) {
            if (identity) {
              const std::int64_t logical_input =
                  input_start + output_channel * inner_width + inner;
              value = producer[
                  batch * producer_width + gather_indices[logical_input]];
            } else {
              for (std::int64_t input_channel = 0;
                   input_channel < input_channels;
                   ++input_channel) {
                const std::int64_t logical_input =
                    input_start + input_channel * inner_width + inner;
                value += channel_maps[
                    map_start + output_channel * input_channels +
                    input_channel] * producer[
                    batch * producer_width + gather_indices[logical_input]];
              }
            }
          }
          output[
              batch * output_width + output_start +
              output_channel * inner_width + inner] = value;
        }
      }
    }
  }
}

template <typename Scalar, typename Control>
void source_arena_channel_transform_adjoint(
    const Scalar* output_adjoint,
    const Scalar* producer,
    const Control* channel_maps,
    const std::int64_t* gather_indices,
    const std::int64_t* reverse_offsets,
    const std::int64_t* reverse_output_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t block_count,
    Scalar* producer_adjoint,
    Control* channel_maps_adjoint) {
  const std::int64_t output_width = output_feature_offsets[block_count];
  const std::int64_t map_count = map_offsets[block_count];
  std::fill(
      producer_adjoint,
      producer_adjoint + batch_size * producer_width,
      Scalar{});
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_count,
      Control{});
  static_cast<void>(reverse_offsets);
  static_cast<void>(reverse_output_indices);
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const std::int64_t atom_type = atom_types[batch];
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      const std::int64_t map_start = map_offsets[block];
      const bool identity = map_offsets[block + 1] == map_start;
      const std::int64_t center_type = center_types[input_start];
      if (center_type >= 0 && center_type != atom_type) {
        continue;
      }
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t inner = 0; inner < inner_width; ++inner) {
          const Scalar gradient = output_adjoint[
              batch * output_width + output_start +
              output_channel * inner_width + inner];
          if (identity) {
            const std::int64_t logical_input =
                input_start + output_channel * inner_width + inner;
            producer_adjoint[
                batch * producer_width + gather_indices[logical_input]] +=
                gradient;
            continue;
          }
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            const std::int64_t logical_input =
                input_start + input_channel * inner_width + inner;
            const std::int64_t producer_index = gather_indices[logical_input];
            const std::int64_t map_index =
                map_start + output_channel * input_channels + input_channel;
            producer_adjoint[
                batch * producer_width + producer_index] +=
                gradient * conjugate(channel_maps[map_index]);
            channel_maps_adjoint[map_index] += project_control_gradient(
                gradient * conjugate(
                    producer[batch * producer_width + producer_index]));
          }
        }
      }
    }
  }
}

template <typename Scalar, typename Control>
void source_arena_channel_transform_double_backward(
    const Scalar* output_adjoint,
    const Scalar* producer,
    const Control* channel_maps,
    const Scalar* producer_adjoint_tangent,
    const Control* channel_maps_adjoint_tangent,
    const std::int64_t* gather_indices,
    const std::int64_t* reverse_offsets,
    const std::int64_t* reverse_output_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t block_count,
    Scalar* output_adjoint_tangent,
    Scalar* producer_second_adjoint,
    Control* channel_maps_second_adjoint) {
  const std::int64_t output_width = output_feature_offsets[block_count];
  const std::int64_t map_count = map_offsets[block_count];
  std::fill(
      output_adjoint_tangent,
      output_adjoint_tangent + batch_size * output_width,
      Scalar{});
  std::fill(
      producer_second_adjoint,
      producer_second_adjoint + batch_size * producer_width,
      Scalar{});
  std::fill(
      channel_maps_second_adjoint,
      channel_maps_second_adjoint + map_count,
      Control{});
  static_cast<void>(reverse_offsets);
  static_cast<void>(reverse_output_indices);
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const std::int64_t atom_type = atom_types[batch];
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      const std::int64_t map_start = map_offsets[block];
      const bool identity = map_offsets[block + 1] == map_start;
      const std::int64_t center_type = center_types[input_start];
      if (center_type >= 0 && center_type != atom_type) {
        continue;
      }
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t inner = 0; inner < inner_width; ++inner) {
          const std::int64_t output_index =
              batch * output_width + output_start +
              output_channel * inner_width + inner;
          const Scalar gradient = output_adjoint[output_index];
          if (identity) {
            const std::int64_t logical_input =
                input_start + output_channel * inner_width + inner;
            output_adjoint_tangent[output_index] = producer_adjoint_tangent[
                batch * producer_width + gather_indices[logical_input]];
            continue;
          }
          Scalar output_tangent{};
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            const std::int64_t logical_input =
                input_start + input_channel * inner_width + inner;
            const std::int64_t producer_index = gather_indices[logical_input];
            const std::int64_t map_index =
                map_start + output_channel * input_channels + input_channel;
            const Scalar input_value =
                producer[batch * producer_width + producer_index];
            const Scalar input_tangent = producer_adjoint_tangent[
                batch * producer_width + producer_index];
            output_tangent +=
                channel_maps[map_index] * input_tangent +
                channel_maps_adjoint_tangent[map_index] * input_value;
            producer_second_adjoint[
                batch * producer_width + producer_index] +=
                gradient * conjugate(
                    channel_maps_adjoint_tangent[map_index]);
            channel_maps_second_adjoint[map_index] +=
                project_control_gradient(
                    gradient * conjugate(input_tangent));
          }
          output_adjoint_tangent[output_index] = output_tangent;
        }
      }
    }
  }
}

template <typename Scalar>
void carrier_channel_update_forward(
    const Scalar* values,
    const Scalar* gates,
    const Scalar* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t feature_count = feature_offsets[block_count];
  const std::int64_t channel_count = channel_offsets[block_count];
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* value_row = values + batch * feature_count;
    const Scalar* gate_row = gates + batch * channel_count;
    Scalar* output_row = output + batch * feature_count;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t feature_start = feature_offsets[block];
      const std::int64_t channel_start = channel_offsets[block];
      const std::int64_t local_channels =
          channel_offsets[block + 1] - channel_start;
      const std::int64_t inner_width =
          (feature_offsets[block + 1] - feature_start) /
          local_channels;
      const std::int64_t map_start = map_offsets[block];
      for (std::int64_t output_channel = 0;
           output_channel < local_channels;
           ++output_channel) {
        const Scalar gate =
            gate_row[channel_start + output_channel];
        for (std::int64_t inner = 0;
             inner < inner_width;
             ++inner) {
          Scalar mixed = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < local_channels;
               ++input_channel) {
            mixed +=
                channel_maps[
                    map_start +
                    output_channel * local_channels +
                    input_channel] *
                value_row[
                    feature_start +
                    input_channel * inner_width +
                    inner];
          }
          const std::int64_t output_feature =
              feature_start +
              output_channel * inner_width +
              inner;
          output_row[output_feature] =
              value_row[output_feature] + gate * mixed;
        }
      }
    }
  }
}

template <typename Scalar>
void carrier_channel_transform_forward(
    const Scalar* values,
    const Scalar* channel_maps,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = values + batch * input_width;
    Scalar* output_row = output + batch * output_width;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t inner = 0; inner < inner_width; ++inner) {
          Scalar value = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            value += channel_maps[
                map_offsets[block] + output_channel * input_channels +
                input_channel] * input_row[
                input_start + input_channel * inner_width + inner];
          }
          output_row[
              output_start + output_channel * inner_width + inner] = value;
        }
      }
    }
  }
}

template <typename Scalar>
void carrier_channel_transform_adjoint(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Scalar* channel_maps,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* values_adjoint,
    Scalar* channel_maps_adjoint) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  const std::int64_t map_count = map_offsets[block_count];
  std::fill(
      values_adjoint,
      values_adjoint + batch_size * input_width,
      Scalar(0));
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_count,
      Scalar(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row = output_adjoint + batch * output_width;
    const Scalar* input_row = values + batch * input_width;
    Scalar* input_adjoint_row = values_adjoint + batch * input_width;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t input_channel = 0;
             input_channel < input_channels;
             ++input_channel) {
          const std::int64_t map_index =
              map_offsets[block] + output_channel * input_channels +
              input_channel;
          const Scalar map_value = channel_maps[map_index];
          for (std::int64_t inner = 0; inner < inner_width; ++inner) {
            const Scalar gradient = output_row[
                output_start + output_channel * inner_width + inner];
            const Scalar input_value = input_row[
                input_start + input_channel * inner_width + inner];
            input_adjoint_row[
                input_start + input_channel * inner_width + inner] +=
                gradient * conjugate(map_value);
            channel_maps_adjoint[map_index] +=
                gradient * conjugate(input_value);
          }
        }
      }
    }
  }
}

template <typename Scalar>
void carrier_channel_transform_double_backward(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Scalar* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Scalar* channel_maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output_adjoint_tangent,
    Scalar* values_second_adjoint,
    Scalar* channel_maps_second_adjoint) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  const std::int64_t map_count = map_offsets[block_count];
  std::fill(
      output_adjoint_tangent,
      output_adjoint_tangent + batch_size * output_width,
      Scalar(0));
  std::fill(
      values_second_adjoint,
      values_second_adjoint + batch_size * input_width,
      Scalar(0));
  std::fill(
      channel_maps_second_adjoint,
      channel_maps_second_adjoint + map_count,
      Scalar(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row = output_adjoint + batch * output_width;
    const Scalar* input_row = values + batch * input_width;
    const Scalar* input_tangent_row =
        values_adjoint_tangent + batch * input_width;
    Scalar* output_tangent_row =
        output_adjoint_tangent + batch * output_width;
    Scalar* input_second_row =
        values_second_adjoint + batch * input_width;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t input_channel = 0;
             input_channel < input_channels;
             ++input_channel) {
          const std::int64_t map_index =
              map_offsets[block] + output_channel * input_channels +
              input_channel;
          const Scalar map_value = channel_maps[map_index];
          const Scalar map_tangent = channel_maps_adjoint_tangent[map_index];
          for (std::int64_t inner = 0; inner < inner_width; ++inner) {
            const std::int64_t input_index =
                input_start + input_channel * inner_width + inner;
            const std::int64_t output_index =
                output_start + output_channel * inner_width + inner;
            const Scalar gradient = output_row[output_index];
            const Scalar input_value = input_row[input_index];
            const Scalar input_tangent = input_tangent_row[input_index];
            output_tangent_row[output_index] +=
                map_value * input_tangent + map_tangent * input_value;
            input_second_row[input_index] +=
                gradient * conjugate(map_tangent);
            channel_maps_second_adjoint[map_index] +=
                gradient * conjugate(input_tangent);
          }
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_channel_transform_real_maps_forward(
    const Scalar* values,
    const Real* channel_maps,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = values + batch * input_width;
    Scalar* output_row = output + batch * output_width;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t inner = 0; inner < inner_width; ++inner) {
          Scalar value = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            value += channel_maps[
                map_offsets[block] + output_channel * input_channels +
                input_channel] * input_row[
                input_start + input_channel * inner_width + inner];
          }
          output_row[
              output_start + output_channel * inner_width + inner] = value;
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_channel_transform_real_maps_adjoint(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Real* channel_maps,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* values_adjoint,
    Real* channel_maps_adjoint) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  std::fill(
      values_adjoint,
      values_adjoint + batch_size * input_width,
      Scalar(0));
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_offsets[block_count],
      Real(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row = output_adjoint + batch * output_width;
    const Scalar* input_row = values + batch * input_width;
    Scalar* input_adjoint_row = values_adjoint + batch * input_width;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t input_channel = 0;
             input_channel < input_channels;
             ++input_channel) {
          const std::int64_t map_index =
              map_offsets[block] + output_channel * input_channels +
              input_channel;
          const Real map_value = channel_maps[map_index];
          for (std::int64_t inner = 0; inner < inner_width; ++inner) {
            const Scalar gradient = output_row[
                output_start + output_channel * inner_width + inner];
            const Scalar input_value = input_row[
                input_start + input_channel * inner_width + inner];
            input_adjoint_row[
                input_start + input_channel * inner_width + inner] +=
                gradient * map_value;
            channel_maps_adjoint[map_index] +=
                real_component(gradient * conjugate(input_value));
          }
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_channel_transform_real_maps_double_backward(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Real* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Real* channel_maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output_adjoint_tangent,
    Scalar* values_second_adjoint,
    Real* channel_maps_second_adjoint) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  std::fill(
      output_adjoint_tangent,
      output_adjoint_tangent + batch_size * output_width,
      Scalar(0));
  std::fill(
      values_second_adjoint,
      values_second_adjoint + batch_size * input_width,
      Scalar(0));
  std::fill(
      channel_maps_second_adjoint,
      channel_maps_second_adjoint + map_offsets[block_count],
      Real(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row = output_adjoint + batch * output_width;
    const Scalar* input_row = values + batch * input_width;
    const Scalar* input_tangent_row =
        values_adjoint_tangent + batch * input_width;
    Scalar* output_tangent_row =
        output_adjoint_tangent + batch * output_width;
    Scalar* input_second_row =
        values_second_adjoint + batch * input_width;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) / input_channels;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        for (std::int64_t input_channel = 0;
             input_channel < input_channels;
             ++input_channel) {
          const std::int64_t map_index =
              map_offsets[block] + output_channel * input_channels +
              input_channel;
          const Real map_value = channel_maps[map_index];
          const Real map_tangent = channel_maps_adjoint_tangent[map_index];
          for (std::int64_t inner = 0; inner < inner_width; ++inner) {
            const std::int64_t input_index =
                input_start + input_channel * inner_width + inner;
            const std::int64_t output_index =
                output_start + output_channel * inner_width + inner;
            const Scalar gradient = output_row[output_index];
            const Scalar input_value = input_row[input_index];
            const Scalar input_tangent = input_tangent_row[input_index];
            output_tangent_row[output_index] +=
                map_value * input_tangent + map_tangent * input_value;
            input_second_row[input_index] += gradient * map_tangent;
            channel_maps_second_adjoint[map_index] += real_component(
                gradient * conjugate(input_tangent));
          }
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_role_channel_map_adjoint(
    const Scalar* edge_values,
    const Real* role_weights,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t block_count,
    Scalar* channel_maps_adjoint) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_offsets[block_count],
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t atom = atom_centers[edge];
    if (atom < 0 || atom >= atom_count) {
      continue;
    }
    const Scalar* input_row = edge_values + edge * input_width;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Real weight = role_weights[edge * role_count + role];
      const Scalar* output_row = atomic_output_adjoint +
          (atom * role_count + role) * output_width;
      for (std::int64_t block = 0; block < block_count; ++block) {
        const std::int64_t input_start = input_feature_offsets[block];
        const std::int64_t output_start = output_feature_offsets[block];
        const std::int64_t input_channels =
            input_channel_offsets[block + 1] - input_channel_offsets[block];
        const std::int64_t output_channels =
            output_channel_offsets[block + 1] - output_channel_offsets[block];
        const std::int64_t inner_width =
            (input_feature_offsets[block + 1] - input_start) /
            input_channels;
        for (std::int64_t output_channel = 0;
             output_channel < output_channels;
             ++output_channel) {
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            const std::int64_t map_index = map_offsets[block] +
                output_channel * input_channels + input_channel;
            for (std::int64_t inner = 0; inner < inner_width; ++inner) {
              channel_maps_adjoint[map_index] += weight * output_row[
                  output_start + output_channel * inner_width + inner] *
                  conjugate(input_row[
                      input_start + input_channel * inner_width + inner]);
            }
          }
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_role_channel_real_map_adjoint(
    const Scalar* edge_values,
    const Real* role_weights,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t block_count,
    Real* channel_maps_adjoint) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_offsets[block_count],
      Real(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t atom = atom_centers[edge];
    if (atom < 0 || atom >= atom_count) {
      continue;
    }
    const Scalar* input_row = edge_values + edge * input_width;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Real weight = role_weights[edge * role_count + role];
      const Scalar* output_row = atomic_output_adjoint +
          (atom * role_count + role) * output_width;
      for (std::int64_t block = 0; block < block_count; ++block) {
        const std::int64_t input_start = input_feature_offsets[block];
        const std::int64_t output_start = output_feature_offsets[block];
        const std::int64_t input_channels =
            input_channel_offsets[block + 1] - input_channel_offsets[block];
        const std::int64_t output_channels =
            output_channel_offsets[block + 1] - output_channel_offsets[block];
        const std::int64_t inner_width =
            (input_feature_offsets[block + 1] - input_start) /
            input_channels;
        for (std::int64_t output_channel = 0;
             output_channel < output_channels;
             ++output_channel) {
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            const std::int64_t map_index = map_offsets[block] +
                output_channel * input_channels + input_channel;
            for (std::int64_t inner = 0; inner < inner_width; ++inner) {
              channel_maps_adjoint[map_index] += weight * real_component(
                  output_row[
                      output_start + output_channel * inner_width + inner] *
                  conjugate(input_row[
                      input_start + input_channel * inner_width + inner]));
            }
          }
        }
      }
    }
  }
}

template <typename Scalar, typename Real, typename Control>
void carrier_role_channel_map_adjoint_double_backward_impl(
    const Scalar* edge_values,
    const Real* role_weights,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const Control* channel_maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t block_count,
    Scalar* edge_values_second_adjoint,
    Real* role_weights_second_adjoint,
    Scalar* atomic_output_adjoint_tangent) {
  const std::int64_t input_width = input_feature_offsets[block_count];
  const std::int64_t output_width = output_feature_offsets[block_count];
  std::fill(
      edge_values_second_adjoint,
      edge_values_second_adjoint + edge_count * input_width,
      Scalar(0));
  std::fill(
      role_weights_second_adjoint,
      role_weights_second_adjoint + edge_count * role_count,
      Real(0));
  std::fill(
      atomic_output_adjoint_tangent,
      atomic_output_adjoint_tangent +
          atom_count * role_count * output_width,
      Scalar(0));
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const std::int64_t atom = atom_centers[edge];
    if (atom < 0 || atom >= atom_count) {
      continue;
    }
    const Scalar* input_row = edge_values + edge * input_width;
    Scalar* input_second_row =
        edge_values_second_adjoint + edge * input_width;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Real weight = role_weights[edge * role_count + role];
      const Scalar* output_row = atomic_output_adjoint +
          (atom * role_count + role) * output_width;
      Scalar* output_tangent_row = atomic_output_adjoint_tangent +
          (atom * role_count + role) * output_width;
      Real role_second = Real(0);
      for (std::int64_t block = 0; block < block_count; ++block) {
        const std::int64_t input_start = input_feature_offsets[block];
        const std::int64_t output_start = output_feature_offsets[block];
        const std::int64_t input_channels =
            input_channel_offsets[block + 1] - input_channel_offsets[block];
        const std::int64_t output_channels =
            output_channel_offsets[block + 1] - output_channel_offsets[block];
        const std::int64_t inner_width =
            (input_feature_offsets[block + 1] - input_start) /
            input_channels;
        for (std::int64_t output_channel = 0;
             output_channel < output_channels;
             ++output_channel) {
          for (std::int64_t input_channel = 0;
               input_channel < input_channels;
               ++input_channel) {
            const std::int64_t map_index = map_offsets[block] +
                output_channel * input_channels + input_channel;
            const Control map_tangent =
                channel_maps_adjoint_tangent[map_index];
            for (std::int64_t inner = 0; inner < inner_width; ++inner) {
              const std::int64_t input_index = input_start +
                  input_channel * inner_width + inner;
              const std::int64_t output_index = output_start +
                  output_channel * inner_width + inner;
              const Scalar input_value = input_row[input_index];
              const Scalar output_value = output_row[output_index];
              output_tangent_row[output_index] +=
                  weight * map_tangent * input_value;
              input_second_row[input_index] +=
                  weight * output_value * conjugate(map_tangent);
              const Scalar role_contribution =
                  map_tangent * input_value * conjugate(output_value);
              if constexpr (std::is_same_v<Control, Scalar>) {
                if constexpr (
                    std::is_same_v<Scalar, std::complex<float>> ||
                    std::is_same_v<Scalar, std::complex<double>>) {
                  role_second += real_component(role_contribution);
                } else {
                  role_second += role_contribution;
                }
              } else {
                role_second += real_component(role_contribution);
              }
            }
          }
        }
      }
      role_weights_second_adjoint[edge * role_count + role] = role_second;
    }
  }
}

template <typename Scalar, typename Real>
void carrier_role_channel_map_adjoint_double_backward(
    const Scalar* edge_values,
    const Real* role_weights,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const Scalar* channel_maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t block_count,
    Scalar* edge_values_second_adjoint,
    Real* role_weights_second_adjoint,
    Scalar* atomic_output_adjoint_tangent) {
  carrier_role_channel_map_adjoint_double_backward_impl<
      Scalar, Real, Scalar>(
      edge_values,
      role_weights,
      atomic_output_adjoint,
      atom_centers,
      channel_maps_adjoint_tangent,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets,
      edge_count,
      atom_count,
      role_count,
      block_count,
      edge_values_second_adjoint,
      role_weights_second_adjoint,
      atomic_output_adjoint_tangent);
}

template <typename Scalar, typename Real>
void carrier_role_channel_real_map_adjoint_double_backward(
    const Scalar* edge_values,
    const Real* role_weights,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const Real* channel_maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t block_count,
    Scalar* edge_values_second_adjoint,
    Real* role_weights_second_adjoint,
    Scalar* atomic_output_adjoint_tangent) {
  carrier_role_channel_map_adjoint_double_backward_impl<
      Scalar, Real, Real>(
      edge_values,
      role_weights,
      atomic_output_adjoint,
      atom_centers,
      channel_maps_adjoint_tangent,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets,
      edge_count,
      atom_count,
      role_count,
      block_count,
      edge_values_second_adjoint,
      role_weights_second_adjoint,
      atomic_output_adjoint_tangent);
}

template <typename Scalar>
void carrier_channel_update_adjoint(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Scalar* gates,
    const Scalar* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* values_adjoint,
    Scalar* gates_adjoint,
    Scalar* channel_maps_adjoint) {
  const std::int64_t feature_count = feature_offsets[block_count];
  const std::int64_t channel_count = channel_offsets[block_count];
  std::fill(
      values_adjoint,
      values_adjoint + batch_size * feature_count,
      Scalar(0));
  std::fill(
      gates_adjoint,
      gates_adjoint + batch_size * channel_count,
      Scalar(0));
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_offsets[block_count],
      Scalar(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row =
        output_adjoint + batch * feature_count;
    const Scalar* value_row = values + batch * feature_count;
    const Scalar* gate_row = gates + batch * channel_count;
    Scalar* value_adjoint_row =
        values_adjoint + batch * feature_count;
    Scalar* gate_adjoint_row =
        gates_adjoint + batch * channel_count;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t feature_start = feature_offsets[block];
      const std::int64_t channel_start = channel_offsets[block];
      const std::int64_t local_channels =
          channel_offsets[block + 1] - channel_start;
      const std::int64_t inner_width =
          (feature_offsets[block + 1] - feature_start) /
          local_channels;
      const std::int64_t map_start = map_offsets[block];
      for (std::int64_t output_channel = 0;
           output_channel < local_channels;
           ++output_channel) {
        const Scalar gate =
            gate_row[channel_start + output_channel];
        for (std::int64_t inner = 0;
             inner < inner_width;
             ++inner) {
          const std::int64_t output_feature =
              feature_start +
              output_channel * inner_width +
              inner;
          const Scalar gradient = output_row[output_feature];
          value_adjoint_row[output_feature] += gradient;
          Scalar mixed = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < local_channels;
               ++input_channel) {
            const std::int64_t input_feature =
                feature_start +
                input_channel * inner_width +
                inner;
            const std::int64_t map_index =
                map_start +
                output_channel * local_channels +
                input_channel;
            const Scalar map_value = channel_maps[map_index];
            const Scalar input_value = value_row[input_feature];
            mixed += map_value * input_value;
            value_adjoint_row[input_feature] +=
                gradient * conjugate(gate * map_value);
            channel_maps_adjoint[map_index] +=
                gradient * conjugate(gate * input_value);
          }
          gate_adjoint_row[channel_start + output_channel] +=
              gradient * conjugate(mixed);
        }
      }
    }
  }
}

template <typename Scalar>
void carrier_channel_update_double_backward(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Scalar* gates,
    const Scalar* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Scalar* gates_adjoint_tangent,
    const Scalar* channel_maps_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output_adjoint_tangent,
    Scalar* values_second_adjoint,
    Scalar* gates_second_adjoint,
    Scalar* channel_maps_second_adjoint) {
  const std::int64_t feature_count = feature_offsets[block_count];
  const std::int64_t channel_count = channel_offsets[block_count];
  std::copy(
      values_adjoint_tangent,
      values_adjoint_tangent + batch_size * feature_count,
      output_adjoint_tangent);
  std::fill(
      values_second_adjoint,
      values_second_adjoint + batch_size * feature_count,
      Scalar(0));
  std::fill(
      gates_second_adjoint,
      gates_second_adjoint + batch_size * channel_count,
      Scalar(0));
  std::fill(
      channel_maps_second_adjoint,
      channel_maps_second_adjoint + map_offsets[block_count],
      Scalar(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row =
        output_adjoint + batch * feature_count;
    const Scalar* value_row = values + batch * feature_count;
    const Scalar* gate_row = gates + batch * channel_count;
    const Scalar* value_tangent_row =
        values_adjoint_tangent + batch * feature_count;
    Scalar* output_tangent_row =
        output_adjoint_tangent + batch * feature_count;
    Scalar* value_second_row =
        values_second_adjoint + batch * feature_count;
    Scalar* gate_second_row =
        gates_second_adjoint + batch * channel_count;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t feature_start = feature_offsets[block];
      const std::int64_t channel_start = channel_offsets[block];
      const std::int64_t local_channels =
          channel_offsets[block + 1] - channel_start;
      const std::int64_t inner_width =
          (feature_offsets[block + 1] - feature_start) /
          local_channels;
      const std::int64_t map_start = map_offsets[block];
      for (std::int64_t output_channel = 0;
           output_channel < local_channels;
           ++output_channel) {
        const Scalar gate =
            gate_row[channel_start + output_channel];
        const Scalar gate_tangent =
            gates_adjoint_tangent[
                batch * channel_count +
                channel_start +
                output_channel];
        for (std::int64_t inner = 0;
             inner < inner_width;
             ++inner) {
          const std::int64_t output_feature =
              feature_start +
              output_channel * inner_width +
              inner;
          const Scalar gradient = output_row[output_feature];
          Scalar mixed = Scalar(0);
          Scalar mixed_adjoint_tangent = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < local_channels;
               ++input_channel) {
            const std::int64_t input_feature =
                feature_start +
                input_channel * inner_width +
                inner;
            const std::int64_t map_index =
                map_start +
                output_channel * local_channels +
                input_channel;
            const Scalar map_value = channel_maps[map_index];
            const Scalar input_value = value_row[input_feature];
            const Scalar value_tangent =
                value_tangent_row[input_feature];
            const Scalar map_tangent =
                channel_maps_adjoint_tangent[map_index];
            mixed += map_value * input_value;
            mixed_adjoint_tangent +=
                map_value * value_tangent +
                input_value * map_tangent;
            const Scalar mixed_adjoint =
                gradient * conjugate(gate);
            const Scalar mixed_second =
                gradient * conjugate(gate_tangent);
            value_second_row[input_feature] +=
                mixed_adjoint * conjugate(map_tangent) +
                mixed_second * conjugate(map_value);
            channel_maps_second_adjoint[map_index] +=
                mixed_adjoint * conjugate(value_tangent) +
                mixed_second * conjugate(input_value);
          }
          output_tangent_row[output_feature] +=
              gate * mixed_adjoint_tangent +
              gate_tangent * mixed;
          gate_second_row[channel_start + output_channel] +=
              gradient * conjugate(mixed_adjoint_tangent);
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_channel_update_real_controls_forward(
    const Scalar* values,
    const Real* gates,
    const Real* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t feature_count = feature_offsets[block_count];
  const std::int64_t channel_count = channel_offsets[block_count];
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* value_row = values + batch * feature_count;
    const Real* gate_row = gates + batch * channel_count;
    Scalar* output_row = output + batch * feature_count;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t feature_start = feature_offsets[block];
      const std::int64_t channel_start = channel_offsets[block];
      const std::int64_t local_channels =
          channel_offsets[block + 1] - channel_start;
      const std::int64_t inner_width =
          (feature_offsets[block + 1] - feature_start) /
          local_channels;
      const std::int64_t map_start = map_offsets[block];
      for (std::int64_t output_channel = 0;
           output_channel < local_channels;
           ++output_channel) {
        const Real gate = gate_row[channel_start + output_channel];
        for (std::int64_t inner = 0;
             inner < inner_width;
             ++inner) {
          Scalar mixed = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < local_channels;
               ++input_channel) {
            mixed +=
                channel_maps[
                    map_start +
                    output_channel * local_channels +
                    input_channel] *
                value_row[
                    feature_start +
                    input_channel * inner_width +
                    inner];
          }
          const std::int64_t output_feature =
              feature_start +
              output_channel * inner_width +
              inner;
          output_row[output_feature] =
              value_row[output_feature] + gate * mixed;
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_channel_update_real_controls_adjoint(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Real* gates,
    const Real* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* values_adjoint,
    Real* gates_adjoint,
    Real* channel_maps_adjoint) {
  const std::int64_t feature_count = feature_offsets[block_count];
  const std::int64_t channel_count = channel_offsets[block_count];
  std::fill(
      values_adjoint,
      values_adjoint + batch_size * feature_count,
      Scalar(0));
  std::fill(
      gates_adjoint,
      gates_adjoint + batch_size * channel_count,
      Real(0));
  std::fill(
      channel_maps_adjoint,
      channel_maps_adjoint + map_offsets[block_count],
      Real(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row =
        output_adjoint + batch * feature_count;
    const Scalar* value_row = values + batch * feature_count;
    const Real* gate_row = gates + batch * channel_count;
    Scalar* value_adjoint_row =
        values_adjoint + batch * feature_count;
    Real* gate_adjoint_row =
        gates_adjoint + batch * channel_count;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t feature_start = feature_offsets[block];
      const std::int64_t channel_start = channel_offsets[block];
      const std::int64_t local_channels =
          channel_offsets[block + 1] - channel_start;
      const std::int64_t inner_width =
          (feature_offsets[block + 1] - feature_start) /
          local_channels;
      const std::int64_t map_start = map_offsets[block];
      for (std::int64_t output_channel = 0;
           output_channel < local_channels;
           ++output_channel) {
        const Real gate = gate_row[channel_start + output_channel];
        for (std::int64_t inner = 0;
             inner < inner_width;
             ++inner) {
          const std::int64_t output_feature =
              feature_start +
              output_channel * inner_width +
              inner;
          const Scalar gradient = output_row[output_feature];
          value_adjoint_row[output_feature] += gradient;
          Scalar mixed = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < local_channels;
               ++input_channel) {
            const std::int64_t input_feature =
                feature_start +
                input_channel * inner_width +
                inner;
            const std::int64_t map_index =
                map_start +
                output_channel * local_channels +
                input_channel;
            const Real map_value = channel_maps[map_index];
            const Scalar input_value = value_row[input_feature];
            mixed += map_value * input_value;
            value_adjoint_row[input_feature] +=
                gradient * gate * map_value;
            channel_maps_adjoint[map_index] += real_component(
                gradient * conjugate(gate * input_value));
          }
          gate_adjoint_row[channel_start + output_channel] +=
              real_component(gradient * conjugate(mixed));
        }
      }
    }
  }
}

template <typename Scalar, typename Real>
void carrier_channel_update_real_controls_double_backward(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Real* gates,
    const Real* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Real* gates_adjoint_tangent,
    const Real* channel_maps_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t block_count,
    Scalar* output_adjoint_tangent,
    Scalar* values_second_adjoint,
    Real* gates_second_adjoint,
    Real* channel_maps_second_adjoint) {
  const std::int64_t feature_count = feature_offsets[block_count];
  const std::int64_t channel_count = channel_offsets[block_count];
  std::copy(
      values_adjoint_tangent,
      values_adjoint_tangent + batch_size * feature_count,
      output_adjoint_tangent);
  std::fill(
      values_second_adjoint,
      values_second_adjoint + batch_size * feature_count,
      Scalar(0));
  std::fill(
      gates_second_adjoint,
      gates_second_adjoint + batch_size * channel_count,
      Real(0));
  std::fill(
      channel_maps_second_adjoint,
      channel_maps_second_adjoint + map_offsets[block_count],
      Real(0));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row =
        output_adjoint + batch * feature_count;
    const Scalar* value_row = values + batch * feature_count;
    const Real* gate_row = gates + batch * channel_count;
    const Scalar* value_tangent_row =
        values_adjoint_tangent + batch * feature_count;
    Scalar* output_tangent_row =
        output_adjoint_tangent + batch * feature_count;
    Scalar* value_second_row =
        values_second_adjoint + batch * feature_count;
    Real* gate_second_row =
        gates_second_adjoint + batch * channel_count;
    for (std::int64_t block = 0; block < block_count; ++block) {
      const std::int64_t feature_start = feature_offsets[block];
      const std::int64_t channel_start = channel_offsets[block];
      const std::int64_t local_channels =
          channel_offsets[block + 1] - channel_start;
      const std::int64_t inner_width =
          (feature_offsets[block + 1] - feature_start) /
          local_channels;
      const std::int64_t map_start = map_offsets[block];
      for (std::int64_t output_channel = 0;
           output_channel < local_channels;
           ++output_channel) {
        const Real gate = gate_row[channel_start + output_channel];
        const Real gate_tangent =
            gates_adjoint_tangent[
                batch * channel_count +
                channel_start +
                output_channel];
        for (std::int64_t inner = 0;
             inner < inner_width;
             ++inner) {
          const std::int64_t output_feature =
              feature_start +
              output_channel * inner_width +
              inner;
          const Scalar gradient = output_row[output_feature];
          Scalar mixed = Scalar(0);
          Scalar mixed_adjoint_tangent = Scalar(0);
          for (std::int64_t input_channel = 0;
               input_channel < local_channels;
               ++input_channel) {
            const std::int64_t input_feature =
                feature_start +
                input_channel * inner_width +
                inner;
            const std::int64_t map_index =
                map_start +
                output_channel * local_channels +
                input_channel;
            const Real map_value = channel_maps[map_index];
            const Scalar input_value = value_row[input_feature];
            const Scalar value_tangent =
                value_tangent_row[input_feature];
            const Real map_tangent =
                channel_maps_adjoint_tangent[map_index];
            mixed += map_value * input_value;
            mixed_adjoint_tangent +=
                map_value * value_tangent +
                input_value * map_tangent;
            const Scalar mixed_adjoint = gradient * gate;
            const Scalar mixed_second = gradient * gate_tangent;
            value_second_row[input_feature] +=
                mixed_adjoint * map_tangent +
                mixed_second * map_value;
            channel_maps_second_adjoint[map_index] +=
                real_component(
                    mixed_adjoint * conjugate(value_tangent) +
                    mixed_second * conjugate(input_value));
          }
          output_tangent_row[output_feature] +=
              gate * mixed_adjoint_tangent +
              gate_tangent * mixed;
          gate_second_row[channel_start + output_channel] +=
              real_component(
                  gradient * conjugate(mixed_adjoint_tangent));
        }
      }
    }
  }
}

template <typename Scalar>
void source_analysis_forward(
    const Scalar* source,
    std::int64_t batch_size,
    std::int64_t source_dimension,
    const std::int64_t* assembly_rows,
    const std::int64_t* assembly_columns,
    const Scalar* assembly_values,
    std::int64_t assembly_nnz,
    std::int64_t induced_dimension,
    const std::int64_t* synthesis_rows,
    const std::int64_t* synthesis_columns,
    const Scalar* synthesis_values,
    std::int64_t synthesis_nnz,
    std::int64_t output_dimension,
    Scalar* output) {
  std::vector<Scalar> ambient(static_cast<std::size_t>(induced_dimension));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    std::fill(ambient.begin(), ambient.end(), Scalar(0));
    const Scalar* source_row = source + batch * source_dimension;
    Scalar* output_row = output + batch * output_dimension;
    std::fill(output_row, output_row + output_dimension, Scalar(0));
    for (std::int64_t entry = 0; entry < assembly_nnz; ++entry) {
      const std::int64_t row = assembly_rows[entry];
      const std::int64_t column = assembly_columns[entry];
      ambient[static_cast<std::size_t>(row)] +=
          assembly_values[entry] * source_row[column];
    }
    for (std::int64_t entry = 0; entry < synthesis_nnz; ++entry) {
      const std::int64_t row = synthesis_rows[entry];
      const std::int64_t column = synthesis_columns[entry];
      output_row[column] +=
          conjugate(synthesis_values[entry]) *
          ambient[static_cast<std::size_t>(row)];
    }
  }
}

template <typename Scalar>
void source_analysis_adjoint(
    const Scalar* output_adjoint,
    std::int64_t batch_size,
    std::int64_t output_dimension,
    const std::int64_t* assembly_rows,
    const std::int64_t* assembly_columns,
    const Scalar* assembly_values,
    std::int64_t assembly_nnz,
    std::int64_t source_dimension,
    std::int64_t induced_dimension,
    const std::int64_t* synthesis_rows,
    const std::int64_t* synthesis_columns,
    const Scalar* synthesis_values,
    std::int64_t synthesis_nnz,
    Scalar* source_adjoint) {
  std::vector<Scalar> ambient_adjoint(
      static_cast<std::size_t>(induced_dimension)
  );
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    std::fill(ambient_adjoint.begin(), ambient_adjoint.end(), Scalar(0));
    const Scalar* output_row = output_adjoint + batch * output_dimension;
    Scalar* source_row = source_adjoint + batch * source_dimension;
    std::fill(source_row, source_row + source_dimension, Scalar(0));
    for (std::int64_t entry = 0; entry < synthesis_nnz; ++entry) {
      const std::int64_t row = synthesis_rows[entry];
      const std::int64_t column = synthesis_columns[entry];
      ambient_adjoint[static_cast<std::size_t>(row)] +=
          synthesis_values[entry] * output_row[column];
    }
    for (std::int64_t entry = 0; entry < assembly_nnz; ++entry) {
      const std::int64_t row = assembly_rows[entry];
      const std::int64_t column = assembly_columns[entry];
      source_row[column] +=
          conjugate(assembly_values[entry]) *
          ambient_adjoint[static_cast<std::size_t>(row)];
    }
  }
}

template <typename Scalar>
void source_analysis_linear_forward(
    const Scalar* source,
    std::int64_t batch_size,
    std::int64_t source_dimension,
    const std::int64_t* assembly_rows,
    const std::int64_t* assembly_columns,
    const Scalar* assembly_values,
    std::int64_t assembly_nnz,
    std::int64_t induced_dimension,
    const std::int64_t* synthesis_rows,
    const std::int64_t* synthesis_columns,
    const Scalar* synthesis_values,
    std::int64_t synthesis_nnz,
    std::int64_t output_dimension,
    const Scalar* weight,
    Scalar bias,
    Scalar* output) {
  std::vector<Scalar> induced_weight(
      static_cast<std::size_t>(induced_dimension), Scalar(0));
  for (std::int64_t entry = 0; entry < synthesis_nnz; ++entry) {
    induced_weight[static_cast<std::size_t>(synthesis_rows[entry])] +=
        conjugate(synthesis_values[entry]) *
        weight[synthesis_columns[entry]];
  }
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    Scalar value = bias;
    const Scalar* source_row = source + batch * source_dimension;
    for (std::int64_t entry = 0; entry < assembly_nnz; ++entry) {
      value +=
          source_row[assembly_columns[entry]] *
          assembly_values[entry] *
          induced_weight[static_cast<std::size_t>(assembly_rows[entry])];
    }
    output[batch] = value;
  }
}

template <typename Scalar>
void source_analysis_linear_adjoint(
    const Scalar* output_adjoint,
    const Scalar* source,
    std::int64_t batch_size,
    std::int64_t source_dimension,
    const std::int64_t* assembly_rows,
    const std::int64_t* assembly_columns,
    const Scalar* assembly_values,
    std::int64_t assembly_nnz,
    std::int64_t induced_dimension,
    const std::int64_t* synthesis_rows,
    const std::int64_t* synthesis_columns,
    const Scalar* synthesis_values,
    std::int64_t synthesis_nnz,
    std::int64_t output_dimension,
    const Scalar* weight,
    Scalar* source_adjoint,
    Scalar* weight_adjoint,
    Scalar* bias_adjoint) {
  std::vector<Scalar> induced(
      static_cast<std::size_t>(induced_dimension));
  std::vector<Scalar> induced_adjoint_weight(
      static_cast<std::size_t>(induced_dimension), Scalar(0));
  std::vector<Scalar> source_adjoint_weight(
      static_cast<std::size_t>(source_dimension), Scalar(0));
  for (std::int64_t entry = 0; entry < synthesis_nnz; ++entry) {
    induced_adjoint_weight[
        static_cast<std::size_t>(synthesis_rows[entry])] +=
        synthesis_values[entry] *
        conjugate(weight[synthesis_columns[entry]]);
  }
  for (std::int64_t entry = 0; entry < assembly_nnz; ++entry) {
    source_adjoint_weight[
        static_cast<std::size_t>(assembly_columns[entry])] +=
        conjugate(assembly_values[entry]) *
        induced_adjoint_weight[
            static_cast<std::size_t>(assembly_rows[entry])];
  }
  std::fill(
      weight_adjoint,
      weight_adjoint + output_dimension,
      Scalar(0));
  *bias_adjoint = Scalar(0);
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar gradient = output_adjoint[batch];
    const Scalar* source_row = source + batch * source_dimension;
    Scalar* source_gradient =
        source_adjoint + batch * source_dimension;
    *bias_adjoint += gradient;
    for (std::int64_t source_index = 0;
         source_index < source_dimension;
         ++source_index) {
      source_gradient[source_index] =
          gradient *
          source_adjoint_weight[static_cast<std::size_t>(source_index)];
    }
    std::fill(induced.begin(), induced.end(), Scalar(0));
    for (std::int64_t entry = 0; entry < assembly_nnz; ++entry) {
      induced[static_cast<std::size_t>(assembly_rows[entry])] +=
          assembly_values[entry] *
          source_row[assembly_columns[entry]];
    }
    for (std::int64_t entry = 0; entry < synthesis_nnz; ++entry) {
      weight_adjoint[synthesis_columns[entry]] +=
          gradient *
          conjugate(
              induced[static_cast<std::size_t>(synthesis_rows[entry])]) *
          synthesis_values[entry];
    }
  }
}

template <typename Scalar>
void compact_pair_product_forward(
    const Scalar* left,
    const Scalar* right,
    std::int64_t batch_size,
    std::int64_t dimension,
    bool antisymmetric,
    Scalar* output) {
  const Scalar inverse_sqrt_two =
      Scalar(1.0 / std::sqrt(2.0));
  const std::int64_t output_dimension = antisymmetric
      ? dimension * (dimension - 1) / 2
      : dimension * (dimension + 1) / 2;
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* left_row = left + batch * dimension;
    const Scalar* right_row = right + batch * dimension;
    Scalar* output_row = output + batch * output_dimension;
    std::int64_t output_index = 0;
    for (std::int64_t i = 0; i < dimension; ++i) {
      const std::int64_t start = antisymmetric ? i + 1 : i;
      for (std::int64_t j = start; j < dimension; ++j) {
        if (!antisymmetric && i == j) {
          output_row[output_index] = left_row[i] * right_row[i];
        } else if (antisymmetric) {
          output_row[output_index] = inverse_sqrt_two *
              (left_row[i] * right_row[j] -
               left_row[j] * right_row[i]);
        } else {
          output_row[output_index] = inverse_sqrt_two *
              (left_row[i] * right_row[j] +
               left_row[j] * right_row[i]);
        }
        ++output_index;
      }
    }
  }
}

template <typename Scalar>
void compact_pair_product_adjoint(
    const Scalar* output_adjoint,
    const Scalar* left,
    const Scalar* right,
    std::int64_t batch_size,
    std::int64_t dimension,
    bool antisymmetric,
    Scalar* left_adjoint,
    Scalar* right_adjoint) {
  const Scalar inverse_sqrt_two =
      Scalar(1.0 / std::sqrt(2.0));
  const std::int64_t output_dimension = antisymmetric
      ? dimension * (dimension - 1) / 2
      : dimension * (dimension + 1) / 2;
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* output_row = output_adjoint + batch * output_dimension;
    const Scalar* left_row = left + batch * dimension;
    const Scalar* right_row = right + batch * dimension;
    Scalar* left_gradient = left_adjoint + batch * dimension;
    Scalar* right_gradient = right_adjoint + batch * dimension;
    std::fill(left_gradient, left_gradient + dimension, Scalar(0));
    std::fill(right_gradient, right_gradient + dimension, Scalar(0));
    std::int64_t output_index = 0;
    for (std::int64_t i = 0; i < dimension; ++i) {
      const std::int64_t start = antisymmetric ? i + 1 : i;
      for (std::int64_t j = start; j < dimension; ++j) {
        const Scalar gradient = output_row[output_index];
        if (!antisymmetric && i == j) {
          left_gradient[i] += gradient * conjugate(right_row[i]);
          right_gradient[i] += gradient * conjugate(left_row[i]);
        } else if (antisymmetric) {
          left_gradient[i] +=
              inverse_sqrt_two * gradient * conjugate(right_row[j]);
          left_gradient[j] -=
              inverse_sqrt_two * gradient * conjugate(right_row[i]);
          right_gradient[j] +=
              inverse_sqrt_two * gradient * conjugate(left_row[i]);
          right_gradient[i] -=
              inverse_sqrt_two * gradient * conjugate(left_row[j]);
        } else {
          left_gradient[i] +=
              inverse_sqrt_two * gradient * conjugate(right_row[j]);
          left_gradient[j] +=
              inverse_sqrt_two * gradient * conjugate(right_row[i]);
          right_gradient[j] +=
              inverse_sqrt_two * gradient * conjugate(left_row[i]);
          right_gradient[i] +=
              inverse_sqrt_two * gradient * conjugate(left_row[j]);
        }
        ++output_index;
      }
    }
  }
}

template <typename Scalar>
void compact_exterior_power_forward(
    const Scalar* factors,
    std::int64_t batch_size,
    std::int64_t order,
    std::int64_t dimension,
    Scalar* output) {
  const std::int64_t output_dimension =
      binomial_coefficient(dimension, order);
  if (output_dimension == 0) {
    return;
  }
  const Scalar normalization =
      Scalar(1.0 / std::sqrt(std::tgamma(double(order + 1))));
  std::vector<std::int64_t> combination(
      static_cast<std::size_t>(order));
  std::vector<Scalar> matrix(
      static_cast<std::size_t>(order * order));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    for (std::int64_t index = 0; index < order; ++index) {
      combination[static_cast<std::size_t>(index)] = index;
    }
    const Scalar* factor_batch =
        factors + batch * order * dimension;
    Scalar* output_batch = output + batch * output_dimension;
    std::int64_t output_index = 0;
    do {
      for (std::int64_t row = 0; row < order; ++row) {
        for (std::int64_t column = 0; column < order; ++column) {
          matrix[static_cast<std::size_t>(row * order + column)] =
              factor_batch[
                  row * dimension +
                  combination[static_cast<std::size_t>(column)]];
        }
      }
      output_batch[output_index] =
          normalization * determinant_subset_dp(matrix, order);
      ++output_index;
    } while (next_combination(&combination, dimension));
  }
}

template <typename Scalar>
void compact_exterior_power_adjoint(
    const Scalar* output_adjoint,
    const Scalar* factors,
    std::int64_t batch_size,
    std::int64_t order,
    std::int64_t dimension,
    Scalar* factors_adjoint) {
  const std::int64_t output_dimension =
      binomial_coefficient(dimension, order);
  std::fill(
      factors_adjoint,
      factors_adjoint + batch_size * order * dimension,
      Scalar(0));
  if (output_dimension == 0) {
    return;
  }
  const Scalar normalization =
      Scalar(1.0 / std::sqrt(std::tgamma(double(order + 1))));
  std::vector<std::int64_t> combination(
      static_cast<std::size_t>(order));
  std::vector<Scalar> matrix(
      static_cast<std::size_t>(order * order));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    for (std::int64_t index = 0; index < order; ++index) {
      combination[static_cast<std::size_t>(index)] = index;
    }
    const Scalar* factor_batch =
        factors + batch * order * dimension;
    Scalar* factor_gradient =
        factors_adjoint + batch * order * dimension;
    const Scalar* output_gradient =
        output_adjoint + batch * output_dimension;
    std::int64_t output_index = 0;
    do {
      for (std::int64_t row = 0; row < order; ++row) {
        for (std::int64_t column = 0; column < order; ++column) {
          matrix[static_cast<std::size_t>(row * order + column)] =
              factor_batch[
                  row * dimension +
                  combination[static_cast<std::size_t>(column)]];
        }
      }
      for (std::int64_t row = 0; row < order; ++row) {
        for (std::int64_t column = 0; column < order; ++column) {
          Scalar cofactor = determinant_minor(
              matrix,
              order,
              row,
              column);
          if (((row + column) & 1) != 0) {
            cofactor = -cofactor;
          }
          factor_gradient[
              row * dimension +
              combination[static_cast<std::size_t>(column)]] +=
              output_gradient[output_index] *
              conjugate(normalization * cofactor);
        }
      }
      ++output_index;
    } while (next_combination(&combination, dimension));
  }
}

template <typename Scalar>
void symmetric_power_monomial_forward(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    const std::int64_t* output_offsets,
    const Scalar* monomial_values,
    std::int64_t term_count,
    std::int64_t output_dimension,
    Scalar* output) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = input + batch * input_dimension;
    Scalar* output_row = output + batch * output_dimension;
    for (std::int64_t output_index = 0;
         output_index < output_dimension;
         ++output_index) {
      Scalar value = Scalar(0);
      const std::int64_t start = output_offsets[output_index];
      const std::int64_t finish = output_offsets[output_index + 1];
      for (std::int64_t term = start; term < finish; ++term) {
        Scalar monomial = monomial_values[term];
        for (std::int64_t component = 0;
             component < input_dimension;
             ++component) {
          const std::int64_t exponent =
              monomial_counts[term * input_dimension + component];
          monomial *= integer_power(
              input_row[component],
              exponent);
        }
        value += monomial;
      }
      output_row[output_index] = value;
    }
  }
  (void)term_count;
}

template <typename Scalar>
void symmetric_power_monomial_adjoint(
    const Scalar* output_adjoint,
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    const std::int64_t* output_indices,
    const Scalar* monomial_values,
    std::int64_t term_count,
    std::int64_t output_dimension,
    Scalar* input_adjoint) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = input + batch * input_dimension;
    Scalar* input_gradient =
        input_adjoint + batch * input_dimension;
    std::fill(
        input_gradient,
        input_gradient + input_dimension,
        Scalar(0));
    for (std::int64_t term = 0; term < term_count; ++term) {
      Scalar monomial = monomial_values[term];
      std::int64_t zero_count = 0;
      std::int64_t zero_component = -1;
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        if (input_row[component] == Scalar(0)) {
          ++zero_count;
          zero_component = component;
          continue;
        }
        monomial *= integer_power(
            input_row[component],
            exponent);
      }
      if (zero_count >= 2) {
        continue;
      }
      const Scalar output_gradient =
          output_adjoint[
              batch * output_dimension + output_indices[term]];
      if (zero_count == 1) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + zero_component];
        if (exponent == 1) {
          input_gradient[zero_component] +=
              output_gradient * conjugate(monomial);
        }
        continue;
      }
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        const Scalar derivative =
            Scalar(exponent) * monomial / input_row[component];
        input_gradient[component] +=
            output_gradient * conjugate(derivative);
      }
    }
  }
}

template <typename Scalar>
void symmetric_power_shared_monomial_forward(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    std::int64_t monomial_count,
    const std::int64_t* output_offsets,
    const std::int64_t* coefficient_terms,
    const Scalar* coefficient_values,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* output) {
  std::vector<Scalar> monomials(
      static_cast<std::size_t>(monomial_count));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = input + batch * input_dimension;
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      Scalar value = Scalar(1);
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        value *= integer_power(
            input_row[component],
            monomial_counts[
                term * input_dimension + component]);
      }
      monomials[static_cast<std::size_t>(term)] = value;
    }
    Scalar* output_row = output + batch * output_dimension;
    for (std::int64_t output_index = 0;
         output_index < output_dimension;
         ++output_index) {
      Scalar value = Scalar(0);
      const std::int64_t start = output_offsets[output_index];
      const std::int64_t finish = output_offsets[output_index + 1];
      for (std::int64_t coefficient = start;
           coefficient < finish;
           ++coefficient) {
        value +=
            coefficient_values[coefficient] *
            monomials[static_cast<std::size_t>(
                coefficient_terms[coefficient])];
      }
      output_row[output_index] = value;
    }
  }
  (void)coefficient_count;
}

template <typename Scalar>
void symmetric_power_shared_monomial_adjoint(
    const Scalar* output_adjoint,
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    std::int64_t monomial_count,
    const std::int64_t* output_offsets,
    const std::int64_t* coefficient_terms,
    const Scalar* coefficient_values,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* input_adjoint) {
  std::vector<Scalar> monomial_adjoint(
      static_cast<std::size_t>(monomial_count));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    std::fill(
        monomial_adjoint.begin(),
        monomial_adjoint.end(),
        Scalar(0));
    const Scalar* output_gradient =
        output_adjoint + batch * output_dimension;
    for (std::int64_t output_index = 0;
         output_index < output_dimension;
         ++output_index) {
      const std::int64_t start = output_offsets[output_index];
      const std::int64_t finish = output_offsets[output_index + 1];
      for (std::int64_t coefficient = start;
           coefficient < finish;
           ++coefficient) {
        monomial_adjoint[static_cast<std::size_t>(
            coefficient_terms[coefficient])] +=
            output_gradient[output_index] *
            conjugate(coefficient_values[coefficient]);
      }
    }
    const Scalar* input_row = input + batch * input_dimension;
    Scalar* input_gradient =
        input_adjoint + batch * input_dimension;
    std::fill(
        input_gradient,
        input_gradient + input_dimension,
        Scalar(0));
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      const Scalar term_gradient =
          monomial_adjoint[static_cast<std::size_t>(term)];
      Scalar monomial = Scalar(1);
      std::int64_t zero_count = 0;
      std::int64_t zero_component = -1;
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        if (input_row[component] == Scalar(0)) {
          ++zero_count;
          zero_component = component;
          continue;
        }
        monomial *= integer_power(
            input_row[component],
            exponent);
      }
      if (zero_count >= 2) {
        continue;
      }
      if (zero_count == 1) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + zero_component];
        if (exponent == 1) {
          input_gradient[zero_component] +=
              term_gradient * conjugate(monomial);
        }
        continue;
      }
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        const Scalar derivative =
            Scalar(exponent) * monomial / input_row[component];
        input_gradient[component] +=
            term_gradient * conjugate(derivative);
      }
    }
  }
  (void)coefficient_count;
}

template <typename Scalar>
void symmetric_power_shared_monomial_batched_adjoint(
    const Scalar* output_adjoint,
    const Scalar* input,
    std::int64_t seed_count,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    std::int64_t monomial_count,
    const std::int64_t* output_offsets,
    const std::int64_t* coefficient_terms,
    const Scalar* coefficient_values,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* input_adjoint) {
  std::fill(
      input_adjoint,
      input_adjoint +
          seed_count * batch_size * input_dimension,
      Scalar(0));
  std::vector<Scalar> monomial_adjoint(
      static_cast<std::size_t>(seed_count * monomial_count));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    std::fill(
        monomial_adjoint.begin(),
        monomial_adjoint.end(),
        Scalar(0));
    for (std::int64_t output_index = 0;
         output_index < output_dimension;
         ++output_index) {
      const std::int64_t start = output_offsets[output_index];
      const std::int64_t finish = output_offsets[output_index + 1];
      for (std::int64_t coefficient = start;
           coefficient < finish;
           ++coefficient) {
        const std::int64_t term = coefficient_terms[coefficient];
        const Scalar coefficient_adjoint =
            conjugate(coefficient_values[coefficient]);
        for (std::int64_t seed = 0; seed < seed_count; ++seed) {
          monomial_adjoint[
              seed * monomial_count + term] +=
              output_adjoint[
                  (seed * batch_size + batch) *
                      output_dimension +
                  output_index] *
              coefficient_adjoint;
        }
      }
    }
    const Scalar* input_row = input + batch * input_dimension;
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      Scalar monomial = Scalar(1);
      std::int64_t zero_count = 0;
      std::int64_t zero_component = -1;
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        if (input_row[component] == Scalar(0)) {
          ++zero_count;
          zero_component = component;
          continue;
        }
        monomial *= integer_power(
            input_row[component],
            exponent);
      }
      if (zero_count >= 2) {
        continue;
      }
      if (zero_count == 1) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + zero_component];
        if (exponent != 1) {
          continue;
        }
        const Scalar derivative = conjugate(monomial);
        for (std::int64_t seed = 0; seed < seed_count; ++seed) {
          input_adjoint[
              (seed * batch_size + batch) *
                  input_dimension +
              zero_component] +=
              monomial_adjoint[
                  seed * monomial_count + term] *
              derivative;
        }
        continue;
      }
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[
                term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        const Scalar derivative = conjugate(
            Scalar(exponent) * monomial / input_row[component]);
        for (std::int64_t seed = 0; seed < seed_count; ++seed) {
          input_adjoint[
              (seed * batch_size + batch) *
                  input_dimension +
              component] +=
              monomial_adjoint[
                  seed * monomial_count + term] *
              derivative;
        }
      }
    }
  }
  (void)coefficient_count;
}

template <typename Scalar>
void factorized_angular_forward(
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    std::int64_t source_dimension,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t node_count,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    std::int64_t root_count,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t total_projection_dimension,
    std::int64_t output_dimension,
    Scalar* output) {
  const std::int64_t workspace_dimension =
      node_offsets[node_count - 1] + node_dimensions[node_count - 1];
  std::vector<Scalar> workspace(
      static_cast<std::size_t>(workspace_dimension)
  );
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    std::fill(workspace.begin(), workspace.end(), Scalar(0));
    Scalar* output_row = output + batch * output_dimension;
    std::fill(output_row, output_row + output_dimension, Scalar(0));
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      std::fill(workspace.begin(), workspace.end(), Scalar(0));
      const Scalar* input_row =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output =
            workspace.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          std::copy(
              input_row + leaf_offset,
              input_row + leaf_offset + node_dimensions[node],
              node_output);
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          node_output[column] +=
              conjugate(coefficient_values[entry]) *
              left[left_index] * right[right_index];
        }
      }
      for (std::int64_t root = 0; root < root_count; ++root) {
        const std::int64_t root_node = root_nodes[root];
        const std::int64_t root_dimension =
            node_dimensions[root_node];
        const std::int64_t projection_start =
            root_projection_starts[root];
        const std::int64_t projection_dimension =
            root_projection_dimensions[root];
        const Scalar* root_value =
            workspace.data() + node_offsets[root_node];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          Scalar* output_block =
              output_row +
              root_output_offsets[root] +
              projection * root_dimension;
          const Scalar projection_value =
              projection_values[
                  source * total_projection_dimension +
                  projection_start + projection];
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            output_block[magnetic] +=
                root_value[magnetic] * projection_value;
          }
        }
      }
    }
  }
}

namespace {

template <typename Scalar>
void sparse_monomial_linear_forward_adjoint_impl(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* factor_offsets,
    const std::int64_t* factor_indices,
    const std::int64_t* factor_exponents,
    const Scalar* monomial_coefficients,
    std::int64_t monomial_count,
    std::int64_t maximum_term_factors,
    Scalar* workspace,
    Scalar* output,
    Scalar* input_adjoint) {
  Scalar* powers = workspace;
  Scalar* power_derivatives = powers + maximum_term_factors;
  Scalar* prefix = power_derivatives + maximum_term_factors;
  Scalar* suffix = prefix + maximum_term_factors + 1;
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = input + batch * input_dimension;
    Scalar* input_gradient = input_adjoint + batch * input_dimension;
    std::fill(
        input_gradient,
        input_gradient + input_dimension,
        Scalar(0));
    Scalar value = Scalar(0);
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      const std::int64_t start = factor_offsets[term];
      const std::int64_t finish = factor_offsets[term + 1];
      const std::int64_t term_factor_count = finish - start;
      prefix[0] = Scalar(1);
      for (std::int64_t factor = start; factor < finish; ++factor) {
        const std::int64_t local = factor - start;
        const std::int64_t component = factor_indices[factor];
        const std::int64_t exponent = factor_exponents[factor];
        powers[local] = integer_power(input_row[component], exponent);
        power_derivatives[local] =
            Scalar(exponent) *
            integer_power(input_row[component], exponent - 1);
        prefix[local + 1] = prefix[local] * powers[local];
      }
      const Scalar coefficient = monomial_coefficients[term];
      value += coefficient * prefix[term_factor_count];
      suffix[term_factor_count] = Scalar(1);
      for (std::int64_t local = term_factor_count; local > 0; --local) {
        suffix[local - 1] = powers[local - 1] * suffix[local];
      }
      for (std::int64_t local = 0; local < term_factor_count; ++local) {
        const std::int64_t component = factor_indices[start + local];
        const Scalar derivative =
            prefix[local] * power_derivatives[local] * suffix[local + 1];
        input_gradient[component] +=
            conjugate(coefficient * derivative);
      }
    }
    output[batch] = value;
  }
}

}  // namespace

template <typename Scalar>
void symmetric_power_sparse_monomial_linear_forward_adjoint(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* factor_offsets,
    const std::int64_t* factor_indices,
    const std::int64_t* factor_exponents,
    std::int64_t factor_count,
    const Scalar* monomial_coefficients,
    std::int64_t monomial_count,
    Scalar* output,
    Scalar* input_adjoint) {
  if (batch_size < 0 || input_dimension <= 0 || monomial_count < 0 ||
      factor_count < 0) {
    throw std::invalid_argument(
        "Invalid sparse symmetric-power dimensions");
  }
  if (batch_size == 0) {
    return;
  }
  if (input == nullptr || factor_offsets == nullptr ||
      monomial_coefficients == nullptr || output == nullptr ||
      input_adjoint == nullptr ||
      (factor_count > 0 &&
       (factor_indices == nullptr || factor_exponents == nullptr))) {
    throw std::invalid_argument(
        "Sparse symmetric-power evaluation received a null array");
  }
  if (factor_offsets[0] != 0 ||
      factor_offsets[monomial_count] != factor_count) {
    throw std::invalid_argument(
        "Sparse symmetric-power factor offsets do not cover the factors");
  }
  std::int64_t maximum_term_factors = 0;
  for (std::int64_t term = 0; term < monomial_count; ++term) {
    if (factor_offsets[term] > factor_offsets[term + 1]) {
      throw std::invalid_argument(
          "Sparse symmetric-power factor offsets are not monotone");
    }
    for (std::int64_t factor = factor_offsets[term];
         factor < factor_offsets[term + 1]; ++factor) {
      if (factor_indices[factor] < 0 ||
          factor_indices[factor] >= input_dimension ||
          factor_exponents[factor] <= 0) {
        throw std::invalid_argument(
            "Sparse symmetric-power factor is out of range");
      }
    }
    maximum_term_factors = std::max(
        maximum_term_factors,
        factor_offsets[term + 1] - factor_offsets[term]);
  }

  std::vector<Scalar> workspace(
      static_cast<std::size_t>(4 * maximum_term_factors + 2));
  sparse_monomial_linear_forward_adjoint_impl(
      input, batch_size, input_dimension, factor_offsets, factor_indices,
      factor_exponents, monomial_coefficients, monomial_count,
      maximum_term_factors, workspace.data(), output, input_adjoint);
}

template <typename Scalar>
void symmetric_power_sparse_monomial_linear_forward_adjoint_prevalidated(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* factor_offsets,
    const std::int64_t* factor_indices,
    const std::int64_t* factor_exponents,
    std::int64_t factor_count,
    const Scalar* monomial_coefficients,
    std::int64_t monomial_count,
    std::int64_t maximum_term_factors,
    Scalar* workspace,
    std::int64_t workspace_size,
    Scalar* output,
    Scalar* input_adjoint) {
  if (batch_size < 0 || input_dimension <= 0 || monomial_count < 0 ||
      factor_count < 0 || maximum_term_factors < 0 ||
      workspace_size < 4 * maximum_term_factors + 2) {
    throw std::invalid_argument(
        "Invalid prevalidated sparse symmetric-power dimensions");
  }
  if (batch_size == 0) {
    return;
  }
  if (input == nullptr || factor_offsets == nullptr ||
      monomial_coefficients == nullptr || workspace == nullptr ||
      output == nullptr || input_adjoint == nullptr ||
      (factor_count > 0 &&
       (factor_indices == nullptr || factor_exponents == nullptr)) ||
      factor_offsets[0] != 0 ||
      factor_offsets[monomial_count] != factor_count) {
    throw std::invalid_argument(
        "Prevalidated sparse symmetric-power evaluation received an invalid array");
  }
  sparse_monomial_linear_forward_adjoint_impl(
      input, batch_size, input_dimension, factor_offsets, factor_indices,
      factor_exponents, monomial_coefficients, monomial_count,
      maximum_term_factors, workspace, output, input_adjoint);
}

template <typename Scalar, typename CoefficientScalar>
void symmetric_power_sparse_monomial_dag_linear_forward_adjoint_prevalidated(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* power_channels,
    const std::int64_t* power_exponents,
    std::int64_t power_count,
    const std::int64_t* node_parents,
    const std::int64_t* node_powers,
    std::int64_t node_count,
    std::int64_t root_node_count,
    const std::int64_t* monomial_nodes,
    const CoefficientScalar* monomial_coefficients,
    std::int64_t monomial_count,
    Scalar* workspace,
    std::int64_t workspace_size,
    Scalar* output,
    Scalar* input_adjoint) {
  const std::int64_t required_workspace =
      2 * power_count + 2 * node_count;
  if (batch_size < 0 || input_dimension <= 0 || power_count < 0 ||
      node_count <= 0 || root_node_count < 0 ||
      root_node_count >= node_count || monomial_count < 0 ||
      workspace_size < required_workspace) {
    throw std::invalid_argument(
        "Invalid prevalidated sparse symmetric-power DAG dimensions");
  }
  if (batch_size == 0) {
    return;
  }
  if (input == nullptr || power_channels == nullptr ||
      power_exponents == nullptr || node_parents == nullptr ||
      node_powers == nullptr || monomial_nodes == nullptr ||
      monomial_coefficients == nullptr || workspace == nullptr ||
      output == nullptr || input_adjoint == nullptr) {
    throw std::invalid_argument(
        "Prevalidated sparse symmetric-power DAG received a null array");
  }

  Scalar* power_values = workspace;
  Scalar* power_adjoint = power_values + power_count;
  Scalar* node_values = power_adjoint + power_count;
  Scalar* node_adjoint = node_values + node_count;

  // Algorithmic reference: reverse accumulation through an ACE product DAG.
  // Paper/reference: R. Drautz, Phys. Rev. B 99, 014104 (2019), ACE
  // polynomial basis construction. Independent product-rule implementation;
  // no external source code was copied or adapted.
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = input + batch * input_dimension;
    Scalar* input_gradient = input_adjoint + batch * input_dimension;
    std::fill(
        input_gradient,
        input_gradient + input_dimension,
        Scalar(0));
    std::fill(
        power_adjoint,
        power_adjoint + power_count,
        Scalar(0));
    std::fill(
        node_adjoint,
        node_adjoint + node_count,
        Scalar(0));

    for (std::int64_t power = 0; power < power_count; ++power) {
      power_values[power] = integer_power(
          input_row[power_channels[power]], power_exponents[power]);
    }
    node_values[0] = Scalar(1);
    for (std::int64_t node = 1; node <= root_node_count; ++node)
      node_values[node] = power_values[node_powers[node]];
    for (std::int64_t node = root_node_count + 1;
         node < node_count;
         ++node) {
      node_values[node] =
          node_values[node_parents[node]] * power_values[node_powers[node]];
    }

    Scalar value = Scalar(0);
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      const std::int64_t node = monomial_nodes[term];
      const CoefficientScalar coefficient = monomial_coefficients[term];
      value += coefficient * node_values[node];
      node_adjoint[node] += conjugate(coefficient);
    }
    output[batch] = value;

    for (std::int64_t node = node_count - 1;
         node > root_node_count;
         --node) {
      const std::int64_t parent = node_parents[node];
      const std::int64_t power = node_powers[node];
      const Scalar gradient = node_adjoint[node];
      node_adjoint[parent] += gradient * conjugate(power_values[power]);
      power_adjoint[power] += gradient * conjugate(node_values[parent]);
    }
    for (std::int64_t node = root_node_count; node > 0; --node)
      power_adjoint[node_powers[node]] += node_adjoint[node];
    for (std::int64_t power = 0; power < power_count; ++power) {
      const std::int64_t channel = power_channels[power];
      const std::int64_t exponent = power_exponents[power];
      input_gradient[channel] += power_adjoint[power] * conjugate(
          Scalar(exponent) * integer_power(input_row[channel], exponent - 1));
    }
  }
}

template <typename Scalar, typename CoefficientScalar>
void symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_prevalidated(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* power_channels,
    const std::int64_t* power_exponents,
    std::int64_t power_count,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    std::int64_t node_count,
    const std::int64_t* monomial_operands,
    const CoefficientScalar* monomial_coefficients,
    std::int64_t monomial_count,
    Scalar* workspace,
    std::int64_t workspace_size,
    Scalar* output,
    Scalar* input_adjoint) {
  const std::int64_t value_count = power_count + node_count;
  const std::int64_t required_workspace = 2 * value_count;
  if (batch_size < 0 || input_dimension <= 0 || power_count < 0 ||
      node_count < 0 || monomial_count < 0 ||
      workspace_size < required_workspace) {
    throw std::invalid_argument(
        "Invalid prevalidated sparse symmetric-power binary DAG dimensions");
  }
  if (batch_size == 0) {
    return;
  }
  if (input == nullptr || power_channels == nullptr ||
      power_exponents == nullptr || node_left == nullptr ||
      node_right == nullptr || monomial_operands == nullptr ||
      monomial_coefficients == nullptr || workspace == nullptr ||
      output == nullptr || input_adjoint == nullptr) {
    throw std::invalid_argument(
        "Prevalidated sparse symmetric-power binary DAG received a null array");
  }

  Scalar* values = workspace;
  Scalar* adjoints = values + value_count;
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar* input_row = input + batch * input_dimension;
    Scalar* input_gradient = input_adjoint + batch * input_dimension;
    std::fill(
        input_gradient,
        input_gradient + input_dimension,
        Scalar(0));
    std::fill(adjoints, adjoints + value_count, Scalar(0));

    for (std::int64_t power = 0; power < power_count; ++power) {
      values[power] = integer_power(
          input_row[power_channels[power]], power_exponents[power]);
    }
    for (std::int64_t node = 0; node < node_count; ++node) {
      values[power_count + node] =
          values[node_left[node]] * values[node_right[node]];
    }

    Scalar value = Scalar(0);
    for (std::int64_t term = 0; term < monomial_count; ++term) {
      const std::int64_t operand = monomial_operands[term];
      const CoefficientScalar coefficient = monomial_coefficients[term];
      if (operand < 0) {
        value += coefficient;
      } else {
        value += coefficient * values[operand];
        adjoints[operand] += conjugate(coefficient);
      }
    }
    output[batch] = value;

    for (std::int64_t node = node_count - 1; node >= 0; --node) {
      const std::int64_t output_operand = power_count + node;
      const std::int64_t left = node_left[node];
      const std::int64_t right = node_right[node];
      const Scalar gradient = adjoints[output_operand];
      adjoints[left] += gradient * conjugate(values[right]);
      adjoints[right] += gradient * conjugate(values[left]);
    }
    for (std::int64_t power = 0; power < power_count; ++power) {
      const std::int64_t channel = power_channels[power];
      const std::int64_t exponent = power_exponents[power];
      input_gradient[channel] +=
          adjoints[power] *
          conjugate(Scalar(exponent) *
                    integer_power(input_row[channel], exponent - 1));
    }
  }
}

template <typename Real>
void symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
    const std::complex<Real> *input, std::int64_t batch_size,
    std::int64_t input_dimension, const std::int64_t *power_channels,
    const std::int64_t *power_exponents, std::int64_t power_count,
    const std::int64_t *node_left, const std::int64_t *node_right,
    std::int64_t node_count, const std::int64_t *monomial_operands,
    const Real *monomial_coefficients, std::int64_t monomial_count,
    std::int64_t tile_size, Real *workspace, std::int64_t workspace_size,
    std::complex<Real> *output, std::complex<Real> *input_adjoint) {
  if (batch_size < 0 || input_dimension <= 0 || power_count < 0 ||
      node_count < 0 || monomial_count < 0 || tile_size <= 0) {
    throw std::invalid_argument(
        "Invalid prevalidated split-real tiled binary DAG dimensions");
  }
  const std::int64_t value_count = power_count + node_count;
  const std::int64_t tile_value_count = value_count * tile_size;
  const std::int64_t required_workspace = 4 * tile_value_count + 2 * tile_size;
  if (workspace_size < required_workspace) {
    throw std::invalid_argument(
        "Invalid prevalidated split-real tiled binary DAG workspace");
  }
  if (batch_size == 0) {
    return;
  }
  if (input == nullptr || power_channels == nullptr ||
      power_exponents == nullptr || node_left == nullptr ||
      node_right == nullptr || monomial_operands == nullptr ||
      monomial_coefficients == nullptr || workspace == nullptr ||
      output == nullptr || input_adjoint == nullptr) {
    throw std::invalid_argument(
        "Prevalidated split-real tiled binary DAG received a null array");
  }

  Real *values_real = workspace;
  Real *values_imaginary = values_real + tile_value_count;
  Real *adjoints_real = values_imaginary + tile_value_count;
  Real *adjoints_imaginary = adjoints_real + tile_value_count;
  Real *output_real = adjoints_imaginary + tile_value_count;
  Real *output_imaginary = output_real + tile_size;

  // Algorithmic reference: reverse accumulation through an ACE product DAG.
  // Paper/reference: R. Drautz, Phys. Rev. B 99, 014104 (2019), ACE
  // polynomial basis construction. Independent split-real implementation;
  // no external source code was copied or adapted.
  std::fill(input_adjoint, input_adjoint + batch_size * input_dimension,
            std::complex<Real>(Real(0), Real(0)));
  for (std::int64_t tile_begin = 0; tile_begin < batch_size;
       tile_begin += tile_size) {
    const std::int64_t lane_count =
        std::min(tile_size, batch_size - tile_begin);
    std::fill(adjoints_real, adjoints_real + tile_value_count, Real(0));
    std::fill(adjoints_imaginary, adjoints_imaginary + tile_value_count,
              Real(0));
    std::fill(output_real, output_real + lane_count, Real(0));
    std::fill(output_imaginary, output_imaginary + lane_count, Real(0));

    for (std::int64_t power = 0; power < power_count; ++power) {
      const std::int64_t offset = power * tile_size;
      const std::int64_t channel = power_channels[power];
      const std::int64_t exponent = power_exponents[power];
      YE3T_LANE_LOOP
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const std::complex<Real> value = integer_power(
            input[(tile_begin + lane) * input_dimension + channel], exponent);
        values_real[offset + lane] = value.real();
        values_imaginary[offset + lane] = value.imag();
      }
    }
    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t output_offset = (power_count + node) * tile_size;
      const std::int64_t left_offset = node_left[node] * tile_size;
      const std::int64_t right_offset = node_right[node] * tile_size;
      YE3T_LANE_LOOP
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const Real left_real = values_real[left_offset + lane];
        const Real left_imaginary = values_imaginary[left_offset + lane];
        const Real right_real = values_real[right_offset + lane];
        const Real right_imaginary = values_imaginary[right_offset + lane];
        values_real[output_offset + lane] =
            left_real * right_real - left_imaginary * right_imaginary;
        values_imaginary[output_offset + lane] =
            left_real * right_imaginary + left_imaginary * right_real;
      }
    }

    for (std::int64_t term = 0; term < monomial_count; ++term) {
      const std::int64_t operand = monomial_operands[term];
      const Real coefficient = monomial_coefficients[term];
      if (operand < 0) {
        YE3T_LANE_LOOP
        for (std::int64_t lane = 0; lane < lane_count; ++lane)
          output_real[lane] += coefficient;
        continue;
      }
      const std::int64_t offset = operand * tile_size;
      YE3T_LANE_LOOP
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        output_real[lane] += coefficient * values_real[offset + lane];
        output_imaginary[lane] += coefficient * values_imaginary[offset + lane];
        adjoints_real[offset + lane] += coefficient;
      }
    }
    for (std::int64_t lane = 0; lane < lane_count; ++lane) {
      output[tile_begin + lane] =
          std::complex<Real>(output_real[lane], output_imaginary[lane]);
    }

    for (std::int64_t node = node_count - 1; node >= 0; --node) {
      const std::int64_t output_offset = (power_count + node) * tile_size;
      const std::int64_t left_offset = node_left[node] * tile_size;
      const std::int64_t right_offset = node_right[node] * tile_size;
      YE3T_LANE_LOOP
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const Real gradient_real = adjoints_real[output_offset + lane];
        const Real gradient_imaginary =
            adjoints_imaginary[output_offset + lane];
        const Real left_real = values_real[left_offset + lane];
        const Real left_imaginary = values_imaginary[left_offset + lane];
        const Real right_real = values_real[right_offset + lane];
        const Real right_imaginary = values_imaginary[right_offset + lane];
        adjoints_real[left_offset + lane] +=
            gradient_real * right_real + gradient_imaginary * right_imaginary;
        adjoints_imaginary[left_offset + lane] +=
            gradient_imaginary * right_real - gradient_real * right_imaginary;
        adjoints_real[right_offset + lane] +=
            gradient_real * left_real + gradient_imaginary * left_imaginary;
        adjoints_imaginary[right_offset + lane] +=
            gradient_imaginary * left_real - gradient_real * left_imaginary;
      }
    }
    for (std::int64_t power = 0; power < power_count; ++power) {
      const std::int64_t offset = power * tile_size;
      const std::int64_t channel = power_channels[power];
      const std::int64_t exponent = power_exponents[power];
      YE3T_LANE_LOOP
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        const std::complex<Real> derivative =
            Real(exponent) *
            integer_power(
                input[(tile_begin + lane) * input_dimension + channel],
                exponent - 1);
        const Real gradient_real = adjoints_real[offset + lane];
        const Real gradient_imaginary = adjoints_imaginary[offset + lane];
        input_adjoint[(tile_begin + lane) * input_dimension + channel] +=
            std::complex<Real>(gradient_real * derivative.real() +
                                   gradient_imaginary * derivative.imag(),
                               gradient_imaginary * derivative.real() -
                                   gradient_real * derivative.imag());
      }
    }
  }
}

template <typename Real>
void ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
    const std::complex<Real> *input, std::int64_t batch_size,
    std::int64_t input_stride, const std::int64_t *node_offsets,
    const std::int64_t *node_dimensions, const std::int64_t *node_leaf_offsets,
    const std::int64_t *leaf_input_components,
    std::int64_t leaf_component_count,
    const std::int64_t *node_coefficient_offsets, std::int64_t node_count,
    const std::int64_t *coefficient_left_components,
    const std::int64_t *coefficient_right_components,
    const std::int64_t *coefficient_output_components,
    const Real *coefficient_values, std::int64_t coefficient_count,
    const std::int64_t *readout_components, const Real *readout_coefficients,
    std::int64_t readout_count, std::int64_t tile_size, Real *workspace,
    std::int64_t workspace_size, std::complex<Real> *output,
    std::complex<Real> *input_adjoint) {
  if (batch_size < 0 || input_stride <= 0 || leaf_component_count < 0 ||
      node_count <= 0 || coefficient_count < 0 || readout_count <= 0 ||
      tile_size <= 0) {
    throw std::invalid_argument(
        "Invalid prevalidated ACE coupled-product DAG dimensions");
  }
  if (node_offsets == nullptr || node_dimensions == nullptr ||
      node_leaf_offsets == nullptr || node_coefficient_offsets == nullptr) {
    throw std::invalid_argument(
        "Prevalidated ACE coupled-product DAG received a null plan array");
  }
  const std::int64_t total_node_components =
      node_offsets[node_count - 1] + node_dimensions[node_count - 1];
  const std::int64_t tile_component_count = total_node_components * tile_size;
  const std::int64_t required_workspace =
      4 * tile_component_count + 2 * tile_size;
  if (total_node_components <= 0 || workspace_size < required_workspace) {
    throw std::invalid_argument(
        "Invalid prevalidated ACE coupled-product DAG workspace");
  }
  if (batch_size == 0)
    return;
  if (input == nullptr || readout_components == nullptr ||
      readout_coefficients == nullptr || workspace == nullptr ||
      output == nullptr || input_adjoint == nullptr ||
      (leaf_component_count > 0 && leaf_input_components == nullptr) ||
      (coefficient_count > 0 && (coefficient_left_components == nullptr ||
                                 coefficient_right_components == nullptr ||
                                 coefficient_output_components == nullptr ||
                                 coefficient_values == nullptr))) {
    throw std::invalid_argument(
        "Prevalidated ACE coupled-product DAG received a null array");
  }

  Real *values_real = workspace;
  Real *values_imaginary = values_real + tile_component_count;
  Real *adjoints_real = values_imaginary + tile_component_count;
  Real *adjoints_imaginary = adjoints_real + tile_component_count;
  Real *output_real = adjoints_imaginary + tile_component_count;
  Real *output_imaginary = output_real + tile_size;

  // Algorithmic reference: reverse accumulation through an ACE product DAG.
  // Paper/reference: R. Drautz, Phys. Rev. B 99, 014104 (2019).
  // Independent split-real implementation; no external code was adapted.
  for (std::int64_t tile_begin = 0; tile_begin < batch_size;
       tile_begin += tile_size) {
    const std::int64_t lane_count =
        std::min(tile_size, batch_size - tile_begin);
    std::fill(adjoints_real, adjoints_real + tile_component_count, Real(0));
    std::fill(adjoints_imaginary, adjoints_imaginary + tile_component_count,
              Real(0));
    std::fill(output_real, output_real + lane_count, Real(0));
    std::fill(output_imaginary, output_imaginary + lane_count, Real(0));

    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t node_offset = node_offsets[node];
      const std::int64_t node_dimension = node_dimensions[node];
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset >= 0) {
        for (std::int64_t component = 0; component < node_dimension;
             ++component) {
          const std::int64_t value_offset =
              (node_offset + component) * tile_size;
          const std::int64_t input_component =
              leaf_input_components[leaf_offset + component];
          for (std::int64_t lane = 0; lane < lane_count; ++lane) {
            const std::complex<Real> value =
                input[(tile_begin + lane) * input_stride + input_component];
            values_real[value_offset + lane] = value.real();
            values_imaginary[value_offset + lane] = value.imag();
          }
        }
        continue;
      }

      for (std::int64_t component = 0; component < node_dimension;
           ++component) {
        const std::int64_t value_offset = (node_offset + component) * tile_size;
        std::fill(values_real + value_offset,
                  values_real + value_offset + lane_count, Real(0));
        std::fill(values_imaginary + value_offset,
                  values_imaginary + value_offset + lane_count, Real(0));
      }
      for (std::int64_t entry = node_coefficient_offsets[node];
           entry < node_coefficient_offsets[node + 1]; ++entry) {
        const std::int64_t left_offset =
            coefficient_left_components[entry] * tile_size;
        const std::int64_t right_offset =
            coefficient_right_components[entry] * tile_size;
        const std::int64_t result_offset =
            coefficient_output_components[entry] * tile_size;
        const Real coefficient = coefficient_values[entry];
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          const Real left_real = values_real[left_offset + lane];
          const Real left_imaginary = values_imaginary[left_offset + lane];
          const Real right_real = values_real[right_offset + lane];
          const Real right_imaginary = values_imaginary[right_offset + lane];
          values_real[result_offset + lane] +=
              coefficient *
              (left_real * right_real - left_imaginary * right_imaginary);
          values_imaginary[result_offset + lane] +=
              coefficient *
              (left_real * right_imaginary + left_imaginary * right_real);
        }
      }
    }

    for (std::int64_t term = 0; term < readout_count; ++term) {
      const std::int64_t value_offset = readout_components[term] * tile_size;
      const Real coefficient = readout_coefficients[term];
      for (std::int64_t lane = 0; lane < lane_count; ++lane) {
        output_real[lane] += coefficient * values_real[value_offset + lane];
        output_imaginary[lane] +=
            coefficient * values_imaginary[value_offset + lane];
        adjoints_real[value_offset + lane] += coefficient;
      }
    }
    for (std::int64_t lane = 0; lane < lane_count; ++lane) {
      output[tile_begin + lane] +=
          std::complex<Real>(output_real[lane], output_imaginary[lane]);
    }

    for (std::int64_t node = node_count; node-- > 0;) {
      if (node_leaf_offsets[node] >= 0)
        continue;
      for (std::int64_t entry = node_coefficient_offsets[node + 1];
           entry-- > node_coefficient_offsets[node];) {
        const std::int64_t left_offset =
            coefficient_left_components[entry] * tile_size;
        const std::int64_t right_offset =
            coefficient_right_components[entry] * tile_size;
        const std::int64_t result_offset =
            coefficient_output_components[entry] * tile_size;
        const Real coefficient = coefficient_values[entry];
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          const Real gradient_real =
              coefficient * adjoints_real[result_offset + lane];
          const Real gradient_imaginary =
              coefficient * adjoints_imaginary[result_offset + lane];
          const Real left_real = values_real[left_offset + lane];
          const Real left_imaginary = values_imaginary[left_offset + lane];
          const Real right_real = values_real[right_offset + lane];
          const Real right_imaginary = values_imaginary[right_offset + lane];
          adjoints_real[left_offset + lane] +=
              gradient_real * right_real + gradient_imaginary * right_imaginary;
          adjoints_imaginary[left_offset + lane] +=
              gradient_imaginary * right_real - gradient_real * right_imaginary;
          adjoints_real[right_offset + lane] +=
              gradient_real * left_real + gradient_imaginary * left_imaginary;
          adjoints_imaginary[right_offset + lane] +=
              gradient_imaginary * left_real - gradient_real * left_imaginary;
        }
      }
    }
    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0)
        continue;
      for (std::int64_t component = 0; component < node_dimensions[node];
           ++component) {
        const std::int64_t value_offset =
            (node_offsets[node] + component) * tile_size;
        const std::int64_t input_component =
            leaf_input_components[leaf_offset + component];
        for (std::int64_t lane = 0; lane < lane_count; ++lane) {
          input_adjoint[(tile_begin + lane) * input_stride + input_component] +=
              std::complex<Real>(adjoints_real[value_offset + lane],
                                 adjoints_imaginary[value_offset + lane]);
        }
      }
    }
  }
}

template <typename Scalar>
void factorized_angular_adjoint(
    const Scalar *output_adjoint, const Scalar *packed_slots,
    std::int64_t batch_size, std::int64_t input_dimension,
    std::int64_t source_dimension, const std::int64_t *node_offsets,
    const std::int64_t *node_dimensions, const std::int64_t *node_leaf_offsets,
    const std::int64_t *node_left, const std::int64_t *node_right,
    const std::int64_t *node_coefficient_offsets, std::int64_t node_count,
    const std::int64_t *coefficient_rows,
    const std::int64_t *coefficient_columns, const Scalar *coefficient_values,
    const std::int64_t *root_nodes, std::int64_t root_count,
    const std::int64_t *root_projection_starts,
    const std::int64_t *root_projection_dimensions,
    const std::int64_t *root_output_offsets, const Scalar *projection_values,
    std::int64_t total_projection_dimension, std::int64_t output_dimension,
    Scalar *packed_slots_adjoint) {
  const std::int64_t workspace_dimension =
      node_offsets[node_count - 1] + node_dimensions[node_count - 1];
  std::vector<Scalar> workspace(static_cast<std::size_t>(workspace_dimension));
  std::vector<Scalar> workspace_adjoint(
      static_cast<std::size_t>(workspace_dimension));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    std::fill(workspace.begin(), workspace.end(), Scalar(0));
    std::fill(
        workspace_adjoint.begin(),
        workspace_adjoint.end(),
        Scalar(0));
    const Scalar* output_row =
        output_adjoint + batch * output_dimension;
    Scalar* input_adjoint =
        packed_slots_adjoint +
        batch * source_dimension * input_dimension;
    std::fill(
        input_adjoint,
        input_adjoint + source_dimension * input_dimension,
        Scalar(0));
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      std::fill(workspace.begin(), workspace.end(), Scalar(0));
      std::fill(
          workspace_adjoint.begin(),
          workspace_adjoint.end(),
          Scalar(0));
      const Scalar* input_source =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output =
            workspace.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          std::copy(
              input_source + leaf_offset,
              input_source + leaf_offset + node_dimensions[node],
              node_output);
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          node_output[column] +=
              conjugate(coefficient_values[entry]) *
              left[left_index] * right[right_index];
        }
      }
      for (std::int64_t root = 0; root < root_count; ++root) {
        const std::int64_t root_node = root_nodes[root];
        const std::int64_t root_dimension =
            node_dimensions[root_node];
        const std::int64_t projection_start =
            root_projection_starts[root];
        const std::int64_t projection_dimension =
            root_projection_dimensions[root];
        Scalar* root_adjoint =
            workspace_adjoint.data() + node_offsets[root_node];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          const Scalar* output_block =
              output_row +
              root_output_offsets[root] +
              projection * root_dimension;
          const Scalar projection_value =
              projection_values[
                  source * total_projection_dimension +
                  projection_start + projection];
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            root_adjoint[magnetic] +=
                output_block[magnetic] *
                conjugate(projection_value);
          }
        }
      }
      for (std::int64_t node = node_count; node-- > 0;) {
        Scalar* node_adjoint =
            workspace_adjoint.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        Scalar* left_adjoint =
            workspace_adjoint.data() + node_offsets[left_node];
        Scalar* right_adjoint =
            workspace_adjoint.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          const Scalar gradient = node_adjoint[column];
          left_adjoint[left_index] +=
              gradient * coefficient_values[entry] *
              conjugate(right[right_index]);
          right_adjoint[right_index] +=
              gradient * coefficient_values[entry] *
              conjugate(left[left_index]);
        }
      }
      Scalar* input_source_adjoint =
          input_adjoint + source * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset < 0) {
          continue;
        }
        const Scalar* leaf_adjoint =
            workspace_adjoint.data() + node_offsets[node];
        for (
            std::int64_t index = 0;
            index < node_dimensions[node];
            ++index) {
          input_source_adjoint[leaf_offset + index] +=
              leaf_adjoint[index];
        }
      }
    }
  }
}

template <typename Scalar>
void factorized_angular_double_backward(
    const Scalar* packed_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    std::int64_t source_dimension,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t node_count,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    std::int64_t root_count,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t total_projection_dimension,
    std::int64_t output_dimension,
    Scalar* output_tangent,
    Scalar* packed_slots_tangent) {
  const std::int64_t workspace_dimension =
      node_offsets[node_count - 1] + node_dimensions[node_count - 1];
  std::vector<Scalar> workspace(
      static_cast<std::size_t>(workspace_dimension));
  std::vector<Scalar> workspace_adjoint(
      static_cast<std::size_t>(workspace_dimension));
  std::vector<Scalar> adjoint_tangent(
      static_cast<std::size_t>(workspace_dimension));
  std::vector<Scalar> primal_tangent(
      static_cast<std::size_t>(workspace_dimension));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    Scalar* output_tangent_row =
        output_tangent + batch * output_dimension;
    Scalar* packed_tangent_row =
        packed_slots_tangent +
        batch * source_dimension * input_dimension;
    std::fill(
        output_tangent_row,
        output_tangent_row + output_dimension,
        Scalar(0));
    std::fill(
        packed_tangent_row,
        packed_tangent_row + source_dimension * input_dimension,
        Scalar(0));
    const Scalar* output_adjoint_row =
        output_adjoint + batch * output_dimension;
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      std::fill(workspace.begin(), workspace.end(), Scalar(0));
      std::fill(
          workspace_adjoint.begin(),
          workspace_adjoint.end(),
          Scalar(0));
      std::fill(
          adjoint_tangent.begin(),
          adjoint_tangent.end(),
          Scalar(0));
      std::fill(
          primal_tangent.begin(),
          primal_tangent.end(),
          Scalar(0));
      const Scalar* input_source =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      const Scalar* input_adjoint_tangent =
          packed_adjoint_tangent +
          (batch * source_dimension + source) * input_dimension;

      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output =
            workspace.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          std::copy(
              input_source + leaf_offset,
              input_source + leaf_offset + node_dimensions[node],
              node_output);
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          node_output[column] +=
              conjugate(coefficient_values[entry]) *
              left[row / right_dimension] *
              right[row % right_dimension];
        }
      }

      for (std::int64_t root = 0; root < root_count; ++root) {
        const std::int64_t root_node = root_nodes[root];
        const std::int64_t root_dimension =
            node_dimensions[root_node];
        const std::int64_t projection_start =
            root_projection_starts[root];
        const std::int64_t projection_dimension =
            root_projection_dimensions[root];
        Scalar* root_adjoint =
            workspace_adjoint.data() + node_offsets[root_node];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          const Scalar* output_block =
              output_adjoint_row +
              root_output_offsets[root] +
              projection * root_dimension;
          const Scalar projection_value =
              projection_values[
                  source * total_projection_dimension +
                  projection_start + projection];
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            root_adjoint[magnetic] +=
                output_block[magnetic] *
                conjugate(projection_value);
          }
        }
      }

      for (std::int64_t node = node_count; node-- > 0;) {
        if (node_leaf_offsets[node] >= 0) {
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        Scalar* left_adjoint =
            workspace_adjoint.data() + node_offsets[left_node];
        Scalar* right_adjoint =
            workspace_adjoint.data() + node_offsets[right_node];
        const Scalar* node_adjoint =
            workspace_adjoint.data() + node_offsets[node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          const Scalar gradient = node_adjoint[column];
          left_adjoint[left_index] +=
              gradient * coefficient_values[entry] *
              conjugate(right[right_index]);
          right_adjoint[right_index] +=
              gradient * coefficient_values[entry] *
              conjugate(left[left_index]);
        }
      }

      for (std::int64_t node = 0; node < node_count; ++node) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset < 0) {
          continue;
        }
        Scalar* leaf_tangent =
            adjoint_tangent.data() + node_offsets[node];
        for (
            std::int64_t component = 0;
            component < node_dimensions[node];
            ++component) {
          leaf_tangent[component] +=
              input_adjoint_tangent[leaf_offset + component];
        }
      }

      for (std::int64_t node = 0; node < node_count; ++node) {
        if (node_leaf_offsets[node] >= 0) {
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        const Scalar* node_adjoint =
            workspace_adjoint.data() + node_offsets[node];
        Scalar* node_adjoint_tangent =
            adjoint_tangent.data() + node_offsets[node];
        const Scalar* left_adjoint_tangent =
            adjoint_tangent.data() + node_offsets[left_node];
        const Scalar* right_adjoint_tangent =
            adjoint_tangent.data() + node_offsets[right_node];
        Scalar* left_primal_tangent =
            primal_tangent.data() + node_offsets[left_node];
        Scalar* right_primal_tangent =
            primal_tangent.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          const Scalar coefficient = coefficient_values[entry];
          node_adjoint_tangent[column] +=
              conjugate(coefficient) *
              (
                  left_adjoint_tangent[left_index] *
                      right[right_index] +
                  right_adjoint_tangent[right_index] *
                      left[left_index]);
          left_primal_tangent[left_index] +=
              conjugate(right_adjoint_tangent[right_index]) *
              node_adjoint[column] * coefficient;
          right_primal_tangent[right_index] +=
              conjugate(left_adjoint_tangent[left_index]) *
              node_adjoint[column] * coefficient;
        }
      }

      for (std::int64_t root = 0; root < root_count; ++root) {
        const std::int64_t root_node = root_nodes[root];
        const std::int64_t root_dimension =
            node_dimensions[root_node];
        const std::int64_t projection_start =
            root_projection_starts[root];
        const std::int64_t projection_dimension =
            root_projection_dimensions[root];
        const Scalar* root_adjoint_tangent =
            adjoint_tangent.data() + node_offsets[root_node];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          Scalar* output_block =
              output_tangent_row +
              root_output_offsets[root] +
              projection * root_dimension;
          const Scalar projection_value =
              projection_values[
                  source * total_projection_dimension +
                  projection_start + projection];
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            output_block[magnetic] +=
                root_adjoint_tangent[magnetic] * projection_value;
          }
        }
      }

      for (std::int64_t node = node_count; node-- > 0;) {
        if (node_leaf_offsets[node] >= 0) {
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            workspace.data() + node_offsets[left_node];
        const Scalar* right =
            workspace.data() + node_offsets[right_node];
        const Scalar* node_tangent =
            primal_tangent.data() + node_offsets[node];
        Scalar* left_tangent =
            primal_tangent.data() + node_offsets[left_node];
        Scalar* right_tangent =
            primal_tangent.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          const Scalar gradient = node_tangent[column];
          left_tangent[left_index] +=
              gradient * coefficient_values[entry] *
              conjugate(right[right_index]);
          right_tangent[right_index] +=
              gradient * coefficient_values[entry] *
              conjugate(left[left_index]);
        }
      }

      Scalar* input_tangent =
          packed_tangent_row + source * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset < 0) {
          continue;
        }
        const Scalar* leaf_tangent =
            primal_tangent.data() + node_offsets[node];
        for (
            std::int64_t component = 0;
            component < node_dimensions[node];
            ++component) {
          input_tangent[leaf_offset + component] +=
              leaf_tangent[component];
        }
      }
    }
  }
}

template <typename Scalar>
void factorized_angular_segmented_forward(
    const Scalar* packed_slots,
    std::int64_t batch_size,
    const FactorizedAngularSegmentedPlanView<Scalar>& plan,
    Scalar* output) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    for (std::int64_t segment = 0;
         segment < plan.segment_count;
         ++segment) {
      const std::int64_t source_dimension =
          plan.segment_source_offsets[segment + 1] -
          plan.segment_source_offsets[segment];
      const std::int64_t node_start =
          plan.segment_node_offsets[segment];
      const std::int64_t root_start =
          plan.segment_root_offsets[segment];
      const std::int64_t output_start =
          plan.segment_output_offsets[segment];
      factorized_angular_forward<Scalar>(
          packed_slots +
              batch * plan.packed_input_dimension +
              plan.segment_input_offsets[segment],
          1,
          plan.segment_input_dimensions[segment],
          source_dimension,
          plan.node_offsets + node_start,
          plan.node_dimensions + node_start,
          plan.node_leaf_offsets + node_start,
          plan.node_left + node_start,
          plan.node_right + node_start,
          plan.node_coefficient_offsets + node_start,
          plan.segment_node_offsets[segment + 1] - node_start,
          plan.coefficient_rows,
          plan.coefficient_columns,
          plan.coefficient_values,
          plan.root_nodes + root_start,
          plan.segment_root_offsets[segment + 1] - root_start,
          plan.root_projection_starts + root_start,
          plan.root_projection_dimensions + root_start,
          plan.root_output_offsets + root_start,
          plan.projection_values +
              plan.segment_projection_offsets[segment],
          plan.segment_projection_dimensions[segment],
          plan.segment_output_offsets[segment + 1] - output_start,
          output + batch * plan.output_dimension + output_start);
    }
  }
}

template <typename Scalar>
void factorized_angular_segmented_adjoint(
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    const FactorizedAngularSegmentedPlanView<Scalar>& plan,
    Scalar* packed_slots_adjoint) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    for (std::int64_t segment = 0;
         segment < plan.segment_count;
         ++segment) {
      const std::int64_t source_dimension =
          plan.segment_source_offsets[segment + 1] -
          plan.segment_source_offsets[segment];
      const std::int64_t node_start =
          plan.segment_node_offsets[segment];
      const std::int64_t root_start =
          plan.segment_root_offsets[segment];
      const std::int64_t output_start =
          plan.segment_output_offsets[segment];
      factorized_angular_adjoint<Scalar>(
          output_adjoint +
              batch * plan.output_dimension + output_start,
          packed_slots +
              batch * plan.packed_input_dimension +
              plan.segment_input_offsets[segment],
          1,
          plan.segment_input_dimensions[segment],
          source_dimension,
          plan.node_offsets + node_start,
          plan.node_dimensions + node_start,
          plan.node_leaf_offsets + node_start,
          plan.node_left + node_start,
          plan.node_right + node_start,
          plan.node_coefficient_offsets + node_start,
          plan.segment_node_offsets[segment + 1] - node_start,
          plan.coefficient_rows,
          plan.coefficient_columns,
          plan.coefficient_values,
          plan.root_nodes + root_start,
          plan.segment_root_offsets[segment + 1] - root_start,
          plan.root_projection_starts + root_start,
          plan.root_projection_dimensions + root_start,
          plan.root_output_offsets + root_start,
          plan.projection_values +
              plan.segment_projection_offsets[segment],
          plan.segment_projection_dimensions[segment],
          plan.segment_output_offsets[segment + 1] - output_start,
          packed_slots_adjoint +
              batch * plan.packed_input_dimension +
              plan.segment_input_offsets[segment]);
    }
  }
}

template <typename Scalar>
void factorized_angular_segmented_double_backward(
    const Scalar* packed_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    const FactorizedAngularSegmentedPlanView<Scalar>& plan,
    Scalar* output_tangent,
    Scalar* packed_slots_tangent) {
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    for (std::int64_t segment = 0;
         segment < plan.segment_count;
         ++segment) {
      const std::int64_t source_dimension =
          plan.segment_source_offsets[segment + 1] -
          plan.segment_source_offsets[segment];
      const std::int64_t node_start =
          plan.segment_node_offsets[segment];
      const std::int64_t root_start =
          plan.segment_root_offsets[segment];
      const std::int64_t output_start =
          plan.segment_output_offsets[segment];
      factorized_angular_double_backward<Scalar>(
          packed_adjoint_tangent +
              batch * plan.packed_input_dimension +
              plan.segment_input_offsets[segment],
          output_adjoint +
              batch * plan.output_dimension + output_start,
          packed_slots +
              batch * plan.packed_input_dimension +
              plan.segment_input_offsets[segment],
          1,
          plan.segment_input_dimensions[segment],
          source_dimension,
          plan.node_offsets + node_start,
          plan.node_dimensions + node_start,
          plan.node_leaf_offsets + node_start,
          plan.node_left + node_start,
          plan.node_right + node_start,
          plan.node_coefficient_offsets + node_start,
          plan.segment_node_offsets[segment + 1] - node_start,
          plan.coefficient_rows,
          plan.coefficient_columns,
          plan.coefficient_values,
          plan.root_nodes + root_start,
          plan.segment_root_offsets[segment + 1] - root_start,
          plan.root_projection_starts + root_start,
          plan.root_projection_dimensions + root_start,
          plan.root_output_offsets + root_start,
          plan.projection_values +
              plan.segment_projection_offsets[segment],
          plan.segment_projection_dimensions[segment],
          plan.segment_output_offsets[segment + 1] - output_start,
          output_tangent +
              batch * plan.output_dimension + output_start,
          packed_slots_tangent +
              batch * plan.packed_input_dimension +
              plan.segment_input_offsets[segment]);
    }
  }
}

template <typename Scalar>
void factorized_angular_linear_forward(
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    std::int64_t source_dimension,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t node_count,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    std::int64_t root_count,
    const Scalar* projection_values,
    std::int64_t projection_dimension,
    const Scalar* weight,
    Scalar bias,
    Scalar* output) {
  const std::int64_t workspace_dimension =
      node_offsets[node_count - 1] + node_dimensions[node_count - 1];
  const std::int64_t root_dimension =
      node_dimensions[root_nodes[0]];
  std::vector<Scalar> workspace(
      static_cast<std::size_t>(workspace_dimension));
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    Scalar value = bias;
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      std::fill(workspace.begin(), workspace.end(), Scalar(0));
      const Scalar* input_row =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output = workspace.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          std::copy(
              input_row + leaf_offset,
              input_row + leaf_offset + node_dimensions[node],
              node_output);
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left = workspace.data() + node_offsets[left_node];
        const Scalar* right = workspace.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          node_output[column] +=
              conjugate(coefficient_values[entry]) *
              left[row / right_dimension] *
              right[row % right_dimension];
        }
      }
      for (std::int64_t root = 0; root < root_count; ++root) {
        const Scalar* root_value =
            workspace.data() + node_offsets[root_nodes[root]];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          const Scalar projection_value =
              projection_values[
                  source * projection_dimension + projection];
          const std::int64_t feature_offset =
              (root * projection_dimension + projection) *
              root_dimension;
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            value +=
                root_value[magnetic] *
                projection_value *
                weight[feature_offset + magnetic];
          }
        }
      }
    }
    output[batch] = value;
  }
}

template <typename Scalar>
void factorized_angular_linear_adjoint(
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    std::int64_t source_dimension,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t node_count,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    std::int64_t root_count,
    const Scalar* projection_values,
    std::int64_t projection_dimension,
    const Scalar* weight,
    Scalar* packed_slots_adjoint,
    Scalar* weight_adjoint,
    Scalar* bias_adjoint) {
  const std::int64_t workspace_dimension =
      node_offsets[node_count - 1] + node_dimensions[node_count - 1];
  const std::int64_t root_dimension =
      node_dimensions[root_nodes[0]];
  const std::int64_t output_dimension =
      root_count * projection_dimension * root_dimension;
  std::vector<Scalar> workspace(
      static_cast<std::size_t>(workspace_dimension));
  std::vector<Scalar> workspace_adjoint(
      static_cast<std::size_t>(workspace_dimension));
  std::fill(
      weight_adjoint,
      weight_adjoint + output_dimension,
      Scalar(0));
  *bias_adjoint = Scalar(0);
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Scalar gradient = output_adjoint[batch];
    *bias_adjoint += gradient;
    Scalar* input_adjoint =
        packed_slots_adjoint +
        batch * source_dimension * input_dimension;
    std::fill(
        input_adjoint,
        input_adjoint + source_dimension * input_dimension,
        Scalar(0));
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      std::fill(workspace.begin(), workspace.end(), Scalar(0));
      std::fill(
          workspace_adjoint.begin(),
          workspace_adjoint.end(),
          Scalar(0));
      const Scalar* input_source =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output = workspace.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          std::copy(
              input_source + leaf_offset,
              input_source + leaf_offset + node_dimensions[node],
              node_output);
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left = workspace.data() + node_offsets[left_node];
        const Scalar* right = workspace.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          node_output[column] +=
              conjugate(coefficient_values[entry]) *
              left[row / right_dimension] *
              right[row % right_dimension];
        }
      }
      for (std::int64_t root = 0; root < root_count; ++root) {
        const Scalar* root_value =
            workspace.data() + node_offsets[root_nodes[root]];
        Scalar* root_adjoint =
            workspace_adjoint.data() + node_offsets[root_nodes[root]];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          const Scalar projection_value =
              projection_values[
                  source * projection_dimension + projection];
          const std::int64_t feature_offset =
              (root * projection_dimension + projection) *
              root_dimension;
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            const std::int64_t feature = feature_offset + magnetic;
            weight_adjoint[feature] +=
                gradient *
                conjugate(
                    root_value[magnetic] * projection_value);
            root_adjoint[magnetic] +=
                gradient *
                conjugate(weight[feature]) *
                conjugate(projection_value);
          }
        }
      }
      for (std::int64_t node = node_count; node-- > 0;) {
        Scalar* node_adjoint =
            workspace_adjoint.data() + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left = workspace.data() + node_offsets[left_node];
        const Scalar* right = workspace.data() + node_offsets[right_node];
        Scalar* left_adjoint =
            workspace_adjoint.data() + node_offsets[left_node];
        Scalar* right_adjoint =
            workspace_adjoint.data() + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          const std::int64_t left_index = row / right_dimension;
          const std::int64_t right_index = row % right_dimension;
          const Scalar node_gradient = node_adjoint[column];
          left_adjoint[left_index] +=
              node_gradient *
              coefficient_values[entry] *
              conjugate(right[right_index]);
          right_adjoint[right_index] +=
              node_gradient *
              coefficient_values[entry] *
              conjugate(left[left_index]);
        }
      }
      Scalar* input_source_adjoint =
          input_adjoint + source * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset < 0) {
          continue;
        }
        const Scalar* leaf_adjoint =
            workspace_adjoint.data() + node_offsets[node];
        for (
            std::int64_t index = 0;
            index < node_dimensions[node];
            ++index) {
          input_source_adjoint[leaf_offset + index] +=
              leaf_adjoint[index];
        }
      }
    }
  }
}

template void source_analysis_forward<float>(
    const float*, std::int64_t, std::int64_t, const std::int64_t*,
    const std::int64_t*, const float*, std::int64_t, std::int64_t,
    const std::int64_t*, const std::int64_t*, const float*, std::int64_t,
    std::int64_t, float*);
template void source_analysis_forward<double>(
    const double*, std::int64_t, std::int64_t, const std::int64_t*,
    const std::int64_t*, const double*, std::int64_t, std::int64_t,
    const std::int64_t*, const std::int64_t*, const double*, std::int64_t,
    std::int64_t, double*);
template void source_analysis_forward<std::complex<float>>(
    const std::complex<float>*, std::int64_t, std::int64_t,
    const std::int64_t*, const std::int64_t*, const std::complex<float>*,
    std::int64_t, std::int64_t, const std::int64_t*, const std::int64_t*,
    const std::complex<float>*, std::int64_t, std::int64_t,
    std::complex<float>*);
template void source_analysis_forward<std::complex<double>>(
    const std::complex<double>*, std::int64_t, std::int64_t,
    const std::int64_t*, const std::int64_t*, const std::complex<double>*,
    std::int64_t, std::int64_t, const std::int64_t*, const std::int64_t*,
    const std::complex<double>*, std::int64_t, std::int64_t,
    std::complex<double>*);

template void source_analysis_adjoint<float>(
    const float*, std::int64_t, std::int64_t, const std::int64_t*,
    const std::int64_t*, const float*, std::int64_t, std::int64_t,
    std::int64_t, const std::int64_t*, const std::int64_t*, const float*,
    std::int64_t, float*);
template void source_analysis_adjoint<double>(
    const double*, std::int64_t, std::int64_t, const std::int64_t*,
    const std::int64_t*, const double*, std::int64_t, std::int64_t,
    std::int64_t, const std::int64_t*, const std::int64_t*, const double*,
    std::int64_t, double*);
template void source_analysis_adjoint<std::complex<float>>(
    const std::complex<float>*, std::int64_t, std::int64_t,
    const std::int64_t*, const std::int64_t*, const std::complex<float>*,
    std::int64_t, std::int64_t, std::int64_t, const std::int64_t*,
    const std::int64_t*, const std::complex<float>*, std::int64_t,
    std::complex<float>*);
template void source_analysis_adjoint<std::complex<double>>(
    const std::complex<double>*, std::int64_t, std::int64_t,
    const std::int64_t*, const std::int64_t*, const std::complex<double>*,
    std::int64_t, std::int64_t, std::int64_t, const std::int64_t*,
    const std::int64_t*, const std::complex<double>*, std::int64_t,
    std::complex<double>*);

#define YE3T_INSTANTIATE_SOURCE_LINEAR(Scalar) \
template void source_analysis_linear_forward<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const Scalar*, std::int64_t, std::int64_t, \
    const std::int64_t*, const std::int64_t*, const Scalar*, \
    std::int64_t, std::int64_t, const Scalar*, Scalar, Scalar*); \
template void source_analysis_linear_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, std::int64_t, \
    const std::int64_t*, const std::int64_t*, const Scalar*, \
    std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const Scalar*, std::int64_t, std::int64_t, \
    const Scalar*, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_SOURCE_LINEAR(float)
YE3T_INSTANTIATE_SOURCE_LINEAR(double)
YE3T_INSTANTIATE_SOURCE_LINEAR(std::complex<float>)
YE3T_INSTANTIATE_SOURCE_LINEAR(std::complex<double>)

#undef YE3T_INSTANTIATE_SOURCE_LINEAR

template void compact_pair_product_forward<float>(
    const float*, const float*, std::int64_t, std::int64_t, bool, float*);
template void compact_pair_product_forward<double>(
    const double*, const double*, std::int64_t, std::int64_t, bool, double*);
template void compact_pair_product_forward<std::complex<float>>(
    const std::complex<float>*, const std::complex<float>*, std::int64_t,
    std::int64_t, bool, std::complex<float>*);
template void compact_pair_product_forward<std::complex<double>>(
    const std::complex<double>*, const std::complex<double>*, std::int64_t,
    std::int64_t, bool, std::complex<double>*);

template void compact_pair_product_adjoint<float>(
    const float*, const float*, const float*, std::int64_t, std::int64_t,
    bool, float*, float*);
template void compact_pair_product_adjoint<double>(
    const double*, const double*, const double*, std::int64_t, std::int64_t,
    bool, double*, double*);
template void compact_pair_product_adjoint<std::complex<float>>(
    const std::complex<float>*, const std::complex<float>*,
    const std::complex<float>*, std::int64_t, std::int64_t, bool,
    std::complex<float>*, std::complex<float>*);
template void compact_pair_product_adjoint<std::complex<double>>(
    const std::complex<double>*, const std::complex<double>*,
    const std::complex<double>*, std::int64_t, std::int64_t, bool,
    std::complex<double>*, std::complex<double>*);

#define YE3T_INSTANTIATE_EXTERIOR_POWER(Scalar) \
template void compact_exterior_power_forward<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void compact_exterior_power_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, std::int64_t, \
    std::int64_t, Scalar*);

YE3T_INSTANTIATE_EXTERIOR_POWER(float)
YE3T_INSTANTIATE_EXTERIOR_POWER(double)
YE3T_INSTANTIATE_EXTERIOR_POWER(std::complex<float>)
YE3T_INSTANTIATE_EXTERIOR_POWER(std::complex<double>)

#undef YE3T_INSTANTIATE_EXTERIOR_POWER

#define YE3T_INSTANTIATE_SYMMETRIC_POWER(Scalar) \
template void symmetric_power_monomial_forward<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const Scalar*, std::int64_t, std::int64_t, \
    Scalar*); \
template void symmetric_power_monomial_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, std::int64_t, \
    const std::int64_t*, const std::int64_t*, const Scalar*, \
    std::int64_t, std::int64_t, Scalar*); \
template void symmetric_power_shared_monomial_forward<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    std::int64_t, const std::int64_t*, const std::int64_t*, \
    const Scalar*, std::int64_t, std::int64_t, Scalar*); \
template void symmetric_power_shared_monomial_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, std::int64_t, \
    const std::int64_t*, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const Scalar*, std::int64_t, std::int64_t, \
    Scalar*); \
template void symmetric_power_shared_monomial_batched_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, std::int64_t, \
    std::int64_t, const std::int64_t*, std::int64_t, \
    const std::int64_t*, const std::int64_t*, const Scalar*, \
    std::int64_t, std::int64_t, Scalar*); \
template void symmetric_power_sparse_monomial_linear_forward_adjoint<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    const Scalar*, std::int64_t, Scalar*, Scalar*); \
template void symmetric_power_sparse_monomial_linear_forward_adjoint_prevalidated<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    const Scalar*, std::int64_t, std::int64_t, Scalar*, std::int64_t, \
    Scalar*, Scalar*);

YE3T_INSTANTIATE_SYMMETRIC_POWER(float)
YE3T_INSTANTIATE_SYMMETRIC_POWER(double)
YE3T_INSTANTIATE_SYMMETRIC_POWER(std::complex<float>)
YE3T_INSTANTIATE_SYMMETRIC_POWER(std::complex<double>)

#undef YE3T_INSTANTIATE_SYMMETRIC_POWER

#define YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(Scalar, CoefficientScalar) \
template void symmetric_power_sparse_monomial_dag_linear_forward_adjoint_prevalidated<Scalar, CoefficientScalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, std::int64_t, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, const std::int64_t*, \
    const CoefficientScalar*, std::int64_t, Scalar*, std::int64_t, Scalar*, \
    Scalar*);

YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(float, float)
YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(double, double)
YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(std::complex<float>, std::complex<float>)
YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(std::complex<double>, std::complex<double>)
YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(std::complex<float>, float)
YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG(std::complex<double>, double)

#undef YE3T_INSTANTIATE_SYMMETRIC_POWER_DAG

#define YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(Scalar, CoefficientScalar) \
template void symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_prevalidated<Scalar, CoefficientScalar>( \
    const Scalar*, std::int64_t, std::int64_t, const std::int64_t*, \
    const std::int64_t*, std::int64_t, const std::int64_t*, \
    const std::int64_t*, std::int64_t, const std::int64_t*, \
    const CoefficientScalar*, std::int64_t, Scalar*, std::int64_t, Scalar*, \
    Scalar*);

YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(float, float)
YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(double, double)
YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(std::complex<float>,
                                            std::complex<float>)
YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(std::complex<double>,
                                            std::complex<double>)
YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(std::complex<float>, float)
YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG(std::complex<double>, double)

#undef YE3T_INSTANTIATE_SYMMETRIC_POWER_BINARY_DAG

template void
symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated<
    float>(const std::complex<float> *, std::int64_t, std::int64_t,
           const std::int64_t *, const std::int64_t *, std::int64_t,
           const std::int64_t *, const std::int64_t *, std::int64_t,
           const std::int64_t *, const float *, std::int64_t, std::int64_t,
           float *, std::int64_t, std::complex<float> *, std::complex<float> *);
template void
symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated<
    double>(const std::complex<double> *, std::int64_t, std::int64_t,
            const std::int64_t *, const std::int64_t *, std::int64_t,
            const std::int64_t *, const std::int64_t *, std::int64_t,
            const std::int64_t *, const double *, std::int64_t, std::int64_t,
            double *, std::int64_t, std::complex<double> *,
            std::complex<double> *);

template void
ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated<
    float>(const std::complex<float> *, std::int64_t, std::int64_t,
           const std::int64_t *, const std::int64_t *, const std::int64_t *,
           const std::int64_t *, std::int64_t, const std::int64_t *,
           std::int64_t, const std::int64_t *, const std::int64_t *,
           const std::int64_t *, const float *, std::int64_t,
           const std::int64_t *, const float *, std::int64_t, std::int64_t,
           float *, std::int64_t, std::complex<float> *, std::complex<float> *);
template void
ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated<
    double>(const std::complex<double> *, std::int64_t, std::int64_t,
            const std::int64_t *, const std::int64_t *, const std::int64_t *,
            const std::int64_t *, std::int64_t, const std::int64_t *,
            std::int64_t, const std::int64_t *, const std::int64_t *,
            const std::int64_t *, const double *, std::int64_t,
            const std::int64_t *, const double *, std::int64_t, std::int64_t,
            double *, std::int64_t, std::complex<double> *,
            std::complex<double> *);

#define YE3T_INSTANTIATE_FACTORIZED(Scalar)                                    \
  template void factorized_angular_forward<Scalar>(                            \
      const Scalar *, std::int64_t, std::int64_t, std::int64_t,                \
      const std::int64_t *, const std::int64_t *, const std::int64_t *,        \
      const std::int64_t *, const std::int64_t *, const std::int64_t *,        \
      std::int64_t, const std::int64_t *, const std::int64_t *,                \
      const Scalar *, const std::int64_t *, std::int64_t,                      \
      const std::int64_t *, const std::int64_t *, const std::int64_t *,        \
      const Scalar *, std::int64_t, std::int64_t, Scalar *);                   \
  template void factorized_angular_adjoint<Scalar>(                            \
      const Scalar *, const Scalar *, std::int64_t, std::int64_t,              \
      std::int64_t, const std::int64_t *, const std::int64_t *,                \
      const std::int64_t *, const std::int64_t *, const std::int64_t *,        \
      const std::int64_t *, std::int64_t, const std::int64_t *,                \
      const std::int64_t *, const Scalar *, const std::int64_t *,              \
      std::int64_t, const std::int64_t *, const std::int64_t *,                \
      const std::int64_t *, const Scalar *, std::int64_t, std::int64_t,        \
      Scalar *);                                                               \
  template void factorized_angular_double_backward<Scalar>(                    \
      const Scalar *, const Scalar *, const Scalar *, std::int64_t,            \
      std::int64_t, std::int64_t, const std::int64_t *, const std::int64_t *,  \
      const std::int64_t *, const std::int64_t *, const std::int64_t *,        \
      const std::int64_t *, std::int64_t, const std::int64_t *,                \
      const std::int64_t *, const Scalar *, const std::int64_t *,              \
      std::int64_t, const std::int64_t *, const std::int64_t *,                \
      const std::int64_t *, const Scalar *, std::int64_t, std::int64_t,        \
      Scalar *, Scalar *);

YE3T_INSTANTIATE_FACTORIZED(float)
YE3T_INSTANTIATE_FACTORIZED(double)
YE3T_INSTANTIATE_FACTORIZED(std::complex<float>)
YE3T_INSTANTIATE_FACTORIZED(std::complex<double>)

#undef YE3T_INSTANTIATE_FACTORIZED

#define YE3T_INSTANTIATE_SEGMENTED_FACTORIZED(Scalar) \
template void factorized_angular_segmented_forward<Scalar>( \
    const Scalar*, std::int64_t, \
    const FactorizedAngularSegmentedPlanView<Scalar>&, Scalar*); \
template void factorized_angular_segmented_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, \
    const FactorizedAngularSegmentedPlanView<Scalar>&, Scalar*); \
template void factorized_angular_segmented_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, std::int64_t, \
    const FactorizedAngularSegmentedPlanView<Scalar>&, Scalar*, Scalar*);

YE3T_INSTANTIATE_SEGMENTED_FACTORIZED(float)
YE3T_INSTANTIATE_SEGMENTED_FACTORIZED(double)
YE3T_INSTANTIATE_SEGMENTED_FACTORIZED(std::complex<float>)
YE3T_INSTANTIATE_SEGMENTED_FACTORIZED(std::complex<double>)

#undef YE3T_INSTANTIATE_SEGMENTED_FACTORIZED

#define YE3T_INSTANTIATE_FACTORIZED_LINEAR(Scalar) \
template void factorized_angular_linear_forward<Scalar>( \
    const Scalar*, std::int64_t, std::int64_t, std::int64_t, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, const std::int64_t*, const std::int64_t*, \
    const Scalar*, const std::int64_t*, std::int64_t, const Scalar*, \
    std::int64_t, const Scalar*, Scalar, Scalar*); \
template void factorized_angular_linear_adjoint<Scalar>( \
    const Scalar*, const Scalar*, std::int64_t, std::int64_t, \
    std::int64_t, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, const std::int64_t*, \
    const std::int64_t*, const Scalar*, const std::int64_t*, \
    std::int64_t, const Scalar*, std::int64_t, const Scalar*, \
    Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_FACTORIZED_LINEAR(float)
YE3T_INSTANTIATE_FACTORIZED_LINEAR(double)
YE3T_INSTANTIATE_FACTORIZED_LINEAR(std::complex<float>)
YE3T_INSTANTIATE_FACTORIZED_LINEAR(std::complex<double>)

#undef YE3T_INSTANTIATE_FACTORIZED_LINEAR

template void cheb_exp_cos_radial_with_derivative<float>(
    const float*, const float*, const float*, std::int64_t, std::int64_t,
    float*, float*);
template void cheb_exp_cos_radial_with_derivative<double>(
    const double*, const double*, const double*, std::int64_t, std::int64_t,
    double*, double*);
template void cheb_exp_cos_radial_table_with_derivative<float>(
    const float*, const float*, const float*, std::int64_t, std::int64_t,
    float*, float*);
template void cheb_exp_cos_radial_table_with_derivative<double>(
    const double*, const double*, const double*, std::int64_t, std::int64_t,
    double*, double*);
template void pace_cheb_exp_cos_radial_table_with_derivative<float>(
    const float*, const float*, const float*, const float*, std::int64_t,
    std::int64_t, float*, float*);
template void pace_cheb_exp_cos_radial_table_with_derivative<double>(
    const double*, const double*, const double*, const double*, std::int64_t,
    std::int64_t, double*, double*);
template std::int64_t pace_uniform_spline_interval_count<float>(float, float);
template std::int64_t pace_uniform_spline_interval_count<double>(
    double, double);
template void pace_uniform_cubic_spline_build<float>(
    const float*, const float*, std::int64_t, std::int64_t, float, float*);
template void pace_uniform_cubic_spline_build<double>(
    const double*, const double*, std::int64_t, std::int64_t, double,
    double*);
template void pace_uniform_cubic_spline_evaluate_with_derivative<float>(
    const float*, const float*, std::int64_t, std::int64_t, std::int64_t,
    float, float*, float*);
template void pace_uniform_cubic_spline_evaluate_with_derivative<double>(
    const double*, const double*, std::int64_t, std::int64_t, std::int64_t,
    double, double*, double*);
template void pace_radial_channel_contraction_with_derivative<float>(
    const float*, const float*, const float*, std::int64_t, std::int64_t,
    std::int64_t, float*, float*);
template void pace_radial_channel_contraction_with_derivative<double>(
    const double*, const double*, const double*, std::int64_t, std::int64_t,
    std::int64_t, double*, double*);
template void cheb_exp_cos_radial_table_double_backward<float>(
    const float*, const float*, const float*, const float*, const float*,
    const float*, std::int64_t, std::int64_t, float*, float*);
template void cheb_exp_cos_radial_table_double_backward<double>(
    const double*, const double*, const double*, const double*, const double*,
    const double*, std::int64_t, std::int64_t, double*, double*);

template void real_spherical_harmonics_with_derivative<float>(
    const float*, std::int64_t, std::int64_t, float, float*, float*);
template void real_spherical_harmonics_with_derivative<double>(
    const double*, std::int64_t, std::int64_t, double, double*, double*);
template void complex_spherical_harmonics_with_derivative<float>(
    const float*, std::int64_t, std::int64_t, float,
    std::complex<float>*, std::complex<float>*);
template void complex_spherical_harmonics_with_derivative<double>(
    const double*, std::int64_t, std::int64_t, double,
    std::complex<double>*, std::complex<double>*);
template void real_spherical_harmonics_table_with_derivative<float>(
    const float*, std::int64_t, std::int64_t, float, float*, float*);
template void real_spherical_harmonics_table_with_derivative<double>(
    const double*, std::int64_t, std::int64_t, double, double*, double*);
template void complex_spherical_harmonics_table_with_derivative<float>(
    const float*, std::int64_t, std::int64_t, float,
    std::complex<float>*, std::complex<float>*, float);
template void complex_spherical_harmonics_table_with_derivative<double>(
    const double*, std::int64_t, std::int64_t, double,
    std::complex<double>*, std::complex<double>*, double);
template void build_complex_spherical_harmonics_table_plan<float>(
    std::int64_t, float*, std::int64_t, float);
template void build_complex_spherical_harmonics_table_plan<double>(
    std::int64_t, double*, std::int64_t, double);
template void
complex_spherical_harmonics_table_with_derivative_prevalidated<float>(
    const float*, std::int64_t, std::int64_t, float, const float*,
    std::int64_t, std::complex<float>*, std::int64_t,
    std::complex<float>*, std::complex<float>*, float);
template void
complex_spherical_harmonics_table_with_derivative_prevalidated<double>(
    const double*, std::int64_t, std::int64_t, double, const double*,
    std::int64_t, std::complex<double>*, std::int64_t,
    std::complex<double>*, std::complex<double>*, double);
template void
complex_spherical_harmonics_nonnegative_table_with_derivative_prevalidated<float>(
    const float*, std::int64_t, std::int64_t, float, const float*,
    std::int64_t, std::complex<float>*, std::int64_t,
    std::complex<float>*, std::complex<float>*, float);
template void
complex_spherical_harmonics_nonnegative_table_with_derivative_prevalidated<double>(
    const double*, std::int64_t, std::int64_t, double, const double*,
    std::int64_t, std::complex<double>*, std::int64_t,
    std::complex<double>*, std::complex<double>*, double);
template void
complex_spherical_harmonics_nonnegative_unit_table_with_derivative_prevalidated<float>(
    const float*, const float*, std::int64_t, std::int64_t, float,
    const float*, std::int64_t, std::complex<float>*, std::int64_t,
    std::complex<float>*, std::complex<float>*);
template void
complex_spherical_harmonics_nonnegative_unit_table_with_derivative_prevalidated<double>(
    const double*, const double*, std::int64_t, std::int64_t, double,
    const double*, std::int64_t, std::complex<double>*, std::int64_t,
    std::complex<double>*, std::complex<double>*);
template void build_complex_spherical_harmonics_recurrence_plan<float>(
    std::int64_t, float*, std::int64_t, float);
template void build_complex_spherical_harmonics_recurrence_plan<double>(
    std::int64_t, double*, std::int64_t, double);
template void
complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<float>(
    const float*, const float*, std::int64_t, std::int64_t, const float*,
    std::int64_t, std::complex<float>*, std::complex<float>*);
template void
complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<double>(
    const double*, const double*, std::int64_t, std::int64_t, const double*,
    std::int64_t, std::complex<double>*, std::complex<double>*);

#define YE3T_INSTANTIATE_PLAIN_SITE_BASIS(Scalar) \
template void plain_site_basis_product_with_derivative<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Scalar*, \
    Scalar*, Scalar*); \
template void plain_site_basis_product_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const Scalar*, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_PLAIN_SITE_BASIS(float)
YE3T_INSTANTIATE_PLAIN_SITE_BASIS(double)
YE3T_INSTANTIATE_PLAIN_SITE_BASIS(std::complex<float>)
YE3T_INSTANTIATE_PLAIN_SITE_BASIS(std::complex<double>)

#undef YE3T_INSTANTIATE_PLAIN_SITE_BASIS

template void scheduled_radial_angular_channels_with_derivative<float>(
    const float*, const float*, const float*, const float*,
    const float*, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, const std::int64_t*, const float*,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    float*, float*);
template void scheduled_radial_angular_channels_with_derivative<double>(
    const double*, const double*, const double*, const double*,
    const double*, const std::int64_t*, const std::int64_t*,
    const std::int64_t*, const std::int64_t*, const double*,
    std::int64_t, std::int64_t, std::int64_t, std::int64_t,
    double*, double*);

#define YE3T_INSTANTIATE_DENSITY(Scalar) \
template void density_accumulate_forward<Scalar>( \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    std::int64_t, Scalar*); \
template void density_accumulate_adjoint<Scalar>( \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    Scalar*);

YE3T_INSTANTIATE_DENSITY(float)
YE3T_INSTANTIATE_DENSITY(double)
YE3T_INSTANTIATE_DENSITY(std::complex<float>)
YE3T_INSTANTIATE_DENSITY(std::complex<double>)

#undef YE3T_INSTANTIATE_DENSITY

#define YE3T_INSTANTIATE_EDGE_OUTER(Scalar) \
template void edge_outer_accumulate_forward<Scalar>( \
    const Scalar*, const Scalar*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void edge_outer_accumulate_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Scalar*); \
template void edge_outer_accumulate_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_EDGE_OUTER(float)
YE3T_INSTANTIATE_EDGE_OUTER(double)

#undef YE3T_INSTANTIATE_EDGE_OUTER

#define YE3T_INSTANTIATE_SOFTMAX_GAUSSIAN_ROLE_DENSITY(Scalar) \
template void softmax_gaussian_role_density_forward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, Scalar, \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*); \
template void softmax_gaussian_role_density_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, Scalar, \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    std::int64_t, Scalar*, Scalar*); \
template void softmax_gaussian_role_density_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, Scalar, \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_SOFTMAX_GAUSSIAN_ROLE_DENSITY(float)
YE3T_INSTANTIATE_SOFTMAX_GAUSSIAN_ROLE_DENSITY(double)

#undef YE3T_INSTANTIATE_SOFTMAX_GAUSSIAN_ROLE_DENSITY

#define YE3T_INSTANTIATE_SCHEDULED_ROLE_DENSITY(Scalar) \
template void scheduled_softmax_gaussian_role_density_forward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, Scalar, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void scheduled_softmax_gaussian_role_density_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, Scalar, const Scalar*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const Scalar*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    std::int64_t, Scalar*, Scalar*, Scalar*, Scalar*); \
template void scheduled_softmax_gaussian_role_density_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, Scalar, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const Scalar*, const std::int64_t*, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, Scalar*, \
    Scalar*, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_SCHEDULED_ROLE_DENSITY(float)
YE3T_INSTANTIATE_SCHEDULED_ROLE_DENSITY(double)

#undef YE3T_INSTANTIATE_SCHEDULED_ROLE_DENSITY

#define YE3T_INSTANTIATE_CARRIER_GATED_SCATTER(Scalar) \
template void carrier_gated_scatter_forward<Scalar>( \
    const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void carrier_gated_scatter_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Scalar*); \
template void carrier_gated_scatter_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Scalar*); \
template void carrier_residual_gated_scatter_forward<Scalar>( \
    const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void carrier_residual_gated_scatter_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Scalar*); \
template void carrier_residual_gated_scatter_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, \
    std::int64_t, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_CARRIER_GATED_SCATTER(float)
YE3T_INSTANTIATE_CARRIER_GATED_SCATTER(double)
YE3T_INSTANTIATE_CARRIER_GATED_SCATTER(std::complex<float>)
YE3T_INSTANTIATE_CARRIER_GATED_SCATTER(std::complex<double>)

#undef YE3T_INSTANTIATE_CARRIER_GATED_SCATTER

#define YE3T_INSTANTIATE_CARRIER_GATED_SCATTER_REAL_GATES(Scalar, Real) \
template void carrier_gated_scatter_real_gates_forward<Scalar, Real>( \
    const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void carrier_gated_scatter_real_gates_adjoint<Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Real*); \
template void carrier_gated_scatter_real_gates_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const Scalar*, \
    const Real*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Real*); \
template void carrier_residual_gated_scatter_real_gates_forward< \
    Scalar, Real>( \
    const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void carrier_residual_gated_scatter_real_gates_adjoint< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Real*); \
template void carrier_residual_gated_scatter_real_gates_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const Scalar*, \
    const Real*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, \
    std::int64_t, Scalar*, Scalar*, Real*);

YE3T_INSTANTIATE_CARRIER_GATED_SCATTER_REAL_GATES(
    std::complex<float>, float)
YE3T_INSTANTIATE_CARRIER_GATED_SCATTER_REAL_GATES(
    std::complex<double>, double)

#undef YE3T_INSTANTIATE_CARRIER_GATED_SCATTER_REAL_GATES

#define YE3T_INSTANTIATE_SEGMENTED_CARRIER_SCATTER(Scalar) \
template void carrier_segmented_residual_gated_scatter_forward<Scalar>( \
    const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*); \
template void carrier_segmented_residual_gated_scatter_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Scalar*); \
template void carrier_segmented_residual_gated_scatter_double_backward< \
    Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_SEGMENTED_CARRIER_SCATTER(float)
YE3T_INSTANTIATE_SEGMENTED_CARRIER_SCATTER(double)
YE3T_INSTANTIATE_SEGMENTED_CARRIER_SCATTER(std::complex<float>)
YE3T_INSTANTIATE_SEGMENTED_CARRIER_SCATTER(std::complex<double>)

#undef YE3T_INSTANTIATE_SEGMENTED_CARRIER_SCATTER

#define YE3T_INSTANTIATE_SEGMENTED_CARRIER_REAL_GATES(Scalar, Real) \
template void \
carrier_segmented_residual_gated_scatter_real_gates_forward< \
    Scalar, Real>( \
    const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*); \
template void \
carrier_segmented_residual_gated_scatter_real_gates_adjoint< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Real*); \
template void \
carrier_segmented_residual_gated_scatter_real_gates_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const Scalar*, \
    const Real*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Scalar*, Real*);

YE3T_INSTANTIATE_SEGMENTED_CARRIER_REAL_GATES(
    std::complex<float>, float)
YE3T_INSTANTIATE_SEGMENTED_CARRIER_REAL_GATES(
    std::complex<double>, double)

#undef YE3T_INSTANTIATE_SEGMENTED_CARRIER_REAL_GATES

#define YE3T_INSTANTIATE_SOURCE_ARENA_GATHER(Scalar) \
template void source_arena_gather_forward<Scalar>( \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*); \
template void source_arena_gather_adjoint<Scalar>( \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*);

YE3T_INSTANTIATE_SOURCE_ARENA_GATHER(float)
YE3T_INSTANTIATE_SOURCE_ARENA_GATHER(double)
YE3T_INSTANTIATE_SOURCE_ARENA_GATHER(std::complex<float>)
YE3T_INSTANTIATE_SOURCE_ARENA_GATHER(std::complex<double>)

#undef YE3T_INSTANTIATE_SOURCE_ARENA_GATHER

#define YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(Scalar, Control) \
template void source_arena_channel_transform_forward<Scalar, Control>( \
    const Scalar*, const Control*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*); \
template void source_arena_channel_transform_adjoint<Scalar, Control>( \
    const Scalar*, const Scalar*, const Control*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*, Control*); \
template void source_arena_channel_transform_double_backward< \
    Scalar, Control>( \
    const Scalar*, const Scalar*, const Control*, const Scalar*, \
    const Control*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Control*);

YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(float, float)
YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(double, double)
YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(
    std::complex<float>, std::complex<float>)
YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(
    std::complex<double>, std::complex<double>)
YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(
    std::complex<float>, float)
YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM(
    std::complex<double>, double)

#undef YE3T_INSTANTIATE_SOURCE_ARENA_CHANNEL_TRANSFORM

#define YE3T_INSTANTIATE_CARRIER_CHANNEL_UPDATE(Scalar) \
template void carrier_channel_update_forward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, Scalar*); \
template void carrier_channel_update_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Scalar*); \
template void carrier_channel_update_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, Scalar*, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_CARRIER_CHANNEL_UPDATE(float)
YE3T_INSTANTIATE_CARRIER_CHANNEL_UPDATE(double)
YE3T_INSTANTIATE_CARRIER_CHANNEL_UPDATE(std::complex<float>)
YE3T_INSTANTIATE_CARRIER_CHANNEL_UPDATE(std::complex<double>)

#undef YE3T_INSTANTIATE_CARRIER_CHANNEL_UPDATE

#define YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM(Scalar) \
template void carrier_channel_transform_forward<Scalar>( \
    const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, Scalar*); \
template void carrier_channel_transform_adjoint<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, Scalar*, Scalar*); \
template void carrier_channel_transform_double_backward<Scalar>( \
    const Scalar*, const Scalar*, const Scalar*, const Scalar*, \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, Scalar*, Scalar*, Scalar*);

YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM(float)
YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM(double)
YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM(std::complex<float>)
YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM(std::complex<double>)

#undef YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM

#define YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM_REAL(Scalar, Real) \
template void carrier_channel_transform_real_maps_forward<Scalar, Real>( \
    const Scalar*, const Real*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, Scalar*); \
template void carrier_channel_transform_real_maps_adjoint<Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, std::int64_t, std::int64_t, Scalar*, Real*); \
template void carrier_channel_transform_real_maps_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const Scalar*, const Real*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, std::int64_t, \
    Scalar*, Scalar*, Real*);

YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM_REAL(
    std::complex<float>, float)
YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM_REAL(
    std::complex<double>, double)

#undef YE3T_INSTANTIATE_CARRIER_CHANNEL_TRANSFORM_REAL

#define YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_MAP(Scalar, Real) \
template void carrier_role_channel_map_adjoint<Scalar, Real>( \
    const Scalar*, const Real*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Scalar*); \
template void carrier_role_channel_map_adjoint_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Real*, const Scalar*, const std::int64_t*, \
    const Scalar*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Real*, Scalar*);

YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_MAP(float, float)
YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_MAP(double, double)
YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_MAP(std::complex<float>, float)
YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_MAP(std::complex<double>, double)

#undef YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_MAP

#define YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_REAL_MAP(Scalar, Real) \
template void carrier_role_channel_real_map_adjoint<Scalar, Real>( \
    const Scalar*, const Real*, const Scalar*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, std::int64_t, std::int64_t, Real*); \
template void carrier_role_channel_real_map_adjoint_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Real*, const Scalar*, const std::int64_t*, \
    const Real*, const std::int64_t*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, std::int64_t, std::int64_t, \
    Scalar*, Real*, Scalar*);

YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_REAL_MAP(
    std::complex<float>, float)
YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_REAL_MAP(
    std::complex<double>, double)

#undef YE3T_INSTANTIATE_CARRIER_ROLE_CHANNEL_REAL_MAP

#define YE3T_INSTANTIATE_CARRIER_CHANNEL_REAL_CONTROLS(Scalar, Real) \
template void carrier_channel_update_real_controls_forward< \
    Scalar, Real>( \
    const Scalar*, const Real*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, Scalar*); \
template void carrier_channel_update_real_controls_adjoint< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const Real*, \
    const std::int64_t*, const std::int64_t*, const std::int64_t*, \
    std::int64_t, std::int64_t, Scalar*, Real*, Real*); \
template void carrier_channel_update_real_controls_double_backward< \
    Scalar, Real>( \
    const Scalar*, const Scalar*, const Real*, const Real*, \
    const Scalar*, const Real*, const Real*, const std::int64_t*, \
    const std::int64_t*, const std::int64_t*, std::int64_t, \
    std::int64_t, Scalar*, Scalar*, Real*, Real*);

YE3T_INSTANTIATE_CARRIER_CHANNEL_REAL_CONTROLS(
    std::complex<float>, float)
YE3T_INSTANTIATE_CARRIER_CHANNEL_REAL_CONTROLS(
    std::complex<double>, double)

#undef YE3T_INSTANTIATE_CARRIER_CHANNEL_REAL_CONTROLS

}  // namespace runtime
}  // namespace ye3t
