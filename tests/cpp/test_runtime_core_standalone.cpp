#include "ye3t_runtime_core.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Fixture {
  std::string schema;
  std::string scope;
  std::string compiler_api;
  std::string compiler_owner;
  std::string compiler_revision;
  std::string basis_convention;
  std::string normalization_convention;
  std::string convention_hash;
  std::string table_hash;
  std::string scalar_type;
  std::int64_t power = 0;
  std::int64_t batch_size = 0;
  std::int64_t input_dimension = 0;
  std::int64_t monomial_count = 0;
  std::int64_t coefficient_count = 0;
  std::int64_t output_dimension = 0;
  std::vector<std::int64_t> monomial_counts;
  std::vector<std::int64_t> output_offsets;
  std::vector<std::int64_t> coefficient_terms;
  std::vector<std::int64_t> coefficient_outputs;
  std::vector<double> coefficient_values;
  std::vector<double> input;
  std::vector<double> output_adjoint;
  std::vector<double> expected_output;
  std::vector<double> expected_input_adjoint;
};

struct RadialSplineFixture {
  std::string schema;
  std::string scope;
  std::string pace_revision;
  std::string oracle_sha256;
  std::string model_sha256;
  std::string environment_sha256;
  double cutoff = 0.0;
  double requested_spacing = 0.0;
  double cutoff_width = 0.0;
  double radial_lambda = 0.0;
  std::int64_t interval_count = 0;
  std::int64_t radial_count = 0;
  std::int64_t output_count = 0;
  std::vector<double> radii;
  std::vector<double> radial_coefficients;
  std::vector<double> expected_g;
  std::vector<double> expected_dg;
  std::vector<double> expected_R;
  std::vector<double> expected_dR;
};

void require(bool condition, const std::string &message)
{
  if (!condition) throw std::runtime_error(message);
}

void expect_key(std::istream &stream, const std::string &expected)
{
  std::string actual;
  require(static_cast<bool>(stream >> actual), "missing key: " + expected);
  require(actual == expected,
          "expected key '" + expected + "', found '" + actual + "'");
}

template <typename Value>
void read_scalar(std::istream &stream, const std::string &key, Value &value)
{
  expect_key(stream, key);
  require(static_cast<bool>(stream >> value), "invalid scalar: " + key);
}

template <typename Value>
void read_vector(std::istream &stream, const std::string &key,
                 std::vector<Value> &values)
{
  expect_key(stream, key);
  std::int64_t count = 0;
  require(static_cast<bool>(stream >> count), "invalid vector size: " + key);
  require(count >= 0, "negative vector size: " + key);
  values.resize(static_cast<std::size_t>(count));
  for (Value &value : values)
    require(static_cast<bool>(stream >> value), "invalid vector value: " + key);
}

bool is_lower_hex(const std::string &value, std::size_t length)
{
  return value.size() == length &&
      std::all_of(value.begin(), value.end(), [](unsigned char character) {
        return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
      });
}

void validate_fixture(const Fixture &fixture)
{
  require(fixture.schema == "ye3t_shared_monomial_fixture_v1",
          "unsupported fixture schema");
  require(fixture.scope == "low_rank_abi_not_ta_catalogue",
          "unexpected fixture scope");
  require(fixture.compiler_api ==
              "ye3t.couplings.symmetric_power_product_plan",
          "fixture lacks compiler-plan provenance");
  require(fixture.compiler_owner == "ye3t", "unexpected compiler owner");
  require(is_lower_hex(fixture.compiler_revision, 40),
          "invalid compiler revision encoding");
  require(fixture.basis_convention == "real_tesseral",
          "unsupported basis convention");
  require(fixture.normalization_convention == "none",
          "standalone parity fixture must disable normalization");
  require(fixture.scalar_type == "float64", "unsupported scalar type");
  require(is_lower_hex(fixture.convention_hash, 16),
          "invalid convention hash encoding");
  require(is_lower_hex(fixture.table_hash, 64),
          "invalid table hash encoding");
  require(fixture.power > 0 && fixture.batch_size > 0,
          "invalid power or batch size");
  require(fixture.input_dimension > 0 && fixture.monomial_count > 0 &&
              fixture.coefficient_count > 0 && fixture.output_dimension > 0,
          "invalid table dimensions");

  const auto input_size = fixture.batch_size * fixture.input_dimension;
  const auto output_size = fixture.batch_size * fixture.output_dimension;
  require(static_cast<std::int64_t>(fixture.monomial_counts.size()) ==
              fixture.monomial_count * fixture.input_dimension,
          "monomial count table has the wrong shape");
  require(static_cast<std::int64_t>(fixture.output_offsets.size()) ==
              fixture.output_dimension + 1,
          "output offset table has the wrong shape");
  require(static_cast<std::int64_t>(fixture.coefficient_terms.size()) ==
              fixture.coefficient_count,
          "coefficient-term table has the wrong shape");
  require(static_cast<std::int64_t>(fixture.coefficient_outputs.size()) ==
              fixture.coefficient_count,
          "coefficient-output table has the wrong shape");
  require(static_cast<std::int64_t>(fixture.coefficient_values.size()) ==
              fixture.coefficient_count,
          "coefficient table has the wrong shape");
  require(static_cast<std::int64_t>(fixture.input.size()) == input_size,
          "input fixture has the wrong shape");
  require(static_cast<std::int64_t>(fixture.output_adjoint.size()) == output_size,
          "output adjoint fixture has the wrong shape");
  require(static_cast<std::int64_t>(fixture.expected_output.size()) == output_size,
          "expected output fixture has the wrong shape");
  require(static_cast<std::int64_t>(fixture.expected_input_adjoint.size()) ==
              input_size,
          "expected input adjoint fixture has the wrong shape");

  require(fixture.output_offsets.front() == 0,
          "output offsets must start at zero");
  require(fixture.output_offsets.back() == fixture.coefficient_count,
          "output offsets must end at coefficient_count");
  for (std::int64_t output = 0; output < fixture.output_dimension; ++output) {
    const auto start = fixture.output_offsets[static_cast<std::size_t>(output)];
    const auto finish =
        fixture.output_offsets[static_cast<std::size_t>(output + 1)];
    require(start <= finish, "output offsets are not monotone");
    for (std::int64_t coefficient = start; coefficient < finish; ++coefficient)
      require(fixture.coefficient_outputs[static_cast<std::size_t>(coefficient)] ==
                  output,
              "coefficient output disagrees with output offsets");
  }
  for (std::int64_t term : fixture.coefficient_terms)
    require(term >= 0 && term < fixture.monomial_count,
            "coefficient references an invalid monomial");
  for (std::int64_t term = 0; term < fixture.monomial_count; ++term) {
    const auto begin = fixture.monomial_counts.begin() +
        term * fixture.input_dimension;
    const auto end = begin + fixture.input_dimension;
    require(std::all_of(begin, end, [](std::int64_t exponent) {
              return exponent >= 0;
            }),
            "monomial exponent is negative");
    require(std::accumulate(begin, end, std::int64_t{0}) == fixture.power,
            "monomial degree disagrees with power");
  }
}

Fixture load_fixture(const std::string &path)
{
  std::ifstream stream(path);
  require(stream.good(), "could not open fixture: " + path);
  Fixture fixture;
  read_scalar(stream, "schema", fixture.schema);
  read_scalar(stream, "scope", fixture.scope);
  read_scalar(stream, "compiler_api", fixture.compiler_api);
  read_scalar(stream, "compiler_owner", fixture.compiler_owner);
  read_scalar(stream, "compiler_revision", fixture.compiler_revision);
  read_scalar(stream, "basis_convention", fixture.basis_convention);
  read_scalar(stream, "normalization_convention",
              fixture.normalization_convention);
  read_scalar(stream, "convention_hash", fixture.convention_hash);
  read_scalar(stream, "table_hash", fixture.table_hash);
  read_scalar(stream, "scalar_type", fixture.scalar_type);
  read_scalar(stream, "power", fixture.power);
  read_scalar(stream, "batch_size", fixture.batch_size);
  read_scalar(stream, "input_dimension", fixture.input_dimension);
  read_scalar(stream, "monomial_count", fixture.monomial_count);
  read_scalar(stream, "coefficient_count", fixture.coefficient_count);
  read_scalar(stream, "output_dimension", fixture.output_dimension);
  read_vector(stream, "monomial_counts", fixture.monomial_counts);
  read_vector(stream, "output_offsets", fixture.output_offsets);
  read_vector(stream, "coefficient_terms", fixture.coefficient_terms);
  read_vector(stream, "coefficient_outputs", fixture.coefficient_outputs);
  read_vector(stream, "coefficient_values", fixture.coefficient_values);
  read_vector(stream, "input", fixture.input);
  read_vector(stream, "output_adjoint", fixture.output_adjoint);
  read_vector(stream, "expected_output", fixture.expected_output);
  read_vector(stream, "expected_input_adjoint",
              fixture.expected_input_adjoint);
  expect_key(stream, "end");
  std::string trailing;
  require(!(stream >> trailing), "unexpected trailing fixture data");
  validate_fixture(fixture);
  return fixture;
}

RadialSplineFixture load_radial_spline_fixture(const std::string &path)
{
  std::ifstream stream(path);
  require(stream.good(), "could not open radial spline fixture: " + path);
  RadialSplineFixture fixture;
  read_scalar(stream, "schema", fixture.schema);
  read_scalar(stream, "scope", fixture.scope);
  read_scalar(stream, "pace_revision", fixture.pace_revision);
  read_scalar(stream, "oracle_sha256", fixture.oracle_sha256);
  read_scalar(stream, "model_sha256", fixture.model_sha256);
  read_scalar(stream, "environment_sha256", fixture.environment_sha256);
  read_scalar(stream, "cutoff", fixture.cutoff);
  read_scalar(stream, "requested_spacing", fixture.requested_spacing);
  read_scalar(stream, "cutoff_width", fixture.cutoff_width);
  read_scalar(stream, "radial_lambda", fixture.radial_lambda);
  read_scalar(stream, "interval_count", fixture.interval_count);
  read_scalar(stream, "radial_count", fixture.radial_count);
  read_scalar(stream, "output_count", fixture.output_count);
  read_vector(stream, "radii", fixture.radii);
  read_vector(stream, "radial_coefficients", fixture.radial_coefficients);
  read_vector(stream, "expected_g", fixture.expected_g);
  read_vector(stream, "expected_dg", fixture.expected_dg);
  read_vector(stream, "expected_R", fixture.expected_R);
  read_vector(stream, "expected_dR", fixture.expected_dR);
  expect_key(stream, "end");
  std::string trailing;
  require(!(stream >> trailing), "unexpected trailing radial spline data");

  require(fixture.schema == "pace_radial_spline_fixture_v1",
          "unsupported radial spline fixture schema");
  require(fixture.scope == "pinned_product_oracle_nonidentity_radial",
          "unexpected radial spline fixture scope");
  require(fixture.pace_revision ==
              "5844770144643f872db8aa06df02d64192a8160f",
          "radial spline fixture has the wrong PACE revision");
  require(is_lower_hex(fixture.oracle_sha256, 64) &&
              is_lower_hex(fixture.model_sha256, 64) &&
              is_lower_hex(fixture.environment_sha256, 64),
          "radial spline fixture has an invalid provenance hash");
  require(std::isfinite(fixture.cutoff) && fixture.cutoff > 0.0 &&
              std::isfinite(fixture.requested_spacing) &&
              fixture.requested_spacing > 0.0 &&
              std::isfinite(fixture.cutoff_width) &&
              fixture.cutoff_width >= 0.0 &&
              std::isfinite(fixture.radial_lambda),
          "radial spline fixture has invalid parameters");
  require(fixture.interval_count > 0 && fixture.radial_count > 1 &&
              fixture.output_count > 1 && !fixture.radii.empty(),
          "radial spline fixture has invalid dimensions");
  require(
      ye3t::runtime::pace_uniform_spline_interval_count<double>(
          fixture.requested_spacing, fixture.cutoff) == fixture.interval_count,
      "radial spline fixture interval count disagrees with the runtime");
  const auto point_count = static_cast<std::int64_t>(fixture.radii.size());
  require(static_cast<std::int64_t>(fixture.radial_coefficients.size()) ==
              fixture.output_count * fixture.radial_count,
          "radial coefficient fixture has the wrong shape");
  require(static_cast<std::int64_t>(fixture.expected_g.size()) ==
              point_count * fixture.radial_count &&
              fixture.expected_dg.size() == fixture.expected_g.size(),
          "radial spline base oracle has the wrong shape");
  require(static_cast<std::int64_t>(fixture.expected_R.size()) ==
              point_count * fixture.output_count &&
              fixture.expected_dR.size() == fixture.expected_R.size(),
          "radial spline contraction oracle has the wrong shape");
  return fixture;
}

bool close(double actual, double expected)
{
  const double tolerance = 5.0e-13 * (1.0 + std::abs(expected));
  return std::abs(actual - expected) <= tolerance;
}

void compare(const std::vector<double> &actual,
             const std::vector<double> &expected,
             const std::string &label)
{
  require(actual.size() == expected.size(), label + " size mismatch");
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!close(actual[index], expected[index]))
      throw std::runtime_error(label + " mismatch at index " +
                               std::to_string(index));
  }
}

template <typename Function>
void require_invalid_argument(Function function, const std::string &label)
{
  bool rejected = false;
  try {
    function();
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, label + " was not rejected");
}

double spline_test_value(double radius, std::int64_t function)
{
  if (function == 0)
    return 0.75 - 1.25 * radius + 0.5 * radius * radius +
        0.125 * radius * radius * radius;
  return -0.4 + 0.3 * radius - 0.2 * radius * radius +
      0.05 * radius * radius * radius;
}

double spline_test_derivative(double radius, std::int64_t function)
{
  if (function == 0)
    return -1.25 + radius + 0.375 * radius * radius;
  return 0.3 - 0.4 * radius + 0.15 * radius * radius;
}

void run_pace_spline_and_contraction_fixture()
{
  constexpr double cutoff = 4.0;
  constexpr double requested_spacing = 0.7;
  constexpr std::int64_t function_count = 2;
  const std::int64_t interval_count =
      ye3t::runtime::pace_uniform_spline_interval_count<double>(
          requested_spacing, cutoff);
  require(interval_count == 5, "PACE spline interval count did not truncate");
  const double spacing = cutoff / static_cast<double>(interval_count);
  require(spacing == 0.8 && spacing != requested_spacing,
          "PACE spline effective spacing was not distinguished from its request");

  std::vector<double> node_values(
      static_cast<std::size_t>(interval_count * function_count));
  std::vector<double> node_derivatives(node_values.size());
  for (std::int64_t node = 1; node <= interval_count; ++node) {
    const double radius = spacing * static_cast<double>(node);
    for (std::int64_t function = 0; function < function_count; ++function) {
      const std::int64_t index = (node - 1) * function_count + function;
      node_values[static_cast<std::size_t>(index)] =
          spline_test_value(radius, function);
      node_derivatives[static_cast<std::size_t>(index)] =
          spline_test_derivative(radius, function);
    }
  }
  std::vector<double> table(
      static_cast<std::size_t>((interval_count + 1) * function_count * 4),
      std::numeric_limits<double>::quiet_NaN());
  ye3t::runtime::pace_uniform_cubic_spline_build<double>(
      node_values.data(), node_derivatives.data(), interval_count,
      function_count, cutoff, table.data());
  require(std::all_of(table.begin(), table.begin() + function_count * 4,
                      [](double value) { return value == 0.0; }),
          "PACE spline interval zero must remain unused");

  const std::vector<double> radii{0.8, 1.17, 3.2, 3.55, 3.999, 4.0, 4.3};
  std::vector<double> values(radii.size() * function_count);
  std::vector<double> derivatives(values.size());
  ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
      radii.data(), table.data(), static_cast<std::int64_t>(radii.size()),
      function_count, interval_count, cutoff, values.data(),
      derivatives.data());
  for (std::size_t point = 0; point < radii.size(); ++point) {
    for (std::int64_t function = 0; function < function_count; ++function) {
      const std::size_t index = point * function_count +
          static_cast<std::size_t>(function);
      const bool outside = radii[point] >= cutoff;
      const double expected_value = outside
          ? 0.0
          : spline_test_value(radii[point], function);
      const double expected_derivative = outside
          ? 0.0
          : spline_test_derivative(radii[point], function);
      require(close(values[index], expected_value),
              "PACE Hermite cubic value mismatch");
      require(close(derivatives[index], expected_derivative),
              "PACE Hermite cubic derivative mismatch");
    }
  }

  double below_first_interval = std::nextafter(spacing, 0.0);
  double rejected_value = 0.0;
  double rejected_derivative = 0.0;
  require_invalid_argument(
      [&]() {
        ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
            &below_first_interval, table.data(), 1, function_count,
            interval_count, cutoff, &rejected_value, &rejected_derivative);
      },
      "PACE spline small-radius query");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::pace_uniform_spline_interval_count<double>(0.0, cutoff);
      },
      "zero PACE spline spacing");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::pace_uniform_spline_interval_count<double>(5.0, cutoff);
      },
      "empty PACE spline grid");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::pace_uniform_cubic_spline_build<double>(
            node_values.data(), node_derivatives.data(), 0, function_count,
            cutoff, table.data());
      },
      "empty PACE spline table");

  constexpr std::int64_t radial_count = 3;
  constexpr std::int64_t output_count = 6;
  std::vector<double> node_radii(static_cast<std::size_t>(interval_count));
  std::vector<double> cutoffs(node_radii.size(), cutoff);
  std::vector<double> cutoff_widths(node_radii.size(), 0.8);
  std::vector<double> lambdas(node_radii.size(), 0.5723);
  for (std::int64_t node = 1; node <= interval_count; ++node)
    node_radii[static_cast<std::size_t>(node - 1)] = spacing * node;
  std::vector<double> radial_values(
      node_radii.size() * radial_count);
  std::vector<double> radial_derivatives(radial_values.size());
  ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
      node_radii.data(), cutoffs.data(), cutoff_widths.data(), lambdas.data(),
      interval_count, radial_count, radial_values.data(),
      radial_derivatives.data());
  const std::vector<double> radial_coefficients{
      0.2, -1.1, 0.7,
      1.3, 0.4, -0.9,
      -0.6, 1.7, 0.25,
      2.0, -0.3, 0.1,
      -1.4, -0.8, 1.9,
      0.75, 1.25, -0.5,
  };
  std::vector<double> contracted_values(
      node_radii.size() * output_count);
  std::vector<double> contracted_derivatives(contracted_values.size());
  ye3t::runtime::pace_radial_channel_contraction_with_derivative<double>(
      radial_values.data(), radial_derivatives.data(),
      radial_coefficients.data(), interval_count, radial_count, output_count,
      contracted_values.data(), contracted_derivatives.data());

  std::vector<double> radial_table(
      static_cast<std::size_t>((interval_count + 1) * radial_count * 4));
  std::vector<double> contracted_table(
      static_cast<std::size_t>((interval_count + 1) * output_count * 4));
  ye3t::runtime::pace_uniform_cubic_spline_build<double>(
      radial_values.data(), radial_derivatives.data(), interval_count,
      radial_count, cutoff, radial_table.data());
  ye3t::runtime::pace_uniform_cubic_spline_build<double>(
      contracted_values.data(), contracted_derivatives.data(), interval_count,
      output_count, cutoff, contracted_table.data());

  std::vector<double> spline_radial_values(radii.size() * radial_count);
  std::vector<double> spline_radial_derivatives(spline_radial_values.size());
  std::vector<double> spline_contracted_values(radii.size() * output_count);
  std::vector<double> spline_contracted_derivatives(
      spline_contracted_values.size());
  ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
      radii.data(), radial_table.data(), static_cast<std::int64_t>(radii.size()),
      radial_count, interval_count, cutoff, spline_radial_values.data(),
      spline_radial_derivatives.data());
  ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
      radii.data(), contracted_table.data(),
      static_cast<std::int64_t>(radii.size()), output_count, interval_count,
      cutoff, spline_contracted_values.data(),
      spline_contracted_derivatives.data());
  std::vector<double> contraction_after_spline(
      spline_contracted_values.size());
  std::vector<double> derivative_contraction_after_spline(
      spline_contracted_values.size());
  ye3t::runtime::pace_radial_channel_contraction_with_derivative<double>(
      spline_radial_values.data(), spline_radial_derivatives.data(),
      radial_coefficients.data(), static_cast<std::int64_t>(radii.size()),
      radial_count, output_count, contraction_after_spline.data(),
      derivative_contraction_after_spline.data());
  compare(spline_contracted_values, contraction_after_spline,
          "PACE spline radial contraction values");
  compare(spline_contracted_derivatives, derivative_contraction_after_spline,
          "PACE spline radial contraction derivatives");

  const double step = 1.0e-7;
  for (double radius : {1.17, 3.55}) {
    const double plus_radius = radius + step;
    const double minus_radius = radius - step;
    std::vector<double> center(output_count);
    std::vector<double> center_derivative(output_count);
    std::vector<double> plus(output_count);
    std::vector<double> scratch(output_count);
    std::vector<double> minus(output_count);
    ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
        &radius, contracted_table.data(), 1, output_count, interval_count,
        cutoff, center.data(), center_derivative.data());
    ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
        &plus_radius, contracted_table.data(), 1, output_count, interval_count,
        cutoff, plus.data(), scratch.data());
    ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
        &minus_radius, contracted_table.data(), 1, output_count, interval_count,
        cutoff, minus.data(), scratch.data());
    for (std::int64_t output = 0; output < output_count; ++output) {
      const double finite_difference =
          (plus[static_cast<std::size_t>(output)] -
           minus[static_cast<std::size_t>(output)]) /
          (2.0 * step);
      const double analytic = center_derivative[static_cast<std::size_t>(output)];
      const double tolerance = 2.0e-7 * (1.0 + std::abs(analytic));
      require(std::abs(analytic - finite_difference) <= tolerance,
              "PACE contracted spline finite-difference mismatch");
    }
  }

  require_invalid_argument(
      [&]() {
        ye3t::runtime::pace_radial_channel_contraction_with_derivative<double>(
            radial_values.data(), radial_derivatives.data(),
            radial_coefficients.data(), interval_count, radial_count, 0,
            contracted_values.data(), contracted_derivatives.data());
      },
      "empty PACE radial contraction");
}

void run_pinned_pace_radial_spline_fixture(
    const RadialSplineFixture &fixture)
{
  const double scale =
      static_cast<double>(fixture.interval_count) / fixture.cutoff;
  const double spacing = 1.0 / scale;
  std::vector<double> node_radii(
      static_cast<std::size_t>(fixture.interval_count));
  std::vector<double> cutoffs(node_radii.size(), fixture.cutoff);
  std::vector<double> cutoff_widths(
      node_radii.size(), fixture.cutoff_width);
  std::vector<double> lambdas(node_radii.size(), fixture.radial_lambda);
  for (std::int64_t node = 1; node <= fixture.interval_count; ++node)
    node_radii[static_cast<std::size_t>(node - 1)] = spacing * node;

  std::vector<double> node_g(
      node_radii.size() * static_cast<std::size_t>(fixture.radial_count));
  std::vector<double> node_dg(node_g.size());
  ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
      node_radii.data(), cutoffs.data(), cutoff_widths.data(), lambdas.data(),
      fixture.interval_count, fixture.radial_count, node_g.data(),
      node_dg.data());
  std::vector<double> node_R(
      node_radii.size() * static_cast<std::size_t>(fixture.output_count));
  std::vector<double> node_dR(node_R.size());
  ye3t::runtime::pace_radial_channel_contraction_with_derivative<double>(
      node_g.data(), node_dg.data(), fixture.radial_coefficients.data(),
      fixture.interval_count, fixture.radial_count, fixture.output_count,
      node_R.data(), node_dR.data());

  std::vector<double> g_table(
      static_cast<std::size_t>(
          (fixture.interval_count + 1) * fixture.radial_count * 4));
  std::vector<double> R_table(
      static_cast<std::size_t>(
          (fixture.interval_count + 1) * fixture.output_count * 4));
  ye3t::runtime::pace_uniform_cubic_spline_build<double>(
      node_g.data(), node_dg.data(), fixture.interval_count,
      fixture.radial_count, fixture.cutoff, g_table.data());
  ye3t::runtime::pace_uniform_cubic_spline_build<double>(
      node_R.data(), node_dR.data(), fixture.interval_count,
      fixture.output_count, fixture.cutoff, R_table.data());

  std::vector<double> actual_g(fixture.expected_g.size());
  std::vector<double> actual_dg(fixture.expected_dg.size());
  std::vector<double> actual_R(fixture.expected_R.size());
  std::vector<double> actual_dR(fixture.expected_dR.size());
  const auto point_count = static_cast<std::int64_t>(fixture.radii.size());
  ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
      fixture.radii.data(), g_table.data(), point_count,
      fixture.radial_count, fixture.interval_count, fixture.cutoff,
      actual_g.data(), actual_dg.data());
  ye3t::runtime::pace_uniform_cubic_spline_evaluate_with_derivative<double>(
      fixture.radii.data(), R_table.data(), point_count,
      fixture.output_count, fixture.interval_count, fixture.cutoff,
      actual_R.data(), actual_dR.data());
  compare(actual_g, fixture.expected_g, "pinned PACE spline g");
  compare(actual_dg, fixture.expected_dg, "pinned PACE spline dg");
  compare(actual_R, fixture.expected_R, "pinned PACE spline R");
  compare(actual_dR, fixture.expected_dR, "pinned PACE spline dR");
}

void run_pace_radial_fixture()
{
  const std::vector<double> radii{1.37, 3.995, 4.0, 4.01};
  const std::vector<double> cutoffs(radii.size(), 4.0);
  const std::vector<double> cutoff_widths(radii.size(), 0.01);
  const std::vector<double> lambdas(radii.size(), 0.5723);
  constexpr std::int64_t radial_count = 3;
  std::vector<double> values(
      radii.size() * radial_count,
      std::numeric_limits<double>::quiet_NaN());
  std::vector<double> derivatives(
      values.size(), std::numeric_limits<double>::quiet_NaN());
  ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
      radii.data(), cutoffs.data(), cutoff_widths.data(), lambdas.data(),
      static_cast<std::int64_t>(radii.size()), radial_count, values.data(),
      derivatives.data());

  compare(
      values,
      {
          0.7374281949352972,
          0.4362147381563176,
          0.7127134551156157,
          1.9276546323610626e-06,
          1.7861195058139943e-09,
          7.1378581177282385e-09,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      },
      "PACE radial values");
  compare(
      derivatives,
      {
          -0.3456000490756779,
          -0.40345387377939357,
          -0.18827917863553195,
          -1.3766514252119698e-03,
          -1.6329245668914275e-06,
          -6.5243216946332245e-06,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      },
      "PACE radial derivatives");

  const double step = 1.0e-7;
  for (std::int64_t edge = 0; edge < 2; ++edge) {
    for (std::int64_t radial = 0; radial < radial_count; ++radial) {
      double plus_radius = radii[static_cast<std::size_t>(edge)] + step;
      double minus_radius = radii[static_cast<std::size_t>(edge)] - step;
      double plus_values[radial_count];
      double plus_derivatives[radial_count];
      double minus_values[radial_count];
      double minus_derivatives[radial_count];
      ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
          &plus_radius, &cutoffs[static_cast<std::size_t>(edge)],
          &cutoff_widths[static_cast<std::size_t>(edge)],
          &lambdas[static_cast<std::size_t>(edge)], 1, radial_count,
          plus_values, plus_derivatives);
      ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
          &minus_radius, &cutoffs[static_cast<std::size_t>(edge)],
          &cutoff_widths[static_cast<std::size_t>(edge)],
          &lambdas[static_cast<std::size_t>(edge)], 1, radial_count,
          minus_values, minus_derivatives);
      const double finite_difference =
          (plus_values[radial] - minus_values[radial]) / (2.0 * step);
      const double analytic =
          derivatives[static_cast<std::size_t>(edge * radial_count + radial)];
      const double tolerance = 2.0e-7 * (1.0 + std::abs(analytic));
      require(std::abs(analytic - finite_difference) <= tolerance,
              "PACE radial finite-difference mismatch");
    }
  }

  double join_radius = 3.99;
  double join_plus_radius = join_radius + step;
  double join_minus_radius = join_radius - step;
  double join_values[radial_count];
  double join_derivatives[radial_count];
  double join_plus_values[radial_count];
  double join_plus_derivatives[radial_count];
  double join_minus_values[radial_count];
  double join_minus_derivatives[radial_count];
  ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
      &join_radius, cutoffs.data(), cutoff_widths.data(), lambdas.data(), 1,
      radial_count, join_values, join_derivatives);
  ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
      &join_plus_radius, cutoffs.data(), cutoff_widths.data(), lambdas.data(),
      1, radial_count, join_plus_values, join_plus_derivatives);
  ye3t::runtime::pace_cheb_exp_cos_radial_table_with_derivative<double>(
      &join_minus_radius, cutoffs.data(), cutoff_widths.data(), lambdas.data(),
      1, radial_count, join_minus_values, join_minus_derivatives);
  for (std::int64_t radial = 0; radial < radial_count; ++radial) {
    require(
        std::isfinite(join_values[radial]) &&
            std::isfinite(join_plus_derivatives[radial]) &&
            std::isfinite(join_minus_derivatives[radial]),
        "PACE radial taper-join result is not finite");
    const double finite_difference =
        (join_plus_values[radial] - join_minus_values[radial]) /
        (2.0 * step);
    const double tolerance =
        2.0e-7 * (1.0 + std::abs(join_derivatives[radial]));
    require(
        std::abs(join_derivatives[radial] - finite_difference) <= tolerance,
        "PACE radial taper-join finite-difference mismatch");
  }
}

std::vector<double> forward(const Fixture &fixture,
                            const std::vector<double> &input)
{
  std::vector<double> output(
      static_cast<std::size_t>(fixture.batch_size * fixture.output_dimension),
      std::numeric_limits<double>::quiet_NaN());
  ye3t::runtime::symmetric_power_shared_monomial_forward<double>(
      input.data(), fixture.batch_size, fixture.input_dimension,
      fixture.monomial_counts.data(), fixture.monomial_count,
      fixture.output_offsets.data(), fixture.coefficient_terms.data(),
      fixture.coefficient_values.data(), fixture.coefficient_count,
      fixture.output_dimension, output.data());
  return output;
}

double objective(const Fixture &fixture, const std::vector<double> &input)
{
  const auto output = forward(fixture, input);
  return std::inner_product(output.begin(), output.end(),
                            fixture.output_adjoint.begin(), 0.0);
}

void run(const Fixture &fixture)
{
  compare(forward(fixture, fixture.input), fixture.expected_output, "forward");

  std::vector<double> input_adjoint(
      static_cast<std::size_t>(fixture.batch_size * fixture.input_dimension),
      std::numeric_limits<double>::quiet_NaN());
  ye3t::runtime::symmetric_power_shared_monomial_adjoint<double>(
      fixture.output_adjoint.data(), fixture.input.data(), fixture.batch_size,
      fixture.input_dimension, fixture.monomial_counts.data(),
      fixture.monomial_count, fixture.output_offsets.data(),
      fixture.coefficient_terms.data(), fixture.coefficient_values.data(),
      fixture.coefficient_count, fixture.output_dimension,
      input_adjoint.data());
  compare(input_adjoint, fixture.expected_input_adjoint, "adjoint");

  const double step = 1.0e-6;
  std::vector<double> finite_difference(input_adjoint.size(), 0.0);
  for (std::size_t index = 0; index < fixture.input.size(); ++index) {
    auto plus = fixture.input;
    auto minus = fixture.input;
    plus[index] += step;
    minus[index] -= step;
    finite_difference[index] =
        (objective(fixture, plus) - objective(fixture, minus)) /
        (2.0 * step);
  }
  for (std::size_t index = 0; index < input_adjoint.size(); ++index) {
    const double tolerance = 2.0e-9 * (1.0 + std::abs(input_adjoint[index]));
    require(std::abs(input_adjoint[index] - finite_difference[index]) <=
                tolerance,
            "finite-difference adjoint mismatch at index " +
                std::to_string(index));
  }

  Fixture invalid_schema = fixture;
  invalid_schema.schema = "unknown";
  bool rejected = false;
  try {
    validate_fixture(invalid_schema);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "unknown schema was not rejected");

  Fixture invalid_term = fixture;
  invalid_term.coefficient_terms[0] = fixture.monomial_count;
  rejected = false;
  try {
    validate_fixture(invalid_term);
  } catch (const std::runtime_error &) {
    rejected = true;
  }
  require(rejected, "invalid monomial reference was not rejected");
}

void run_complex_spherical_table_fixture()
{
  using Complex = std::complex<double>;
  constexpr std::int64_t edge_count = 4;
  constexpr std::int64_t maximum_angular_momentum = 4;
  constexpr std::int64_t width =
      (maximum_angular_momentum + 1) * (maximum_angular_momentum + 1);
  const std::vector<double> vectors{
      1.2, -0.7, 0.5,
      0.0, 0.0, 1.3,
      -0.4, 0.9, 0.0,
      0.0, 0.0, 0.0,
  };
  std::vector<Complex> table(edge_count * width);
  std::vector<Complex> derivatives(edge_count * width * 3);
  std::vector<Complex> scaled_table(edge_count * width);
  std::vector<Complex> scaled_derivatives(edge_count * width * 3);
  std::vector<Complex> planned_table(edge_count * width);
  std::vector<Complex> planned_derivatives(edge_count * width * 3);
  const std::int64_t nonnegative_width =
      ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(
          maximum_angular_momentum);
  require(nonnegative_width == 15,
          "complex spherical nonnegative table width mismatch");
  std::vector<Complex> nonnegative_table(edge_count * nonnegative_width);
  std::vector<Complex> nonnegative_derivatives(
      edge_count * nonnegative_width * 3);
  constexpr std::int64_t physical_edge_count = edge_count - 1;
  std::vector<double> radii(physical_edge_count);
  std::vector<double> unit_vectors(physical_edge_count * 3);
  for (std::int64_t edge = 0; edge < physical_edge_count; ++edge) {
    const double x = vectors[edge * 3];
    const double y = vectors[edge * 3 + 1];
    const double z = vectors[edge * 3 + 2];
    radii[edge] = std::sqrt(x * x + y * y + z * z);
    for (std::int64_t axis = 0; axis < 3; ++axis)
      unit_vectors[edge * 3 + axis] =
          vectors[edge * 3 + axis] / radii[edge];
  }
  std::vector<Complex> unit_table(physical_edge_count * nonnegative_width);
  std::vector<Complex> unit_derivatives(
      physical_edge_count * nonnegative_width * 3);
  const double scale = 3.25;
  ye3t::runtime::complex_spherical_harmonics_table_with_derivative<double>(
      vectors.data(), edge_count, maximum_angular_momentum, 1.0e-14,
      table.data(), derivatives.data());
  ye3t::runtime::complex_spherical_harmonics_table_with_derivative<double>(
      vectors.data(), edge_count, maximum_angular_momentum, 1.0e-14,
      scaled_table.data(), scaled_derivatives.data(), scale);
  const std::int64_t plan_size =
      ye3t::runtime::complex_spherical_harmonics_table_plan_size(
          maximum_angular_momentum);
  std::vector<double> plan(static_cast<std::size_t>(plan_size));
  std::vector<Complex> workspace(
      static_cast<std::size_t>(maximum_angular_momentum + 1));
  ye3t::runtime::build_complex_spherical_harmonics_table_plan<double>(
      maximum_angular_momentum, plan.data(), plan_size);
  ye3t::runtime::
      complex_spherical_harmonics_table_with_derivative_prevalidated<double>(
          vectors.data(), edge_count, maximum_angular_momentum, 1.0e-14,
          plan.data(), plan_size, workspace.data(),
          static_cast<std::int64_t>(workspace.size()), planned_table.data(),
          planned_derivatives.data(), scale);
  ye3t::runtime::
      complex_spherical_harmonics_nonnegative_table_with_derivative_prevalidated<
          double>(
          vectors.data(), edge_count, maximum_angular_momentum, 1.0e-14,
          plan.data(), plan_size, workspace.data(),
          static_cast<std::int64_t>(workspace.size()),
          nonnegative_table.data(), nonnegative_derivatives.data(), scale);
  ye3t::runtime::
      complex_spherical_harmonics_nonnegative_unit_table_with_derivative_prevalidated<
          double>(
          unit_vectors.data(), radii.data(), physical_edge_count,
          maximum_angular_momentum, 1.0e-14, plan.data(), plan_size,
          workspace.data(), static_cast<std::int64_t>(workspace.size()),
          unit_table.data(), unit_derivatives.data());

  for (std::int64_t angular_momentum = 0;
       angular_momentum <= maximum_angular_momentum;
       ++angular_momentum) {
    const std::int64_t degree_width = 2 * angular_momentum + 1;
    std::vector<Complex> degree(edge_count * degree_width);
    std::vector<Complex> degree_derivatives(edge_count * degree_width * 3);
    ye3t::runtime::complex_spherical_harmonics_with_derivative<double>(
        vectors.data(), edge_count, angular_momentum, 1.0e-14,
        degree.data(), degree_derivatives.data());
    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
      for (std::int64_t component = 0; component < degree_width; ++component) {
        const std::int64_t source = edge * degree_width + component;
        const std::int64_t destination =
            edge * width + angular_momentum * angular_momentum + component;
        require(std::abs(table[destination] - degree[source]) <= 2.0e-12,
                "complex spherical table value mismatch");
        require(
            std::abs(scaled_table[destination] - scale * degree[source]) <=
                7.0e-12,
            "scaled complex spherical table value mismatch");
        require(
            std::abs(planned_table[destination] - scaled_table[destination]) <=
                2.0e-15,
            "prevalidated complex spherical table value mismatch");
        const std::int64_t magnetic = component - angular_momentum;
        const std::int64_t nonnegative =
            edge * nonnegative_width +
            angular_momentum * (angular_momentum + 1) / 2 +
            std::abs(magnetic);
        const double sign = std::abs(magnetic) % 2 == 0 ? 1.0 : -1.0;
        const Complex reconstructed = magnetic < 0
            ? sign * std::conj(nonnegative_table[nonnegative])
            : nonnegative_table[nonnegative];
        require(std::abs(reconstructed - planned_table[destination]) <= 2.0e-15,
                "nonnegative complex spherical reconstruction mismatch");
        if (edge < physical_edge_count) {
          require(
              std::abs(
                  scale * unit_table[nonnegative] -
                  nonnegative_table[nonnegative]) <= 2.0e-15,
              "precomputed-unit complex spherical value mismatch");
        }
        for (std::int64_t axis = 0; axis < 3; ++axis) {
          const double derivative_residual = std::abs(
              derivatives[destination * 3 + axis] -
              degree_derivatives[source * 3 + axis]);
          require(
              derivative_residual <= 2.0e-12,
              "complex spherical table derivative mismatch at l=" +
                  std::to_string(angular_momentum) + ", edge=" +
                  std::to_string(edge) + ", component=" +
                  std::to_string(component) + ", axis=" +
                  std::to_string(axis) + ", residual=" +
                  std::to_string(derivative_residual));
          require(
              std::abs(
                  scaled_derivatives[destination * 3 + axis] -
                  scale * degree_derivatives[source * 3 + axis]) <= 7.0e-12,
              "scaled complex spherical table derivative mismatch");
          require(
              std::abs(
                  planned_derivatives[destination * 3 + axis] -
                  scaled_derivatives[destination * 3 + axis]) <= 2.0e-15,
              "prevalidated complex spherical table derivative mismatch");
          const Complex reconstructed_derivative = magnetic < 0
              ? sign * std::conj(
                    nonnegative_derivatives[nonnegative * 3 + axis])
              : nonnegative_derivatives[nonnegative * 3 + axis];
          require(
              std::abs(
                  reconstructed_derivative -
                  planned_derivatives[destination * 3 + axis]) <= 2.0e-15,
              "nonnegative complex spherical derivative reconstruction mismatch");
          if (edge < physical_edge_count) {
            require(
                std::abs(
                    scale * unit_derivatives[nonnegative * 3 + axis] -
                    nonnegative_derivatives[nonnegative * 3 + axis]) <=
                    2.0e-15,
                "precomputed-unit complex spherical derivative mismatch");
          }
        }
      }
    }
  }

  bool rejected = false;
  try {
    ye3t::runtime::
        complex_spherical_harmonics_table_with_derivative_prevalidated<double>(
            vectors.data(), edge_count, maximum_angular_momentum, 1.0e-14,
            plan.data(), plan_size - 1, workspace.data(),
            static_cast<std::int64_t>(workspace.size()), planned_table.data(),
            planned_derivatives.data(), scale);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  require(rejected, "undersized complex spherical plan was not rejected");
}

void run_ace_coupled_product_dag_linear_fixture() {
  using Complex = std::complex<double>;
  constexpr std::int64_t batch_size = 3;
  constexpr std::int64_t input_stride = 4;
  constexpr std::int64_t tile_size = 2;
  const std::vector<Complex> input{
      {0.8, -0.3}, {1.2, 0.4},  {-0.5, 0.7}, {2.0, -1.0},
      {0.0, 0.0},  {1.0, 0.0},  {1.0, 0.0},  {1.0, 2.0},
      {-0.2, 0.9}, {0.6, -0.8}, {1.1, 0.2},  {-0.7, 0.5},
  };
  const std::vector<std::int64_t> node_offsets{0, 2, 4, 6};
  const std::vector<std::int64_t> node_dimensions{2, 2, 2, 1};
  const std::vector<std::int64_t> node_leaf_offsets{0, 2, -1, -1};
  const std::vector<std::int64_t> leaf_input_components{0, 2, 1, 0};
  const std::vector<std::int64_t> node_coefficient_offsets{0, 0, 0, 4, 7};
  const std::vector<std::int64_t> coefficient_left_components{0, 1, 0, 1,
                                                              4, 5, 4};
  const std::vector<std::int64_t> coefficient_right_components{2, 3, 3, 2,
                                                               4, 5, 5};
  const std::vector<std::int64_t> coefficient_output_components{4, 4, 5, 5,
                                                                6, 6, 6};
  const std::vector<double> coefficient_values{0.5, -0.25, 0.75, 0.125,
                                               1.2, -0.4,  0.3};
  const std::vector<std::int64_t> readout_components{6};
  const std::vector<double> readout_coefficients{1.1};
  std::vector<double> workspace(
      static_cast<std::size_t>((4 * 7 + 2) * tile_size));
  std::vector<Complex> output(batch_size, Complex(0.0, 0.0));
  std::vector<Complex> input_adjoint(batch_size * input_stride,
                                     Complex(0.0, 0.0));

  auto evaluate = [](const Complex *row) {
    const Complex u = 0.5 * row[0] * row[1] - 0.25 * row[2] * row[0];
    const Complex v = 0.75 * row[0] * row[0] + 0.125 * row[2] * row[1];
    return 1.1 * (1.2 * u * u - 0.4 * v * v + 0.3 * u * v);
  };

  ye3t::runtime::
      ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
          input.data(), batch_size, input_stride, node_offsets.data(),
          node_dimensions.data(), node_leaf_offsets.data(),
          leaf_input_components.data(),
          static_cast<std::int64_t>(leaf_input_components.size()),
          node_coefficient_offsets.data(),
          static_cast<std::int64_t>(node_offsets.size()),
          coefficient_left_components.data(),
          coefficient_right_components.data(),
          coefficient_output_components.data(), coefficient_values.data(),
          static_cast<std::int64_t>(coefficient_values.size()),
          readout_components.data(), readout_coefficients.data(),
          static_cast<std::int64_t>(readout_components.size()), tile_size,
          workspace.data(), static_cast<std::int64_t>(workspace.size()),
          output.data(), input_adjoint.data());

  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Complex expected = evaluate(input.data() + batch * input_stride);
    require(std::abs(output[batch] - expected) <= 3.0e-14,
            "ACE coupled-product DAG forward mismatch");
  }

  const double step = 1.0e-7;
  for (std::size_t index = 0; index < input.size(); ++index) {
    const std::int64_t batch = static_cast<std::int64_t>(index) / input_stride;
    for (int axis = 0; axis < 2; ++axis) {
      auto plus = input;
      auto minus = input;
      const Complex delta = axis == 0 ? Complex(step, 0.0) : Complex(0.0, step);
      plus[index] += delta;
      minus[index] -= delta;
      const double finite_difference =
          (evaluate(plus.data() + batch * input_stride).real() -
           evaluate(minus.data() + batch * input_stride).real()) /
          (2.0 * step);
      const double analytic =
          axis == 0 ? input_adjoint[index].real() : input_adjoint[index].imag();
      const double tolerance = 5.0e-8 * (1.0 + std::abs(analytic));
      require(std::abs(analytic - finite_difference) <= tolerance,
              "ACE coupled-product DAG adjoint mismatch at index " +
                  std::to_string(index));
    }
  }

  const std::vector<Complex> output_seed(batch_size, Complex(0.3, -0.2));
  const std::vector<Complex> adjoint_seed(batch_size * input_stride,
                                          Complex(-0.4, 0.1));
  auto accumulated_output = output_seed;
  auto accumulated_adjoint = adjoint_seed;
  ye3t::runtime::
      ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
          input.data(), batch_size, input_stride, node_offsets.data(),
          node_dimensions.data(), node_leaf_offsets.data(),
          leaf_input_components.data(),
          static_cast<std::int64_t>(leaf_input_components.size()),
          node_coefficient_offsets.data(),
          static_cast<std::int64_t>(node_offsets.size()),
          coefficient_left_components.data(),
          coefficient_right_components.data(),
          coefficient_output_components.data(), coefficient_values.data(),
          static_cast<std::int64_t>(coefficient_values.size()),
          readout_components.data(), readout_coefficients.data(),
          static_cast<std::int64_t>(readout_components.size()), tile_size,
          workspace.data(), static_cast<std::int64_t>(workspace.size()),
          accumulated_output.data(), accumulated_adjoint.data());
  for (std::size_t index = 0; index < output.size(); ++index)
    require(std::abs(accumulated_output[index] - output_seed[index] -
                     output[index]) <= 3.0e-14,
            "ACE coupled-product DAG output did not accumulate");
  for (std::size_t index = 0; index < input_adjoint.size(); ++index)
    require(std::abs(accumulated_adjoint[index] - adjoint_seed[index] -
                     input_adjoint[index]) <= 3.0e-14,
            "ACE coupled-product DAG input adjoint did not accumulate");

  require_invalid_argument(
      [&]() {
        ye3t::runtime::
            ace_coupled_product_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
                input.data(), batch_size, input_stride, node_offsets.data(),
                node_dimensions.data(), node_leaf_offsets.data(),
                leaf_input_components.data(),
                static_cast<std::int64_t>(leaf_input_components.size()),
                node_coefficient_offsets.data(),
                static_cast<std::int64_t>(node_offsets.size()),
                coefficient_left_components.data(),
                coefficient_right_components.data(),
                coefficient_output_components.data(), coefficient_values.data(),
                static_cast<std::int64_t>(coefficient_values.size()),
                readout_components.data(), readout_coefficients.data(),
                static_cast<std::int64_t>(readout_components.size()), tile_size,
                workspace.data(),
                static_cast<std::int64_t>(workspace.size() - 1), output.data(),
                input_adjoint.data());
      },
      "undersized ACE coupled-product DAG workspace");
}

void run_sparse_symmetric_power_linear_fixture()
{
  using Complex = std::complex<double>;
  constexpr std::int64_t batch_size = 2;
  constexpr std::int64_t input_dimension = 4;
  const std::vector<Complex> input{
      {2.0, 1.0}, {0.0, 0.0}, {-0.5, 0.25}, {3.0, -2.0},
      {0.0, 0.0}, {1.0, -1.0}, {0.0, 0.0}, {2.0, 0.5},
  };
  const std::vector<std::int64_t> factor_offsets{0, 2, 4, 6};
  const std::vector<std::int64_t> factor_indices{0, 2, 1, 3, 0, 1};
  const std::vector<std::int64_t> factor_exponents{2, 1, 1, 2, 1, 1};
  const std::vector<Complex> coefficients{
      {0.7, 0.0}, {-1.2, 0.0}, {0.4, 0.0}};
  std::vector<Complex> output(batch_size);
  std::vector<Complex> adjoint(batch_size * input_dimension);
  ye3t::runtime::symmetric_power_sparse_monomial_linear_forward_adjoint(
      input.data(), batch_size, input_dimension, factor_offsets.data(),
      factor_indices.data(), factor_exponents.data(),
      static_cast<std::int64_t>(factor_indices.size()), coefficients.data(),
      static_cast<std::int64_t>(coefficients.size()), output.data(),
      adjoint.data());
  std::vector<Complex> prevalidated_output(batch_size);
  std::vector<Complex> prevalidated_adjoint(batch_size * input_dimension);
  std::vector<Complex> workspace(10);
  ye3t::runtime::symmetric_power_sparse_monomial_linear_forward_adjoint_prevalidated(
      input.data(), batch_size, input_dimension, factor_offsets.data(),
      factor_indices.data(), factor_exponents.data(),
      static_cast<std::int64_t>(factor_indices.size()), coefficients.data(),
      static_cast<std::int64_t>(coefficients.size()), 2, workspace.data(),
      static_cast<std::int64_t>(workspace.size()), prevalidated_output.data(),
      prevalidated_adjoint.data());
  require(prevalidated_output == output,
          "prevalidated sparse symmetric-power forward mismatch");
  require(prevalidated_adjoint == adjoint,
          "prevalidated sparse symmetric-power adjoint mismatch");

  const std::vector<std::int64_t> power_channels{0, 2, 1, 3, 0};
  const std::vector<std::int64_t> power_exponents{2, 1, 1, 2, 1};
  const std::vector<std::int64_t> node_parents{-1, 0, 0, 0, 1, 2, 3};
  const std::vector<std::int64_t> node_powers{-1, 0, 2, 4, 1, 3, 2};
  const std::vector<std::int64_t> monomial_nodes{4, 5, 6};
  std::vector<Complex> dag_output(batch_size);
  std::vector<Complex> dag_adjoint(batch_size * input_dimension);
  std::vector<Complex> dag_workspace(24);
  ye3t::runtime::symmetric_power_sparse_monomial_dag_linear_forward_adjoint_prevalidated(
      input.data(), batch_size, input_dimension, power_channels.data(),
      power_exponents.data(), static_cast<std::int64_t>(power_channels.size()),
      node_parents.data(), node_powers.data(),
      static_cast<std::int64_t>(node_parents.size()), 3,
      monomial_nodes.data(),
      coefficients.data(), static_cast<std::int64_t>(coefficients.size()),
      dag_workspace.data(), static_cast<std::int64_t>(dag_workspace.size()),
      dag_output.data(), dag_adjoint.data());
  for (std::size_t index = 0; index < output.size(); ++index)
    require(std::abs(dag_output[index] - output[index]) <= 2.0e-14,
            "product-DAG symmetric-power forward mismatch");
  for (std::size_t index = 0; index < adjoint.size(); ++index)
    require(std::abs(dag_adjoint[index] - adjoint[index]) <= 2.0e-14,
            "product-DAG symmetric-power adjoint mismatch");

  const std::vector<double> real_coefficients{0.7, -1.2, 0.4};
  std::vector<Complex> real_coefficient_output(batch_size);
  std::vector<Complex> real_coefficient_adjoint(
      batch_size * input_dimension);
  ye3t::runtime::symmetric_power_sparse_monomial_dag_linear_forward_adjoint_prevalidated(
      input.data(), batch_size, input_dimension, power_channels.data(),
      power_exponents.data(), static_cast<std::int64_t>(power_channels.size()),
      node_parents.data(), node_powers.data(),
      static_cast<std::int64_t>(node_parents.size()), 3,
      monomial_nodes.data(),
      real_coefficients.data(),
      static_cast<std::int64_t>(real_coefficients.size()),
      dag_workspace.data(), static_cast<std::int64_t>(dag_workspace.size()),
      real_coefficient_output.data(), real_coefficient_adjoint.data());
  require(real_coefficient_output == dag_output,
          "real-coefficient product-DAG forward mismatch");
  require(real_coefficient_adjoint == dag_adjoint,
          "real-coefficient product-DAG adjoint mismatch");

  const std::vector<std::int64_t> binary_left{0, 2, 4};
  const std::vector<std::int64_t> binary_right{1, 3, 2};
  const std::vector<std::int64_t> binary_monomial_operands{5, 6, 7};
  std::vector<Complex> binary_output(batch_size);
  std::vector<Complex> binary_adjoint(batch_size * input_dimension);
  std::vector<Complex> binary_workspace(16);
  ye3t::runtime::
      symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_prevalidated(
          input.data(), batch_size, input_dimension, power_channels.data(),
          power_exponents.data(),
          static_cast<std::int64_t>(power_channels.size()), binary_left.data(),
          binary_right.data(), static_cast<std::int64_t>(binary_left.size()),
          binary_monomial_operands.data(), real_coefficients.data(),
          static_cast<std::int64_t>(real_coefficients.size()),
          binary_workspace.data(),
          static_cast<std::int64_t>(binary_workspace.size()),
          binary_output.data(), binary_adjoint.data());
  for (std::size_t index = 0; index < output.size(); ++index)
    require(std::abs(binary_output[index] - output[index]) <= 2.0e-14,
            "binary product-DAG symmetric-power forward mismatch");
  for (std::size_t index = 0; index < adjoint.size(); ++index)
    require(std::abs(binary_adjoint[index] - adjoint[index]) <= 2.0e-14,
            "binary product-DAG symmetric-power adjoint mismatch");

  const std::int64_t binary_tile_size = 2;
  const std::int64_t binary_value_count =
      static_cast<std::int64_t>(power_channels.size() + binary_left.size());
  std::vector<double> tiled_binary_workspace(static_cast<std::size_t>(
      (4 * binary_value_count + 2) * binary_tile_size));
  std::vector<Complex> tiled_binary_output(batch_size);
  std::vector<Complex> tiled_binary_adjoint(batch_size * input_dimension);
  ye3t::runtime::
      symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
          input.data(), batch_size, input_dimension, power_channels.data(),
          power_exponents.data(),
          static_cast<std::int64_t>(power_channels.size()), binary_left.data(),
          binary_right.data(), static_cast<std::int64_t>(binary_left.size()),
          binary_monomial_operands.data(), real_coefficients.data(),
          static_cast<std::int64_t>(real_coefficients.size()), binary_tile_size,
          tiled_binary_workspace.data(),
          static_cast<std::int64_t>(tiled_binary_workspace.size()),
          tiled_binary_output.data(), tiled_binary_adjoint.data());
  for (std::size_t index = 0; index < output.size(); ++index)
    require(std::abs(tiled_binary_output[index] - output[index]) <= 2.0e-14,
            "split-real tiled binary product-DAG forward mismatch");
  for (std::size_t index = 0; index < adjoint.size(); ++index)
    require(std::abs(tiled_binary_adjoint[index] - adjoint[index]) <= 2.0e-14,
            "split-real tiled binary product-DAG adjoint mismatch");

  auto polynomial = [&](const Complex *row) {
    return coefficients[0] * row[0] * row[0] * row[2] +
           coefficients[1] * row[1] * row[3] * row[3] +
           coefficients[2] * row[0] * row[1];
  };
  for (std::int64_t batch = 0; batch < batch_size; ++batch) {
    const Complex expected = polynomial(input.data() + batch * input_dimension);
    require(std::abs(output[batch] - expected) <= 2.0e-14,
            "sparse symmetric-power forward mismatch");
  }

  const double step = 1.0e-7;
  for (std::size_t index = 0; index < input.size(); ++index) {
    const std::int64_t batch =
        static_cast<std::int64_t>(index) / input_dimension;
    for (int axis = 0; axis < 2; ++axis) {
      auto plus = input;
      auto minus = input;
      const Complex delta = axis == 0 ? Complex(step, 0.0) : Complex(0.0, step);
      plus[index] += delta;
      minus[index] -= delta;
      const double finite_difference =
          (polynomial(plus.data() + batch * input_dimension).real() -
           polynomial(minus.data() + batch * input_dimension).real()) /
          (2.0 * step);
      const double analytic = axis == 0
          ? adjoint[index].real()
          : adjoint[index].imag();
      const double tolerance = 3.0e-8 * (1.0 + std::abs(analytic));
      require(std::abs(analytic - finite_difference) <= tolerance,
              "sparse symmetric-power adjoint mismatch at index " +
                  std::to_string(index));
    }
  }

  require_invalid_argument(
      [&]() {
        auto bad_offsets = factor_offsets;
        bad_offsets.back() -= 1;
        ye3t::runtime::symmetric_power_sparse_monomial_linear_forward_adjoint(
            input.data(), batch_size, input_dimension, bad_offsets.data(),
            factor_indices.data(), factor_exponents.data(),
            static_cast<std::int64_t>(factor_indices.size()),
            coefficients.data(),
            static_cast<std::int64_t>(coefficients.size()), output.data(),
            adjoint.data());
      },
      "incomplete sparse symmetric-power factor offsets");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::symmetric_power_sparse_monomial_linear_forward_adjoint_prevalidated(
            input.data(), batch_size, input_dimension, factor_offsets.data(),
            factor_indices.data(), factor_exponents.data(),
            static_cast<std::int64_t>(factor_indices.size()), coefficients.data(),
            static_cast<std::int64_t>(coefficients.size()), 2,
            workspace.data(), 9, prevalidated_output.data(),
            prevalidated_adjoint.data());
      },
      "undersized prevalidated sparse symmetric-power workspace");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::symmetric_power_sparse_monomial_dag_linear_forward_adjoint_prevalidated(
            input.data(), batch_size, input_dimension, power_channels.data(),
            power_exponents.data(),
            static_cast<std::int64_t>(power_channels.size()),
            node_parents.data(), node_powers.data(),
            static_cast<std::int64_t>(node_parents.size()), 3,
            monomial_nodes.data(), coefficients.data(),
            static_cast<std::int64_t>(coefficients.size()),
            dag_workspace.data(), 23, dag_output.data(), dag_adjoint.data());
      },
      "undersized product-DAG symmetric-power workspace");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::
            symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_prevalidated(
                input.data(), batch_size, input_dimension,
                power_channels.data(), power_exponents.data(),
                static_cast<std::int64_t>(power_channels.size()),
                binary_left.data(), binary_right.data(),
                static_cast<std::int64_t>(binary_left.size()),
                binary_monomial_operands.data(), real_coefficients.data(),
                static_cast<std::int64_t>(real_coefficients.size()),
                binary_workspace.data(), 15, binary_output.data(),
                binary_adjoint.data());
      },
      "undersized binary product-DAG symmetric-power workspace");
  require_invalid_argument(
      [&]() {
        ye3t::runtime::
            symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
                input.data(), batch_size, input_dimension,
                power_channels.data(), power_exponents.data(),
                static_cast<std::int64_t>(power_channels.size()),
                binary_left.data(), binary_right.data(),
                static_cast<std::int64_t>(binary_left.size()),
                binary_monomial_operands.data(), real_coefficients.data(),
                static_cast<std::int64_t>(real_coefficients.size()),
                binary_tile_size, tiled_binary_workspace.data(),
                static_cast<std::int64_t>(tiled_binary_workspace.size()) - 1,
                tiled_binary_output.data(), tiled_binary_adjoint.data());
      },
      "undersized split-real tiled binary product-DAG workspace");
}

template <typename Real>
void check_complex_spherical_recurrence(std::int64_t maximum_angular_momentum,
                                        Real value_tolerance,
                                        Real gradient_tolerance)
{
  using Complex = std::complex<Real>;
  // Generic, axis-aligned, and near-polar directions; radii differ so the
  // 1/radius chain rule is exercised.
  const std::vector<Real> vectors{
      Real(1.2), Real(-0.7), Real(0.5),
      Real(0.0), Real(0.0), Real(1.3),
      Real(0.0), Real(0.0), Real(-2.1),
      Real(0.9), Real(0.0), Real(0.0),
      Real(0.0), Real(-1.7), Real(0.0),
      Real(1.0e-6), Real(2.0e-6), Real(0.75),
      Real(-0.4), Real(0.9), Real(0.0),
      Real(2.5), Real(2.5), Real(-2.5),
  };
  const std::int64_t edge_count =
      static_cast<std::int64_t>(vectors.size() / 3);
  const std::int64_t width =
      ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(
          maximum_angular_momentum);
  std::vector<Real> radii(edge_count);
  std::vector<Real> unit_vectors(edge_count * 3);
  for (std::int64_t edge = 0; edge < edge_count; ++edge) {
    const Real x = vectors[edge * 3];
    const Real y = vectors[edge * 3 + 1];
    const Real z = vectors[edge * 3 + 2];
    radii[edge] = std::sqrt(x * x + y * y + z * z);
    for (std::int64_t axis = 0; axis < 3; ++axis)
      unit_vectors[edge * 3 + axis] = vectors[edge * 3 + axis] / radii[edge];
  }
  const Real scale = std::sqrt(Real(4) * std::acos(Real(-1)));
  const std::int64_t table_plan_size =
      ye3t::runtime::complex_spherical_harmonics_table_plan_size(
          maximum_angular_momentum);
  std::vector<Real> table_plan(static_cast<std::size_t>(table_plan_size));
  ye3t::runtime::build_complex_spherical_harmonics_table_plan<Real>(
      maximum_angular_momentum, table_plan.data(), table_plan_size, scale);
  std::vector<Complex> workspace(
      static_cast<std::size_t>(maximum_angular_momentum + 1));
  std::vector<Complex> table(edge_count * width);
  std::vector<Complex> table_derivatives(edge_count * width * 3);
  ye3t::runtime::
      complex_spherical_harmonics_nonnegative_unit_table_with_derivative_prevalidated<
          Real>(unit_vectors.data(), radii.data(), edge_count,
                maximum_angular_momentum, Real(1.0e-14), table_plan.data(),
                table_plan_size, workspace.data(),
                static_cast<std::int64_t>(workspace.size()), table.data(),
                table_derivatives.data());

  const std::int64_t plan_size =
      ye3t::runtime::complex_spherical_harmonics_recurrence_plan_size(
          maximum_angular_momentum);
  require(plan_size == 1 + 2 * width,
          "complex spherical recurrence plan size mismatch");
  std::vector<Real> plan(static_cast<std::size_t>(plan_size));
  ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<Real>(
      maximum_angular_momentum, plan.data(), plan_size, scale);
  require(std::abs(plan[0] - Real(1)) <= Real(1.0e-6),
          "scaled recurrence plan must start at Y00 = 1");
  std::vector<Complex> recurrence(edge_count * width);
  std::vector<Complex> recurrence_derivatives(edge_count * width * 3);
  ye3t::runtime::
      complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
          Real>(unit_vectors.data(), radii.data(), edge_count,
                maximum_angular_momentum, plan.data(), plan_size,
                recurrence.data(), recurrence_derivatives.data());
  for (std::size_t index = 0; index < recurrence.size(); ++index) {
    const Real magnitude = Real(1) + std::abs(table[index]);
    require(std::abs(recurrence[index] - table[index]) <=
                value_tolerance * magnitude,
            "complex spherical recurrence value mismatch");
    for (std::size_t axis = 0; axis < 3; ++axis) {
      const std::size_t entry = index * 3 + axis;
      const Real derivative_magnitude =
          Real(1) + std::abs(table_derivatives[entry]);
      require(std::abs(recurrence_derivatives[entry] -
                       table_derivatives[entry]) <=
                  gradient_tolerance * derivative_magnitude,
              "complex spherical recurrence derivative mismatch");
    }
  }
}

void run_complex_spherical_recurrence_fixture()
{
  // The recurrence must reproduce the polynomial-table convention, including
  // the Condon-Shortley phase, PACE Y00 = 1 scaling, and the raw-displacement
  // derivative chain rule, through the degrees used by the LAMMPS adapters.
  for (std::int64_t maximum_angular_momentum : {0, 1, 2, 3, 4, 6, 8, 12}) {
    check_complex_spherical_recurrence<double>(
        maximum_angular_momentum, 5.0e-13, 5.0e-13);
  }
  for (std::int64_t maximum_angular_momentum : {0, 1, 4, 8}) {
    check_complex_spherical_recurrence<float>(
        maximum_angular_momentum, 6.0e-5f, 6.0e-5f);
  }

  // Central finite differences of the recurrence values with respect to the
  // raw displacement, independent of the table derivative code.
  constexpr std::int64_t maximum_angular_momentum = 7;
  const std::int64_t width =
      ye3t::runtime::complex_spherical_harmonics_nonnegative_table_width(
          maximum_angular_momentum);
  const std::int64_t plan_size =
      ye3t::runtime::complex_spherical_harmonics_recurrence_plan_size(
          maximum_angular_momentum);
  std::vector<double> plan(static_cast<std::size_t>(plan_size));
  ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<double>(
      maximum_angular_momentum, plan.data(), plan_size);
  auto evaluate = [&](const std::vector<double> &vector,
                      std::vector<std::complex<double>> &values,
                      std::vector<std::complex<double>> &derivatives) {
    const double radius = std::sqrt(vector[0] * vector[0] +
                                    vector[1] * vector[1] +
                                    vector[2] * vector[2]);
    const std::vector<double> unit{vector[0] / radius, vector[1] / radius,
                                   vector[2] / radius};
    values.assign(static_cast<std::size_t>(width), {});
    derivatives.assign(static_cast<std::size_t>(width * 3), {});
    ye3t::runtime::
        complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
            double>(unit.data(), &radius, 1, maximum_angular_momentum,
                    plan.data(), plan_size, values.data(),
                    derivatives.data());
  };
  const std::vector<double> base{0.8, -1.1, 0.6};
  std::vector<std::complex<double>> values, derivatives, plus, minus, unused;
  evaluate(base, values, derivatives);
  const double step = 1.0e-5;
  for (std::size_t axis = 0; axis < 3; ++axis) {
    std::vector<double> shifted = base;
    shifted[axis] += step;
    evaluate(shifted, plus, unused);
    shifted[axis] -= 2.0 * step;
    evaluate(shifted, minus, unused);
    for (std::int64_t index = 0; index < width; ++index) {
      const std::complex<double> finite =
          (plus[index] - minus[index]) / (2.0 * step);
      const std::complex<double> analytic = derivatives[index * 3 + axis];
      require(std::abs(finite - analytic) <=
                  1.0e-7 * (1.0 + std::abs(analytic)),
              "complex spherical recurrence finite-difference mismatch");
    }
  }

  require_invalid_argument(
      [&]() {
        std::vector<double> undersized(static_cast<std::size_t>(plan_size - 1));
        ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<
            double>(maximum_angular_momentum, undersized.data(),
                    plan_size - 1);
      },
      "undersized complex spherical recurrence plan");
  require_invalid_argument(
      [&]() {
        const double unit[3] = {0.0, 0.0, 1.0};
        const double radius = 1.0;
        ye3t::runtime::
            complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
                double>(unit, &radius, 1, maximum_angular_momentum,
                        plan.data(), plan_size - 1, values.data(),
                        derivatives.data());
      },
      "undersized complex spherical recurrence plan at evaluation");
  require_invalid_argument(
      [&]() {
        const double radius = 1.0;
        ye3t::runtime::
            complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
                double>(nullptr, &radius, 1, maximum_angular_momentum,
                        plan.data(), plan_size, values.data(),
                        derivatives.data());
      },
      "null complex spherical recurrence input");
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: ye3t_runtime_core_standalone_test "
                 "MONOMIAL_FIXTURE RADIAL_SPLINE_FIXTURE\n";
    return 2;
  }
  try {
    run_pace_radial_fixture();
    run_pace_spline_and_contraction_fixture();
    run_pinned_pace_radial_spline_fixture(
        load_radial_spline_fixture(argv[2]));
    run(load_fixture(argv[1]));
    run_complex_spherical_table_fixture();
    run_complex_spherical_recurrence_fixture();
    run_sparse_symmetric_power_linear_fixture();
    run_ace_coupled_product_dag_linear_fixture();
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
