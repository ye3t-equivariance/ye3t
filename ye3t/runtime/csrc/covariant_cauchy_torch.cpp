// Torch adapter for the torch-free covariant lifted-Cauchy kernels.
//
// Registered as a fragment of the ``ye3t_runtime`` operator namespace so that
// it can be built and loaded independently of the execution-plan extension.

#include <torch/extension.h>
#include <torch/library.h>

#include <tuple>

#include "ye3t_covariant_cauchy_core.h"

namespace {

void check_cpu_contiguous(const torch::Tensor& tensor, torch::ScalarType dtype,
                          const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be a CPU tensor.");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has the wrong dtype.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

void check_plan(const torch::Tensor& term_row,
                const torch::Tensor& term_coefficient,
                const torch::Tensor& factor_offsets,
                const torch::Tensor& factors, std::int64_t input_count,
                std::int64_t row_count) {
  check_cpu_contiguous(term_row, torch::kInt64, "term_row");
  check_cpu_contiguous(term_coefficient, torch::kFloat64, "term_coefficient");
  check_cpu_contiguous(factor_offsets, torch::kInt64, "factor_offsets");
  check_cpu_contiguous(factors, torch::kInt64, "factors");
  const std::int64_t term_count = term_row.numel();
  TORCH_CHECK(term_coefficient.numel() == term_count,
              "term_coefficient must align with term_row.");
  TORCH_CHECK(factor_offsets.numel() == term_count + 1,
              "factor_offsets must have term_count + 1 entries.");
  if (term_count > 0) {
    TORCH_CHECK(term_row.min().item<std::int64_t>() >= 0 &&
                    term_row.max().item<std::int64_t>() < row_count,
                "term_row is out of range.");
    TORCH_CHECK(factor_offsets[0].item<std::int64_t>() == 0 &&
                    factor_offsets[term_count].item<std::int64_t>() ==
                        factors.numel(),
                "factor_offsets must span factors exactly.");
    TORCH_CHECK((factor_offsets.diff() >= 0).all().item<bool>(),
                "factor_offsets must be nondecreasing.");
  }
  if (factors.numel() > 0) {
    TORCH_CHECK(factors.min().item<std::int64_t>() >= 0 &&
                    factors.max().item<std::int64_t>() < input_count,
                "factors are out of range.");
  }
}

std::tuple<torch::Tensor, torch::Tensor> covariant_cauchy_forward_vjp(
    const torch::Tensor& inputs, const torch::Tensor& cotangent,
    const torch::Tensor& term_row, const torch::Tensor& term_coefficient,
    const torch::Tensor& factor_offsets, const torch::Tensor& factors,
    std::int64_t row_count) {
  check_cpu_contiguous(inputs, torch::kFloat64, "inputs");
  TORCH_CHECK(inputs.dim() == 2, "inputs must be [site_count, input_count].");
  TORCH_CHECK(row_count >= 0, "row_count must be nonnegative.");
  const std::int64_t site_count = inputs.size(0);
  const std::int64_t input_count = inputs.size(1);
  check_plan(term_row, term_coefficient, factor_offsets, factors, input_count,
             row_count);
  const bool reverse = cotangent.numel() > 0;
  if (reverse) {
    check_cpu_contiguous(cotangent, torch::kFloat64, "cotangent");
    TORCH_CHECK(cotangent.dim() == 2 && cotangent.size(0) == site_count &&
                    cotangent.size(1) == row_count,
                "cotangent must be [site_count, row_count].");
  }
  torch::Tensor outputs = torch::empty({site_count, row_count}, inputs.options());
  torch::Tensor gradient =
      reverse ? torch::empty_like(inputs) : torch::empty({0}, inputs.options());
  ye3t::covariant_cauchy::forward_vjp(
      site_count, input_count, row_count, term_row.numel(),
      term_row.data_ptr<std::int64_t>(), term_coefficient.data_ptr<double>(),
      factor_offsets.data_ptr<std::int64_t>(), factors.data_ptr<std::int64_t>(),
      inputs.data_ptr<double>(),
      reverse ? cotangent.data_ptr<double>() : nullptr,
      outputs.data_ptr<double>(),
      reverse ? gradient.data_ptr<double>() : nullptr);
  return std::make_tuple(outputs, gradient);
}

void covariant_cauchy_accumulate_normal_equations(
    const torch::Tensor& features, const torch::Tensor& targets,
    const torch::Tensor& site_weights, torch::Tensor gram, torch::Tensor rhs) {
  check_cpu_contiguous(features, torch::kFloat64, "features");
  check_cpu_contiguous(targets, torch::kFloat64, "targets");
  check_cpu_contiguous(gram, torch::kFloat64, "gram");
  check_cpu_contiguous(rhs, torch::kFloat64, "rhs");
  TORCH_CHECK(features.dim() == 3,
              "features must be [site_count, multiplet_count, width].");
  const std::int64_t site_count = features.size(0);
  const std::int64_t multiplet_count = features.size(1);
  const std::int64_t width = features.size(2);
  TORCH_CHECK(targets.dim() == 2 && targets.size(0) == site_count &&
                  targets.size(1) == width,
              "targets must be [site_count, width].");
  TORCH_CHECK(gram.dim() == 2 && gram.size(0) == multiplet_count &&
                  gram.size(1) == multiplet_count,
              "gram must be [multiplet_count, multiplet_count].");
  TORCH_CHECK(rhs.dim() == 1 && rhs.size(0) == multiplet_count,
              "rhs must be [multiplet_count].");
  const bool weighted = site_weights.numel() > 0;
  if (weighted) {
    check_cpu_contiguous(site_weights, torch::kFloat64, "site_weights");
    TORCH_CHECK(site_weights.dim() == 1 && site_weights.size(0) == site_count,
                "site_weights must be [site_count].");
  }
  ye3t::covariant_cauchy::accumulate_normal_equations(
      site_count, multiplet_count, width, features.data_ptr<double>(),
      targets.data_ptr<double>(),
      weighted ? site_weights.data_ptr<double>() : nullptr,
      gram.data_ptr<double>(), rhs.data_ptr<double>());
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(ye3t_runtime, library) {
  library.def(
      "covariant_cauchy_forward_vjp(Tensor inputs, Tensor cotangent, "
      "Tensor term_row, Tensor term_coefficient, Tensor factor_offsets, "
      "Tensor factors, int row_count) -> (Tensor, Tensor)",
      &covariant_cauchy_forward_vjp);
  library.def(
      "covariant_cauchy_accumulate_normal_equations(Tensor features, "
      "Tensor targets, Tensor site_weights, Tensor(a!) gram, Tensor(b!) rhs) "
      "-> ()",
      &covariant_cauchy_accumulate_normal_equations);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("covariant_cauchy_core_abi_version", []() { return 1; });
}
