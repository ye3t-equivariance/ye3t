#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include "ye3t_runtime_core.h"

namespace {

template <typename Scalar>
struct CoreScalar {
  using type = Scalar;
};

template <>
struct CoreScalar<c10::complex<float>> {
  using type = std::complex<float>;
};

template <>
struct CoreScalar<c10::complex<double>> {
  using type = std::complex<double>;
};

void check_data_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be a two-dimensional tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64 ||
          tensor.scalar_type() == torch::kComplexFloat ||
          tensor.scalar_type() == torch::kComplexDouble,
      name,
      " must use float32, float64, complex64, or complex128");
}

void check_data_vector(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64 ||
          tensor.scalar_type() == torch::kComplexFloat ||
          tensor.scalar_type() == torch::kComplexDouble,
      name,
      " must use float32, float64, complex64, or complex128");
}

void check_factor_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 3, name, " must be a three-dimensional tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64 ||
          tensor.scalar_type() == torch::kComplexFloat ||
          tensor.scalar_type() == torch::kComplexDouble,
      name,
      " must use float32, float64, complex64, or complex128");
}

void check_batched_adjoint_tensor(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 3, name, " must be a three-dimensional tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64 ||
          tensor.scalar_type() == torch::kComplexFloat ||
          tensor.scalar_type() == torch::kComplexDouble,
      name,
      " must use float32, float64, complex64, or complex128");
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

void check_indices(
    const torch::Tensor& rows,
    const torch::Tensor& columns,
    const torch::Tensor& values,
    const torch::Tensor& data,
    const char* prefix) {
  TORCH_CHECK(rows.device().is_cpu(), prefix, " rows must be on CPU");
  TORCH_CHECK(columns.device().is_cpu(), prefix, " columns must be on CPU");
  TORCH_CHECK(values.device().is_cpu(), prefix, " values must be on CPU");
  TORCH_CHECK(rows.is_contiguous(), prefix, " rows must be contiguous");
  TORCH_CHECK(columns.is_contiguous(), prefix, " columns must be contiguous");
  TORCH_CHECK(values.is_contiguous(), prefix, " values must be contiguous");
  TORCH_CHECK(rows.dim() == 1, prefix, " rows must be one-dimensional");
  TORCH_CHECK(columns.dim() == 1, prefix, " columns must be one-dimensional");
  TORCH_CHECK(values.dim() == 1, prefix, " values must be one-dimensional");
  TORCH_CHECK(rows.scalar_type() == torch::kInt64, prefix, " rows must be int64");
  TORCH_CHECK(columns.scalar_type() == torch::kInt64, prefix, " columns must be int64");
  TORCH_CHECK(
      values.scalar_type() == data.scalar_type(),
      prefix,
      " values must have the same dtype as data");
  TORCH_CHECK(
      rows.numel() == columns.numel() && rows.numel() == values.numel(),
      prefix,
      " COO arrays must have equal length");
}

void check_int64_vector(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt64,
      name,
      " must use int64");
}

void check_int64_matrix(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt64,
      name,
      " must use int64");
}

void check_real_vector(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.device().is_cpu(), name, " must be on CPU");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64,
      name,
      " must use float32 or float64");
}

void check_value_vector(
    const torch::Tensor& tensor,
    const torch::Tensor& data,
    const char* name) {
  TORCH_CHECK(tensor.device() == data.device(), name, " device mismatch");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == data.scalar_type(),
      name,
      " dtype must match input");
}

void check_factorized_workspace_dimension(
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    std::int64_t workspace_dimension) {
  TORCH_CHECK(
      workspace_dimension > 0,
      "factorized workspace_dimension must be positive");
  TORCH_CHECK(
      node_offsets.numel() > 0 &&
          node_dimensions.numel() == node_offsets.numel(),
      "factorized workspace metadata requires matching node arrays");
  const std::int64_t final_node = node_offsets.numel() - 1;
  const std::int64_t expected =
      node_offsets.data_ptr<std::int64_t>()[final_node] +
      node_dimensions.data_ptr<std::int64_t>()[final_node];
  TORCH_CHECK(
      workspace_dimension == expected,
      "factorized workspace_dimension does not match compiler metadata");
}

struct FactorizedRootMetadata {
  std::vector<std::int64_t> projection_starts;
  std::vector<std::int64_t> projection_dimensions;
  std::vector<std::int64_t> output_offsets;
  std::int64_t total_projection_dimension;
  std::int64_t output_dimension;
};

FactorizedRootMetadata uniform_factorized_root_metadata(
    const torch::Tensor& root_nodes,
    const torch::Tensor& node_dimensions,
    std::int64_t projection_dimension,
    std::int64_t output_dimension) {
  TORCH_CHECK(
      root_nodes.numel() > 0,
      "factorized plan requires at least one root");
  TORCH_CHECK(
      projection_dimension > 0,
      "factorized projection dimension must be positive");
  const auto* roots = root_nodes.data_ptr<std::int64_t>();
  const auto* dimensions = node_dimensions.data_ptr<std::int64_t>();
  const std::int64_t node_count = node_dimensions.numel();
  TORCH_CHECK(
      roots[0] >= 0 && roots[0] < node_count,
      "factorized root index is out of bounds");
  const std::int64_t root_dimension = dimensions[roots[0]];
  TORCH_CHECK(
      root_dimension > 0,
      "factorized root dimension must be positive");

  FactorizedRootMetadata metadata;
  metadata.total_projection_dimension = projection_dimension;
  metadata.projection_starts.resize(root_nodes.numel(), 0);
  metadata.projection_dimensions.resize(
      root_nodes.numel(),
      projection_dimension);
  metadata.output_offsets.resize(root_nodes.numel());
  for (std::int64_t root = 0; root < root_nodes.numel(); ++root) {
    TORCH_CHECK(
        roots[root] >= 0 && roots[root] < node_count,
        "factorized root index is out of bounds");
    TORCH_CHECK(
        dimensions[roots[root]] == root_dimension,
        "uniform factorized roots must have identical dimensions");
    metadata.output_offsets[root] =
        root * projection_dimension * root_dimension;
  }
  metadata.output_dimension =
      root_nodes.numel() * projection_dimension * root_dimension;
  TORCH_CHECK(
      output_dimension == metadata.output_dimension,
      "output width does not match uniform factorized root metadata");
  return metadata;
}

std::pair<std::int64_t, std::int64_t>
check_heterogeneous_factorized_root_metadata(
    const torch::Tensor& root_nodes,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t output_dimension) {
  check_int64_vector(
      root_projection_starts,
      "root_projection_starts");
  check_int64_vector(
      root_projection_dimensions,
      "root_projection_dimensions");
  check_int64_vector(root_output_offsets, "root_output_offsets");
  const std::int64_t root_count = root_nodes.numel();
  TORCH_CHECK(root_count > 0, "factorized plan requires roots");
  TORCH_CHECK(
      root_projection_starts.numel() == root_count &&
          root_projection_dimensions.numel() == root_count &&
          root_output_offsets.numel() == root_count,
      "heterogeneous factorized root metadata must match root count");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      projection_values.numel() > 0 &&
          projection_values.numel() % source_dimension == 0,
      "projection weights must be nonempty and divisible by source_dimension");
  const std::int64_t total_projection_dimension =
      projection_values.numel() / source_dimension;
  TORCH_CHECK(
      output_dimension > 0,
      "heterogeneous factorized output dimension must be positive");

  const auto* roots = root_nodes.data_ptr<std::int64_t>();
  const auto* dimensions = node_dimensions.data_ptr<std::int64_t>();
  const auto* projection_starts =
      root_projection_starts.data_ptr<std::int64_t>();
  const auto* projection_dimensions =
      root_projection_dimensions.data_ptr<std::int64_t>();
  const auto* output_offsets =
      root_output_offsets.data_ptr<std::int64_t>();
  const std::int64_t node_count = node_dimensions.numel();
  std::vector<std::pair<std::int64_t, std::int64_t>> intervals;
  intervals.reserve(root_count);
  for (std::int64_t root = 0; root < root_count; ++root) {
    TORCH_CHECK(
        roots[root] >= 0 && roots[root] < node_count,
        "factorized root index is out of bounds");
    const std::int64_t root_dimension = dimensions[roots[root]];
    const std::int64_t projection_start = projection_starts[root];
    const std::int64_t projection_dimension =
        projection_dimensions[root];
    TORCH_CHECK(
        root_dimension > 0,
        "factorized root dimension must be positive");
    TORCH_CHECK(
        projection_start >= 0 && projection_dimension > 0 &&
            projection_start <=
                total_projection_dimension - projection_dimension,
        "heterogeneous root projection interval is out of bounds");
    const std::int64_t block_dimension =
        projection_dimension * root_dimension;
    TORCH_CHECK(
        output_offsets[root] >= 0 &&
            block_dimension <= output_dimension &&
            output_offsets[root] <= output_dimension - block_dimension,
        "heterogeneous root output interval is out of bounds");
    intervals.emplace_back(
        output_offsets[root],
        output_offsets[root] + block_dimension);
  }
  std::sort(intervals.begin(), intervals.end());
  std::int64_t output_cursor = 0;
  for (const auto& interval : intervals) {
    TORCH_CHECK(
        interval.first == output_cursor,
        "heterogeneous root output intervals must exactly partition output");
    output_cursor = interval.second;
  }
  TORCH_CHECK(
      output_cursor == output_dimension,
      "heterogeneous root output intervals must cover output");
  return std::make_pair(
      total_projection_dimension,
      output_dimension);
}

struct HeterogeneousFactorizedDimensions {
  std::int64_t input_dimension;
  std::int64_t total_projection_dimension;
  std::int64_t output_dimension;
};

HeterogeneousFactorizedDimensions
check_heterogeneous_factorized_cpu_inputs(
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension) {
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");
  const std::int64_t node_count = node_offsets.numel();
  TORCH_CHECK(node_count > 0, "factorized plan must contain nodes");
  TORCH_CHECK(
      node_dimensions.numel() == node_count &&
          node_leaf_offsets.numel() == node_count &&
          node_left.numel() == node_count &&
          node_right.numel() == node_count,
      "factorized node arrays must have equal length");
  TORCH_CHECK(
      node_coefficient_offsets.numel() == node_count + 1,
      "node_coefficient_offsets must have node_count + 1 entries");
  TORCH_CHECK(
      coefficient_rows.numel() == coefficient_columns.numel() &&
          coefficient_rows.numel() == coefficient_values.numel(),
      "factorized coefficient arrays must have equal length");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      packed_slots.size(1) % source_dimension == 0,
      "packed_slots width must be divisible by source_dimension");
  check_factorized_workspace_dimension(
      node_offsets,
      node_dimensions,
      workspace_dimension);
  const auto projection_and_output =
      check_heterogeneous_factorized_root_metadata(
          root_nodes,
          node_dimensions,
          root_projection_starts,
          root_projection_dimensions,
          root_output_offsets,
          projection_values,
          source_dimension,
          output_dimension);
  return {
      packed_slots.size(1) / source_dimension,
      projection_and_output.first,
      projection_and_output.second};
}

struct SegmentedFactorizedDimensions {
  std::int64_t segment_count;
};

SegmentedFactorizedDimensions check_segmented_factorized_cpu_inputs(
    const torch::Tensor& packed_slots,
    const torch::Tensor& segment_source_offsets,
    const torch::Tensor& segment_input_offsets,
    const torch::Tensor& segment_input_dimensions,
    const torch::Tensor& segment_workspace_offsets,
    const torch::Tensor& segment_workspace_dimensions,
    const torch::Tensor& segment_node_offsets,
    const torch::Tensor& segment_root_offsets,
    const torch::Tensor& segment_projection_offsets,
    const torch::Tensor& segment_projection_dimensions,
    const torch::Tensor& segment_output_offsets,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t total_source_count,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension) {
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(segment_source_offsets, "segment_source_offsets");
  check_int64_vector(segment_input_offsets, "segment_input_offsets");
  check_int64_vector(
      segment_input_dimensions,
      "segment_input_dimensions");
  check_int64_vector(
      segment_workspace_offsets,
      "segment_workspace_offsets");
  check_int64_vector(
      segment_workspace_dimensions,
      "segment_workspace_dimensions");
  check_int64_vector(segment_node_offsets, "segment_node_offsets");
  check_int64_vector(segment_root_offsets, "segment_root_offsets");
  check_int64_vector(
      segment_projection_offsets,
      "segment_projection_offsets");
  check_int64_vector(
      segment_projection_dimensions,
      "segment_projection_dimensions");
  check_int64_vector(segment_output_offsets, "segment_output_offsets");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  check_int64_vector(
      root_projection_starts,
      "root_projection_starts");
  check_int64_vector(
      root_projection_dimensions,
      "root_projection_dimensions");
  check_int64_vector(root_output_offsets, "root_output_offsets");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");

  const std::int64_t segment_count = segment_input_offsets.numel();
  TORCH_CHECK(segment_count > 0, "segmented plan requires segments");
  TORCH_CHECK(
      segment_source_offsets.numel() == segment_count + 1 &&
          segment_output_offsets.numel() == segment_count + 1 &&
          segment_node_offsets.numel() == segment_count + 1 &&
          segment_root_offsets.numel() == segment_count + 1,
      "segmented boundary offsets must contain segment_count + 1 entries");
  TORCH_CHECK(
      segment_input_dimensions.numel() == segment_count &&
          segment_workspace_offsets.numel() == segment_count &&
          segment_workspace_dimensions.numel() == segment_count &&
          segment_projection_offsets.numel() == segment_count &&
          segment_projection_dimensions.numel() == segment_count,
      "segmented per-plan metadata must match segment count");

  const std::int64_t node_count = node_offsets.numel();
  const std::int64_t root_count = root_nodes.numel();
  TORCH_CHECK(node_count > 0, "segmented plan requires nodes");
  TORCH_CHECK(root_count > 0, "segmented plan requires roots");
  TORCH_CHECK(
      node_dimensions.numel() == node_count &&
          node_leaf_offsets.numel() == node_count &&
          node_left.numel() == node_count &&
          node_right.numel() == node_count &&
          node_coefficient_offsets.numel() == node_count + 1,
      "segmented node metadata has inconsistent lengths");
  TORCH_CHECK(
      coefficient_rows.numel() == coefficient_columns.numel() &&
          coefficient_rows.numel() == coefficient_values.numel(),
      "segmented coefficient arrays must have equal length");
  TORCH_CHECK(
      root_projection_starts.numel() == root_count &&
          root_projection_dimensions.numel() == root_count &&
          root_output_offsets.numel() == root_count,
      "segmented root metadata has inconsistent lengths");

  const auto* source_offsets =
      segment_source_offsets.data_ptr<std::int64_t>();
  const auto* input_offsets =
      segment_input_offsets.data_ptr<std::int64_t>();
  const auto* input_dimensions =
      segment_input_dimensions.data_ptr<std::int64_t>();
  const auto* workspace_offsets =
      segment_workspace_offsets.data_ptr<std::int64_t>();
  const auto* workspace_dimensions =
      segment_workspace_dimensions.data_ptr<std::int64_t>();
  const auto* node_group_offsets =
      segment_node_offsets.data_ptr<std::int64_t>();
  const auto* root_group_offsets =
      segment_root_offsets.data_ptr<std::int64_t>();
  const auto* projection_offsets =
      segment_projection_offsets.data_ptr<std::int64_t>();
  const auto* projection_dimensions =
      segment_projection_dimensions.data_ptr<std::int64_t>();
  const auto* output_offsets =
      segment_output_offsets.data_ptr<std::int64_t>();
  const auto* local_node_offsets =
      node_offsets.data_ptr<std::int64_t>();
  const auto* local_node_dimensions =
      node_dimensions.data_ptr<std::int64_t>();
  const auto* leaf_offsets =
      node_leaf_offsets.data_ptr<std::int64_t>();
  const auto* left_nodes = node_left.data_ptr<std::int64_t>();
  const auto* right_nodes = node_right.data_ptr<std::int64_t>();
  const auto* coefficient_offsets =
      node_coefficient_offsets.data_ptr<std::int64_t>();
  const auto* roots = root_nodes.data_ptr<std::int64_t>();
  const auto* projection_starts =
      root_projection_starts.data_ptr<std::int64_t>();
  const auto* root_projection_dims =
      root_projection_dimensions.data_ptr<std::int64_t>();
  const auto* root_output_starts =
      root_output_offsets.data_ptr<std::int64_t>();

  TORCH_CHECK(
      source_offsets[0] == 0 && node_group_offsets[0] == 0 &&
          root_group_offsets[0] == 0 && output_offsets[0] == 0,
      "segmented boundary offsets must start at zero");
  std::int64_t expected_input_offset = 0;
  std::int64_t expected_workspace_offset = 0;
  std::int64_t expected_projection_offset = 0;
  for (std::int64_t segment = 0;
       segment < segment_count;
       ++segment) {
    const std::int64_t source_dimension =
        source_offsets[segment + 1] - source_offsets[segment];
    const std::int64_t input_dimension = input_dimensions[segment];
    const std::int64_t segment_workspace_dimension =
        workspace_dimensions[segment];
    const std::int64_t projection_dimension =
        projection_dimensions[segment];
    const std::int64_t local_node_count =
        node_group_offsets[segment + 1] - node_group_offsets[segment];
    const std::int64_t local_root_count =
        root_group_offsets[segment + 1] - root_group_offsets[segment];
    const std::int64_t local_output_dimension =
        output_offsets[segment + 1] - output_offsets[segment];
    TORCH_CHECK(
        source_dimension > 0 && input_dimension > 0 &&
            segment_workspace_dimension > 0 &&
            projection_dimension > 0 && local_node_count > 0 &&
            local_root_count > 0 && local_output_dimension > 0,
        "segmented plan dimensions must be positive");
    TORCH_CHECK(
        input_offsets[segment] == expected_input_offset,
        "segmented input intervals must exactly partition packed input");
    TORCH_CHECK(
        workspace_offsets[segment] == expected_workspace_offset,
        "segmented workspace intervals must exactly partition workspace");
    TORCH_CHECK(
        projection_offsets[segment] == expected_projection_offset,
        "segmented projection intervals must exactly partition coefficients");
    expected_input_offset += source_dimension * input_dimension;
    expected_workspace_offset +=
        source_dimension * segment_workspace_dimension;
    expected_projection_offset +=
        source_dimension * projection_dimension;

    const std::int64_t node_start = node_group_offsets[segment];
    for (std::int64_t local_node = 0;
         local_node < local_node_count;
         ++local_node) {
      const std::int64_t node = node_start + local_node;
      TORCH_CHECK(
          local_node_dimensions[node] > 0 &&
              local_node_offsets[node] >= 0 &&
              local_node_offsets[node] + local_node_dimensions[node] <=
                  segment_workspace_dimension,
          "segmented node interval is outside its workspace");
      TORCH_CHECK(
          coefficient_offsets[node] >= 0 &&
              coefficient_offsets[node] <= coefficient_offsets[node + 1] &&
              coefficient_offsets[node + 1] <=
                  coefficient_values.numel(),
          "segmented coefficient interval is invalid");
      if (leaf_offsets[node] >= 0) {
        TORCH_CHECK(
            leaf_offsets[node] + local_node_dimensions[node] <=
                input_dimension,
            "segmented leaf interval is outside its input");
      } else {
        TORCH_CHECK(
            left_nodes[node] >= 0 && left_nodes[node] < local_node &&
                right_nodes[node] >= 0 && right_nodes[node] < local_node,
            "segmented merge children must precede their parent");
      }
    }

    const std::int64_t root_start = root_group_offsets[segment];
    std::vector<std::pair<std::int64_t, std::int64_t>> intervals;
    intervals.reserve(local_root_count);
    for (std::int64_t local_root = 0;
         local_root < local_root_count;
         ++local_root) {
      const std::int64_t root = root_start + local_root;
      TORCH_CHECK(
          roots[root] >= 0 && roots[root] < local_node_count,
          "segmented root index is outside its local node range");
      TORCH_CHECK(
          projection_starts[root] >= 0 &&
              root_projection_dims[root] > 0 &&
              projection_starts[root] + root_projection_dims[root] <=
                  projection_dimension,
          "segmented root projection interval is invalid");
      const std::int64_t block_dimension =
          root_projection_dims[root] *
          local_node_dimensions[node_start + roots[root]];
      TORCH_CHECK(
          root_output_starts[root] >= 0 &&
              root_output_starts[root] + block_dimension <=
                  local_output_dimension,
          "segmented root output interval is invalid");
      intervals.emplace_back(
          root_output_starts[root],
          root_output_starts[root] + block_dimension);
    }
    std::sort(intervals.begin(), intervals.end());
    std::int64_t output_cursor = 0;
    for (const auto& interval : intervals) {
      TORCH_CHECK(
          interval.first == output_cursor,
          "segmented root outputs must exactly partition each destination");
      output_cursor = interval.second;
    }
    TORCH_CHECK(
        output_cursor == local_output_dimension,
        "segmented root outputs must cover each destination");
  }
  TORCH_CHECK(
      source_offsets[segment_count] == total_source_count,
      "total_source_count does not match segmented source offsets");
  TORCH_CHECK(
      node_group_offsets[segment_count] == node_count &&
          root_group_offsets[segment_count] == root_count,
      "segmented node/root offsets do not cover metadata");
  TORCH_CHECK(
      expected_input_offset == packed_slots.size(1),
      "segmented input intervals do not cover packed input");
  TORCH_CHECK(
      expected_workspace_offset == workspace_dimension,
      "segmented workspace intervals do not match workspace dimension");
  TORCH_CHECK(
      expected_projection_offset == projection_values.numel(),
      "segmented projection intervals do not cover projection values");
  TORCH_CHECK(
      output_offsets[segment_count] == output_dimension,
      "segmented output intervals do not match output dimension");
  TORCH_CHECK(
      coefficient_offsets[node_count] == coefficient_values.numel(),
      "segmented coefficient offsets do not cover coefficients");
  return {segment_count};
}

template <typename TorchScalar>
ye3t::runtime::FactorizedAngularSegmentedPlanView<
    typename CoreScalar<TorchScalar>::type>
segmented_factorized_plan_view(
    const torch::Tensor& segment_source_offsets,
    const torch::Tensor& segment_input_offsets,
    const torch::Tensor& segment_input_dimensions,
    const torch::Tensor& segment_workspace_offsets,
    const torch::Tensor& segment_workspace_dimensions,
    const torch::Tensor& segment_node_offsets,
    const torch::Tensor& segment_root_offsets,
    const torch::Tensor& segment_projection_offsets,
    const torch::Tensor& segment_projection_dimensions,
    const torch::Tensor& segment_output_offsets,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t segment_count,
    std::int64_t packed_input_dimension,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension) {
  using core_t = typename CoreScalar<TorchScalar>::type;
  return {
      segment_source_offsets.data_ptr<std::int64_t>(),
      segment_input_offsets.data_ptr<std::int64_t>(),
      segment_input_dimensions.data_ptr<std::int64_t>(),
      segment_workspace_offsets.data_ptr<std::int64_t>(),
      segment_workspace_dimensions.data_ptr<std::int64_t>(),
      segment_node_offsets.data_ptr<std::int64_t>(),
      segment_root_offsets.data_ptr<std::int64_t>(),
      segment_projection_offsets.data_ptr<std::int64_t>(),
      segment_projection_dimensions.data_ptr<std::int64_t>(),
      segment_output_offsets.data_ptr<std::int64_t>(),
      node_offsets.data_ptr<std::int64_t>(),
      node_dimensions.data_ptr<std::int64_t>(),
      node_leaf_offsets.data_ptr<std::int64_t>(),
      node_left.data_ptr<std::int64_t>(),
      node_right.data_ptr<std::int64_t>(),
      node_coefficient_offsets.data_ptr<std::int64_t>(),
      coefficient_rows.data_ptr<std::int64_t>(),
      coefficient_columns.data_ptr<std::int64_t>(),
      reinterpret_cast<const core_t*>(
          coefficient_values.data_ptr<TorchScalar>()),
      root_nodes.data_ptr<std::int64_t>(),
      root_projection_starts.data_ptr<std::int64_t>(),
      root_projection_dimensions.data_ptr<std::int64_t>(),
      root_output_offsets.data_ptr<std::int64_t>(),
      reinterpret_cast<const core_t*>(
          projection_values.data_ptr<TorchScalar>()),
      segment_count,
      packed_input_dimension,
      workspace_dimension,
      output_dimension};
}

std::tuple<torch::Tensor, torch::Tensor>
cheb_exp_cos_radial_with_derivative_cpu(
    const torch::Tensor& radii,
    const torch::Tensor& cutoffs,
    const torch::Tensor& lambdas,
    std::int64_t radial_index) {
  check_real_vector(radii, "radii");
  check_real_vector(cutoffs, "cutoffs");
  check_real_vector(lambdas, "lambdas");
  TORCH_CHECK(
      radii.sizes() == cutoffs.sizes() &&
          radii.sizes() == lambdas.sizes(),
      "radii, cutoffs, and lambdas must have identical shapes");
  TORCH_CHECK(
      radii.scalar_type() == cutoffs.scalar_type() &&
          radii.scalar_type() == lambdas.scalar_type(),
      "radii, cutoffs, and lambdas must have identical dtypes");
  TORCH_CHECK(radial_index >= 0, "radial_index must be non-negative");
  auto values = torch::empty_like(radii);
  auto derivatives = torch::empty_like(radii);
  AT_DISPATCH_FLOATING_TYPES(
      radii.scalar_type(),
      "ye3t_cheb_exp_cos_radial_with_derivative_cpu",
      [&] {
        ye3t::runtime::cheb_exp_cos_radial_with_derivative<scalar_t>(
            radii.data_ptr<scalar_t>(),
            cutoffs.data_ptr<scalar_t>(),
            lambdas.data_ptr<scalar_t>(),
            radii.numel(),
            radial_index,
            values.data_ptr<scalar_t>(),
            derivatives.data_ptr<scalar_t>());
      });
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
cheb_exp_cos_radial_table_with_derivative_cpu(
    const torch::Tensor& radii,
    const torch::Tensor& cutoffs,
    const torch::Tensor& lambdas,
    std::int64_t maximum_radial_index) {
  check_real_vector(radii, "radii");
  check_real_vector(cutoffs, "cutoffs");
  check_real_vector(lambdas, "lambdas");
  TORCH_CHECK(
      radii.sizes() == cutoffs.sizes() &&
          radii.sizes() == lambdas.sizes(),
      "radii, cutoffs, and lambdas must have identical shapes");
  TORCH_CHECK(
      radii.scalar_type() == cutoffs.scalar_type() &&
          radii.scalar_type() == lambdas.scalar_type(),
      "radii, cutoffs, and lambdas must have identical dtypes");
  TORCH_CHECK(
      maximum_radial_index >= 0,
      "maximum_radial_index must be non-negative");
  const std::int64_t width = maximum_radial_index + 1;
  auto values = torch::empty(
      {radii.numel(), width},
      radii.options());
  auto derivatives = torch::empty_like(values);
  AT_DISPATCH_FLOATING_TYPES(
      radii.scalar_type(),
      "ye3t_cheb_exp_cos_radial_table_with_derivative_cpu",
      [&] {
        ye3t::runtime::cheb_exp_cos_radial_table_with_derivative<scalar_t>(
            radii.data_ptr<scalar_t>(),
            cutoffs.data_ptr<scalar_t>(),
            lambdas.data_ptr<scalar_t>(),
            radii.numel(),
            maximum_radial_index,
            values.data_ptr<scalar_t>(),
            derivatives.data_ptr<scalar_t>());
      });
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
cheb_exp_cos_radial_table_double_backward_cpu(
    const torch::Tensor& values_adjoint,
    const torch::Tensor& radial_derivatives,
    const torch::Tensor& radii,
    const torch::Tensor& cutoffs,
    const torch::Tensor& lambdas,
    const torch::Tensor& grad_grad_radii,
    std::int64_t maximum_radial_index) {
  check_real_vector(radii, "radii");
  check_real_vector(cutoffs, "cutoffs");
  check_real_vector(lambdas, "lambdas");
  check_real_vector(grad_grad_radii, "grad_grad_radii");
  const std::int64_t width = maximum_radial_index + 1;
  TORCH_CHECK(
      maximum_radial_index >= 0,
      "maximum_radial_index must be non-negative");
  TORCH_CHECK(
      values_adjoint.device().is_cpu() &&
          values_adjoint.is_contiguous() &&
          values_adjoint.dim() == 2 &&
          values_adjoint.size(0) == radii.numel() &&
          values_adjoint.size(1) == width,
      "values_adjoint must be a contiguous CPU radial table");
  TORCH_CHECK(
      radial_derivatives.sizes() == values_adjoint.sizes() &&
          radial_derivatives.device().is_cpu() &&
          radial_derivatives.is_contiguous(),
      "radial_derivatives shape and device must match values_adjoint");
  TORCH_CHECK(
      radii.sizes() == cutoffs.sizes() &&
          radii.sizes() == lambdas.sizes() &&
          radii.sizes() == grad_grad_radii.sizes(),
      "radial vectors must have identical shapes");
  TORCH_CHECK(
      values_adjoint.scalar_type() == radii.scalar_type() &&
          radial_derivatives.scalar_type() == radii.scalar_type() &&
          cutoffs.scalar_type() == radii.scalar_type() &&
          lambdas.scalar_type() == radii.scalar_type() &&
          grad_grad_radii.scalar_type() == radii.scalar_type(),
      "radial double-backward tensors must have identical dtypes");
  auto values_adjoint_gradient = torch::empty_like(values_adjoint);
  auto radii_gradient = torch::empty_like(radii);
  AT_DISPATCH_FLOATING_TYPES(
      radii.scalar_type(),
      "ye3t_cheb_exp_cos_radial_table_double_backward_cpu",
      [&] {
        ye3t::runtime::
            cheb_exp_cos_radial_table_double_backward<scalar_t>(
                values_adjoint.data_ptr<scalar_t>(),
                radial_derivatives.data_ptr<scalar_t>(),
                radii.data_ptr<scalar_t>(),
                cutoffs.data_ptr<scalar_t>(),
                lambdas.data_ptr<scalar_t>(),
                grad_grad_radii.data_ptr<scalar_t>(),
                radii.numel(),
                maximum_radial_index,
                values_adjoint_gradient.data_ptr<scalar_t>(),
                radii_gradient.data_ptr<scalar_t>());
      });
  return std::make_tuple(values_adjoint_gradient, radii_gradient);
}

std::tuple<torch::Tensor, torch::Tensor>
spherical_harmonics_with_derivative_cpu(
    const torch::Tensor& edge_vectors,
    std::int64_t angular_momentum,
    bool real_output,
    double epsilon) {
  TORCH_CHECK(edge_vectors.device().is_cpu(), "edge_vectors must be on CPU");
  TORCH_CHECK(
      edge_vectors.is_contiguous(),
      "edge_vectors must be contiguous");
  TORCH_CHECK(
      edge_vectors.dim() == 2 && edge_vectors.size(1) == 3,
      "edge_vectors must have shape [edge_count, 3]");
  TORCH_CHECK(
      edge_vectors.scalar_type() == torch::kFloat32 ||
          edge_vectors.scalar_type() == torch::kFloat64,
      "edge_vectors must use float32 or float64");
  TORCH_CHECK(
      angular_momentum >= 0,
      "angular_momentum must be non-negative");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  const std::int64_t width = 2 * angular_momentum + 1;
  if (real_output) {
    auto values = torch::empty(
        {edge_vectors.size(0), width},
        edge_vectors.options());
    auto derivatives = torch::empty(
        {edge_vectors.size(0), width, 3},
        edge_vectors.options());
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_real_spherical_harmonics_with_derivative_cpu",
        [&] {
          ye3t::runtime::real_spherical_harmonics_with_derivative<scalar_t>(
              edge_vectors.data_ptr<scalar_t>(),
              edge_vectors.size(0),
              angular_momentum,
              static_cast<scalar_t>(epsilon),
              values.data_ptr<scalar_t>(),
              derivatives.data_ptr<scalar_t>());
        });
    return std::make_tuple(values, derivatives);
  }

  const auto complex_type =
      edge_vectors.scalar_type() == torch::kFloat32
      ? torch::kComplexFloat
      : torch::kComplexDouble;
  auto complex_options = edge_vectors.options().dtype(complex_type);
  auto values = torch::empty(
      {edge_vectors.size(0), width},
      complex_options);
  auto derivatives = torch::empty(
      {edge_vectors.size(0), width, 3},
      complex_options);
  AT_DISPATCH_FLOATING_TYPES(
      edge_vectors.scalar_type(),
      "ye3t_complex_spherical_harmonics_with_derivative_cpu",
      [&] {
        using complex_t = std::complex<scalar_t>;
        using torch_complex_t = c10::complex<scalar_t>;
        static_assert(sizeof(complex_t) == sizeof(torch_complex_t));
        ye3t::runtime::complex_spherical_harmonics_with_derivative<scalar_t>(
            edge_vectors.data_ptr<scalar_t>(),
            edge_vectors.size(0),
            angular_momentum,
            static_cast<scalar_t>(epsilon),
            reinterpret_cast<complex_t*>(
                values.data_ptr<torch_complex_t>()),
            reinterpret_cast<complex_t*>(
                derivatives.data_ptr<torch_complex_t>()));
      });
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
spherical_harmonics_table_with_derivative_cpu(
    const torch::Tensor& edge_vectors,
    std::int64_t maximum_angular_momentum,
    bool real_output,
    double epsilon) {
  TORCH_CHECK(edge_vectors.device().is_cpu(), "edge_vectors must be on CPU");
  TORCH_CHECK(
      edge_vectors.is_contiguous(),
      "edge_vectors must be contiguous");
  TORCH_CHECK(
      edge_vectors.dim() == 2 && edge_vectors.size(1) == 3,
      "edge_vectors must have shape [edge_count, 3]");
  TORCH_CHECK(
      edge_vectors.scalar_type() == torch::kFloat32 ||
          edge_vectors.scalar_type() == torch::kFloat64,
      "edge_vectors must use float32 or float64");
  TORCH_CHECK(
      maximum_angular_momentum >= 0,
      "maximum_angular_momentum must be non-negative");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  if (real_output) {
    auto values = torch::empty(
        {edge_vectors.size(0), width},
        edge_vectors.options());
    auto derivatives = torch::empty(
        {edge_vectors.size(0), width, 3},
        edge_vectors.options());
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_real_spherical_harmonics_table_with_derivative_cpu",
        [&] {
          ye3t::runtime::
              real_spherical_harmonics_table_with_derivative<scalar_t>(
                  edge_vectors.data_ptr<scalar_t>(),
                  edge_vectors.size(0),
                  maximum_angular_momentum,
                  static_cast<scalar_t>(epsilon),
                  values.data_ptr<scalar_t>(),
                  derivatives.data_ptr<scalar_t>());
        });
    return std::make_tuple(values, derivatives);
  }

  const auto complex_type =
      edge_vectors.scalar_type() == torch::kFloat32
      ? torch::kComplexFloat
      : torch::kComplexDouble;
  auto complex_options = edge_vectors.options().dtype(complex_type);
  auto values = torch::empty(
      {edge_vectors.size(0), width},
      complex_options);
  auto derivatives = torch::empty(
      {edge_vectors.size(0), width, 3},
      complex_options);
  AT_DISPATCH_FLOATING_TYPES(
      edge_vectors.scalar_type(),
      "ye3t_complex_spherical_harmonics_table_with_derivative_cpu",
      [&] {
        using complex_t = std::complex<scalar_t>;
        using torch_complex_t = c10::complex<scalar_t>;
        static_assert(sizeof(complex_t) == sizeof(torch_complex_t));
        ye3t::runtime::
            complex_spherical_harmonics_table_with_derivative<scalar_t>(
                edge_vectors.data_ptr<scalar_t>(),
                edge_vectors.size(0),
                maximum_angular_momentum,
                static_cast<scalar_t>(epsilon),
                reinterpret_cast<complex_t*>(
                    values.data_ptr<torch_complex_t>()),
                reinterpret_cast<complex_t*>(
                    derivatives.data_ptr<torch_complex_t>()));
      });
  return std::make_tuple(values, derivatives);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
plain_site_basis_product_with_derivative_cpu(
    const torch::Tensor& radial_values,
    const torch::Tensor& radial_derivatives,
    const torch::Tensor& angular_values,
    const torch::Tensor& angular_derivatives,
    const torch::Tensor& prefactors,
    const torch::Tensor& prefactor_derivatives_center,
    const torch::Tensor& prefactor_derivatives_neighbor,
    const torch::Tensor& radial_directions,
    const torch::Tensor& term_groups,
    const torch::Tensor& term_channels,
    std::int64_t channel_count) {
  check_data_tensor(radial_values, "radial_values");
  check_data_tensor(radial_derivatives, "radial_derivatives");
  check_data_tensor(angular_values, "angular_values");
  check_factor_tensor(angular_derivatives, "angular_derivatives");
  check_data_tensor(prefactors, "prefactors");
  check_data_tensor(
      prefactor_derivatives_center,
      "prefactor_derivatives_center");
  check_data_tensor(
      prefactor_derivatives_neighbor,
      "prefactor_derivatives_neighbor");
  check_data_tensor(radial_directions, "radial_directions");
  check_int64_vector(term_groups, "term_groups");
  check_int64_vector(term_channels, "term_channels");
  TORCH_CHECK(channel_count > 0, "channel_count must be positive");
  const auto dtype = radial_values.scalar_type();
  const auto device = radial_values.device();
  for (const auto& tensor : {
           radial_derivatives,
           angular_values,
           angular_derivatives,
           prefactors,
           prefactor_derivatives_center,
           prefactor_derivatives_neighbor,
           radial_directions}) {
    TORCH_CHECK(
        tensor.scalar_type() == dtype,
        "all source-product data tensors must have the same dtype");
    TORCH_CHECK(
        tensor.device() == device,
        "all source-product data tensors must be on the same device");
  }
  const std::int64_t group_count = radial_values.size(0);
  const std::int64_t edge_count = radial_values.size(1);
  const std::int64_t term_count = angular_values.size(0);
  TORCH_CHECK(
      radial_derivatives.sizes() == radial_values.sizes(),
      "radial derivatives must match radial values");
  TORCH_CHECK(
      prefactors.sizes() == radial_values.sizes() &&
          prefactor_derivatives_center.sizes() ==
              radial_values.sizes() &&
          prefactor_derivatives_neighbor.sizes() ==
              radial_values.sizes(),
      "prefactor tables must have shape [group_count, edge_count]");
  TORCH_CHECK(
      angular_values.size(1) == edge_count,
      "angular values must have shape [term_count, edge_count]");
  TORCH_CHECK(
      angular_derivatives.size(0) == term_count &&
          angular_derivatives.size(1) == edge_count &&
          angular_derivatives.size(2) == 3,
      "angular derivatives must have shape [term_count, edge_count, 3]");
  TORCH_CHECK(
      radial_directions.size(0) == edge_count &&
          radial_directions.size(1) == 3,
      "radial directions must have shape [edge_count, 3]");
  TORCH_CHECK(
      term_groups.numel() == term_count &&
          term_channels.numel() == term_count,
      "term index arrays must have length term_count");
  if (term_count > 0) {
    TORCH_CHECK(
        term_groups.min().item<std::int64_t>() >= 0 &&
            term_groups.max().item<std::int64_t>() < group_count,
        "term_groups entries must lie in [0, group_count)");
    TORCH_CHECK(
        term_channels.min().item<std::int64_t>() >= 0 &&
            term_channels.max().item<std::int64_t>() < channel_count,
        "term_channels entries must lie in [0, channel_count)");
  }
  auto edge_values = torch::empty(
      {edge_count, channel_count},
      radial_values.options());
  auto edge_derivatives = torch::empty(
      {edge_count, channel_count, 3},
      radial_values.options());
  auto charge_center = torch::empty_like(edge_values);
  auto charge_neighbor = torch::empty_like(edge_values);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dtype,
      "ye3t_plain_site_basis_product_with_derivative_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::plain_site_basis_product_with_derivative<core_t>(
            reinterpret_cast<const core_t*>(
                radial_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                radial_derivatives.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                angular_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                angular_derivatives.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                prefactors.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                prefactor_derivatives_center.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                prefactor_derivatives_neighbor.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                radial_directions.data_ptr<scalar_t>()),
            term_groups.data_ptr<std::int64_t>(),
            term_channels.data_ptr<std::int64_t>(),
            edge_count,
            group_count,
            term_count,
            channel_count,
            reinterpret_cast<core_t*>(
                edge_values.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                edge_derivatives.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                charge_center.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                charge_neighbor.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      edge_values,
      edge_derivatives,
      charge_center,
      charge_neighbor);
}

std::tuple<torch::Tensor, torch::Tensor>
scheduled_radial_angular_channels_with_derivative_cpu(
    const torch::Tensor& radial_values,
    const torch::Tensor& radial_derivatives,
    const torch::Tensor& angular_values,
    const torch::Tensor& angular_derivatives,
    const torch::Tensor& radial_directions,
    const torch::Tensor& edge_types,
    const torch::Tensor& channel_radial_indices,
    const torch::Tensor& channel_angular_indices,
    const torch::Tensor& channel_types,
    const torch::Tensor& channel_scales) {
  check_data_tensor(radial_values, "radial_values");
  check_data_tensor(radial_derivatives, "radial_derivatives");
  check_data_tensor(angular_values, "angular_values");
  check_factor_tensor(angular_derivatives, "angular_derivatives");
  check_data_tensor(radial_directions, "radial_directions");
  check_int64_vector(edge_types, "edge_types");
  check_int64_vector(
      channel_radial_indices,
      "channel_radial_indices");
  check_int64_vector(
      channel_angular_indices,
      "channel_angular_indices");
  check_int64_vector(channel_types, "channel_types");
  check_data_vector(channel_scales, "channel_scales");
  const auto dtype = radial_values.scalar_type();
  TORCH_CHECK(
      dtype == torch::kFloat32 || dtype == torch::kFloat64,
      "scheduled radial-angular channels require float32 or float64");
  for (const auto& tensor : {
           radial_derivatives,
           angular_values,
           angular_derivatives,
           radial_directions,
           channel_scales}) {
    TORCH_CHECK(
        tensor.scalar_type() == dtype,
        "scheduled radial-angular data tensors must share one dtype");
  }
  const auto edge_count = radial_values.size(0);
  const auto radial_width = radial_values.size(1);
  const auto angular_width = angular_values.size(1);
  const auto channel_count = channel_scales.numel();
  TORCH_CHECK(
      radial_derivatives.sizes() == radial_values.sizes(),
      "radial derivatives must match radial values");
  TORCH_CHECK(
      angular_derivatives.size(0) == edge_count &&
          angular_derivatives.size(1) == angular_width &&
          angular_derivatives.size(2) == 3,
      "angular derivatives must have shape [edge_count, angular_width, 3]");
  TORCH_CHECK(
      angular_values.size(0) == edge_count,
      "radial and angular tables must share one edge axis");
  TORCH_CHECK(
      radial_directions.size(0) == edge_count &&
          radial_directions.size(1) == 3,
      "radial directions must have shape [edge_count, 3]");
  TORCH_CHECK(
      edge_types.numel() == edge_count,
      "edge_types must have length edge_count");
  TORCH_CHECK(
      channel_radial_indices.numel() == channel_count &&
          channel_angular_indices.numel() == channel_count &&
          channel_types.numel() == channel_count,
      "channel schedule arrays must have length channel_count");
  if (channel_count > 0) {
    TORCH_CHECK(
        channel_radial_indices.min().item<std::int64_t>() >= 0 &&
            channel_radial_indices.max().item<std::int64_t>() <
                radial_width,
        "channel radial indices must lie in the radial table");
    TORCH_CHECK(
        channel_angular_indices.min().item<std::int64_t>() >= 0 &&
            channel_angular_indices.max().item<std::int64_t>() <
                angular_width,
        "channel angular indices must lie in the angular table");
    TORCH_CHECK(
        channel_types.min().item<std::int64_t>() >= -1,
        "channel types must be -1 or a nonnegative edge type");
  }
  auto edge_values = torch::empty(
      {edge_count, channel_count},
      radial_values.options());
  auto edge_derivatives = torch::empty(
      {edge_count, channel_count, 3},
      radial_values.options());
  AT_DISPATCH_FLOATING_TYPES(
      dtype,
      "ye3t_scheduled_radial_angular_channels_with_derivative_cpu",
      [&] {
        ye3t::runtime::
            scheduled_radial_angular_channels_with_derivative<scalar_t>(
                radial_values.data_ptr<scalar_t>(),
                radial_derivatives.data_ptr<scalar_t>(),
                angular_values.data_ptr<scalar_t>(),
                angular_derivatives.data_ptr<scalar_t>(),
                radial_directions.data_ptr<scalar_t>(),
                edge_types.data_ptr<std::int64_t>(),
                channel_radial_indices.data_ptr<std::int64_t>(),
                channel_angular_indices.data_ptr<std::int64_t>(),
                channel_types.data_ptr<std::int64_t>(),
                channel_scales.data_ptr<scalar_t>(),
                edge_count,
                radial_width,
                angular_width,
                channel_count,
                edge_values.data_ptr<scalar_t>(),
                edge_derivatives.data_ptr<scalar_t>());
      });
  return std::make_tuple(edge_values, edge_derivatives);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
plain_site_basis_product_adjoint_cpu(
    const torch::Tensor& radial_values,
    const torch::Tensor& radial_derivatives,
    const torch::Tensor& angular_values,
    const torch::Tensor& angular_derivatives,
    const torch::Tensor& prefactors,
    const torch::Tensor& prefactor_derivatives_center,
    const torch::Tensor& prefactor_derivatives_neighbor,
    const torch::Tensor& radial_directions,
    const torch::Tensor& edge_weights,
    const torch::Tensor& edge_weight_derivatives,
    const torch::Tensor& term_groups,
    const torch::Tensor& term_channels,
    const torch::Tensor& edge_adjoint) {
  check_data_tensor(radial_values, "radial_values");
  check_data_tensor(radial_derivatives, "radial_derivatives");
  check_data_tensor(angular_values, "angular_values");
  check_factor_tensor(angular_derivatives, "angular_derivatives");
  check_data_tensor(prefactors, "prefactors");
  check_data_tensor(
      prefactor_derivatives_center,
      "prefactor_derivatives_center");
  check_data_tensor(
      prefactor_derivatives_neighbor,
      "prefactor_derivatives_neighbor");
  check_data_tensor(radial_directions, "radial_directions");
  check_value_vector(edge_weights, radial_values, "edge_weights");
  check_data_tensor(
      edge_weight_derivatives,
      "edge_weight_derivatives");
  check_int64_vector(term_groups, "term_groups");
  check_int64_vector(term_channels, "term_channels");
  check_data_tensor(edge_adjoint, "edge_adjoint");
  const auto dtype = radial_values.scalar_type();
  const auto device = radial_values.device();
  for (const auto& tensor : {
           radial_derivatives,
           angular_values,
           angular_derivatives,
           prefactors,
           prefactor_derivatives_center,
           prefactor_derivatives_neighbor,
           radial_directions,
           edge_weights,
           edge_weight_derivatives,
           edge_adjoint}) {
    TORCH_CHECK(
        tensor.scalar_type() == dtype,
        "all source-adjoint data tensors must have the same dtype");
    TORCH_CHECK(
        tensor.device() == device,
        "all source-adjoint data tensors must be on the same device");
  }
  const std::int64_t group_count = radial_values.size(0);
  const std::int64_t edge_count = radial_values.size(1);
  const std::int64_t term_count = angular_values.size(0);
  const std::int64_t channel_count = edge_adjoint.size(1);
  TORCH_CHECK(channel_count > 0, "edge_adjoint must have channels");
  TORCH_CHECK(
      radial_derivatives.sizes() == radial_values.sizes(),
      "radial derivatives must match radial values");
  TORCH_CHECK(
      prefactors.sizes() == radial_values.sizes() &&
          prefactor_derivatives_center.sizes() ==
              radial_values.sizes() &&
          prefactor_derivatives_neighbor.sizes() ==
              radial_values.sizes(),
      "prefactor tables must have shape [group_count, edge_count]");
  TORCH_CHECK(
      angular_values.size(1) == edge_count,
      "angular values must have shape [term_count, edge_count]");
  TORCH_CHECK(
      angular_derivatives.size(0) == term_count &&
          angular_derivatives.size(1) == edge_count &&
          angular_derivatives.size(2) == 3,
      "angular derivatives must have shape [term_count, edge_count, 3]");
  TORCH_CHECK(
      radial_directions.size(0) == edge_count &&
          radial_directions.size(1) == 3,
      "radial directions must have shape [edge_count, 3]");
  TORCH_CHECK(
      edge_weights.numel() == edge_count,
      "edge_weights must have length edge_count");
  TORCH_CHECK(
      edge_weight_derivatives.size(0) == edge_count &&
          edge_weight_derivatives.size(1) == 3,
      "edge weight derivatives must have shape [edge_count, 3]");
  TORCH_CHECK(
      edge_adjoint.size(0) == edge_count,
      "edge_adjoint must have shape [edge_count, channel_count]");
  TORCH_CHECK(
      term_groups.numel() == term_count &&
          term_channels.numel() == term_count,
      "term index arrays must have length term_count");
  if (term_count > 0) {
    TORCH_CHECK(
        term_groups.min().item<std::int64_t>() >= 0 &&
            term_groups.max().item<std::int64_t>() < group_count,
        "term_groups entries must lie in [0, group_count)");
    TORCH_CHECK(
        term_channels.min().item<std::int64_t>() >= 0 &&
            term_channels.max().item<std::int64_t>() < channel_count,
        "term_channels entries must lie in edge_adjoint channel bounds");
  }
  auto edge_position_adjoint = torch::empty(
      {edge_count, 3},
      radial_values.options());
  auto edge_charge_center = torch::empty(
      {edge_count},
      radial_values.options());
  auto edge_charge_neighbor = torch::empty_like(edge_charge_center);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dtype,
      "ye3t_plain_site_basis_product_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::plain_site_basis_product_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                radial_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                radial_derivatives.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                angular_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                angular_derivatives.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                prefactors.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                prefactor_derivatives_center.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                prefactor_derivatives_neighbor.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                radial_directions.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_weights.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_weight_derivatives.data_ptr<scalar_t>()),
            term_groups.data_ptr<std::int64_t>(),
            term_channels.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                edge_adjoint.data_ptr<scalar_t>()),
            edge_count,
            group_count,
            term_count,
            channel_count,
            reinterpret_cast<core_t*>(
                edge_position_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                edge_charge_center.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                edge_charge_neighbor.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      edge_position_adjoint,
      edge_charge_center,
      edge_charge_neighbor);
}

torch::Tensor density_accumulate_cpu(
    const torch::Tensor& edge_values,
    const torch::Tensor& centers,
    std::int64_t atom_count) {
  check_data_tensor(edge_values, "edge_values");
  check_int64_vector(centers, "centers");
  TORCH_CHECK(
      centers.numel() == edge_values.size(0),
      "centers length must match the edge count");
  TORCH_CHECK(atom_count > 0, "atom_count must be positive");
  if (centers.numel() > 0) {
    TORCH_CHECK(
        centers.min().item<std::int64_t>() >= 0 &&
            centers.max().item<std::int64_t>() < atom_count,
        "centers entries must lie in [0, atom_count)");
  }
  auto atomic_values = torch::empty(
      {atom_count, edge_values.size(1)},
      edge_values.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      edge_values.scalar_type(),
      "ye3t_density_accumulate_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::density_accumulate_forward<core_t>(
            reinterpret_cast<const core_t*>(
                edge_values.data_ptr<scalar_t>()),
            centers.data_ptr<std::int64_t>(),
            edge_values.size(0),
            edge_values.size(1),
            atom_count,
            reinterpret_cast<core_t*>(
                atomic_values.data_ptr<scalar_t>()));
      });
  return atomic_values;
}

torch::Tensor density_accumulate_adjoint_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& centers) {
  check_data_tensor(atomic_adjoint, "atomic_adjoint");
  check_int64_vector(centers, "centers");
  if (centers.numel() > 0) {
    TORCH_CHECK(
        centers.min().item<std::int64_t>() >= 0 &&
            centers.max().item<std::int64_t>() < atomic_adjoint.size(0),
        "centers entries must lie in the atomic_adjoint range");
  }
  auto edge_adjoint = torch::empty(
      {centers.numel(), atomic_adjoint.size(1)},
      atomic_adjoint.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      atomic_adjoint.scalar_type(),
      "ye3t_density_accumulate_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::density_accumulate_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                atomic_adjoint.data_ptr<scalar_t>()),
            centers.data_ptr<std::int64_t>(),
            centers.numel(),
            atomic_adjoint.size(1),
            reinterpret_cast<core_t*>(
                edge_adjoint.data_ptr<scalar_t>()));
      });
  return edge_adjoint;
}

void check_edge_outer_inputs(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& centers,
    std::int64_t atom_count) {
  check_data_tensor(left, "left");
  check_data_tensor(right, "right");
  check_int64_vector(centers, "centers");
  TORCH_CHECK(
      left.scalar_type() == torch::kFloat32 ||
          left.scalar_type() == torch::kFloat64,
      "edge outer accumulation requires float32 or float64 inputs");
  TORCH_CHECK(
      right.scalar_type() == left.scalar_type(),
      "left and right must have the same dtype");
  TORCH_CHECK(
      right.size(0) == left.size(0),
      "left and right edge counts must match");
  TORCH_CHECK(
      centers.numel() == left.size(0),
      "centers length must match the edge count");
  TORCH_CHECK(atom_count > 0, "atom_count must be positive");
  if (centers.numel() > 0) {
    TORCH_CHECK(
        centers.min().item<std::int64_t>() >= 0 &&
            centers.max().item<std::int64_t>() < atom_count,
        "centers entries must lie in [0, atom_count)");
  }
}

torch::Tensor edge_outer_accumulate_cpu(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& centers,
    std::int64_t atom_count) {
  check_edge_outer_inputs(left, right, centers, atom_count);
  auto atomic_values = torch::empty(
      {atom_count, left.size(1), right.size(1)},
      left.options());
  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "ye3t_edge_outer_accumulate_cpu",
      [&] {
        ye3t::runtime::edge_outer_accumulate_forward<scalar_t>(
            left.data_ptr<scalar_t>(),
            right.data_ptr<scalar_t>(),
            centers.data_ptr<std::int64_t>(),
            left.size(0),
            left.size(1),
            right.size(1),
            atom_count,
            atomic_values.data_ptr<scalar_t>());
      });
  return atomic_values;
}

std::tuple<torch::Tensor, torch::Tensor>
edge_outer_accumulate_adjoint_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& centers) {
  check_edge_outer_inputs(
      left,
      right,
      centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.device().is_cpu(),
      "atomic_adjoint must be on CPU");
  TORCH_CHECK(
      atomic_adjoint.is_contiguous(),
      "atomic_adjoint must be contiguous");
  TORCH_CHECK(
      atomic_adjoint.dim() == 3,
      "atomic_adjoint must be three-dimensional");
  TORCH_CHECK(
      atomic_adjoint.scalar_type() == left.scalar_type(),
      "atomic_adjoint must have the same dtype as left");
  TORCH_CHECK(
      atomic_adjoint.size(1) == left.size(1) &&
          atomic_adjoint.size(2) == right.size(1),
      "atomic_adjoint dimensions must match left and right");
  auto left_adjoint = torch::empty_like(left);
  auto right_adjoint = torch::empty_like(right);
  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "ye3t_edge_outer_accumulate_adjoint_cpu",
      [&] {
        ye3t::runtime::edge_outer_accumulate_adjoint<scalar_t>(
            atomic_adjoint.data_ptr<scalar_t>(),
            left.data_ptr<scalar_t>(),
            right.data_ptr<scalar_t>(),
            centers.data_ptr<std::int64_t>(),
            left.size(0),
            left.size(1),
            right.size(1),
            left_adjoint.data_ptr<scalar_t>(),
            right_adjoint.data_ptr<scalar_t>());
      });
  return std::make_tuple(left_adjoint, right_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
edge_outer_accumulate_double_backward_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& left_adjoint_tangent,
    const torch::Tensor& right_adjoint_tangent,
    const torch::Tensor& centers) {
  check_edge_outer_inputs(
      left,
      right,
      centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.device().is_cpu() &&
          atomic_adjoint.is_contiguous() &&
          atomic_adjoint.dim() == 3,
      "atomic_adjoint must be a contiguous three-dimensional CPU tensor");
  TORCH_CHECK(
      atomic_adjoint.scalar_type() == left.scalar_type() &&
          atomic_adjoint.size(1) == left.size(1) &&
          atomic_adjoint.size(2) == right.size(1),
      "atomic_adjoint must match left and right dimensions and dtype");
  TORCH_CHECK(
      left_adjoint_tangent.device().is_cpu() &&
          left_adjoint_tangent.is_contiguous() &&
          left_adjoint_tangent.sizes() == left.sizes() &&
          left_adjoint_tangent.scalar_type() == left.scalar_type(),
      "left_adjoint_tangent must match left");
  TORCH_CHECK(
      right_adjoint_tangent.device().is_cpu() &&
          right_adjoint_tangent.is_contiguous() &&
          right_adjoint_tangent.sizes() == right.sizes() &&
          right_adjoint_tangent.scalar_type() == right.scalar_type(),
      "right_adjoint_tangent must match right");
  auto atomic_adjoint_tangent = torch::empty_like(atomic_adjoint);
  auto left_second_adjoint = torch::empty_like(left);
  auto right_second_adjoint = torch::empty_like(right);
  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "ye3t_edge_outer_accumulate_double_backward_cpu",
      [&] {
        ye3t::runtime::edge_outer_accumulate_double_backward<scalar_t>(
            atomic_adjoint.data_ptr<scalar_t>(),
            left.data_ptr<scalar_t>(),
            right.data_ptr<scalar_t>(),
            left_adjoint_tangent.data_ptr<scalar_t>(),
            right_adjoint_tangent.data_ptr<scalar_t>(),
            centers.data_ptr<std::int64_t>(),
            left.size(0),
            left.size(1),
            right.size(1),
            atomic_adjoint.size(0),
            atomic_adjoint_tangent.data_ptr<scalar_t>(),
            left_second_adjoint.data_ptr<scalar_t>(),
            right_second_adjoint.data_ptr<scalar_t>());
      });
  return std::make_tuple(
      atomic_adjoint_tangent,
      left_second_adjoint,
      right_second_adjoint);
}

void check_softmax_gaussian_role_density_inputs(
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& atom_centers,
    std::int64_t atom_count) {
  check_real_vector(distances, "distances");
  check_real_vector(cutoffs, "cutoffs");
  check_real_vector(filter_centers, "filter_centers");
  check_data_tensor(edge_values, "edge_values");
  check_int64_vector(atom_centers, "atom_centers");
  TORCH_CHECK(
      distances.dim() == 1 && cutoffs.dim() == 1 &&
          filter_centers.dim() == 1 && edge_values.dim() == 2,
      "softmax-Gaussian role density requires one-dimensional distances, "
      "cutoffs, and filter centers plus two-dimensional edge values");
  TORCH_CHECK(
      distances.scalar_type() == torch::kFloat32 ||
          distances.scalar_type() == torch::kFloat64,
      "softmax-Gaussian role density requires float32 or float64 inputs");
  TORCH_CHECK(
      cutoffs.scalar_type() == distances.scalar_type() &&
          filter_centers.scalar_type() == distances.scalar_type() &&
          edge_values.scalar_type() == distances.scalar_type(),
      "softmax-Gaussian role-density floating inputs must share one dtype");
  TORCH_CHECK(
      cutoffs.numel() == distances.numel() &&
          edge_values.size(0) == distances.numel() &&
          atom_centers.numel() == distances.numel(),
      "softmax-Gaussian role-density edge counts must match");
  TORCH_CHECK(
      filter_centers.numel() > 0 && edge_values.size(1) > 0,
      "softmax-Gaussian role density requires roles and edge channels");
  TORCH_CHECK(filter_width > 0.0, "filter_width must be positive");
  TORCH_CHECK(atom_count > 0, "atom_count must be positive");
  if (distances.numel() > 0) {
    TORCH_CHECK(
        cutoffs.min().item<double>() > 0.0,
        "cutoffs must be positive");
    TORCH_CHECK(
        atom_centers.min().item<std::int64_t>() >= 0 &&
            atom_centers.max().item<std::int64_t>() < atom_count,
        "atom_centers entries must lie in [0, atom_count)");
  }
}

torch::Tensor softmax_gaussian_role_density_cpu(
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& atom_centers,
    std::int64_t atom_count) {
  check_softmax_gaussian_role_density_inputs(
      distances, cutoffs, filter_centers, filter_width,
      edge_values, atom_centers, atom_count);
  auto atomic_values = torch::empty(
      {atom_count, filter_centers.numel(), edge_values.size(1)},
      edge_values.options());
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_softmax_gaussian_role_density_cpu",
      [&] {
        ye3t::runtime::softmax_gaussian_role_density_forward<scalar_t>(
            distances.data_ptr<scalar_t>(),
            cutoffs.data_ptr<scalar_t>(),
            filter_centers.data_ptr<scalar_t>(),
            static_cast<scalar_t>(filter_width),
            edge_values.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(),
            distances.numel(),
            filter_centers.numel(),
            edge_values.size(1),
            atom_count,
            atomic_values.data_ptr<scalar_t>());
      });
  return atomic_values;
}

std::tuple<torch::Tensor, torch::Tensor>
softmax_gaussian_role_density_adjoint_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& atom_centers) {
  check_softmax_gaussian_role_density_inputs(
      distances, cutoffs, filter_centers, filter_width,
      edge_values, atom_centers, atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.device().is_cpu() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.dim() == 3 &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == edge_values.size(1),
      "atomic_adjoint must match role-density output layout and dtype");
  auto distance_adjoint = torch::empty_like(distances);
  auto edge_adjoint = torch::empty_like(edge_values);
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_softmax_gaussian_role_density_adjoint_cpu",
      [&] {
        ye3t::runtime::softmax_gaussian_role_density_adjoint<scalar_t>(
            atomic_adjoint.data_ptr<scalar_t>(),
            distances.data_ptr<scalar_t>(),
            cutoffs.data_ptr<scalar_t>(),
            filter_centers.data_ptr<scalar_t>(),
            static_cast<scalar_t>(filter_width),
            edge_values.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(),
            distances.numel(),
            filter_centers.numel(),
            edge_values.size(1),
            distance_adjoint.data_ptr<scalar_t>(),
            edge_adjoint.data_ptr<scalar_t>());
      });
  return std::make_tuple(distance_adjoint, edge_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
softmax_gaussian_role_density_double_backward_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& distance_adjoint_tangent,
    const torch::Tensor& edge_adjoint_tangent,
    const torch::Tensor& atom_centers) {
  check_softmax_gaussian_role_density_inputs(
      distances, cutoffs, filter_centers, filter_width,
      edge_values, atom_centers, atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.device().is_cpu() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.dim() == 3 &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == edge_values.size(1),
      "atomic_adjoint must match role-density output layout and dtype");
  TORCH_CHECK(
      distance_adjoint_tangent.device().is_cpu() &&
          distance_adjoint_tangent.is_contiguous() &&
          distance_adjoint_tangent.sizes() == distances.sizes() &&
          distance_adjoint_tangent.scalar_type() == distances.scalar_type(),
      "distance_adjoint_tangent must match distances");
  TORCH_CHECK(
      edge_adjoint_tangent.device().is_cpu() &&
          edge_adjoint_tangent.is_contiguous() &&
          edge_adjoint_tangent.sizes() == edge_values.sizes() &&
          edge_adjoint_tangent.scalar_type() == edge_values.scalar_type(),
      "edge_adjoint_tangent must match edge_values");
  auto atomic_adjoint_tangent = torch::empty_like(atomic_adjoint);
  auto distance_second_adjoint = torch::empty_like(distances);
  auto edge_second_adjoint = torch::empty_like(edge_values);
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_softmax_gaussian_role_density_double_backward_cpu",
      [&] {
        ye3t::runtime::softmax_gaussian_role_density_double_backward<scalar_t>(
            atomic_adjoint.data_ptr<scalar_t>(),
            distances.data_ptr<scalar_t>(),
            cutoffs.data_ptr<scalar_t>(),
            filter_centers.data_ptr<scalar_t>(),
            static_cast<scalar_t>(filter_width),
            edge_values.data_ptr<scalar_t>(),
            distance_adjoint_tangent.data_ptr<scalar_t>(),
            edge_adjoint_tangent.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(),
            distances.numel(),
            filter_centers.numel(),
            edge_values.size(1),
            atomic_adjoint.size(0),
            atomic_adjoint_tangent.data_ptr<scalar_t>(),
            distance_second_adjoint.data_ptr<scalar_t>(),
            edge_second_adjoint.data_ptr<scalar_t>());
      });
  return std::make_tuple(
      atomic_adjoint_tangent,
      distance_second_adjoint,
      edge_second_adjoint);
}

void check_scheduled_role_density_inputs(
    const torch::Tensor& radial_values,
    const torch::Tensor& angular_values,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& soft_weights,
    const torch::Tensor& edge_types,
    const torch::Tensor& channel_radial_indices,
    const torch::Tensor& channel_angular_indices,
    const torch::Tensor& channel_types,
    const torch::Tensor& channel_scales,
    const torch::Tensor& atom_centers,
    std::int64_t atom_count) {
  check_data_tensor(radial_values, "radial_values");
  check_data_tensor(angular_values, "angular_values");
  check_real_vector(distances, "distances");
  check_real_vector(cutoffs, "cutoffs");
  check_real_vector(filter_centers, "filter_centers");
  check_real_vector(soft_weights, "soft_weights");
  check_int64_vector(edge_types, "edge_types");
  check_int64_vector(
      channel_radial_indices, "channel_radial_indices");
  check_int64_vector(
      channel_angular_indices, "channel_angular_indices");
  check_int64_vector(channel_types, "channel_types");
  check_real_vector(channel_scales, "channel_scales");
  check_int64_vector(atom_centers, "atom_centers");
  TORCH_CHECK(
      radial_values.dim() == 2 && angular_values.dim() == 2,
      "scheduled role density requires two-dimensional radial and angular tables");
  TORCH_CHECK(
      distances.scalar_type() == torch::kFloat32 ||
          distances.scalar_type() == torch::kFloat64,
      "scheduled role density requires float32 or float64 inputs");
  const auto dtype = distances.scalar_type();
  TORCH_CHECK(
      radial_values.scalar_type() == dtype &&
          angular_values.scalar_type() == dtype &&
          cutoffs.scalar_type() == dtype &&
          filter_centers.scalar_type() == dtype &&
          soft_weights.scalar_type() == dtype &&
          channel_scales.scalar_type() == dtype,
      "scheduled role-density floating inputs must share one dtype");
  const std::int64_t edge_count = distances.numel();
  TORCH_CHECK(
      radial_values.size(0) == edge_count &&
          angular_values.size(0) == edge_count &&
          cutoffs.numel() == edge_count &&
          soft_weights.numel() == edge_count &&
          edge_types.numel() == edge_count &&
          atom_centers.numel() == edge_count,
      "scheduled role-density edge counts must match");
  const std::int64_t channel_count = channel_scales.numel();
  TORCH_CHECK(
      channel_count > 0 &&
          channel_radial_indices.numel() == channel_count &&
          channel_angular_indices.numel() == channel_count &&
          channel_types.numel() == channel_count,
      "scheduled role-density channel schedules must match");
  TORCH_CHECK(
      filter_centers.numel() > 0 && filter_width > 0.0 &&
          atom_count > 0,
      "scheduled role density requires roles, positive width, and atoms");
  if (edge_count > 0) {
    TORCH_CHECK(cutoffs.min().item<double>() > 0.0, "cutoffs must be positive");
    TORCH_CHECK(
        atom_centers.min().item<std::int64_t>() >= 0 &&
            atom_centers.max().item<std::int64_t>() < atom_count,
        "atom_centers entries must lie in [0, atom_count)");
    TORCH_CHECK(
        channel_radial_indices.min().item<std::int64_t>() >= 0 &&
            channel_radial_indices.max().item<std::int64_t>() <
                radial_values.size(1),
        "channel radial indices are out of bounds");
    TORCH_CHECK(
        channel_angular_indices.min().item<std::int64_t>() >= 0 &&
            channel_angular_indices.max().item<std::int64_t>() <
                angular_values.size(1),
        "channel angular indices are out of bounds");
  }
}

torch::Tensor scheduled_softmax_gaussian_role_density_cpu(
    const torch::Tensor& radial_values,
    const torch::Tensor& angular_values,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& soft_weights,
    const torch::Tensor& edge_types,
    const torch::Tensor& channel_radial_indices,
    const torch::Tensor& channel_angular_indices,
    const torch::Tensor& channel_types,
    const torch::Tensor& channel_scales,
    const torch::Tensor& atom_centers,
    std::int64_t atom_count) {
  check_scheduled_role_density_inputs(
      radial_values, angular_values, distances, cutoffs, filter_centers,
      filter_width, soft_weights, edge_types, channel_radial_indices,
      channel_angular_indices, channel_types, channel_scales, atom_centers,
      atom_count);
  auto output = torch::empty(
      {atom_count, filter_centers.numel(), channel_scales.numel() + 1},
      distances.options());
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_scheduled_softmax_gaussian_role_density_cpu",
      [&] {
        ye3t::runtime::scheduled_softmax_gaussian_role_density_forward<scalar_t>(
            radial_values.data_ptr<scalar_t>(),
            angular_values.data_ptr<scalar_t>(),
            distances.data_ptr<scalar_t>(), cutoffs.data_ptr<scalar_t>(),
            filter_centers.data_ptr<scalar_t>(),
            static_cast<scalar_t>(filter_width),
            soft_weights.data_ptr<scalar_t>(), edge_types.data_ptr<std::int64_t>(),
            channel_radial_indices.data_ptr<std::int64_t>(),
            channel_angular_indices.data_ptr<std::int64_t>(),
            channel_types.data_ptr<std::int64_t>(),
            channel_scales.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(), distances.numel(),
            radial_values.size(1), angular_values.size(1),
            filter_centers.numel(), channel_scales.numel(), atom_count,
            output.data_ptr<scalar_t>());
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
scheduled_softmax_gaussian_role_density_adjoint_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& radial_values,
    const torch::Tensor& angular_values,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& soft_weights,
    const torch::Tensor& edge_types,
    const torch::Tensor& channel_radial_indices,
    const torch::Tensor& channel_angular_indices,
    const torch::Tensor& channel_types,
    const torch::Tensor& channel_scales,
    const torch::Tensor& atom_centers) {
  check_scheduled_role_density_inputs(
      radial_values, angular_values, distances, cutoffs, filter_centers,
      filter_width, soft_weights, edge_types, channel_radial_indices,
      channel_angular_indices, channel_types, channel_scales, atom_centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.device().is_cpu() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.dim() == 3 &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == channel_scales.numel() + 1,
      "atomic_adjoint must match scheduled role-density output");
  auto radial_adjoint = torch::empty_like(radial_values);
  auto angular_adjoint = torch::empty_like(angular_values);
  auto distance_adjoint = torch::empty_like(distances);
  auto soft_adjoint = torch::empty_like(soft_weights);
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_scheduled_softmax_gaussian_role_density_adjoint_cpu",
      [&] {
        ye3t::runtime::scheduled_softmax_gaussian_role_density_adjoint<scalar_t>(
            atomic_adjoint.data_ptr<scalar_t>(),
            radial_values.data_ptr<scalar_t>(), angular_values.data_ptr<scalar_t>(),
            distances.data_ptr<scalar_t>(), cutoffs.data_ptr<scalar_t>(),
            filter_centers.data_ptr<scalar_t>(),
            static_cast<scalar_t>(filter_width),
            soft_weights.data_ptr<scalar_t>(), edge_types.data_ptr<std::int64_t>(),
            channel_radial_indices.data_ptr<std::int64_t>(),
            channel_angular_indices.data_ptr<std::int64_t>(),
            channel_types.data_ptr<std::int64_t>(),
            channel_scales.data_ptr<scalar_t>(), atom_centers.data_ptr<std::int64_t>(),
            distances.numel(), radial_values.size(1), angular_values.size(1),
            filter_centers.numel(), channel_scales.numel(),
            radial_adjoint.data_ptr<scalar_t>(), angular_adjoint.data_ptr<scalar_t>(),
            distance_adjoint.data_ptr<scalar_t>(), soft_adjoint.data_ptr<scalar_t>());
      });
  return std::make_tuple(
      radial_adjoint, angular_adjoint, distance_adjoint, soft_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
scheduled_softmax_gaussian_role_density_double_backward_cpu(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& radial_values,
    const torch::Tensor& angular_values,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& soft_weights,
    const torch::Tensor& radial_adjoint_tangent,
    const torch::Tensor& angular_adjoint_tangent,
    const torch::Tensor& distance_adjoint_tangent,
    const torch::Tensor& soft_adjoint_tangent,
    const torch::Tensor& edge_types,
    const torch::Tensor& channel_radial_indices,
    const torch::Tensor& channel_angular_indices,
    const torch::Tensor& channel_types,
    const torch::Tensor& channel_scales,
    const torch::Tensor& atom_centers) {
  check_scheduled_role_density_inputs(
      radial_values, angular_values, distances, cutoffs, filter_centers,
      filter_width, soft_weights, edge_types, channel_radial_indices,
      channel_angular_indices, channel_types, channel_scales, atom_centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.device().is_cpu() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.dim() == 3 &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == channel_scales.numel() + 1,
      "atomic_adjoint must match scheduled role-density output");
  TORCH_CHECK(
      radial_adjoint_tangent.sizes() == radial_values.sizes() &&
          angular_adjoint_tangent.sizes() == angular_values.sizes() &&
          distance_adjoint_tangent.sizes() == distances.sizes() &&
          soft_adjoint_tangent.sizes() == soft_weights.sizes(),
      "scheduled role-density adjoint tangents must match primal inputs");
  auto atomic_tangent = torch::empty_like(atomic_adjoint);
  auto radial_second = torch::empty_like(radial_values);
  auto angular_second = torch::empty_like(angular_values);
  auto distance_second = torch::empty_like(distances);
  auto soft_second = torch::empty_like(soft_weights);
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_scheduled_softmax_gaussian_role_density_double_backward_cpu",
      [&] {
        ye3t::runtime::scheduled_softmax_gaussian_role_density_double_backward<scalar_t>(
            atomic_adjoint.data_ptr<scalar_t>(), radial_values.data_ptr<scalar_t>(),
            angular_values.data_ptr<scalar_t>(), distances.data_ptr<scalar_t>(),
            cutoffs.data_ptr<scalar_t>(), filter_centers.data_ptr<scalar_t>(),
            static_cast<scalar_t>(filter_width), soft_weights.data_ptr<scalar_t>(),
            radial_adjoint_tangent.data_ptr<scalar_t>(),
            angular_adjoint_tangent.data_ptr<scalar_t>(),
            distance_adjoint_tangent.data_ptr<scalar_t>(),
            soft_adjoint_tangent.data_ptr<scalar_t>(),
            edge_types.data_ptr<std::int64_t>(),
            channel_radial_indices.data_ptr<std::int64_t>(),
            channel_angular_indices.data_ptr<std::int64_t>(),
            channel_types.data_ptr<std::int64_t>(), channel_scales.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(), distances.numel(),
            radial_values.size(1), angular_values.size(1), filter_centers.numel(),
            channel_scales.numel(), atomic_adjoint.size(0),
            atomic_tangent.data_ptr<scalar_t>(), radial_second.data_ptr<scalar_t>(),
            angular_second.data_ptr<scalar_t>(), distance_second.data_ptr<scalar_t>(),
            soft_second.data_ptr<scalar_t>());
      });
  return std::make_tuple(
      atomic_tangent, radial_second, angular_second, distance_second, soft_second);
}

void check_carrier_gated_scatter_inputs(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    std::int64_t target_count) {
  check_data_tensor(node_values, "node_values");
  check_data_tensor(edge_gates, "edge_gates");
  check_int64_vector(edge_sources, "edge_sources");
  check_int64_vector(edge_targets, "edge_targets");
  check_int64_vector(feature_channels, "feature_channels");
  const bool real_gates_for_complex_values =
      (node_values.scalar_type() == torch::kComplexFloat &&
       edge_gates.scalar_type() == torch::kFloat) ||
      (node_values.scalar_type() == torch::kComplexDouble &&
       edge_gates.scalar_type() == torch::kDouble);
  TORCH_CHECK(
      edge_gates.scalar_type() == node_values.scalar_type() ||
          real_gates_for_complex_values,
      "edge_gates must match node_values dtype or use its real component "
      "dtype for complex carriers");
  TORCH_CHECK(
      edge_sources.numel() == edge_gates.size(0) &&
          edge_targets.numel() == edge_gates.size(0),
      "edge indices must contain one entry per gate row");
  TORCH_CHECK(
      feature_channels.numel() == node_values.size(1),
      "feature_channels must contain one entry per feature");
  TORCH_CHECK(
      node_values.size(0) > 0,
      "node_values must contain at least one node");
  TORCH_CHECK(
      edge_gates.size(1) > 0,
      "edge_gates must contain at least one channel");
  TORCH_CHECK(target_count > 0, "target_count must be positive");
  if (edge_sources.numel() > 0) {
    TORCH_CHECK(
        edge_sources.min().item<std::int64_t>() >= 0 &&
            edge_sources.max().item<std::int64_t>() <
                node_values.size(0),
        "edge_sources entries must index node_values");
    TORCH_CHECK(
        edge_targets.min().item<std::int64_t>() >= 0 &&
            edge_targets.max().item<std::int64_t>() < target_count,
        "edge_targets entries must lie in [0, target_count)");
  }
  if (feature_channels.numel() > 0) {
    TORCH_CHECK(
        feature_channels.min().item<std::int64_t>() >= 0 &&
            feature_channels.max().item<std::int64_t>() <
                edge_gates.size(1),
        "feature_channels entries must index edge_gates columns");
  }
}

bool carrier_gated_scatter_uses_real_gates(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates) {
  return
      (node_values.scalar_type() == torch::kComplexFloat &&
       edge_gates.scalar_type() == torch::kFloat) ||
      (node_values.scalar_type() == torch::kComplexDouble &&
       edge_gates.scalar_type() == torch::kDouble);
}

torch::Tensor carrier_gated_scatter_cpu(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    std::int64_t target_count) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_count);
  auto target_values = torch::empty(
      {target_count, node_values.size(1)},
      node_values.options());
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_gated_scatter_real_gates_forward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              node_values.data_ptr<c10::complex<float>>()),
          edge_gates.data_ptr<float>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(1),
          edge_gates.size(1),
          target_count,
          reinterpret_cast<std::complex<float>*>(
              target_values.data_ptr<c10::complex<float>>()));
    } else {
      ye3t::runtime::carrier_gated_scatter_real_gates_forward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              node_values.data_ptr<c10::complex<double>>()),
          edge_gates.data_ptr<double>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(1),
          edge_gates.size(1),
          target_count,
          reinterpret_cast<std::complex<double>*>(
              target_values.data_ptr<c10::complex<double>>()));
    }
    return target_values;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_gated_scatter_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_gated_scatter_forward<core_t>(
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            edge_gates.size(0),
            node_values.size(1),
            edge_gates.size(1),
            target_count,
            reinterpret_cast<core_t*>(
                target_values.data_ptr<scalar_t>()));
      });
  return target_values;
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_gated_scatter_adjoint_cpu(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_adjoint.size(0));
  check_data_tensor(target_adjoint, "target_adjoint");
  TORCH_CHECK(
      target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.size(1) == node_values.size(1),
      "target_adjoint must match the node feature width and dtype");
  auto node_adjoint = torch::empty_like(node_values);
  auto gate_adjoint = torch::empty_like(edge_gates);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_gated_scatter_real_gates_adjoint<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              target_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              node_values.data_ptr<c10::complex<float>>()),
          edge_gates.data_ptr<float>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<float>*>(
              node_adjoint.data_ptr<c10::complex<float>>()),
          gate_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_gated_scatter_real_gates_adjoint<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              target_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              node_values.data_ptr<c10::complex<double>>()),
          edge_gates.data_ptr<double>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<double>*>(
              node_adjoint.data_ptr<c10::complex<double>>()),
          gate_adjoint.data_ptr<double>());
    }
    return std::make_tuple(node_adjoint, gate_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_gated_scatter_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_gated_scatter_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                target_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            edge_gates.size(0),
            node_values.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                node_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gate_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(node_adjoint, gate_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_gated_scatter_double_backward_cpu(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& node_adjoint_tangent,
    const torch::Tensor& gate_adjoint_tangent,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_adjoint.size(0));
  check_data_tensor(target_adjoint, "target_adjoint");
  check_data_tensor(node_adjoint_tangent, "node_adjoint_tangent");
  check_data_tensor(gate_adjoint_tangent, "gate_adjoint_tangent");
  TORCH_CHECK(
      target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.size(1) == node_values.size(1),
      "target_adjoint must match node_values");
  TORCH_CHECK(
      node_adjoint_tangent.scalar_type() ==
              node_values.scalar_type() &&
          node_adjoint_tangent.sizes() == node_values.sizes(),
      "node_adjoint_tangent must match node_values");
  TORCH_CHECK(
      gate_adjoint_tangent.scalar_type() ==
              edge_gates.scalar_type() &&
          gate_adjoint_tangent.sizes() == edge_gates.sizes(),
      "gate_adjoint_tangent must match edge_gates");
  auto target_adjoint_tangent = torch::empty_like(target_adjoint);
  auto node_second_adjoint = torch::empty_like(node_values);
  auto gate_second_adjoint = torch::empty_like(edge_gates);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_gated_scatter_real_gates_double_backward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              target_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              node_values.data_ptr<c10::complex<float>>()),
          edge_gates.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              node_adjoint_tangent.data_ptr<c10::complex<float>>()),
          gate_adjoint_tangent.data_ptr<float>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          target_adjoint.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<float>*>(
              target_adjoint_tangent.data_ptr<c10::complex<float>>()),
          reinterpret_cast<std::complex<float>*>(
              node_second_adjoint.data_ptr<c10::complex<float>>()),
          gate_second_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_gated_scatter_real_gates_double_backward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              target_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              node_values.data_ptr<c10::complex<double>>()),
          edge_gates.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              node_adjoint_tangent.data_ptr<c10::complex<double>>()),
          gate_adjoint_tangent.data_ptr<double>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          target_adjoint.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<double>*>(
              target_adjoint_tangent.data_ptr<c10::complex<double>>()),
          reinterpret_cast<std::complex<double>*>(
              node_second_adjoint.data_ptr<c10::complex<double>>()),
          gate_second_adjoint.data_ptr<double>());
    }
    return std::make_tuple(
        target_adjoint_tangent,
        node_second_adjoint,
        gate_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_gated_scatter_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_gated_scatter_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                target_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gate_adjoint_tangent.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            edge_gates.size(0),
            node_values.size(0),
            target_adjoint.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                target_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                node_second_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gate_second_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
}

torch::Tensor carrier_residual_gated_scatter_cpu(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      node_values.size(0));
  auto target_values = torch::empty_like(node_values);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_residual_gated_scatter_real_gates_forward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              node_values.data_ptr<c10::complex<float>>()),
          edge_gates.data_ptr<float>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<float>*>(
              target_values.data_ptr<c10::complex<float>>()));
    } else {
      ye3t::runtime::carrier_residual_gated_scatter_real_gates_forward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              node_values.data_ptr<c10::complex<double>>()),
          edge_gates.data_ptr<double>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<double>*>(
              target_values.data_ptr<c10::complex<double>>()));
    }
    return target_values;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_residual_gated_scatter_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_residual_gated_scatter_forward<core_t>(
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            edge_gates.size(0),
            node_values.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                target_values.data_ptr<scalar_t>()));
      });
  return target_values;
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_residual_gated_scatter_adjoint_cpu(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      node_values.size(0));
  check_data_tensor(target_adjoint, "target_adjoint");
  TORCH_CHECK(
      target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.sizes() == node_values.sizes(),
      "target_adjoint must match node_values");
  auto node_adjoint = torch::empty_like(node_values);
  auto gate_adjoint = torch::empty_like(edge_gates);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_residual_gated_scatter_real_gates_adjoint<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              target_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              node_values.data_ptr<c10::complex<float>>()),
          edge_gates.data_ptr<float>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<float>*>(
              node_adjoint.data_ptr<c10::complex<float>>()),
          gate_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_residual_gated_scatter_real_gates_adjoint<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              target_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              node_values.data_ptr<c10::complex<double>>()),
          edge_gates.data_ptr<double>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<double>*>(
              node_adjoint.data_ptr<c10::complex<double>>()),
          gate_adjoint.data_ptr<double>());
    }
    return std::make_tuple(node_adjoint, gate_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_residual_gated_scatter_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_residual_gated_scatter_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                target_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            edge_gates.size(0),
            node_values.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                node_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gate_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(node_adjoint, gate_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_residual_gated_scatter_double_backward_cpu(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& node_adjoint_tangent,
    const torch::Tensor& gate_adjoint_tangent,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      node_values.size(0));
  check_data_tensor(target_adjoint, "target_adjoint");
  check_data_tensor(node_adjoint_tangent, "node_adjoint_tangent");
  check_data_tensor(gate_adjoint_tangent, "gate_adjoint_tangent");
  TORCH_CHECK(
      target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.sizes() == node_values.sizes() &&
          node_adjoint_tangent.scalar_type() == node_values.scalar_type() &&
          node_adjoint_tangent.sizes() == node_values.sizes() &&
          gate_adjoint_tangent.scalar_type() == edge_gates.scalar_type() &&
          gate_adjoint_tangent.sizes() == edge_gates.sizes(),
      "residual scatter adjoint tangents must match primals");
  auto target_adjoint_tangent = torch::empty_like(target_adjoint);
  auto node_second_adjoint = torch::empty_like(node_values);
  auto gate_second_adjoint = torch::empty_like(edge_gates);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_residual_gated_scatter_real_gates_double_backward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              target_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              node_values.data_ptr<c10::complex<float>>()),
          edge_gates.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              node_adjoint_tangent.data_ptr<c10::complex<float>>()),
          gate_adjoint_tangent.data_ptr<float>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<float>*>(
              target_adjoint_tangent.data_ptr<c10::complex<float>>()),
          reinterpret_cast<std::complex<float>*>(
              node_second_adjoint.data_ptr<c10::complex<float>>()),
          gate_second_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_residual_gated_scatter_real_gates_double_backward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              target_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              node_values.data_ptr<c10::complex<double>>()),
          edge_gates.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              node_adjoint_tangent.data_ptr<c10::complex<double>>()),
          gate_adjoint_tangent.data_ptr<double>(),
          edge_sources.data_ptr<std::int64_t>(),
          edge_targets.data_ptr<std::int64_t>(),
          feature_channels.data_ptr<std::int64_t>(),
          edge_gates.size(0),
          node_values.size(0),
          node_values.size(1),
          edge_gates.size(1),
          reinterpret_cast<std::complex<double>*>(
              target_adjoint_tangent.data_ptr<c10::complex<double>>()),
          reinterpret_cast<std::complex<double>*>(
              node_second_adjoint.data_ptr<c10::complex<double>>()),
          gate_second_adjoint.data_ptr<double>());
    }
    return std::make_tuple(
        target_adjoint_tangent,
        node_second_adjoint,
        gate_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_residual_gated_scatter_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_residual_gated_scatter_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                target_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gate_adjoint_tangent.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            edge_gates.size(0),
            node_values.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                target_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                node_second_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gate_second_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
}

void check_carrier_segmented_scatter_inputs(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& target_offsets,
    const torch::Tensor& source_offsets,
    const torch::Tensor& source_edges,
    const torch::Tensor& feature_channels,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& channel_features) {
  check_carrier_gated_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      node_values.size(0));
  check_int64_vector(target_offsets, "target_offsets");
  check_int64_vector(source_offsets, "source_offsets");
  check_int64_vector(source_edges, "source_edges");
  check_int64_vector(channel_offsets, "channel_offsets");
  check_int64_vector(channel_features, "channel_features");
  const std::int64_t node_count = node_values.size(0);
  const std::int64_t edge_count = edge_gates.size(0);
  const std::int64_t feature_count = node_values.size(1);
  const std::int64_t channel_count = edge_gates.size(1);
  TORCH_CHECK(
      target_offsets.numel() == node_count + 1 &&
          source_offsets.numel() == node_count + 1,
      "target_offsets and source_offsets must contain node_count + 1 entries");
  TORCH_CHECK(
      source_edges.numel() == edge_count,
      "source_edges must contain one entry per edge");
  TORCH_CHECK(
      channel_offsets.numel() == channel_count + 1 &&
          channel_features.numel() == feature_count,
      "channel segments must cover every feature exactly once");
  const std::int64_t* target_offset_data =
      target_offsets.data_ptr<std::int64_t>();
  const std::int64_t* source_offset_data =
      source_offsets.data_ptr<std::int64_t>();
  const std::int64_t* source_edge_data =
      source_edges.data_ptr<std::int64_t>();
  const std::int64_t* channel_offset_data =
      channel_offsets.data_ptr<std::int64_t>();
  const std::int64_t* channel_feature_data =
      channel_features.data_ptr<std::int64_t>();
  const std::int64_t* edge_source_data =
      edge_sources.data_ptr<std::int64_t>();
  const std::int64_t* edge_target_data =
      edge_targets.data_ptr<std::int64_t>();
  const std::int64_t* feature_channel_data =
      feature_channels.data_ptr<std::int64_t>();
  TORCH_CHECK(
      target_offset_data[0] == 0 &&
          target_offset_data[node_count] == edge_count,
      "target_offsets must span every edge");
  TORCH_CHECK(
      source_offset_data[0] == 0 &&
          source_offset_data[node_count] == edge_count,
      "source_offsets must span every edge");
  TORCH_CHECK(
      channel_offset_data[0] == 0 &&
          channel_offset_data[channel_count] == feature_count,
      "channel_offsets must span every feature");
  std::vector<bool> seen_edges(edge_count, false);
  for (std::int64_t target = 0; target < node_count; ++target) {
    TORCH_CHECK(
        target_offset_data[target] <= target_offset_data[target + 1],
        "target_offsets must be nondecreasing");
    for (std::int64_t edge = target_offset_data[target];
         edge < target_offset_data[target + 1];
         ++edge) {
      TORCH_CHECK(
          edge_target_data[edge] == target,
          "target segments must match edge_targets in target-major order");
    }
  }
  for (std::int64_t source = 0; source < node_count; ++source) {
    TORCH_CHECK(
        source_offset_data[source] <= source_offset_data[source + 1],
        "source_offsets must be nondecreasing");
    for (std::int64_t entry = source_offset_data[source];
         entry < source_offset_data[source + 1];
         ++entry) {
      const std::int64_t edge = source_edge_data[entry];
      TORCH_CHECK(
          edge >= 0 && edge < edge_count && !seen_edges[edge],
          "source_edges must be a permutation of edge indices");
      seen_edges[edge] = true;
      TORCH_CHECK(
          edge_source_data[edge] == source,
          "source segments must match edge_sources");
    }
  }
  std::vector<bool> seen_features(feature_count, false);
  for (std::int64_t channel = 0; channel < channel_count; ++channel) {
    TORCH_CHECK(
        channel_offset_data[channel] <= channel_offset_data[channel + 1],
        "channel_offsets must be nondecreasing");
    for (std::int64_t entry = channel_offset_data[channel];
         entry < channel_offset_data[channel + 1];
         ++entry) {
      const std::int64_t feature = channel_feature_data[entry];
      TORCH_CHECK(
          feature >= 0 && feature < feature_count &&
              !seen_features[feature],
          "channel_features must be a permutation of feature indices");
      seen_features[feature] = true;
      TORCH_CHECK(
          feature_channel_data[feature] == channel,
          "channel segments must match feature_channels");
    }
  }
}

torch::Tensor carrier_segmented_residual_gated_scatter_cpu(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& target_offsets,
    const torch::Tensor& source_offsets,
    const torch::Tensor& source_edges,
    const torch::Tensor& feature_channels,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& channel_features) {
  check_carrier_segmented_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      target_offsets,
      source_offsets,
      source_edges,
      feature_channels,
      channel_offsets,
      channel_features);
  auto target_values = torch::empty_like(node_values);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::
          carrier_segmented_residual_gated_scatter_real_gates_forward<
              std::complex<float>, float>(
              reinterpret_cast<const std::complex<float>*>(
                  node_values.data_ptr<c10::complex<float>>()),
              edge_gates.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              node_values.size(1),
              edge_gates.size(1),
              reinterpret_cast<std::complex<float>*>(
                  target_values.data_ptr<c10::complex<float>>()));
    } else {
      ye3t::runtime::
          carrier_segmented_residual_gated_scatter_real_gates_forward<
              std::complex<double>, double>(
              reinterpret_cast<const std::complex<double>*>(
                  node_values.data_ptr<c10::complex<double>>()),
              edge_gates.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              node_values.size(1),
              edge_gates.size(1),
              reinterpret_cast<std::complex<double>*>(
                  target_values.data_ptr<c10::complex<double>>()));
    }
    return target_values;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_segmented_residual_gated_scatter_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_segmented_residual_gated_scatter_forward<
            core_t>(
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            target_offsets.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            node_values.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                target_values.data_ptr<scalar_t>()));
      });
  return target_values;
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_segmented_residual_gated_scatter_adjoint_cpu(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& target_offsets,
    const torch::Tensor& source_offsets,
    const torch::Tensor& source_edges,
    const torch::Tensor& feature_channels,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& channel_features) {
  check_carrier_segmented_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      target_offsets,
      source_offsets,
      source_edges,
      feature_channels,
      channel_offsets,
      channel_features);
  check_data_tensor(target_adjoint, "target_adjoint");
  TORCH_CHECK(
      target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.sizes() == node_values.sizes(),
      "target_adjoint must match node_values");
  auto node_adjoint = torch::empty_like(node_values);
  auto gate_adjoint = torch::empty_like(edge_gates);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::
          carrier_segmented_residual_gated_scatter_real_gates_adjoint<
              std::complex<float>, float>(
              reinterpret_cast<const std::complex<float>*>(
                  target_adjoint.data_ptr<c10::complex<float>>()),
              reinterpret_cast<const std::complex<float>*>(
                  node_values.data_ptr<c10::complex<float>>()),
              edge_gates.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              channel_offsets.data_ptr<std::int64_t>(),
              channel_features.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              reinterpret_cast<std::complex<float>*>(
                  node_adjoint.data_ptr<c10::complex<float>>()),
              gate_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::
          carrier_segmented_residual_gated_scatter_real_gates_adjoint<
              std::complex<double>, double>(
              reinterpret_cast<const std::complex<double>*>(
                  target_adjoint.data_ptr<c10::complex<double>>()),
              reinterpret_cast<const std::complex<double>*>(
                  node_values.data_ptr<c10::complex<double>>()),
              edge_gates.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              channel_offsets.data_ptr<std::int64_t>(),
              channel_features.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              reinterpret_cast<std::complex<double>*>(
                  node_adjoint.data_ptr<c10::complex<double>>()),
              gate_adjoint.data_ptr<double>());
    }
    return std::make_tuple(node_adjoint, gate_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_segmented_residual_gated_scatter_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::
            carrier_segmented_residual_gated_scatter_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                target_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            source_offsets.data_ptr<std::int64_t>(),
            source_edges.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            channel_features.data_ptr<std::int64_t>(),
            node_values.size(0),
            edge_gates.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                node_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gate_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(node_adjoint, gate_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_segmented_residual_gated_scatter_double_backward_cpu(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& node_adjoint_tangent,
    const torch::Tensor& gate_adjoint_tangent,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& target_offsets,
    const torch::Tensor& source_offsets,
    const torch::Tensor& source_edges,
    const torch::Tensor& feature_channels,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& channel_features) {
  check_carrier_segmented_scatter_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      target_offsets,
      source_offsets,
      source_edges,
      feature_channels,
      channel_offsets,
      channel_features);
  check_data_tensor(target_adjoint, "target_adjoint");
  check_data_tensor(node_adjoint_tangent, "node_adjoint_tangent");
  check_data_tensor(gate_adjoint_tangent, "gate_adjoint_tangent");
  TORCH_CHECK(
      target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.sizes() == node_values.sizes() &&
          node_adjoint_tangent.scalar_type() == node_values.scalar_type() &&
          node_adjoint_tangent.sizes() == node_values.sizes() &&
          gate_adjoint_tangent.scalar_type() == edge_gates.scalar_type() &&
          gate_adjoint_tangent.sizes() == edge_gates.sizes(),
      "segmented scatter adjoint tangents must match primals");
  auto target_adjoint_tangent = torch::empty_like(target_adjoint);
  auto node_second_adjoint = torch::empty_like(node_values);
  auto gate_second_adjoint = torch::empty_like(edge_gates);
  if (carrier_gated_scatter_uses_real_gates(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::
          carrier_segmented_residual_gated_scatter_real_gates_double_backward<
              std::complex<float>, float>(
              reinterpret_cast<const std::complex<float>*>(
                  target_adjoint.data_ptr<c10::complex<float>>()),
              reinterpret_cast<const std::complex<float>*>(
                  node_values.data_ptr<c10::complex<float>>()),
              edge_gates.data_ptr<float>(),
              reinterpret_cast<const std::complex<float>*>(
                  node_adjoint_tangent.data_ptr<c10::complex<float>>()),
              gate_adjoint_tangent.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              channel_offsets.data_ptr<std::int64_t>(),
              channel_features.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              reinterpret_cast<std::complex<float>*>(
                  target_adjoint_tangent.data_ptr<c10::complex<float>>()),
              reinterpret_cast<std::complex<float>*>(
                  node_second_adjoint.data_ptr<c10::complex<float>>()),
              gate_second_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::
          carrier_segmented_residual_gated_scatter_real_gates_double_backward<
              std::complex<double>, double>(
              reinterpret_cast<const std::complex<double>*>(
                  target_adjoint.data_ptr<c10::complex<double>>()),
              reinterpret_cast<const std::complex<double>*>(
                  node_values.data_ptr<c10::complex<double>>()),
              edge_gates.data_ptr<double>(),
              reinterpret_cast<const std::complex<double>*>(
                  node_adjoint_tangent.data_ptr<c10::complex<double>>()),
              gate_adjoint_tangent.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              channel_offsets.data_ptr<std::int64_t>(),
              channel_features.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              reinterpret_cast<std::complex<double>*>(
                  target_adjoint_tangent.data_ptr<c10::complex<double>>()),
              reinterpret_cast<std::complex<double>*>(
                  node_second_adjoint.data_ptr<c10::complex<double>>()),
              gate_second_adjoint.data_ptr<double>());
    }
    return std::make_tuple(
        target_adjoint_tangent,
        node_second_adjoint,
        gate_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_segmented_residual_gated_scatter_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::
            carrier_segmented_residual_gated_scatter_double_backward<
                core_t>(
            reinterpret_cast<const core_t*>(
                target_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                edge_gates.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                node_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gate_adjoint_tangent.data_ptr<scalar_t>()),
            edge_sources.data_ptr<std::int64_t>(),
            edge_targets.data_ptr<std::int64_t>(),
            target_offsets.data_ptr<std::int64_t>(),
            source_offsets.data_ptr<std::int64_t>(),
            source_edges.data_ptr<std::int64_t>(),
            feature_channels.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            channel_features.data_ptr<std::int64_t>(),
            node_values.size(0),
            edge_gates.size(0),
            node_values.size(1),
            edge_gates.size(1),
            reinterpret_cast<core_t*>(
                target_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                node_second_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gate_second_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
}

void check_source_arena_schedule_cpu(
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t output_width) {
  check_int64_vector(gather_indices, "gather_indices");
  check_int64_vector(reverse_offsets, "reverse_offsets");
  check_int64_vector(reverse_output_indices, "reverse_output_indices");
  check_int64_vector(center_types, "center_types");
  check_int64_vector(atom_types, "atom_types");
  TORCH_CHECK(batch_size >= 0, "source arena batch size must be nonnegative");
  TORCH_CHECK(producer_width > 0, "source arena producer width must be positive");
  TORCH_CHECK(output_width > 0, "source arena output width must be positive");
  TORCH_CHECK(
      gather_indices.numel() == output_width &&
          center_types.numel() == output_width,
      "gather_indices and center_types must match source arena output width");
  TORCH_CHECK(
      reverse_offsets.numel() == producer_width + 1,
      "reverse_offsets must have producer_width + 1 entries");
  TORCH_CHECK(
      reverse_output_indices.numel() == output_width,
      "reverse_output_indices must contain each output coordinate once");
  TORCH_CHECK(
      atom_types.numel() == batch_size,
      "atom_types must contain one value per source arena row");
  const auto* gather = gather_indices.data_ptr<std::int64_t>();
  const auto* offsets = reverse_offsets.data_ptr<std::int64_t>();
  const auto* reverse = reverse_output_indices.data_ptr<std::int64_t>();
  TORCH_CHECK(offsets[0] == 0, "reverse_offsets must start at zero");
  TORCH_CHECK(
      offsets[producer_width] == output_width,
      "reverse_offsets must terminate at output_width");
  std::vector<std::int64_t> visits(
      static_cast<std::size_t>(output_width), 0);
  for (std::int64_t producer_index = 0;
       producer_index < producer_width;
       ++producer_index) {
    TORCH_CHECK(
        offsets[producer_index] <= offsets[producer_index + 1],
        "reverse_offsets must be nondecreasing");
    for (std::int64_t entry = offsets[producer_index];
         entry < offsets[producer_index + 1];
         ++entry) {
      const std::int64_t output_index = reverse[entry];
      TORCH_CHECK(
          output_index >= 0 && output_index < output_width,
          "reverse_output_indices contains an out-of-range coordinate");
      TORCH_CHECK(
          gather[output_index] == producer_index,
          "reverse source arena schedule disagrees with gather_indices");
      ++visits[static_cast<std::size_t>(output_index)];
    }
  }
  for (std::int64_t output_index = 0;
       output_index < output_width;
       ++output_index) {
    TORCH_CHECK(
        gather[output_index] >= 0 &&
            gather[output_index] < producer_width,
        "gather_indices contains an out-of-range producer coordinate");
    TORCH_CHECK(
        visits[static_cast<std::size_t>(output_index)] == 1,
        "reverse source arena schedule must visit every output once");
  }
}

torch::Tensor source_arena_gather_cpu(
    const torch::Tensor& producer,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types) {
  check_data_tensor(producer, "producer");
  const std::int64_t output_width = gather_indices.numel();
  check_source_arena_schedule_cpu(
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      producer.size(0),
      producer.size(1),
      output_width);
  auto output = torch::empty(
      {producer.size(0), output_width}, producer.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      producer.scalar_type(),
      "ye3t_source_arena_gather_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::source_arena_gather_forward<core_t>(
            reinterpret_cast<const core_t*>(producer.data_ptr<scalar_t>()),
            gather_indices.data_ptr<std::int64_t>(),
            center_types.data_ptr<std::int64_t>(),
            atom_types.data_ptr<std::int64_t>(),
            producer.size(0),
            producer.size(1),
            output_width,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

torch::Tensor source_arena_gather_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types) {
  check_data_tensor(output_adjoint, "output_adjoint");
  const std::int64_t producer_width = reverse_offsets.numel() - 1;
  check_source_arena_schedule_cpu(
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      output_adjoint.size(0),
      producer_width,
      output_adjoint.size(1));
  auto producer_adjoint = torch::empty(
      {output_adjoint.size(0), producer_width},
      output_adjoint.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      output_adjoint.scalar_type(),
      "ye3t_source_arena_gather_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::source_arena_gather_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reverse_offsets.data_ptr<std::int64_t>(),
            reverse_output_indices.data_ptr<std::int64_t>(),
            center_types.data_ptr<std::int64_t>(),
            atom_types.data_ptr<std::int64_t>(),
            output_adjoint.size(0),
            producer_width,
            output_adjoint.size(1),
            reinterpret_cast<core_t*>(
                producer_adjoint.data_ptr<scalar_t>()));
      });
  return producer_adjoint;
}

torch::Tensor source_arena_gather_double_backward_cpu(
    const torch::Tensor& producer_adjoint_tangent,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types) {
  return source_arena_gather_cpu(
      producer_adjoint_tangent,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types);
}

void check_source_arena_channel_transform_cpu(
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_data_tensor(producer, "producer");
  check_data_vector(channel_maps, "channel_maps");
  check_int64_vector(input_feature_offsets, "input_feature_offsets");
  check_int64_vector(output_feature_offsets, "output_feature_offsets");
  check_int64_vector(input_channel_offsets, "input_channel_offsets");
  check_int64_vector(output_channel_offsets, "output_channel_offsets");
  check_int64_vector(map_offsets, "map_offsets");
  check_source_arena_schedule_cpu(
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      producer.size(0),
      producer.size(1),
      gather_indices.numel());
  const bool real_maps_for_complex_values =
      (producer.scalar_type() == torch::kComplexFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (producer.scalar_type() == torch::kComplexDouble &&
       channel_maps.scalar_type() == torch::kDouble);
  TORCH_CHECK(
      channel_maps.scalar_type() == producer.scalar_type() ||
          real_maps_for_complex_values,
      "source-arena channel maps must match producer dtype or use its real "
      "component dtype");
  const std::int64_t offset_count = input_feature_offsets.numel();
  TORCH_CHECK(
      offset_count >= 2 &&
          output_feature_offsets.numel() == offset_count &&
          input_channel_offsets.numel() == offset_count &&
          output_channel_offsets.numel() == offset_count &&
          map_offsets.numel() == offset_count,
      "source-arena channel-transform offsets must share one block_count + "
      "1 length");
  const auto* input_feature_data =
      input_feature_offsets.data_ptr<std::int64_t>();
  const auto* output_feature_data =
      output_feature_offsets.data_ptr<std::int64_t>();
  const auto* input_channel_data =
      input_channel_offsets.data_ptr<std::int64_t>();
  const auto* output_channel_data =
      output_channel_offsets.data_ptr<std::int64_t>();
  const auto* map_data = map_offsets.data_ptr<std::int64_t>();
  const auto* center_data = center_types.data_ptr<std::int64_t>();
  const std::int64_t block_count = offset_count - 1;
  TORCH_CHECK(
      input_feature_data[0] == 0 && output_feature_data[0] == 0 &&
          input_channel_data[0] == 0 && output_channel_data[0] == 0 &&
          map_data[0] == 0,
      "source-arena channel-transform offsets must begin at zero");
  TORCH_CHECK(
      input_feature_data[block_count] == gather_indices.numel() &&
          map_data[block_count] == channel_maps.numel(),
      "source-arena channel-transform final offsets must match packed "
      "inputs");
  for (std::int64_t block = 0; block < block_count; ++block) {
    const std::int64_t input_start = input_feature_data[block];
    const std::int64_t input_stop = input_feature_data[block + 1];
    const std::int64_t output_start = output_feature_data[block];
    const std::int64_t output_stop = output_feature_data[block + 1];
    const std::int64_t input_channels =
        input_channel_data[block + 1] - input_channel_data[block];
    const std::int64_t output_channels =
        output_channel_data[block + 1] - output_channel_data[block];
    const std::int64_t local_maps = map_data[block + 1] - map_data[block];
    TORCH_CHECK(
        input_stop > input_start && output_stop > output_start &&
            input_channels > 0 && output_channels > 0,
        "source-arena channel-transform blocks require positive dimensions");
    TORCH_CHECK(
        (input_stop - input_start) % input_channels == 0 &&
            (output_stop - output_start) % output_channels == 0 &&
            (input_stop - input_start) / input_channels ==
                (output_stop - output_start) / output_channels,
        "source-arena channel-transform blocks must preserve complete "
        "carrier axes");
    TORCH_CHECK(
        output_channels <= input_channels,
        "source-arena channel-transform blocks must be non-expanding");
    TORCH_CHECK(
        local_maps == input_channels * output_channels ||
            (local_maps == 0 && input_channels == output_channels),
        "source-arena channel-transform blocks require rectangular maps or "
        "an exact identity passthrough");
    const std::int64_t center_type = center_data[input_start];
    for (std::int64_t coordinate = input_start + 1;
         coordinate < input_stop;
         ++coordinate) {
      TORCH_CHECK(
          center_data[coordinate] == center_type,
          "one source-arena transform block cannot mix center types");
    }
  }
}

torch::Tensor source_arena_channel_transform_cpu(
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    std::int64_t declared_output_width) {
  check_source_arena_channel_transform_cpu(
      producer,
      channel_maps,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      declared_output_width == output_width,
      "declared source-arena channel-transform output width does not match "
      "offsets");
  auto output = torch::empty(
      {producer.size(0), output_width}, producer.options());
  if (producer.scalar_type() == torch::kComplexFloat &&
      channel_maps.scalar_type() == torch::kFloat) {
    ye3t::runtime::source_arena_channel_transform_forward<
        std::complex<float>, float>(
        reinterpret_cast<const std::complex<float>*>(
            producer.data_ptr<c10::complex<float>>()),
        channel_maps.data_ptr<float>(),
        gather_indices.data_ptr<std::int64_t>(),
        center_types.data_ptr<std::int64_t>(),
        atom_types.data_ptr<std::int64_t>(),
        input_feature_offsets.data_ptr<std::int64_t>(),
        output_feature_offsets.data_ptr<std::int64_t>(),
        input_channel_offsets.data_ptr<std::int64_t>(),
        output_channel_offsets.data_ptr<std::int64_t>(),
        map_offsets.data_ptr<std::int64_t>(),
        producer.size(0),
        producer.size(1),
        block_count,
        reinterpret_cast<std::complex<float>*>(
            output.data_ptr<c10::complex<float>>()));
  } else if (
      producer.scalar_type() == torch::kComplexDouble &&
      channel_maps.scalar_type() == torch::kDouble) {
    ye3t::runtime::source_arena_channel_transform_forward<
        std::complex<double>, double>(
        reinterpret_cast<const std::complex<double>*>(
            producer.data_ptr<c10::complex<double>>()),
        channel_maps.data_ptr<double>(),
        gather_indices.data_ptr<std::int64_t>(),
        center_types.data_ptr<std::int64_t>(),
        atom_types.data_ptr<std::int64_t>(),
        input_feature_offsets.data_ptr<std::int64_t>(),
        output_feature_offsets.data_ptr<std::int64_t>(),
        input_channel_offsets.data_ptr<std::int64_t>(),
        output_channel_offsets.data_ptr<std::int64_t>(),
        map_offsets.data_ptr<std::int64_t>(),
        producer.size(0),
        producer.size(1),
        block_count,
        reinterpret_cast<std::complex<double>*>(
            output.data_ptr<c10::complex<double>>()));
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        producer.scalar_type(),
        "ye3t_source_arena_channel_transform_cpu",
        [&] {
          using core_t = typename CoreScalar<scalar_t>::type;
          ye3t::runtime::source_arena_channel_transform_forward<
              core_t, core_t>(
              reinterpret_cast<const core_t*>(
                  producer.data_ptr<scalar_t>()),
              reinterpret_cast<const core_t*>(
                  channel_maps.data_ptr<scalar_t>()),
              gather_indices.data_ptr<std::int64_t>(),
              center_types.data_ptr<std::int64_t>(),
              atom_types.data_ptr<std::int64_t>(),
              input_feature_offsets.data_ptr<std::int64_t>(),
              output_feature_offsets.data_ptr<std::int64_t>(),
              input_channel_offsets.data_ptr<std::int64_t>(),
              output_channel_offsets.data_ptr<std::int64_t>(),
              map_offsets.data_ptr<std::int64_t>(),
              producer.size(0),
              producer.size(1),
              block_count,
              reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
        });
  }
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
source_arena_channel_transform_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_source_arena_channel_transform_cpu(
      producer,
      channel_maps,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_data_tensor(output_adjoint, "output_adjoint");
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      output_adjoint.size(0) == producer.size(0) &&
          output_adjoint.size(1) == output_width &&
          output_adjoint.scalar_type() == producer.scalar_type(),
      "output_adjoint must match the fused source-arena output");
  auto producer_adjoint = torch::empty_like(producer);
  auto maps_adjoint = torch::empty_like(channel_maps);
#define YE3T_SOURCE_ARENA_CHANNEL_ADJOINT(Scalar, Control, ScalarTorch, ControlTorch) \
  ye3t::runtime::source_arena_channel_transform_adjoint<Scalar, Control>( \
      reinterpret_cast<const Scalar*>(output_adjoint.data_ptr<ScalarTorch>()), \
      reinterpret_cast<const Scalar*>(producer.data_ptr<ScalarTorch>()), \
      reinterpret_cast<const Control*>(channel_maps.data_ptr<ControlTorch>()), \
      gather_indices.data_ptr<std::int64_t>(), \
      reverse_offsets.data_ptr<std::int64_t>(), \
      reverse_output_indices.data_ptr<std::int64_t>(), \
      center_types.data_ptr<std::int64_t>(), \
      atom_types.data_ptr<std::int64_t>(), \
      input_feature_offsets.data_ptr<std::int64_t>(), \
      output_feature_offsets.data_ptr<std::int64_t>(), \
      input_channel_offsets.data_ptr<std::int64_t>(), \
      output_channel_offsets.data_ptr<std::int64_t>(), \
      map_offsets.data_ptr<std::int64_t>(), producer.size(0), \
      producer.size(1), block_count, \
      reinterpret_cast<Scalar*>(producer_adjoint.data_ptr<ScalarTorch>()), \
      reinterpret_cast<Control*>(maps_adjoint.data_ptr<ControlTorch>()))
  if (producer.scalar_type() == torch::kComplexFloat &&
      channel_maps.scalar_type() == torch::kFloat) {
    YE3T_SOURCE_ARENA_CHANNEL_ADJOINT(
        std::complex<float>, float, c10::complex<float>, float);
  } else if (
      producer.scalar_type() == torch::kComplexDouble &&
      channel_maps.scalar_type() == torch::kDouble) {
    YE3T_SOURCE_ARENA_CHANNEL_ADJOINT(
        std::complex<double>, double, c10::complex<double>, double);
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        producer.scalar_type(),
        "ye3t_source_arena_channel_transform_adjoint_cpu",
        [&] {
          using core_t = typename CoreScalar<scalar_t>::type;
          YE3T_SOURCE_ARENA_CHANNEL_ADJOINT(
              core_t, core_t, scalar_t, scalar_t);
        });
  }
#undef YE3T_SOURCE_ARENA_CHANNEL_ADJOINT
  return std::make_tuple(producer_adjoint, maps_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
source_arena_channel_transform_double_backward_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& producer_adjoint_tangent,
    const torch::Tensor& channel_maps_adjoint_tangent,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_source_arena_channel_transform_cpu(
      producer,
      channel_maps,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(producer_adjoint_tangent, "producer_adjoint_tangent");
  check_data_vector(
      channel_maps_adjoint_tangent,
      "channel_maps_adjoint_tangent");
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      output_adjoint.size(0) == producer.size(0) &&
          output_adjoint.size(1) == output_width &&
          output_adjoint.scalar_type() == producer.scalar_type() &&
          producer_adjoint_tangent.sizes() == producer.sizes() &&
          producer_adjoint_tangent.scalar_type() == producer.scalar_type() &&
          channel_maps_adjoint_tangent.sizes() == channel_maps.sizes() &&
          channel_maps_adjoint_tangent.scalar_type() ==
              channel_maps.scalar_type(),
      "source-arena channel-transform double-adjoint tangents have "
      "incompatible shapes or dtypes");
  auto output_tangent = torch::empty_like(output_adjoint);
  auto producer_second = torch::empty_like(producer);
  auto maps_second = torch::empty_like(channel_maps);
#define YE3T_SOURCE_ARENA_CHANNEL_DOUBLE(Scalar, Control, ScalarTorch, ControlTorch) \
  ye3t::runtime::source_arena_channel_transform_double_backward< \
      Scalar, Control>( \
      reinterpret_cast<const Scalar*>(output_adjoint.data_ptr<ScalarTorch>()), \
      reinterpret_cast<const Scalar*>(producer.data_ptr<ScalarTorch>()), \
      reinterpret_cast<const Control*>(channel_maps.data_ptr<ControlTorch>()), \
      reinterpret_cast<const Scalar*>( \
          producer_adjoint_tangent.data_ptr<ScalarTorch>()), \
      reinterpret_cast<const Control*>( \
          channel_maps_adjoint_tangent.data_ptr<ControlTorch>()), \
      gather_indices.data_ptr<std::int64_t>(), \
      reverse_offsets.data_ptr<std::int64_t>(), \
      reverse_output_indices.data_ptr<std::int64_t>(), \
      center_types.data_ptr<std::int64_t>(), \
      atom_types.data_ptr<std::int64_t>(), \
      input_feature_offsets.data_ptr<std::int64_t>(), \
      output_feature_offsets.data_ptr<std::int64_t>(), \
      input_channel_offsets.data_ptr<std::int64_t>(), \
      output_channel_offsets.data_ptr<std::int64_t>(), \
      map_offsets.data_ptr<std::int64_t>(), producer.size(0), \
      producer.size(1), block_count, \
      reinterpret_cast<Scalar*>(output_tangent.data_ptr<ScalarTorch>()), \
      reinterpret_cast<Scalar*>(producer_second.data_ptr<ScalarTorch>()), \
      reinterpret_cast<Control*>(maps_second.data_ptr<ControlTorch>()))
  if (producer.scalar_type() == torch::kComplexFloat &&
      channel_maps.scalar_type() == torch::kFloat) {
    YE3T_SOURCE_ARENA_CHANNEL_DOUBLE(
        std::complex<float>, float, c10::complex<float>, float);
  } else if (
      producer.scalar_type() == torch::kComplexDouble &&
      channel_maps.scalar_type() == torch::kDouble) {
    YE3T_SOURCE_ARENA_CHANNEL_DOUBLE(
        std::complex<double>, double, c10::complex<double>, double);
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        producer.scalar_type(),
        "ye3t_source_arena_channel_transform_double_backward_cpu",
        [&] {
          using core_t = typename CoreScalar<scalar_t>::type;
          YE3T_SOURCE_ARENA_CHANNEL_DOUBLE(
              core_t, core_t, scalar_t, scalar_t);
        });
  }
#undef YE3T_SOURCE_ARENA_CHANNEL_DOUBLE
  return std::make_tuple(output_tangent, producer_second, maps_second);
}

void check_carrier_channel_update_inputs(
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_data_tensor(values, "values");
  check_data_tensor(gates, "gates");
  check_data_vector(channel_maps, "channel_maps");
  check_int64_vector(feature_offsets, "feature_offsets");
  check_int64_vector(channel_offsets, "channel_offsets");
  check_int64_vector(map_offsets, "map_offsets");
  TORCH_CHECK(
      values.size(0) == gates.size(0),
      "values and gates must share the batch axis");
  const bool real_controls_for_complex_values =
      (values.scalar_type() == torch::kComplexFloat &&
       gates.scalar_type() == torch::kFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (values.scalar_type() == torch::kComplexDouble &&
       gates.scalar_type() == torch::kDouble &&
       channel_maps.scalar_type() == torch::kDouble);
  TORCH_CHECK(
      (gates.scalar_type() == values.scalar_type() &&
       channel_maps.scalar_type() == values.scalar_type()) ||
          real_controls_for_complex_values,
      "gates and channel_maps must either match values dtype or both use "
      "its real component dtype for complex carriers");
  TORCH_CHECK(
      feature_offsets.numel() >= 2 &&
          feature_offsets.numel() == channel_offsets.numel() &&
          feature_offsets.numel() == map_offsets.numel(),
      "carrier update offsets must have one common block_count + 1 length");
  const auto* feature_data = feature_offsets.data_ptr<std::int64_t>();
  const auto* channel_data = channel_offsets.data_ptr<std::int64_t>();
  const auto* map_data = map_offsets.data_ptr<std::int64_t>();
  const std::int64_t block_count = feature_offsets.numel() - 1;
  TORCH_CHECK(
      feature_data[0] == 0 &&
          channel_data[0] == 0 &&
          map_data[0] == 0,
      "carrier update offsets must begin at zero");
  TORCH_CHECK(
      feature_data[block_count] == values.size(1) &&
          channel_data[block_count] == gates.size(1) &&
          map_data[block_count] == channel_maps.numel(),
      "carrier update final offsets must match packed tensor widths");
  for (std::int64_t block = 0; block < block_count; ++block) {
    const std::int64_t local_features =
        feature_data[block + 1] - feature_data[block];
    const std::int64_t local_channels =
        channel_data[block + 1] - channel_data[block];
    const std::int64_t local_maps =
        map_data[block + 1] - map_data[block];
    TORCH_CHECK(
        local_features > 0 && local_channels > 0,
        "carrier update blocks must have positive dimensions");
    TORCH_CHECK(
        local_features % local_channels == 0,
        "each carrier block width must be divisible by its channel count");
    TORCH_CHECK(
        local_maps == local_channels * local_channels,
        "each carrier block must provide one square channel map");
  }
}

void check_carrier_channel_transform_inputs(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_data_tensor(values, "values");
  check_data_vector(channel_maps, "channel_maps");
  check_int64_vector(input_feature_offsets, "input_feature_offsets");
  check_int64_vector(output_feature_offsets, "output_feature_offsets");
  check_int64_vector(input_channel_offsets, "input_channel_offsets");
  check_int64_vector(output_channel_offsets, "output_channel_offsets");
  check_int64_vector(map_offsets, "map_offsets");
  const bool real_maps_for_complex_values =
      (values.scalar_type() == torch::kComplexFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (values.scalar_type() == torch::kComplexDouble &&
       channel_maps.scalar_type() == torch::kDouble);
  TORCH_CHECK(
      channel_maps.scalar_type() == values.scalar_type() ||
          real_maps_for_complex_values,
      "channel_maps must match values dtype or use its real component dtype");
  const auto offset_count = input_feature_offsets.numel();
  TORCH_CHECK(
      offset_count >= 2 &&
          output_feature_offsets.numel() == offset_count &&
          input_channel_offsets.numel() == offset_count &&
          output_channel_offsets.numel() == offset_count &&
          map_offsets.numel() == offset_count,
      "channel-transform offsets must share one block_count + 1 length");
  const auto* input_feature_data =
      input_feature_offsets.data_ptr<std::int64_t>();
  const auto* output_feature_data =
      output_feature_offsets.data_ptr<std::int64_t>();
  const auto* input_channel_data =
      input_channel_offsets.data_ptr<std::int64_t>();
  const auto* output_channel_data =
      output_channel_offsets.data_ptr<std::int64_t>();
  const auto* map_data = map_offsets.data_ptr<std::int64_t>();
  const std::int64_t block_count = offset_count - 1;
  TORCH_CHECK(
      input_feature_data[0] == 0 && output_feature_data[0] == 0 &&
          input_channel_data[0] == 0 && output_channel_data[0] == 0 &&
          map_data[0] == 0,
      "channel-transform offsets must begin at zero");
  TORCH_CHECK(
      input_feature_data[block_count] == values.size(1) &&
          map_data[block_count] == channel_maps.numel(),
      "channel-transform final offsets must match packed inputs");
  for (std::int64_t block = 0; block < block_count; ++block) {
    const std::int64_t input_width =
        input_feature_data[block + 1] - input_feature_data[block];
    const std::int64_t output_width =
        output_feature_data[block + 1] - output_feature_data[block];
    const std::int64_t input_channels =
        input_channel_data[block + 1] - input_channel_data[block];
    const std::int64_t output_channels =
        output_channel_data[block + 1] - output_channel_data[block];
    const std::int64_t map_count = map_data[block + 1] - map_data[block];
    TORCH_CHECK(
        input_width > 0 && output_width > 0 && input_channels > 0 &&
            output_channels > 0,
        "channel-transform blocks must have positive dimensions");
    TORCH_CHECK(
        input_width % input_channels == 0 &&
            output_width % output_channels == 0 &&
            input_width / input_channels == output_width / output_channels,
        "channel-transform blocks must preserve complete carrier axes");
    TORCH_CHECK(
        output_channels <= input_channels,
        "channel-transform blocks must be non-expanding");
    TORCH_CHECK(
        map_count == input_channels * output_channels,
        "channel-transform blocks require rectangular channel maps");
  }
}

bool carrier_channel_transform_uses_real_maps(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps) {
  return
      (values.scalar_type() == torch::kComplexFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (values.scalar_type() == torch::kComplexDouble &&
       channel_maps.scalar_type() == torch::kDouble);
}

torch::Tensor carrier_channel_transform_cpu(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    std::int64_t declared_output_width) {
  check_carrier_channel_transform_inputs(
      values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      declared_output_width == output_width,
      "declared channel-transform output width does not match offsets");
  auto output = torch::empty({values.size(0), output_width}, values.options());
  if (carrier_channel_transform_uses_real_maps(values, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_channel_transform_real_maps_forward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              values.data_ptr<c10::complex<float>>()),
          channel_maps.data_ptr<float>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          block_count,
          reinterpret_cast<std::complex<float>*>(
              output.data_ptr<c10::complex<float>>()));
    } else {
      ye3t::runtime::carrier_channel_transform_real_maps_forward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              values.data_ptr<c10::complex<double>>()),
          channel_maps.data_ptr<double>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          block_count,
          reinterpret_cast<std::complex<double>*>(
              output.data_ptr<c10::complex<double>>()));
    }
    return output;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_transform_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_channel_transform_forward<core_t>(
            reinterpret_cast<const core_t*>(values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps.data_ptr<scalar_t>()),
            input_feature_offsets.data_ptr<std::int64_t>(),
            output_feature_offsets.data_ptr<std::int64_t>(),
            input_channel_offsets.data_ptr<std::int64_t>(),
            output_channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            block_count,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_channel_transform_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_transform_inputs(
      values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_data_tensor(output_adjoint, "output_adjoint");
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      output_adjoint.size(0) == values.size(0) &&
          output_adjoint.size(1) == output_width &&
          output_adjoint.scalar_type() == values.scalar_type(),
      "output_adjoint must match the packed channel-transform output");
  auto values_adjoint = torch::empty_like(values);
  auto maps_adjoint = torch::empty_like(channel_maps);
  if (carrier_channel_transform_uses_real_maps(values, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_channel_transform_real_maps_adjoint<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              output_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              values.data_ptr<c10::complex<float>>()),
          channel_maps.data_ptr<float>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          block_count,
          reinterpret_cast<std::complex<float>*>(
              values_adjoint.data_ptr<c10::complex<float>>()),
          maps_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_channel_transform_real_maps_adjoint<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              output_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              values.data_ptr<c10::complex<double>>()),
          channel_maps.data_ptr<double>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          block_count,
          reinterpret_cast<std::complex<double>*>(
              values_adjoint.data_ptr<c10::complex<double>>()),
          maps_adjoint.data_ptr<double>());
    }
    return std::make_tuple(values_adjoint, maps_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_transform_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_channel_transform_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps.data_ptr<scalar_t>()),
            input_feature_offsets.data_ptr<std::int64_t>(),
            output_feature_offsets.data_ptr<std::int64_t>(),
            input_channel_offsets.data_ptr<std::int64_t>(),
            output_channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            block_count,
            reinterpret_cast<core_t*>(
                values_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(maps_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(values_adjoint, maps_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_channel_transform_double_backward_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& values_adjoint_tangent,
    const torch::Tensor& channel_maps_adjoint_tangent,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_transform_inputs(
      values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(values_adjoint_tangent, "values_adjoint_tangent");
  check_data_vector(
      channel_maps_adjoint_tangent,
      "channel_maps_adjoint_tangent");
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      output_adjoint.size(0) == values.size(0) &&
          output_adjoint.size(1) == output_width &&
          output_adjoint.scalar_type() == values.scalar_type(),
      "output_adjoint must match the packed channel-transform output");
  TORCH_CHECK(
      values_adjoint_tangent.sizes() == values.sizes() &&
          values_adjoint_tangent.scalar_type() == values.scalar_type(),
      "values_adjoint_tangent must match values");
  TORCH_CHECK(
      channel_maps_adjoint_tangent.sizes() == channel_maps.sizes() &&
          channel_maps_adjoint_tangent.scalar_type() ==
              channel_maps.scalar_type(),
      "channel_maps_adjoint_tangent must match channel_maps");
  auto output_tangent = torch::empty_like(output_adjoint);
  auto values_second = torch::empty_like(values);
  auto maps_second = torch::empty_like(channel_maps);
  if (carrier_channel_transform_uses_real_maps(values, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_channel_transform_real_maps_double_backward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              output_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              values.data_ptr<c10::complex<float>>()),
          channel_maps.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              values_adjoint_tangent.data_ptr<c10::complex<float>>()),
          channel_maps_adjoint_tangent.data_ptr<float>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          block_count,
          reinterpret_cast<std::complex<float>*>(
              output_tangent.data_ptr<c10::complex<float>>()),
          reinterpret_cast<std::complex<float>*>(
              values_second.data_ptr<c10::complex<float>>()),
          maps_second.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_channel_transform_real_maps_double_backward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              output_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              values.data_ptr<c10::complex<double>>()),
          channel_maps.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              values_adjoint_tangent.data_ptr<c10::complex<double>>()),
          channel_maps_adjoint_tangent.data_ptr<double>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          block_count,
          reinterpret_cast<std::complex<double>*>(
              output_tangent.data_ptr<c10::complex<double>>()),
          reinterpret_cast<std::complex<double>*>(
              values_second.data_ptr<c10::complex<double>>()),
          maps_second.data_ptr<double>());
    }
    return std::make_tuple(output_tangent, values_second, maps_second);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_transform_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_channel_transform_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                values_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps_adjoint_tangent.data_ptr<scalar_t>()),
            input_feature_offsets.data_ptr<std::int64_t>(),
            output_feature_offsets.data_ptr<std::int64_t>(),
            input_channel_offsets.data_ptr<std::int64_t>(),
            output_channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            block_count,
            reinterpret_cast<core_t*>(output_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(values_second.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(maps_second.data_ptr<scalar_t>()));
      });
  return std::make_tuple(output_tangent, values_second, maps_second);
}

void check_carrier_role_channel_map_adjoint_inputs(
    const torch::Tensor& edge_values,
    const torch::Tensor& role_weights,
    const torch::Tensor& atomic_output_adjoint,
    const torch::Tensor& atom_centers,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_transform_inputs(
      edge_values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_data_tensor(role_weights, "role_weights");
  check_int64_vector(atom_centers, "atom_centers");
  TORCH_CHECK(
      atomic_output_adjoint.device().is_cpu() &&
          atomic_output_adjoint.is_contiguous() &&
          atomic_output_adjoint.dim() == 3,
      "atomic_output_adjoint must be a contiguous three-dimensional CPU "
      "tensor");
  TORCH_CHECK(
      atomic_output_adjoint.scalar_type() == edge_values.scalar_type(),
      "atomic_output_adjoint must match edge_values dtype");
  const auto expected_real_type =
      edge_values.scalar_type() == torch::kComplexFloat
      ? torch::kFloat
      : edge_values.scalar_type() == torch::kComplexDouble
      ? torch::kDouble
      : edge_values.scalar_type();
  TORCH_CHECK(
      role_weights.scalar_type() == expected_real_type,
      "role_weights must use the real component dtype of edge_values");
  TORCH_CHECK(
      role_weights.size(0) == edge_values.size(0) &&
          atom_centers.numel() == edge_values.size(0),
      "role_weights and atom_centers must match the edge count");
  TORCH_CHECK(
      atomic_output_adjoint.size(0) > 0 &&
          role_weights.size(1) > 0 &&
          atomic_output_adjoint.size(1) == role_weights.size(1),
      "atomic_output_adjoint and role_weights must share a positive role "
      "axis");
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  const std::int64_t output_width =
      output_feature_offsets.data_ptr<std::int64_t>()[block_count];
  TORCH_CHECK(
      atomic_output_adjoint.size(2) == output_width,
      "atomic_output_adjoint must match the packed transform output width");
  const auto* center_data = atom_centers.data_ptr<std::int64_t>();
  for (std::int64_t edge = 0; edge < atom_centers.numel(); ++edge) {
    TORCH_CHECK(
        center_data[edge] >= 0 &&
            center_data[edge] < atomic_output_adjoint.size(0),
        "atom_centers contains an out-of-range atom index");
  }
}

torch::Tensor carrier_role_channel_map_adjoint_cpu(
    const torch::Tensor& edge_values,
    const torch::Tensor& role_weights,
    const torch::Tensor& atomic_output_adjoint,
    const torch::Tensor& atom_centers,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_role_channel_map_adjoint_inputs(
      edge_values,
      role_weights,
      atomic_output_adjoint,
      atom_centers,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  auto maps_adjoint = torch::empty_like(channel_maps);
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  if (edge_values.scalar_type() == torch::kComplexFloat) {
    if (channel_maps.scalar_type() == torch::kFloat) {
      ye3t::runtime::carrier_role_channel_real_map_adjoint<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              edge_values.data_ptr<c10::complex<float>>()),
          role_weights.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              atomic_output_adjoint.data_ptr<c10::complex<float>>()),
          atom_centers.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          maps_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_role_channel_map_adjoint<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              edge_values.data_ptr<c10::complex<float>>()),
          role_weights.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              atomic_output_adjoint.data_ptr<c10::complex<float>>()),
          atom_centers.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          reinterpret_cast<std::complex<float>*>(
              maps_adjoint.data_ptr<c10::complex<float>>()));
    }
    return maps_adjoint;
  }
  if (edge_values.scalar_type() == torch::kComplexDouble) {
    if (channel_maps.scalar_type() == torch::kDouble) {
      ye3t::runtime::carrier_role_channel_real_map_adjoint<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              edge_values.data_ptr<c10::complex<double>>()),
          role_weights.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              atomic_output_adjoint.data_ptr<c10::complex<double>>()),
          atom_centers.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          maps_adjoint.data_ptr<double>());
    } else {
      ye3t::runtime::carrier_role_channel_map_adjoint<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              edge_values.data_ptr<c10::complex<double>>()),
          role_weights.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              atomic_output_adjoint.data_ptr<c10::complex<double>>()),
          atom_centers.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          reinterpret_cast<std::complex<double>*>(
              maps_adjoint.data_ptr<c10::complex<double>>()));
    }
    return maps_adjoint;
  }
  AT_DISPATCH_FLOATING_TYPES(
      edge_values.scalar_type(),
      "ye3t_carrier_role_channel_map_adjoint_cpu",
      [&] {
        ye3t::runtime::carrier_role_channel_map_adjoint<scalar_t, scalar_t>(
            edge_values.data_ptr<scalar_t>(),
            role_weights.data_ptr<scalar_t>(),
            atomic_output_adjoint.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(),
            input_feature_offsets.data_ptr<std::int64_t>(),
            output_feature_offsets.data_ptr<std::int64_t>(),
            input_channel_offsets.data_ptr<std::int64_t>(),
            output_channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            edge_values.size(0),
            atomic_output_adjoint.size(0),
            role_weights.size(1),
            block_count,
            maps_adjoint.data_ptr<scalar_t>());
      });
  return maps_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_role_channel_map_adjoint_double_backward_cpu(
    const torch::Tensor& edge_values,
    const torch::Tensor& role_weights,
    const torch::Tensor& atomic_output_adjoint,
    const torch::Tensor& atom_centers,
    const torch::Tensor& channel_maps,
    const torch::Tensor& channel_maps_adjoint_tangent,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_role_channel_map_adjoint_inputs(
      edge_values,
      role_weights,
      atomic_output_adjoint,
      atom_centers,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_data_vector(
      channel_maps_adjoint_tangent,
      "channel_maps_adjoint_tangent");
  TORCH_CHECK(
      channel_maps_adjoint_tangent.sizes() == channel_maps.sizes() &&
          channel_maps_adjoint_tangent.scalar_type() ==
              channel_maps.scalar_type(),
      "channel_maps_adjoint_tangent must match channel_maps");
  auto edge_values_second = torch::empty_like(edge_values);
  auto role_weights_second = torch::empty_like(role_weights);
  auto atomic_output_tangent = torch::empty_like(atomic_output_adjoint);
  const std::int64_t block_count = input_feature_offsets.numel() - 1;
  if (edge_values.scalar_type() == torch::kComplexFloat) {
    if (channel_maps.scalar_type() == torch::kFloat) {
      ye3t::runtime::carrier_role_channel_real_map_adjoint_double_backward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              edge_values.data_ptr<c10::complex<float>>()),
          role_weights.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              atomic_output_adjoint.data_ptr<c10::complex<float>>()),
          atom_centers.data_ptr<std::int64_t>(),
          channel_maps_adjoint_tangent.data_ptr<float>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          reinterpret_cast<std::complex<float>*>(
              edge_values_second.data_ptr<c10::complex<float>>()),
          role_weights_second.data_ptr<float>(),
          reinterpret_cast<std::complex<float>*>(
              atomic_output_tangent.data_ptr<c10::complex<float>>()));
    } else {
      ye3t::runtime::carrier_role_channel_map_adjoint_double_backward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              edge_values.data_ptr<c10::complex<float>>()),
          role_weights.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              atomic_output_adjoint.data_ptr<c10::complex<float>>()),
          atom_centers.data_ptr<std::int64_t>(),
          reinterpret_cast<const std::complex<float>*>(
              channel_maps_adjoint_tangent.data_ptr<c10::complex<float>>()),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          reinterpret_cast<std::complex<float>*>(
              edge_values_second.data_ptr<c10::complex<float>>()),
          role_weights_second.data_ptr<float>(),
          reinterpret_cast<std::complex<float>*>(
              atomic_output_tangent.data_ptr<c10::complex<float>>()));
    }
    return std::make_tuple(
        edge_values_second, role_weights_second, atomic_output_tangent);
  }
  if (edge_values.scalar_type() == torch::kComplexDouble) {
    if (channel_maps.scalar_type() == torch::kDouble) {
      ye3t::runtime::carrier_role_channel_real_map_adjoint_double_backward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              edge_values.data_ptr<c10::complex<double>>()),
          role_weights.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              atomic_output_adjoint.data_ptr<c10::complex<double>>()),
          atom_centers.data_ptr<std::int64_t>(),
          channel_maps_adjoint_tangent.data_ptr<double>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          reinterpret_cast<std::complex<double>*>(
              edge_values_second.data_ptr<c10::complex<double>>()),
          role_weights_second.data_ptr<double>(),
          reinterpret_cast<std::complex<double>*>(
              atomic_output_tangent.data_ptr<c10::complex<double>>()));
    } else {
      ye3t::runtime::carrier_role_channel_map_adjoint_double_backward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              edge_values.data_ptr<c10::complex<double>>()),
          role_weights.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              atomic_output_adjoint.data_ptr<c10::complex<double>>()),
          atom_centers.data_ptr<std::int64_t>(),
          reinterpret_cast<const std::complex<double>*>(
              channel_maps_adjoint_tangent.data_ptr<c10::complex<double>>()),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          block_count,
          reinterpret_cast<std::complex<double>*>(
              edge_values_second.data_ptr<c10::complex<double>>()),
          role_weights_second.data_ptr<double>(),
          reinterpret_cast<std::complex<double>*>(
              atomic_output_tangent.data_ptr<c10::complex<double>>()));
    }
    return std::make_tuple(
        edge_values_second, role_weights_second, atomic_output_tangent);
  }
  AT_DISPATCH_FLOATING_TYPES(
      edge_values.scalar_type(),
      "ye3t_carrier_role_channel_map_adjoint_double_backward_cpu",
      [&] {
        ye3t::runtime::carrier_role_channel_map_adjoint_double_backward<
            scalar_t, scalar_t>(
            edge_values.data_ptr<scalar_t>(),
            role_weights.data_ptr<scalar_t>(),
            atomic_output_adjoint.data_ptr<scalar_t>(),
            atom_centers.data_ptr<std::int64_t>(),
            channel_maps_adjoint_tangent.data_ptr<scalar_t>(),
            input_feature_offsets.data_ptr<std::int64_t>(),
            output_feature_offsets.data_ptr<std::int64_t>(),
            input_channel_offsets.data_ptr<std::int64_t>(),
            output_channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            edge_values.size(0),
            atomic_output_adjoint.size(0),
            role_weights.size(1),
            block_count,
            edge_values_second.data_ptr<scalar_t>(),
            role_weights_second.data_ptr<scalar_t>(),
            atomic_output_tangent.data_ptr<scalar_t>());
      });
  return std::make_tuple(
      edge_values_second, role_weights_second, atomic_output_tangent);
}

bool carrier_channel_update_uses_real_controls(
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps) {
  return
      (values.scalar_type() == torch::kComplexFloat &&
       gates.scalar_type() == torch::kFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (values.scalar_type() == torch::kComplexDouble &&
       gates.scalar_type() == torch::kDouble &&
       channel_maps.scalar_type() == torch::kDouble);
}

torch::Tensor carrier_channel_update_cpu(
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_update_inputs(
      values,
      gates,
      channel_maps,
      feature_offsets,
      channel_offsets,
      map_offsets);
  auto output = torch::empty_like(values);
  if (carrier_channel_update_uses_real_controls(
          values, gates, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_channel_update_real_controls_forward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              values.data_ptr<c10::complex<float>>()),
          gates.data_ptr<float>(),
          channel_maps.data_ptr<float>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          feature_offsets.numel() - 1,
          reinterpret_cast<std::complex<float>*>(
              output.data_ptr<c10::complex<float>>()));
    } else {
      ye3t::runtime::carrier_channel_update_real_controls_forward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              values.data_ptr<c10::complex<double>>()),
          gates.data_ptr<double>(),
          channel_maps.data_ptr<double>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          feature_offsets.numel() - 1,
          reinterpret_cast<std::complex<double>*>(
              output.data_ptr<c10::complex<double>>()));
    }
    return output;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_update_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_channel_update_forward<core_t>(
            reinterpret_cast<const core_t*>(
                values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gates.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps.data_ptr<scalar_t>()),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            feature_offsets.numel() - 1,
            reinterpret_cast<core_t*>(
                output.data_ptr<scalar_t>()));
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_channel_update_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_update_inputs(
      values,
      gates,
      channel_maps,
      feature_offsets,
      channel_offsets,
      map_offsets);
  check_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      output_adjoint.sizes() == values.sizes() &&
          output_adjoint.scalar_type() == values.scalar_type(),
      "output_adjoint must match values");
  auto values_adjoint = torch::empty_like(values);
  auto gates_adjoint = torch::empty_like(gates);
  auto channel_maps_adjoint = torch::empty_like(channel_maps);
  if (carrier_channel_update_uses_real_controls(
          values, gates, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_channel_update_real_controls_adjoint<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              output_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              values.data_ptr<c10::complex<float>>()),
          gates.data_ptr<float>(),
          channel_maps.data_ptr<float>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          feature_offsets.numel() - 1,
          reinterpret_cast<std::complex<float>*>(
              values_adjoint.data_ptr<c10::complex<float>>()),
          gates_adjoint.data_ptr<float>(),
          channel_maps_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_channel_update_real_controls_adjoint<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              output_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              values.data_ptr<c10::complex<double>>()),
          gates.data_ptr<double>(),
          channel_maps.data_ptr<double>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          feature_offsets.numel() - 1,
          reinterpret_cast<std::complex<double>*>(
              values_adjoint.data_ptr<c10::complex<double>>()),
          gates_adjoint.data_ptr<double>(),
          channel_maps_adjoint.data_ptr<double>());
    }
    return std::make_tuple(
        values_adjoint,
        gates_adjoint,
        channel_maps_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_update_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_channel_update_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gates.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps.data_ptr<scalar_t>()),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            feature_offsets.numel() - 1,
            reinterpret_cast<core_t*>(
                values_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gates_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                channel_maps_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      values_adjoint,
      gates_adjoint,
      channel_maps_adjoint);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
carrier_channel_update_double_backward_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& values_adjoint_tangent,
    const torch::Tensor& gates_adjoint_tangent,
    const torch::Tensor& channel_maps_adjoint_tangent,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_update_inputs(
      values,
      gates,
      channel_maps,
      feature_offsets,
      channel_offsets,
      map_offsets);
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(values_adjoint_tangent, "values_adjoint_tangent");
  check_data_tensor(gates_adjoint_tangent, "gates_adjoint_tangent");
  check_data_vector(
      channel_maps_adjoint_tangent,
      "channel_maps_adjoint_tangent");
  TORCH_CHECK(
      output_adjoint.sizes() == values.sizes() &&
          output_adjoint.scalar_type() == values.scalar_type() &&
          values_adjoint_tangent.sizes() == values.sizes() &&
          values_adjoint_tangent.scalar_type() == values.scalar_type(),
      "output_adjoint and values_adjoint_tangent must match values");
  TORCH_CHECK(
      gates_adjoint_tangent.sizes() == gates.sizes() &&
          gates_adjoint_tangent.scalar_type() == gates.scalar_type(),
      "gates_adjoint_tangent must match gates");
  TORCH_CHECK(
      channel_maps_adjoint_tangent.sizes() == channel_maps.sizes() &&
          channel_maps_adjoint_tangent.scalar_type() ==
              channel_maps.scalar_type(),
      "channel_maps_adjoint_tangent must match channel_maps");
  auto output_adjoint_tangent = torch::empty_like(output_adjoint);
  auto values_second_adjoint = torch::empty_like(values);
  auto gates_second_adjoint = torch::empty_like(gates);
  auto channel_maps_second_adjoint = torch::empty_like(channel_maps);
  if (carrier_channel_update_uses_real_controls(
          values, gates, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      ye3t::runtime::carrier_channel_update_real_controls_double_backward<
          std::complex<float>, float>(
          reinterpret_cast<const std::complex<float>*>(
              output_adjoint.data_ptr<c10::complex<float>>()),
          reinterpret_cast<const std::complex<float>*>(
              values.data_ptr<c10::complex<float>>()),
          gates.data_ptr<float>(),
          channel_maps.data_ptr<float>(),
          reinterpret_cast<const std::complex<float>*>(
              values_adjoint_tangent.data_ptr<c10::complex<float>>()),
          gates_adjoint_tangent.data_ptr<float>(),
          channel_maps_adjoint_tangent.data_ptr<float>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          feature_offsets.numel() - 1,
          reinterpret_cast<std::complex<float>*>(
              output_adjoint_tangent.data_ptr<c10::complex<float>>()),
          reinterpret_cast<std::complex<float>*>(
              values_second_adjoint.data_ptr<c10::complex<float>>()),
          gates_second_adjoint.data_ptr<float>(),
          channel_maps_second_adjoint.data_ptr<float>());
    } else {
      ye3t::runtime::carrier_channel_update_real_controls_double_backward<
          std::complex<double>, double>(
          reinterpret_cast<const std::complex<double>*>(
              output_adjoint.data_ptr<c10::complex<double>>()),
          reinterpret_cast<const std::complex<double>*>(
              values.data_ptr<c10::complex<double>>()),
          gates.data_ptr<double>(),
          channel_maps.data_ptr<double>(),
          reinterpret_cast<const std::complex<double>*>(
              values_adjoint_tangent.data_ptr<c10::complex<double>>()),
          gates_adjoint_tangent.data_ptr<double>(),
          channel_maps_adjoint_tangent.data_ptr<double>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          feature_offsets.numel() - 1,
          reinterpret_cast<std::complex<double>*>(
              output_adjoint_tangent.data_ptr<c10::complex<double>>()),
          reinterpret_cast<std::complex<double>*>(
              values_second_adjoint.data_ptr<c10::complex<double>>()),
          gates_second_adjoint.data_ptr<double>(),
          channel_maps_second_adjoint.data_ptr<double>());
    }
    return std::make_tuple(
        output_adjoint_tangent,
        values_second_adjoint,
        gates_second_adjoint,
        channel_maps_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_update_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        ye3t::runtime::carrier_channel_update_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                values.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gates.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                values_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                gates_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                channel_maps_adjoint_tangent.data_ptr<scalar_t>()),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            feature_offsets.numel() - 1,
            reinterpret_cast<core_t*>(
                output_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                values_second_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                gates_second_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                channel_maps_second_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      output_adjoint_tangent,
      values_second_adjoint,
      gates_second_adjoint,
      channel_maps_second_adjoint);
}

torch::Tensor source_analysis_cpu(
    const torch::Tensor& source,
    const torch::Tensor& assembly_rows,
    const torch::Tensor& assembly_columns,
    const torch::Tensor& assembly_values,
    const torch::Tensor& synthesis_rows,
    const torch::Tensor& synthesis_columns,
    const torch::Tensor& synthesis_values,
    std::int64_t induced_dimension,
    std::int64_t output_dimension) {
  check_data_tensor(source, "source");
  check_indices(
      assembly_rows,
      assembly_columns,
      assembly_values,
      source,
      "assembly");
  check_indices(
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      source,
      "synthesis");
  TORCH_CHECK(induced_dimension > 0, "induced_dimension must be positive");
  TORCH_CHECK(output_dimension > 0, "output_dimension must be positive");
  auto output = torch::empty(
      {source.size(0), output_dimension},
      source.options());
  const auto* assembly_row_ptr = assembly_rows.data_ptr<std::int64_t>();
  const auto* assembly_column_ptr = assembly_columns.data_ptr<std::int64_t>();
  const auto* synthesis_row_ptr = synthesis_rows.data_ptr<std::int64_t>();
  const auto* synthesis_column_ptr = synthesis_columns.data_ptr<std::int64_t>();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      source.scalar_type(),
      "ye3t_source_analysis_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::source_analysis_forward<core_t>(
            reinterpret_cast<const core_t*>(source.data_ptr<scalar_t>()),
            source.size(0),
            source.size(1),
            assembly_row_ptr,
            assembly_column_ptr,
            reinterpret_cast<const core_t*>(
                assembly_values.data_ptr<scalar_t>()),
            assembly_rows.numel(),
            induced_dimension,
            synthesis_row_ptr,
            synthesis_column_ptr,
            reinterpret_cast<const core_t*>(
                synthesis_values.data_ptr<scalar_t>()),
            synthesis_rows.numel(),
            output_dimension,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

torch::Tensor source_analysis_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& assembly_rows,
    const torch::Tensor& assembly_columns,
    const torch::Tensor& assembly_values,
    const torch::Tensor& synthesis_rows,
    const torch::Tensor& synthesis_columns,
    const torch::Tensor& synthesis_values,
    std::int64_t source_dimension,
    std::int64_t induced_dimension) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_indices(
      assembly_rows,
      assembly_columns,
      assembly_values,
      output_adjoint,
      "assembly");
  check_indices(
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      output_adjoint,
      "synthesis");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(induced_dimension > 0, "induced_dimension must be positive");
  auto source_adjoint = torch::empty(
      {output_adjoint.size(0), source_dimension},
      output_adjoint.options());
  const auto* assembly_row_ptr = assembly_rows.data_ptr<std::int64_t>();
  const auto* assembly_column_ptr = assembly_columns.data_ptr<std::int64_t>();
  const auto* synthesis_row_ptr = synthesis_rows.data_ptr<std::int64_t>();
  const auto* synthesis_column_ptr = synthesis_columns.data_ptr<std::int64_t>();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      output_adjoint.scalar_type(),
      "ye3t_source_analysis_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::source_analysis_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            output_adjoint.size(0),
            output_adjoint.size(1),
            assembly_row_ptr,
            assembly_column_ptr,
            reinterpret_cast<const core_t*>(
                assembly_values.data_ptr<scalar_t>()),
            assembly_rows.numel(),
            source_dimension,
            induced_dimension,
            synthesis_row_ptr,
            synthesis_column_ptr,
            reinterpret_cast<const core_t*>(
                synthesis_values.data_ptr<scalar_t>()),
            synthesis_rows.numel(),
            reinterpret_cast<core_t*>(
                source_adjoint.data_ptr<scalar_t>()));
      });
  return source_adjoint;
}

torch::Tensor source_analysis_linear_cpu(
    const torch::Tensor& source,
    const torch::Tensor& assembly_rows,
    const torch::Tensor& assembly_columns,
    const torch::Tensor& assembly_values,
    const torch::Tensor& synthesis_rows,
    const torch::Tensor& synthesis_columns,
    const torch::Tensor& synthesis_values,
    std::int64_t induced_dimension,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  check_data_tensor(source, "source");
  check_indices(
      assembly_rows,
      assembly_columns,
      assembly_values,
      source,
      "assembly");
  check_indices(
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      source,
      "synthesis");
  TORCH_CHECK(induced_dimension > 0, "induced_dimension must be positive");
  TORCH_CHECK(
      weight.device().is_cpu() && weight.is_contiguous() &&
          weight.dim() == 1 &&
          weight.scalar_type() == source.scalar_type(),
      "weight must be a contiguous CPU vector with source dtype");
  TORCH_CHECK(
      bias.device().is_cpu() && bias.is_contiguous() &&
          bias.dim() == 0 &&
          bias.scalar_type() == source.scalar_type(),
      "bias must be a contiguous CPU scalar with source dtype");
  auto output = torch::empty({source.size(0)}, source.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      source.scalar_type(),
      "ye3t_source_analysis_linear_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::source_analysis_linear_forward<core_t>(
            reinterpret_cast<const core_t*>(
                source.data_ptr<scalar_t>()),
            source.size(0),
            source.size(1),
            assembly_rows.data_ptr<std::int64_t>(),
            assembly_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                assembly_values.data_ptr<scalar_t>()),
            assembly_rows.numel(),
            induced_dimension,
            synthesis_rows.data_ptr<std::int64_t>(),
            synthesis_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                synthesis_values.data_ptr<scalar_t>()),
            synthesis_rows.numel(),
            weight.numel(),
            reinterpret_cast<const core_t*>(
                weight.data_ptr<scalar_t>()),
            *reinterpret_cast<const core_t*>(
                bias.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
source_analysis_linear_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& source,
    const torch::Tensor& assembly_rows,
    const torch::Tensor& assembly_columns,
    const torch::Tensor& assembly_values,
    const torch::Tensor& synthesis_rows,
    const torch::Tensor& synthesis_columns,
    const torch::Tensor& synthesis_values,
    std::int64_t induced_dimension,
    const torch::Tensor& weight) {
  TORCH_CHECK(
      output_adjoint.device().is_cpu() &&
          output_adjoint.is_contiguous() &&
          output_adjoint.dim() == 1 &&
          output_adjoint.scalar_type() == source.scalar_type() &&
          output_adjoint.numel() == source.size(0),
      "output_adjoint must be a contiguous batch vector with source dtype");
  check_data_tensor(source, "source");
  check_indices(
      assembly_rows,
      assembly_columns,
      assembly_values,
      source,
      "assembly");
  check_indices(
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      source,
      "synthesis");
  TORCH_CHECK(induced_dimension > 0, "induced_dimension must be positive");
  TORCH_CHECK(
      weight.device().is_cpu() && weight.is_contiguous() &&
          weight.dim() == 1 &&
          weight.scalar_type() == source.scalar_type(),
      "weight must be a contiguous CPU vector with source dtype");
  auto source_adjoint = torch::empty_like(source);
  auto weight_adjoint = torch::empty_like(weight);
  auto bias_adjoint = torch::empty({}, weight.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      source.scalar_type(),
      "ye3t_source_analysis_linear_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::source_analysis_linear_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                source.data_ptr<scalar_t>()),
            source.size(0),
            source.size(1),
            assembly_rows.data_ptr<std::int64_t>(),
            assembly_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                assembly_values.data_ptr<scalar_t>()),
            assembly_rows.numel(),
            induced_dimension,
            synthesis_rows.data_ptr<std::int64_t>(),
            synthesis_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                synthesis_values.data_ptr<scalar_t>()),
            synthesis_rows.numel(),
            weight.numel(),
            reinterpret_cast<const core_t*>(
                weight.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                source_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                weight_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                bias_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      source_adjoint,
      weight_adjoint,
      bias_adjoint);
}

torch::Tensor compact_pair_product_cpu(
    const torch::Tensor& left,
    const torch::Tensor& right,
    bool antisymmetric) {
  check_data_tensor(left, "left");
  check_data_tensor(right, "right");
  TORCH_CHECK(
      left.sizes() == right.sizes(),
      "left and right must have identical shapes");
  TORCH_CHECK(
      left.scalar_type() == right.scalar_type(),
      "left and right must have identical dtypes");
  const std::int64_t dimension = left.size(1);
  const std::int64_t output_dimension = antisymmetric
      ? dimension * (dimension - 1) / 2
      : dimension * (dimension + 1) / 2;
  auto output = torch::empty({left.size(0), output_dimension}, left.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      left.scalar_type(),
      "ye3t_compact_pair_product_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::compact_pair_product_forward<core_t>(
            reinterpret_cast<const core_t*>(left.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(right.data_ptr<scalar_t>()),
            left.size(0),
            dimension,
            antisymmetric,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor> compact_pair_product_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& left,
    const torch::Tensor& right,
    bool antisymmetric) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(left, "left");
  check_data_tensor(right, "right");
  TORCH_CHECK(
      left.sizes() == right.sizes(),
      "left and right must have identical shapes");
  TORCH_CHECK(
      output_adjoint.size(0) == left.size(0),
      "output_adjoint batch dimension must match inputs");
  TORCH_CHECK(
      output_adjoint.scalar_type() == left.scalar_type() &&
          left.scalar_type() == right.scalar_type(),
      "adjoint and inputs must have identical dtypes");
  const std::int64_t dimension = left.size(1);
  const std::int64_t expected_output_dimension = antisymmetric
      ? dimension * (dimension - 1) / 2
      : dimension * (dimension + 1) / 2;
  TORCH_CHECK(
      output_adjoint.size(1) == expected_output_dimension,
      "output_adjoint width does not match compact product");
  auto left_adjoint = torch::empty_like(left);
  auto right_adjoint = torch::empty_like(right);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      left.scalar_type(),
      "ye3t_compact_pair_product_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::compact_pair_product_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(left.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(right.data_ptr<scalar_t>()),
            left.size(0),
            dimension,
            antisymmetric,
            reinterpret_cast<core_t*>(
                left_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                right_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(left_adjoint, right_adjoint);
}

torch::Tensor compact_exterior_power_cpu(
    const torch::Tensor& factors) {
  check_factor_tensor(factors, "factors");
  const std::int64_t order = factors.size(1);
  const std::int64_t dimension = factors.size(2);
  TORCH_CHECK(order >= 1, "compact exterior power order must be positive");
  TORCH_CHECK(
      order <= 8,
      "native compact exterior power currently supports order <= 8");
  const std::int64_t output_dimension =
      binomial_coefficient(dimension, order);
  auto output = torch::empty(
      {factors.size(0), output_dimension},
      factors.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      factors.scalar_type(),
      "ye3t_compact_exterior_power_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::compact_exterior_power_forward<core_t>(
            reinterpret_cast<const core_t*>(
                factors.data_ptr<scalar_t>()),
            factors.size(0),
            order,
            dimension,
            reinterpret_cast<core_t*>(
                output.data_ptr<scalar_t>()));
      });
  return output;
}

torch::Tensor compact_exterior_power_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& factors) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_factor_tensor(factors, "factors");
  TORCH_CHECK(
      output_adjoint.scalar_type() == factors.scalar_type(),
      "output_adjoint and factors must have identical dtypes");
  TORCH_CHECK(
      output_adjoint.size(0) == factors.size(0),
      "output_adjoint batch dimension must match factors");
  const std::int64_t order = factors.size(1);
  const std::int64_t dimension = factors.size(2);
  TORCH_CHECK(order >= 1, "compact exterior power order must be positive");
  TORCH_CHECK(
      order <= 8,
      "native compact exterior power currently supports order <= 8");
  TORCH_CHECK(
      output_adjoint.size(1) ==
          binomial_coefficient(dimension, order),
      "output_adjoint width does not match compact exterior power");
  auto factors_adjoint = torch::empty_like(factors);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      factors.scalar_type(),
      "ye3t_compact_exterior_power_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::compact_exterior_power_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                factors.data_ptr<scalar_t>()),
            factors.size(0),
            order,
            dimension,
            reinterpret_cast<core_t*>(
                factors_adjoint.data_ptr<scalar_t>()));
      });
  return factors_adjoint;
}

torch::Tensor symmetric_power_monomial_cpu(
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& output_indices,
    const torch::Tensor& monomial_values) {
  check_data_tensor(input, "input");
  check_int64_matrix(monomial_counts, "monomial_counts");
  check_int64_vector(output_offsets, "output_offsets");
  check_int64_vector(output_indices, "output_indices");
  TORCH_CHECK(
      monomial_values.device().is_cpu(),
      "monomial_values must be on CPU");
  TORCH_CHECK(
      monomial_values.is_contiguous() && monomial_values.dim() == 1,
      "monomial_values must be a contiguous vector");
  TORCH_CHECK(
      monomial_values.scalar_type() == input.scalar_type(),
      "monomial_values and input must have identical dtypes");
  const std::int64_t term_count = monomial_counts.size(0);
  TORCH_CHECK(
      monomial_counts.size(1) == input.size(1),
      "monomial count width must match the input dimension");
  TORCH_CHECK(
      output_indices.numel() == term_count &&
          monomial_values.numel() == term_count,
      "symmetric-power term arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() >= 1,
      "output_offsets must contain at least one entry");
  const std::int64_t output_dimension =
      output_offsets.numel() - 1;
  const auto* offsets = output_offsets.data_ptr<std::int64_t>();
  const auto* indices = output_indices.data_ptr<std::int64_t>();
  TORCH_CHECK(offsets[0] == 0, "output_offsets must start at zero");
  TORCH_CHECK(
      offsets[output_dimension] == term_count,
      "output_offsets must end at the term count");
  for (std::int64_t output = 0; output < output_dimension; ++output) {
    TORCH_CHECK(
        offsets[output] <= offsets[output + 1],
        "output_offsets must be nondecreasing");
  }
  for (std::int64_t term = 0; term < term_count; ++term) {
    TORCH_CHECK(
        indices[term] >= 0 && indices[term] < output_dimension,
        "output_indices contains an out-of-range entry");
  }
  auto output = torch::empty(
      {input.size(0), output_dimension},
      input.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_monomial_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        const auto* input_values = reinterpret_cast<const core_t*>(
            input.data_ptr<scalar_t>());
        auto* output_values = reinterpret_cast<core_t*>(
            output.data_ptr<scalar_t>());
        at::parallel_for(
            0,
            input.size(0),
            1,
            [&](std::int64_t begin, std::int64_t end) {
              ye3t::runtime::symmetric_power_monomial_forward<core_t>(
                  input_values + begin * input.size(1),
                  end - begin,
                  input.size(1),
                  monomial_counts.data_ptr<std::int64_t>(),
                  output_offsets.data_ptr<std::int64_t>(),
                  reinterpret_cast<const core_t*>(
                      monomial_values.data_ptr<scalar_t>()),
                  term_count,
                  output_dimension,
                  output_values + begin * output_dimension);
            });
      });
  return output;
}

torch::Tensor symmetric_power_monomial_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& output_indices,
    const torch::Tensor& monomial_values) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(input, "input");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type(),
      "output_adjoint and input must have identical dtypes");
  TORCH_CHECK(
      output_adjoint.size(0) == input.size(0),
      "output_adjoint batch dimension must match input");
  check_int64_matrix(monomial_counts, "monomial_counts");
  check_int64_vector(output_offsets, "output_offsets");
  check_int64_vector(output_indices, "output_indices");
  TORCH_CHECK(
      monomial_values.device().is_cpu(),
      "monomial_values must be on CPU");
  TORCH_CHECK(
      monomial_values.is_contiguous() && monomial_values.dim() == 1,
      "monomial_values must be a contiguous vector");
  TORCH_CHECK(
      monomial_values.scalar_type() == input.scalar_type(),
      "monomial_values and input must have identical dtypes");
  const std::int64_t term_count = monomial_counts.size(0);
  TORCH_CHECK(
      monomial_counts.size(1) == input.size(1),
      "monomial count width must match the input dimension");
  TORCH_CHECK(
      output_indices.numel() == term_count &&
          monomial_values.numel() == term_count,
      "symmetric-power term arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() == output_adjoint.size(1) + 1,
      "output_offsets length does not match output_adjoint");
  TORCH_CHECK(
      output_offsets.data_ptr<std::int64_t>()[0] == 0 &&
          output_offsets.data_ptr<std::int64_t>()[
              output_adjoint.size(1)] == term_count,
      "output_offsets must span the complete term table");
  const auto* indices = output_indices.data_ptr<std::int64_t>();
  for (std::int64_t term = 0; term < term_count; ++term) {
    TORCH_CHECK(
        indices[term] >= 0 &&
            indices[term] < output_adjoint.size(1),
        "output_indices contains an out-of-range entry");
  }
  auto input_adjoint = torch::empty_like(input);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_monomial_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        const auto* output_gradient =
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>());
        const auto* input_values = reinterpret_cast<const core_t*>(
            input.data_ptr<scalar_t>());
        auto* input_gradient = reinterpret_cast<core_t*>(
            input_adjoint.data_ptr<scalar_t>());
        at::parallel_for(
            0,
            input.size(0),
            1,
            [&](std::int64_t begin, std::int64_t end) {
              ye3t::runtime::symmetric_power_monomial_adjoint<core_t>(
                  output_gradient +
                      begin * output_adjoint.size(1),
                  input_values + begin * input.size(1),
                  end - begin,
                  input.size(1),
                  monomial_counts.data_ptr<std::int64_t>(),
                  output_indices.data_ptr<std::int64_t>(),
                  reinterpret_cast<const core_t*>(
                      monomial_values.data_ptr<scalar_t>()),
                  term_count,
                  output_adjoint.size(1),
                  input_gradient + begin * input.size(1));
            });
      });
  return input_adjoint;
}

torch::Tensor symmetric_power_shared_monomial_cpu(
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_data_tensor(input, "input");
  check_int64_matrix(monomial_counts, "monomial_counts");
  check_int64_vector(output_offsets, "output_offsets");
  check_int64_vector(coefficient_terms, "coefficient_terms");
  check_int64_vector(coefficient_outputs, "coefficient_outputs");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1,
      "coefficient_values must be a contiguous CPU vector");
  TORCH_CHECK(
      coefficient_values.scalar_type() == input.scalar_type(),
      "coefficient_values and input must have identical dtypes");
  const std::int64_t monomial_count = monomial_counts.size(0);
  const std::int64_t coefficient_count =
      coefficient_terms.numel();
  TORCH_CHECK(
      monomial_counts.size(1) == input.size(1),
      "monomial count width must match the input dimension");
  TORCH_CHECK(
      coefficient_outputs.numel() == coefficient_count &&
          coefficient_values.numel() == coefficient_count,
      "shared symmetric-power coefficient arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() >= 1,
      "output_offsets must contain at least one entry");
  const std::int64_t output_dimension =
      output_offsets.numel() - 1;
  const auto* offsets = output_offsets.data_ptr<std::int64_t>();
  const auto* terms = coefficient_terms.data_ptr<std::int64_t>();
  const auto* outputs =
      coefficient_outputs.data_ptr<std::int64_t>();
  TORCH_CHECK(offsets[0] == 0, "output_offsets must start at zero");
  TORCH_CHECK(
      offsets[output_dimension] == coefficient_count,
      "output_offsets must end at the coefficient count");
  for (std::int64_t output = 0; output < output_dimension; ++output) {
    TORCH_CHECK(
        offsets[output] <= offsets[output + 1],
        "output_offsets must be nondecreasing");
    for (std::int64_t coefficient = offsets[output];
         coefficient < offsets[output + 1];
         ++coefficient) {
      TORCH_CHECK(
          outputs[coefficient] == output,
          "coefficient_outputs must agree with output_offsets");
    }
  }
  for (std::int64_t coefficient = 0;
       coefficient < coefficient_count;
       ++coefficient) {
    TORCH_CHECK(
        terms[coefficient] >= 0 &&
            terms[coefficient] < monomial_count,
        "coefficient_terms contains an out-of-range entry");
  }
  auto output = torch::empty(
      {input.size(0), output_dimension},
      input.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        const auto* input_values =
            reinterpret_cast<const core_t*>(
                input.data_ptr<scalar_t>());
        auto* output_values = reinterpret_cast<core_t*>(
            output.data_ptr<scalar_t>());
        at::parallel_for(
            0,
            input.size(0),
            1,
            [&](std::int64_t begin, std::int64_t end) {
              ye3t::runtime::
                  symmetric_power_shared_monomial_forward<core_t>(
                      input_values + begin * input.size(1),
                      end - begin,
                      input.size(1),
                      monomial_counts.data_ptr<std::int64_t>(),
                      monomial_count,
                      output_offsets.data_ptr<std::int64_t>(),
                      coefficient_terms.data_ptr<std::int64_t>(),
                      reinterpret_cast<const core_t*>(
                          coefficient_values.data_ptr<scalar_t>()),
                      coefficient_count,
                      output_dimension,
                      output_values + begin * output_dimension);
            });
      });
  return output;
}

torch::Tensor symmetric_power_shared_monomial_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(input, "input");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type() &&
          output_adjoint.size(0) == input.size(0),
      "output_adjoint dtype and batch must match input");
  check_int64_matrix(monomial_counts, "monomial_counts");
  check_int64_vector(output_offsets, "output_offsets");
  check_int64_vector(coefficient_terms, "coefficient_terms");
  check_int64_vector(coefficient_outputs, "coefficient_outputs");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == input.scalar_type(),
      "coefficient_values must be a matching contiguous CPU vector");
  const std::int64_t monomial_count = monomial_counts.size(0);
  const std::int64_t coefficient_count =
      coefficient_terms.numel();
  TORCH_CHECK(
      monomial_counts.size(1) == input.size(1),
      "monomial count width must match the input dimension");
  TORCH_CHECK(
      coefficient_outputs.numel() == coefficient_count &&
          coefficient_values.numel() == coefficient_count,
      "shared symmetric-power coefficient arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() == output_adjoint.size(1) + 1,
      "output_offsets length must match output_adjoint");
  const auto* offsets = output_offsets.data_ptr<std::int64_t>();
  const auto* terms = coefficient_terms.data_ptr<std::int64_t>();
  const auto* outputs =
      coefficient_outputs.data_ptr<std::int64_t>();
  TORCH_CHECK(
      offsets[0] == 0 &&
          offsets[output_adjoint.size(1)] == coefficient_count,
      "output_offsets must span the coefficient table");
  for (std::int64_t output = 0;
       output < output_adjoint.size(1);
       ++output) {
    TORCH_CHECK(
        offsets[output] <= offsets[output + 1],
        "output_offsets must be nondecreasing");
    for (std::int64_t coefficient = offsets[output];
         coefficient < offsets[output + 1];
         ++coefficient) {
      TORCH_CHECK(
          outputs[coefficient] == output,
          "coefficient_outputs must agree with output_offsets");
    }
  }
  for (std::int64_t coefficient = 0;
       coefficient < coefficient_count;
       ++coefficient) {
    TORCH_CHECK(
        terms[coefficient] >= 0 &&
            terms[coefficient] < monomial_count,
        "coefficient_terms contains an out-of-range entry");
  }
  auto input_adjoint = torch::empty_like(input);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        const auto* output_gradient =
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>());
        const auto* input_values =
            reinterpret_cast<const core_t*>(
                input.data_ptr<scalar_t>());
        auto* input_gradient = reinterpret_cast<core_t*>(
            input_adjoint.data_ptr<scalar_t>());
        at::parallel_for(
            0,
            input.size(0),
            1,
            [&](std::int64_t begin, std::int64_t end) {
              ye3t::runtime::
                  symmetric_power_shared_monomial_adjoint<core_t>(
                      output_gradient +
                          begin * output_adjoint.size(1),
                      input_values + begin * input.size(1),
                      end - begin,
                      input.size(1),
                      monomial_counts.data_ptr<std::int64_t>(),
                      monomial_count,
                      output_offsets.data_ptr<std::int64_t>(),
                      coefficient_terms.data_ptr<std::int64_t>(),
                      reinterpret_cast<const core_t*>(
                          coefficient_values.data_ptr<scalar_t>()),
                      coefficient_count,
                      output_adjoint.size(1),
                      input_gradient + begin * input.size(1));
            });
      });
  return input_adjoint;
}

torch::Tensor symmetric_power_shared_monomial_batched_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_batched_adjoint_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(input, "input");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type() &&
          output_adjoint.size(1) == input.size(0),
      "output_adjoint dtype and batch must match input");
  check_int64_matrix(monomial_counts, "monomial_counts");
  check_int64_vector(output_offsets, "output_offsets");
  check_int64_vector(coefficient_terms, "coefficient_terms");
  check_int64_vector(coefficient_outputs, "coefficient_outputs");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == input.scalar_type(),
      "coefficient_values must be a matching contiguous CPU vector");
  const std::int64_t monomial_count = monomial_counts.size(0);
  const std::int64_t coefficient_count =
      coefficient_terms.numel();
  const std::int64_t output_dimension =
      output_adjoint.size(2);
  TORCH_CHECK(
      monomial_counts.size(1) == input.size(1),
      "monomial count width must match the input dimension");
  TORCH_CHECK(
      coefficient_outputs.numel() == coefficient_count &&
          coefficient_values.numel() == coefficient_count,
      "shared symmetric-power coefficient arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() == output_dimension + 1,
      "output_offsets length must match output_adjoint");
  const auto* offsets = output_offsets.data_ptr<std::int64_t>();
  const auto* terms = coefficient_terms.data_ptr<std::int64_t>();
  const auto* outputs =
      coefficient_outputs.data_ptr<std::int64_t>();
  TORCH_CHECK(
      offsets[0] == 0 &&
          offsets[output_dimension] == coefficient_count,
      "output_offsets must span the coefficient table");
  for (std::int64_t output = 0;
       output < output_dimension;
       ++output) {
    TORCH_CHECK(
        offsets[output] <= offsets[output + 1],
        "output_offsets must be nondecreasing");
    for (std::int64_t coefficient = offsets[output];
         coefficient < offsets[output + 1];
         ++coefficient) {
      TORCH_CHECK(
          outputs[coefficient] == output,
          "coefficient_outputs must agree with output_offsets");
    }
  }
  for (std::int64_t coefficient = 0;
       coefficient < coefficient_count;
       ++coefficient) {
    TORCH_CHECK(
        terms[coefficient] >= 0 &&
            terms[coefficient] < monomial_count,
        "coefficient_terms contains an out-of-range entry");
  }
  auto input_adjoint = torch::empty(
      {
          output_adjoint.size(0),
          input.size(0),
          input.size(1),
      },
      input.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_batched_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::
            symmetric_power_shared_monomial_batched_adjoint<core_t>(
                reinterpret_cast<const core_t*>(
                    output_adjoint.data_ptr<scalar_t>()),
                reinterpret_cast<const core_t*>(
                    input.data_ptr<scalar_t>()),
                output_adjoint.size(0),
                input.size(0),
                input.size(1),
                monomial_counts.data_ptr<std::int64_t>(),
                monomial_count,
                output_offsets.data_ptr<std::int64_t>(),
                coefficient_terms.data_ptr<std::int64_t>(),
                reinterpret_cast<const core_t*>(
                    coefficient_values.data_ptr<scalar_t>()),
                coefficient_count,
                output_dimension,
                reinterpret_cast<core_t*>(
                    input_adjoint.data_ptr<scalar_t>()));
      });
  return input_adjoint;
}

torch::Tensor factorized_angular_cpu(
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension) {
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");
  const std::int64_t node_count = node_offsets.numel();
  TORCH_CHECK(node_count > 0, "factorized plan must contain nodes");
  TORCH_CHECK(
      node_dimensions.numel() == node_count &&
          node_leaf_offsets.numel() == node_count &&
          node_left.numel() == node_count &&
          node_right.numel() == node_count,
      "factorized node arrays must have equal length");
  TORCH_CHECK(
      node_coefficient_offsets.numel() == node_count + 1,
      "node_coefficient_offsets must have node_count + 1 entries");
  TORCH_CHECK(
      coefficient_rows.numel() == coefficient_columns.numel() &&
          coefficient_rows.numel() == coefficient_values.numel(),
      "factorized coefficient arrays must have equal length");
  TORCH_CHECK(root_nodes.numel() > 0, "factorized plan requires roots");
  TORCH_CHECK(
      projection_values.numel() > 0,
      "factorized plan requires projection weights");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      packed_slots.size(1) % source_dimension == 0,
      "packed_slots width must be divisible by source_dimension");
  TORCH_CHECK(
      projection_values.numel() % source_dimension == 0,
      "projection weights must be divisible by source_dimension");
  const std::int64_t input_dimension =
      packed_slots.size(1) / source_dimension;
  const std::int64_t projection_dimension =
      projection_values.numel() / source_dimension;
  check_factorized_workspace_dimension(
      node_offsets,
      node_dimensions,
      workspace_dimension);
  TORCH_CHECK(output_dimension > 0, "output_dimension must be positive");
  const auto root_metadata = uniform_factorized_root_metadata(
      root_nodes,
      node_dimensions,
      projection_dimension,
      output_dimension);
  auto output = torch::empty(
      {packed_slots.size(0), output_dimension},
      packed_slots.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_forward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_count,
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            root_metadata.projection_starts.data(),
            root_metadata.projection_dimensions.data(),
            root_metadata.output_offsets.data(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            root_metadata.total_projection_dimension,
            root_metadata.output_dimension,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

torch::Tensor factorized_angular_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint batch dimension must match packed_slots");
  TORCH_CHECK(
      output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "output_adjoint and packed_slots must have identical dtypes");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");
  const std::int64_t node_count = node_offsets.numel();
  TORCH_CHECK(node_count > 0, "factorized plan must contain nodes");
  TORCH_CHECK(
      node_dimensions.numel() == node_count &&
          node_leaf_offsets.numel() == node_count &&
          node_left.numel() == node_count &&
          node_right.numel() == node_count,
      "factorized node arrays must have equal length");
  TORCH_CHECK(
      node_coefficient_offsets.numel() == node_count + 1,
      "node_coefficient_offsets must have node_count + 1 entries");
  TORCH_CHECK(
      coefficient_rows.numel() == coefficient_columns.numel() &&
          coefficient_rows.numel() == coefficient_values.numel(),
      "factorized coefficient arrays must have equal length");
  TORCH_CHECK(root_nodes.numel() > 0, "factorized plan requires roots");
  TORCH_CHECK(
      projection_values.numel() > 0,
      "factorized plan requires projection weights");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      packed_slots.size(1) % source_dimension == 0,
      "packed_slots width must be divisible by source_dimension");
  TORCH_CHECK(
      projection_values.numel() % source_dimension == 0,
      "projection weights must be divisible by source_dimension");
  const std::int64_t input_dimension =
      packed_slots.size(1) / source_dimension;
  const std::int64_t projection_dimension =
      projection_values.numel() / source_dimension;
  check_factorized_workspace_dimension(
      node_offsets,
      node_dimensions,
      workspace_dimension);
  const auto root_metadata = uniform_factorized_root_metadata(
      root_nodes,
      node_dimensions,
      projection_dimension,
      output_adjoint.size(1));
  auto packed_adjoint = torch::empty_like(packed_slots);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_offsets.numel(),
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            root_metadata.projection_starts.data(),
            root_metadata.projection_dimensions.data(),
            root_metadata.output_offsets.data(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            root_metadata.total_projection_dimension,
            root_metadata.output_dimension,
            reinterpret_cast<core_t*>(
                packed_adjoint.data_ptr<scalar_t>()));
      });
  return packed_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_double_backward_cpu(
    const torch::Tensor& packed_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension) {
  check_data_tensor(packed_adjoint_tangent, "packed_adjoint_tangent");
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  TORCH_CHECK(
      packed_adjoint_tangent.sizes() == packed_slots.sizes(),
      "packed_adjoint_tangent shape must match packed_slots");
  TORCH_CHECK(
      packed_adjoint_tangent.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "factorized double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint batch dimension must match packed_slots");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");
  const std::int64_t node_count = node_offsets.numel();
  TORCH_CHECK(node_count > 0, "factorized plan must contain nodes");
  TORCH_CHECK(
      node_dimensions.numel() == node_count &&
          node_leaf_offsets.numel() == node_count &&
          node_left.numel() == node_count &&
          node_right.numel() == node_count,
      "factorized node arrays must have equal length");
  TORCH_CHECK(
      node_coefficient_offsets.numel() == node_count + 1,
      "node_coefficient_offsets must have node_count + 1 entries");
  TORCH_CHECK(
      coefficient_rows.numel() == coefficient_columns.numel() &&
          coefficient_rows.numel() == coefficient_values.numel(),
      "factorized coefficient arrays must have equal length");
  TORCH_CHECK(root_nodes.numel() > 0, "factorized plan requires roots");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      packed_slots.size(1) % source_dimension == 0,
      "packed_slots width must be divisible by source_dimension");
  TORCH_CHECK(
      projection_values.numel() > 0 &&
          projection_values.numel() % source_dimension == 0,
      "projection weights must be nonempty and divisible by source_dimension");
  const std::int64_t input_dimension =
      packed_slots.size(1) / source_dimension;
  const std::int64_t projection_dimension =
      projection_values.numel() / source_dimension;
  check_factorized_workspace_dimension(
      node_offsets,
      node_dimensions,
      workspace_dimension);
  const auto root_metadata = uniform_factorized_root_metadata(
      root_nodes,
      node_dimensions,
      projection_dimension,
      output_adjoint.size(1));
  auto output_tangent = torch::empty_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_slots);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_count,
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            root_metadata.projection_starts.data(),
            root_metadata.projection_dimensions.data(),
            root_metadata.output_offsets.data(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            root_metadata.total_projection_dimension,
            root_metadata.output_dimension,
            reinterpret_cast<core_t*>(
                output_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                packed_tangent.data_ptr<scalar_t>()));
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

torch::Tensor factorized_angular_heterogeneous_cpu(
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension) {
  const auto dimensions = check_heterogeneous_factorized_cpu_inputs(
      packed_slots,
      node_offsets,
      node_dimensions,
      node_leaf_offsets,
      node_left,
      node_right,
      node_coefficient_offsets,
      coefficient_rows,
      coefficient_columns,
      coefficient_values,
      root_nodes,
      root_projection_starts,
      root_projection_dimensions,
      root_output_offsets,
      projection_values,
      source_dimension,
      workspace_dimension,
      output_dimension);
  auto output = torch::empty(
      {packed_slots.size(0), dimensions.output_dimension},
      packed_slots.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_forward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            dimensions.input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_offsets.numel(),
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            root_projection_starts.data_ptr<std::int64_t>(),
            root_projection_dimensions.data_ptr<std::int64_t>(),
            root_output_offsets.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            dimensions.total_projection_dimension,
            dimensions.output_dimension,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

torch::Tensor factorized_angular_heterogeneous_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension) {
  check_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint batch dimension must match packed_slots");
  TORCH_CHECK(
      output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "output_adjoint and packed_slots must have identical dtypes");
  const auto dimensions = check_heterogeneous_factorized_cpu_inputs(
      packed_slots,
      node_offsets,
      node_dimensions,
      node_leaf_offsets,
      node_left,
      node_right,
      node_coefficient_offsets,
      coefficient_rows,
      coefficient_columns,
      coefficient_values,
      root_nodes,
      root_projection_starts,
      root_projection_dimensions,
      root_output_offsets,
      projection_values,
      source_dimension,
      workspace_dimension,
      output_adjoint.size(1));
  auto packed_adjoint = torch::empty_like(packed_slots);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            dimensions.input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_offsets.numel(),
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            root_projection_starts.data_ptr<std::int64_t>(),
            root_projection_dimensions.data_ptr<std::int64_t>(),
            root_output_offsets.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            dimensions.total_projection_dimension,
            dimensions.output_dimension,
            reinterpret_cast<core_t*>(
                packed_adjoint.data_ptr<scalar_t>()));
      });
  return packed_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_heterogeneous_double_backward_cpu(
    const torch::Tensor& packed_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension) {
  check_data_tensor(packed_adjoint_tangent, "packed_adjoint_tangent");
  check_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      packed_adjoint_tangent.sizes() == packed_slots.sizes(),
      "packed_adjoint_tangent shape must match packed_slots");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint batch dimension must match packed_slots");
  TORCH_CHECK(
      packed_adjoint_tangent.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "factorized double-backward tensors must have identical dtypes");
  const auto dimensions = check_heterogeneous_factorized_cpu_inputs(
      packed_slots,
      node_offsets,
      node_dimensions,
      node_leaf_offsets,
      node_left,
      node_right,
      node_coefficient_offsets,
      coefficient_rows,
      coefficient_columns,
      coefficient_values,
      root_nodes,
      root_projection_starts,
      root_projection_dimensions,
      root_output_offsets,
      projection_values,
      source_dimension,
      workspace_dimension,
      output_adjoint.size(1));
  auto output_tangent = torch::empty_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_slots);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            dimensions.input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_offsets.numel(),
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            root_projection_starts.data_ptr<std::int64_t>(),
            root_projection_dimensions.data_ptr<std::int64_t>(),
            root_output_offsets.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            dimensions.total_projection_dimension,
            dimensions.output_dimension,
            reinterpret_cast<core_t*>(
                output_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                packed_tangent.data_ptr<scalar_t>()));
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

torch::Tensor factorized_angular_segmented_cpu(
    const torch::Tensor& packed_slots,
    const torch::Tensor& segment_source_offsets,
    const torch::Tensor& segment_input_offsets,
    const torch::Tensor& segment_input_dimensions,
    const torch::Tensor& segment_workspace_offsets,
    const torch::Tensor& segment_workspace_dimensions,
    const torch::Tensor& segment_node_offsets,
    const torch::Tensor& segment_root_offsets,
    const torch::Tensor& segment_projection_offsets,
    const torch::Tensor& segment_projection_dimensions,
    const torch::Tensor& segment_output_offsets,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t total_source_count,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension) {
  const auto dimensions = check_segmented_factorized_cpu_inputs(
      packed_slots,
      segment_source_offsets,
      segment_input_offsets,
      segment_input_dimensions,
      segment_workspace_offsets,
      segment_workspace_dimensions,
      segment_node_offsets,
      segment_root_offsets,
      segment_projection_offsets,
      segment_projection_dimensions,
      segment_output_offsets,
      node_offsets,
      node_dimensions,
      node_leaf_offsets,
      node_left,
      node_right,
      node_coefficient_offsets,
      coefficient_rows,
      coefficient_columns,
      coefficient_values,
      root_nodes,
      root_projection_starts,
      root_projection_dimensions,
      root_output_offsets,
      projection_values,
      total_source_count,
      workspace_dimension,
      output_dimension);
  auto output = torch::empty(
      {packed_slots.size(0), output_dimension},
      packed_slots.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_segmented_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        const auto plan = segmented_factorized_plan_view<scalar_t>(
            segment_source_offsets,
            segment_input_offsets,
            segment_input_dimensions,
            segment_workspace_offsets,
            segment_workspace_dimensions,
            segment_node_offsets,
            segment_root_offsets,
            segment_projection_offsets,
            segment_projection_dimensions,
            segment_output_offsets,
            node_offsets,
            node_dimensions,
            node_leaf_offsets,
            node_left,
            node_right,
            node_coefficient_offsets,
            coefficient_rows,
            coefficient_columns,
            coefficient_values,
            root_nodes,
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
            dimensions.segment_count,
            packed_slots.size(1),
            workspace_dimension,
            output_dimension);
        ye3t::runtime::factorized_angular_segmented_forward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            plan,
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

torch::Tensor factorized_angular_segmented_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& segment_source_offsets,
    const torch::Tensor& segment_input_offsets,
    const torch::Tensor& segment_input_dimensions,
    const torch::Tensor& segment_workspace_offsets,
    const torch::Tensor& segment_workspace_dimensions,
    const torch::Tensor& segment_node_offsets,
    const torch::Tensor& segment_root_offsets,
    const torch::Tensor& segment_projection_offsets,
    const torch::Tensor& segment_projection_dimensions,
    const torch::Tensor& segment_output_offsets,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t total_source_count,
    std::int64_t workspace_dimension) {
  check_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0) &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "segmented output adjoint must match packed batch and dtype");
  const std::int64_t output_dimension = output_adjoint.size(1);
  const auto dimensions = check_segmented_factorized_cpu_inputs(
      packed_slots,
      segment_source_offsets,
      segment_input_offsets,
      segment_input_dimensions,
      segment_workspace_offsets,
      segment_workspace_dimensions,
      segment_node_offsets,
      segment_root_offsets,
      segment_projection_offsets,
      segment_projection_dimensions,
      segment_output_offsets,
      node_offsets,
      node_dimensions,
      node_leaf_offsets,
      node_left,
      node_right,
      node_coefficient_offsets,
      coefficient_rows,
      coefficient_columns,
      coefficient_values,
      root_nodes,
      root_projection_starts,
      root_projection_dimensions,
      root_output_offsets,
      projection_values,
      total_source_count,
      workspace_dimension,
      output_dimension);
  auto packed_adjoint = torch::empty_like(packed_slots);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_segmented_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        const auto plan = segmented_factorized_plan_view<scalar_t>(
            segment_source_offsets,
            segment_input_offsets,
            segment_input_dimensions,
            segment_workspace_offsets,
            segment_workspace_dimensions,
            segment_node_offsets,
            segment_root_offsets,
            segment_projection_offsets,
            segment_projection_dimensions,
            segment_output_offsets,
            node_offsets,
            node_dimensions,
            node_leaf_offsets,
            node_left,
            node_right,
            node_coefficient_offsets,
            coefficient_rows,
            coefficient_columns,
            coefficient_values,
            root_nodes,
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
            dimensions.segment_count,
            packed_slots.size(1),
            workspace_dimension,
            output_dimension);
        ye3t::runtime::factorized_angular_segmented_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            plan,
            reinterpret_cast<core_t*>(
                packed_adjoint.data_ptr<scalar_t>()));
      });
  return packed_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_segmented_double_backward_cpu(
    const torch::Tensor& packed_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& segment_source_offsets,
    const torch::Tensor& segment_input_offsets,
    const torch::Tensor& segment_input_dimensions,
    const torch::Tensor& segment_workspace_offsets,
    const torch::Tensor& segment_workspace_dimensions,
    const torch::Tensor& segment_node_offsets,
    const torch::Tensor& segment_root_offsets,
    const torch::Tensor& segment_projection_offsets,
    const torch::Tensor& segment_projection_dimensions,
    const torch::Tensor& segment_output_offsets,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& root_projection_starts,
    const torch::Tensor& root_projection_dimensions,
    const torch::Tensor& root_output_offsets,
    const torch::Tensor& projection_values,
    std::int64_t total_source_count,
    std::int64_t workspace_dimension) {
  check_data_tensor(output_adjoint, "output_adjoint");
  check_data_tensor(packed_adjoint_tangent, "packed_adjoint_tangent");
  TORCH_CHECK(
      packed_adjoint_tangent.sizes() == packed_slots.sizes() &&
          packed_adjoint_tangent.scalar_type() == packed_slots.scalar_type(),
      "segmented packed tangent must match packed input");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0) &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "segmented output adjoint must match packed batch and dtype");
  const std::int64_t output_dimension = output_adjoint.size(1);
  const auto dimensions = check_segmented_factorized_cpu_inputs(
      packed_slots,
      segment_source_offsets,
      segment_input_offsets,
      segment_input_dimensions,
      segment_workspace_offsets,
      segment_workspace_dimensions,
      segment_node_offsets,
      segment_root_offsets,
      segment_projection_offsets,
      segment_projection_dimensions,
      segment_output_offsets,
      node_offsets,
      node_dimensions,
      node_leaf_offsets,
      node_left,
      node_right,
      node_coefficient_offsets,
      coefficient_rows,
      coefficient_columns,
      coefficient_values,
      root_nodes,
      root_projection_starts,
      root_projection_dimensions,
      root_output_offsets,
      projection_values,
      total_source_count,
      workspace_dimension,
      output_dimension);
  auto output_tangent = torch::empty_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_slots);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_segmented_double_backward_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        const auto plan = segmented_factorized_plan_view<scalar_t>(
            segment_source_offsets,
            segment_input_offsets,
            segment_input_dimensions,
            segment_workspace_offsets,
            segment_workspace_dimensions,
            segment_node_offsets,
            segment_root_offsets,
            segment_projection_offsets,
            segment_projection_dimensions,
            segment_output_offsets,
            node_offsets,
            node_dimensions,
            node_leaf_offsets,
            node_left,
            node_right,
            node_coefficient_offsets,
            coefficient_rows,
            coefficient_columns,
            coefficient_values,
            root_nodes,
            root_projection_starts,
            root_projection_dimensions,
            root_output_offsets,
            projection_values,
            dimensions.segment_count,
            packed_slots.size(1),
            workspace_dimension,
            output_dimension);
        ye3t::runtime::factorized_angular_segmented_double_backward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_adjoint_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            plan,
            reinterpret_cast<core_t*>(
                output_tangent.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                packed_tangent.data_ptr<scalar_t>()));
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

torch::Tensor factorized_angular_linear_cpu(
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension,
    const torch::Tensor& weight,
    const torch::Tensor& bias) {
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(node_coefficient_offsets, "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      weight.device().is_cpu() && weight.is_contiguous() &&
          weight.dim() == 1 &&
          weight.scalar_type() == packed_slots.scalar_type(),
      "weight must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      bias.device().is_cpu() && bias.is_contiguous() &&
          bias.dim() == 0 &&
          bias.scalar_type() == packed_slots.scalar_type(),
      "bias must be a contiguous CPU scalar with data dtype");
  const std::int64_t node_count = node_offsets.numel();
  TORCH_CHECK(node_count > 0, "factorized plan must contain nodes");
  TORCH_CHECK(
      node_dimensions.numel() == node_count &&
          node_leaf_offsets.numel() == node_count &&
          node_left.numel() == node_count &&
          node_right.numel() == node_count,
      "factorized node arrays must have equal length");
  TORCH_CHECK(
      node_coefficient_offsets.numel() == node_count + 1,
      "node_coefficient_offsets must have node_count + 1 entries");
  TORCH_CHECK(
      coefficient_rows.numel() == coefficient_columns.numel() &&
          coefficient_rows.numel() == coefficient_values.numel(),
      "factorized coefficient arrays must have equal length");
  TORCH_CHECK(root_nodes.numel() > 0, "factorized plan requires roots");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      packed_slots.size(1) % source_dimension == 0,
      "packed_slots width must be divisible by source_dimension");
  TORCH_CHECK(
      projection_values.numel() % source_dimension == 0,
      "projection weights must be divisible by source_dimension");
  const std::int64_t input_dimension =
      packed_slots.size(1) / source_dimension;
  const std::int64_t projection_dimension =
      projection_values.numel() / source_dimension;
  check_factorized_workspace_dimension(
      node_offsets,
      node_dimensions,
      workspace_dimension);
  const auto* root_ptr = root_nodes.data_ptr<std::int64_t>();
  const auto* dimension_ptr =
      node_dimensions.data_ptr<std::int64_t>();
  const std::int64_t output_dimension =
      root_nodes.numel() * projection_dimension *
      dimension_ptr[root_ptr[0]];
  TORCH_CHECK(
      weight.numel() == output_dimension,
      "weight length must match factorized output dimension");
  auto output = torch::empty(
      {packed_slots.size(0)},
      packed_slots.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_linear_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_linear_forward<core_t>(
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_count,
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            projection_dimension,
            reinterpret_cast<const core_t*>(
                weight.data_ptr<scalar_t>()),
            *reinterpret_cast<const core_t*>(
                bias.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(output.data_ptr<scalar_t>()));
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
factorized_angular_linear_adjoint_cpu(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& packed_slots,
    const torch::Tensor& node_offsets,
    const torch::Tensor& node_dimensions,
    const torch::Tensor& node_leaf_offsets,
    const torch::Tensor& node_left,
    const torch::Tensor& node_right,
    const torch::Tensor& node_coefficient_offsets,
    const torch::Tensor& coefficient_rows,
    const torch::Tensor& coefficient_columns,
    const torch::Tensor& coefficient_values,
    const torch::Tensor& root_nodes,
    const torch::Tensor& projection_values,
    std::int64_t source_dimension,
    std::int64_t workspace_dimension,
    const torch::Tensor& weight) {
  TORCH_CHECK(
      output_adjoint.device().is_cpu() &&
          output_adjoint.is_contiguous() &&
          output_adjoint.dim() == 1 &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "output_adjoint must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      output_adjoint.numel() == packed_slots.size(0),
      "output_adjoint length must match batch size");
  check_data_tensor(packed_slots, "packed_slots");
  check_int64_vector(node_offsets, "node_offsets");
  check_int64_vector(node_dimensions, "node_dimensions");
  check_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_int64_vector(node_left, "node_left");
  check_int64_vector(node_right, "node_right");
  check_int64_vector(node_coefficient_offsets, "node_coefficient_offsets");
  check_int64_vector(coefficient_rows, "coefficient_rows");
  check_int64_vector(coefficient_columns, "coefficient_columns");
  check_int64_vector(root_nodes, "root_nodes");
  TORCH_CHECK(
      coefficient_values.device().is_cpu() &&
          coefficient_values.is_contiguous() &&
          coefficient_values.dim() == 1 &&
          coefficient_values.scalar_type() == packed_slots.scalar_type(),
      "coefficient_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      projection_values.device().is_cpu() &&
          projection_values.is_contiguous() &&
          projection_values.dim() == 1 &&
          projection_values.scalar_type() == packed_slots.scalar_type(),
      "projection_values must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(
      weight.device().is_cpu() && weight.is_contiguous() &&
          weight.dim() == 1 &&
          weight.scalar_type() == packed_slots.scalar_type(),
      "weight must be a contiguous CPU vector with data dtype");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(
      packed_slots.size(1) % source_dimension == 0,
      "packed_slots width must be divisible by source_dimension");
  TORCH_CHECK(
      projection_values.numel() % source_dimension == 0,
      "projection weights must be divisible by source_dimension");
  const std::int64_t input_dimension =
      packed_slots.size(1) / source_dimension;
  const std::int64_t projection_dimension =
      projection_values.numel() / source_dimension;
  check_factorized_workspace_dimension(
      node_offsets,
      node_dimensions,
      workspace_dimension);
  TORCH_CHECK(
      node_offsets.numel() > 0 &&
          node_dimensions.numel() == node_offsets.numel() &&
          node_leaf_offsets.numel() == node_offsets.numel() &&
          node_left.numel() == node_offsets.numel() &&
          node_right.numel() == node_offsets.numel() &&
          node_coefficient_offsets.numel() == node_offsets.numel() + 1,
      "factorized node arrays have inconsistent lengths");
  TORCH_CHECK(root_nodes.numel() > 0, "factorized plan requires roots");
  const std::int64_t expected_weight_dimension =
      root_nodes.numel() * projection_dimension *
      node_dimensions.data_ptr<std::int64_t>()[
          root_nodes.data_ptr<std::int64_t>()[0]];
  TORCH_CHECK(
      weight.numel() == expected_weight_dimension,
      "weight length must match factorized output dimension");
  auto packed_adjoint = torch::empty_like(packed_slots);
  auto weight_adjoint = torch::empty_like(weight);
  auto bias_adjoint = torch::empty({}, weight.options());
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_linear_adjoint_cpu",
      [&] {
        using core_t = typename CoreScalar<scalar_t>::type;
        static_assert(sizeof(core_t) == sizeof(scalar_t));
        ye3t::runtime::factorized_angular_linear_adjoint<core_t>(
            reinterpret_cast<const core_t*>(
                output_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<const core_t*>(
                packed_slots.data_ptr<scalar_t>()),
            packed_slots.size(0),
            input_dimension,
            source_dimension,
            node_offsets.data_ptr<std::int64_t>(),
            node_dimensions.data_ptr<std::int64_t>(),
            node_leaf_offsets.data_ptr<std::int64_t>(),
            node_left.data_ptr<std::int64_t>(),
            node_right.data_ptr<std::int64_t>(),
            node_coefficient_offsets.data_ptr<std::int64_t>(),
            node_offsets.numel(),
            coefficient_rows.data_ptr<std::int64_t>(),
            coefficient_columns.data_ptr<std::int64_t>(),
            reinterpret_cast<const core_t*>(
                coefficient_values.data_ptr<scalar_t>()),
            root_nodes.data_ptr<std::int64_t>(),
            root_nodes.numel(),
            reinterpret_cast<const core_t*>(
                projection_values.data_ptr<scalar_t>()),
            projection_dimension,
            reinterpret_cast<const core_t*>(
                weight.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                packed_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                weight_adjoint.data_ptr<scalar_t>()),
            reinterpret_cast<core_t*>(
                bias_adjoint.data_ptr<scalar_t>()));
      });
  return std::make_tuple(
      packed_adjoint,
      weight_adjoint,
      bias_adjoint);
}

}  // namespace

namespace {

// Streaming degree/order recurrence for the nonnegative-m complex harmonics,
// packed as l*(l+1)/2 + m, with Cartesian derivatives taken with respect to
// the raw displacement. This is the same kernel the native pair style links.
std::tuple<torch::Tensor, torch::Tensor>
complex_spherical_harmonics_recurrence_cpu(
    const torch::Tensor& displacements,
    std::int64_t maximum_angular_momentum,
    double normalization_scale) {
  TORCH_CHECK(displacements.device().is_cpu(), "displacements must be on CPU");
  TORCH_CHECK(
      displacements.dim() == 2 && displacements.size(1) == 3,
      "displacements must have shape [edge_count, 3]");
  TORCH_CHECK(
      displacements.scalar_type() == torch::kFloat64 ||
          displacements.scalar_type() == torch::kFloat32,
      "displacements must be float32 or float64");
  TORCH_CHECK(
      maximum_angular_momentum >= 0,
      "maximum_angular_momentum must be non-negative");
  const auto radii =
      torch::linalg_vector_norm(displacements, 2, {1}, false).contiguous();
  TORCH_CHECK(
      displacements.size(0) == 0 || (radii > 0).all().item<bool>(),
      "displacements must have positive length");
  const auto units = (displacements / radii.unsqueeze(1)).contiguous();
  const std::int64_t edge_count = displacements.size(0);
  const std::int64_t width =
      (maximum_angular_momentum + 1) * (maximum_angular_momentum + 2) / 2;
  const auto complex_options = displacements.options().dtype(
      displacements.scalar_type() == torch::kFloat64 ? torch::kComplexDouble
                                                     : torch::kComplexFloat);
  auto values = torch::empty({edge_count, width}, complex_options);
  auto derivatives = torch::empty({edge_count, width, 3}, complex_options);
  const std::int64_t plan_size =
      ye3t::runtime::complex_spherical_harmonics_recurrence_plan_size(
          maximum_angular_momentum);
  auto plan = torch::empty({plan_size}, displacements.options());
  AT_DISPATCH_FLOATING_TYPES(
      displacements.scalar_type(),
      "ye3t_complex_spherical_harmonics_recurrence_cpu",
      [&] {
        ye3t::runtime::build_complex_spherical_harmonics_recurrence_plan<
            scalar_t>(
            maximum_angular_momentum,
            plan.data_ptr<scalar_t>(),
            plan_size,
            static_cast<scalar_t>(normalization_scale));
        ye3t::runtime::
            complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated<
                scalar_t>(
                units.data_ptr<scalar_t>(),
                radii.data_ptr<scalar_t>(),
                edge_count,
                maximum_angular_momentum,
                plan.data_ptr<scalar_t>(),
                plan_size,
                reinterpret_cast<std::complex<scalar_t>*>(
                    values.data_ptr<c10::complex<scalar_t>>()),
                reinterpret_cast<std::complex<scalar_t>*>(
                    derivatives.data_ptr<c10::complex<scalar_t>>()));
      });
  return std::make_tuple(values, derivatives);
}

}  // namespace

TORCH_LIBRARY(ye3t_runtime, library) {
  library.def(
      "complex_spherical_harmonics_recurrence(Tensor displacements, "
      "int maximum_angular_momentum, float normalization_scale) "
      "-> (Tensor, Tensor)");
  library.def(
      "cheb_exp_cos_radial_with_derivative(Tensor radii, Tensor cutoffs, "
      "Tensor lambdas, int radial_index) -> (Tensor, Tensor)");
  library.def(
      "cheb_exp_cos_radial_table_with_derivative(Tensor radii, "
      "Tensor cutoffs, Tensor lambdas, int maximum_radial_index) "
      "-> (Tensor, Tensor)");
  library.def(
      "cheb_exp_cos_radial_table_double_backward("
      "Tensor values_adjoint, Tensor radial_derivatives, Tensor radii, "
      "Tensor cutoffs, Tensor lambdas, Tensor grad_grad_radii, "
      "int maximum_radial_index) -> (Tensor, Tensor)");
  library.def(
      "spherical_harmonics_with_derivative(Tensor edge_vectors, "
      "int angular_momentum, bool real_output, float epsilon) "
      "-> (Tensor, Tensor)");
  library.def(
      "spherical_harmonics_table_with_derivative(Tensor edge_vectors, "
      "int maximum_angular_momentum, bool real_output, float epsilon) "
      "-> (Tensor, Tensor)");
  library.def(
      "spherical_harmonics_table_double_backward(Tensor edge_vectors, "
      "Tensor value_adjoint, Tensor edge_direction, "
      "int maximum_angular_momentum, float epsilon) "
      "-> (Tensor, Tensor)");
  library.def(
      "plain_site_basis_product_with_derivative("
      "Tensor radial_values, Tensor radial_derivatives, "
      "Tensor angular_values, Tensor angular_derivatives, "
      "Tensor prefactors, Tensor prefactor_derivatives_center, "
      "Tensor prefactor_derivatives_neighbor, Tensor radial_directions, "
      "Tensor term_groups, Tensor term_channels, int channel_count) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  library.def(
      "scheduled_radial_angular_channels_with_derivative("
      "Tensor radial_values, Tensor radial_derivatives, "
      "Tensor angular_values, Tensor angular_derivatives, "
      "Tensor radial_directions, Tensor edge_types, "
      "Tensor channel_radial_indices, Tensor channel_angular_indices, "
      "Tensor channel_types, Tensor channel_scales) "
      "-> (Tensor, Tensor)");
  library.def(
      "plain_site_basis_product_adjoint("
      "Tensor radial_values, Tensor radial_derivatives, "
      "Tensor angular_values, Tensor angular_derivatives, "
      "Tensor prefactors, Tensor prefactor_derivatives_center, "
      "Tensor prefactor_derivatives_neighbor, Tensor radial_directions, "
      "Tensor edge_weights, Tensor edge_weight_derivatives, "
      "Tensor term_groups, Tensor term_channels, Tensor edge_adjoint) "
      "-> (Tensor, Tensor, Tensor)");
  library.def(
      "density_accumulate(Tensor edge_values, Tensor centers, "
      "int atom_count) -> Tensor");
  library.def(
      "density_accumulate_adjoint(Tensor atomic_adjoint, Tensor centers) "
      "-> Tensor");
  library.def(
      "edge_outer_accumulate(Tensor left, Tensor right, Tensor centers, "
      "int atom_count) -> Tensor");
  library.def(
      "edge_outer_accumulate_adjoint(Tensor atomic_adjoint, Tensor left, "
      "Tensor right, Tensor centers) -> (Tensor, Tensor)");
  library.def(
      "edge_outer_accumulate_double_backward("
      "Tensor atomic_adjoint, Tensor left, Tensor right, "
      "Tensor left_adjoint_tangent, Tensor right_adjoint_tangent, "
      "Tensor centers) -> (Tensor, Tensor, Tensor)");
  library.def(
      "softmax_gaussian_role_density("
      "Tensor distances, Tensor cutoffs, Tensor filter_centers, "
      "float filter_width, Tensor edge_values, Tensor atom_centers, "
      "int atom_count) -> Tensor");
  library.def(
      "softmax_gaussian_role_density_adjoint("
      "Tensor atomic_adjoint, Tensor distances, Tensor cutoffs, "
      "Tensor filter_centers, float filter_width, Tensor edge_values, "
      "Tensor atom_centers) -> (Tensor, Tensor)");
  library.def(
      "softmax_gaussian_role_density_double_backward("
      "Tensor atomic_adjoint, Tensor distances, Tensor cutoffs, "
      "Tensor filter_centers, float filter_width, Tensor edge_values, "
      "Tensor distance_adjoint_tangent, Tensor edge_adjoint_tangent, "
      "Tensor atom_centers) -> (Tensor, Tensor, Tensor)");
  library.def(
      "scheduled_softmax_gaussian_role_density("
      "Tensor radial_values, Tensor angular_values, Tensor distances, "
      "Tensor cutoffs, Tensor filter_centers, float filter_width, "
      "Tensor soft_weights, Tensor edge_types, "
      "Tensor channel_radial_indices, Tensor channel_angular_indices, "
      "Tensor channel_types, Tensor channel_scales, Tensor atom_centers, "
      "int atom_count) -> Tensor");
  library.def(
      "scheduled_softmax_gaussian_role_density_adjoint("
      "Tensor atomic_adjoint, Tensor radial_values, Tensor angular_values, "
      "Tensor distances, Tensor cutoffs, Tensor filter_centers, "
      "float filter_width, Tensor soft_weights, Tensor edge_types, "
      "Tensor channel_radial_indices, Tensor channel_angular_indices, "
      "Tensor channel_types, Tensor channel_scales, Tensor atom_centers) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  library.def(
      "scheduled_softmax_gaussian_role_density_double_backward("
      "Tensor atomic_adjoint, Tensor radial_values, Tensor angular_values, "
      "Tensor distances, Tensor cutoffs, Tensor filter_centers, "
      "float filter_width, Tensor soft_weights, "
      "Tensor radial_adjoint_tangent, Tensor angular_adjoint_tangent, "
      "Tensor distance_adjoint_tangent, Tensor soft_adjoint_tangent, "
      "Tensor edge_types, Tensor channel_radial_indices, "
      "Tensor channel_angular_indices, Tensor channel_types, "
      "Tensor channel_scales, Tensor atom_centers) "
      "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
  library.def(
      "carrier_gated_scatter("
      "Tensor node_values, Tensor edge_gates, Tensor edge_sources, "
      "Tensor edge_targets, Tensor feature_channels, int target_count) "
      "-> Tensor");
  library.def(
      "carrier_gated_scatter_adjoint("
      "Tensor target_adjoint, Tensor node_values, Tensor edge_gates, "
      "Tensor edge_sources, Tensor edge_targets, Tensor feature_channels) "
      "-> (Tensor, Tensor)");
  library.def(
      "carrier_gated_scatter_double_backward("
      "Tensor target_adjoint, Tensor node_values, Tensor edge_gates, "
      "Tensor node_adjoint_tangent, Tensor gate_adjoint_tangent, "
      "Tensor edge_sources, Tensor edge_targets, Tensor feature_channels) "
      "-> (Tensor, Tensor, Tensor)");
  library.def(
      "carrier_residual_gated_scatter("
      "Tensor node_values, Tensor edge_gates, Tensor edge_sources, "
      "Tensor edge_targets, Tensor feature_channels) -> Tensor");
  library.def(
      "carrier_residual_gated_scatter_adjoint("
      "Tensor target_adjoint, Tensor node_values, Tensor edge_gates, "
      "Tensor edge_sources, Tensor edge_targets, Tensor feature_channels) "
      "-> (Tensor, Tensor)");
  library.def(
      "carrier_residual_gated_scatter_double_backward("
      "Tensor target_adjoint, Tensor node_values, Tensor edge_gates, "
      "Tensor node_adjoint_tangent, Tensor gate_adjoint_tangent, "
      "Tensor edge_sources, Tensor edge_targets, Tensor feature_channels) "
      "-> (Tensor, Tensor, Tensor)");
  library.def(
      "carrier_segmented_residual_gated_scatter("
      "Tensor node_values, Tensor edge_gates, Tensor edge_sources, "
      "Tensor edge_targets, Tensor target_offsets, Tensor source_offsets, "
      "Tensor source_edges, Tensor feature_channels, "
      "Tensor channel_offsets, Tensor channel_features) -> Tensor");
  library.def(
      "carrier_segmented_residual_gated_scatter_adjoint("
      "Tensor target_adjoint, Tensor node_values, Tensor edge_gates, "
      "Tensor edge_sources, Tensor edge_targets, Tensor target_offsets, "
      "Tensor source_offsets, Tensor source_edges, "
      "Tensor feature_channels, Tensor channel_offsets, "
      "Tensor channel_features) -> (Tensor, Tensor)");
  library.def(
      "carrier_segmented_residual_gated_scatter_double_backward("
      "Tensor target_adjoint, Tensor node_values, Tensor edge_gates, "
      "Tensor node_adjoint_tangent, Tensor gate_adjoint_tangent, "
      "Tensor edge_sources, Tensor edge_targets, Tensor target_offsets, "
      "Tensor source_offsets, Tensor source_edges, "
      "Tensor feature_channels, Tensor channel_offsets, "
      "Tensor channel_features) -> (Tensor, Tensor, Tensor)");
  library.def(
      "source_arena_gather("
      "Tensor producer, Tensor gather_indices, Tensor reverse_offsets, "
      "Tensor reverse_output_indices, Tensor center_types, "
      "Tensor atom_types) -> Tensor");
  library.def(
      "source_arena_gather_adjoint("
      "Tensor output_adjoint, Tensor gather_indices, "
      "Tensor reverse_offsets, Tensor reverse_output_indices, "
      "Tensor center_types, Tensor atom_types) -> Tensor");
  library.def(
      "source_arena_gather_double_backward("
      "Tensor producer_adjoint_tangent, Tensor gather_indices, "
      "Tensor reverse_offsets, Tensor reverse_output_indices, "
      "Tensor center_types, Tensor atom_types) -> Tensor");
  library.def(
      "source_arena_channel_transform("
      "Tensor producer, Tensor channel_maps, Tensor gather_indices, "
      "Tensor reverse_offsets, Tensor reverse_output_indices, "
      "Tensor center_types, Tensor atom_types, "
      "Tensor input_feature_offsets, Tensor output_feature_offsets, "
      "Tensor input_channel_offsets, Tensor output_channel_offsets, "
      "Tensor map_offsets, int output_width) -> Tensor");
  library.def(
      "source_arena_channel_transform_adjoint("
      "Tensor output_adjoint, Tensor producer, Tensor channel_maps, "
      "Tensor gather_indices, Tensor reverse_offsets, "
      "Tensor reverse_output_indices, Tensor center_types, "
      "Tensor atom_types, Tensor input_feature_offsets, "
      "Tensor output_feature_offsets, Tensor input_channel_offsets, "
      "Tensor output_channel_offsets, Tensor map_offsets) "
      "-> (Tensor, Tensor)");
  library.def(
      "source_arena_channel_transform_double_backward("
      "Tensor output_adjoint, Tensor producer, Tensor channel_maps, "
      "Tensor producer_adjoint_tangent, "
      "Tensor channel_maps_adjoint_tangent, Tensor gather_indices, "
      "Tensor reverse_offsets, Tensor reverse_output_indices, "
      "Tensor center_types, Tensor atom_types, "
      "Tensor input_feature_offsets, Tensor output_feature_offsets, "
      "Tensor input_channel_offsets, Tensor output_channel_offsets, "
      "Tensor map_offsets) -> (Tensor, Tensor, Tensor)");
  library.def(
      "carrier_channel_update("
      "Tensor values, Tensor gates, Tensor channel_maps, "
      "Tensor feature_offsets, Tensor channel_offsets, Tensor map_offsets) "
      "-> Tensor");
  library.def(
      "carrier_channel_update_adjoint("
      "Tensor output_adjoint, Tensor values, Tensor gates, "
      "Tensor channel_maps, Tensor feature_offsets, "
      "Tensor channel_offsets, Tensor map_offsets) "
      "-> (Tensor, Tensor, Tensor)");
  library.def(
      "carrier_channel_update_double_backward("
      "Tensor output_adjoint, Tensor values, Tensor gates, "
      "Tensor channel_maps, Tensor values_adjoint_tangent, "
      "Tensor gates_adjoint_tangent, "
      "Tensor channel_maps_adjoint_tangent, Tensor feature_offsets, "
      "Tensor channel_offsets, Tensor map_offsets) "
      "-> (Tensor, Tensor, Tensor, Tensor)");
  library.def(
      "carrier_channel_transform("
      "Tensor values, Tensor channel_maps, Tensor input_feature_offsets, "
      "Tensor output_feature_offsets, Tensor input_channel_offsets, "
      "Tensor output_channel_offsets, Tensor map_offsets, int output_width) "
      "-> Tensor");
  library.def(
      "carrier_channel_transform_adjoint("
      "Tensor output_adjoint, Tensor values, Tensor channel_maps, "
      "Tensor input_feature_offsets, Tensor output_feature_offsets, "
      "Tensor input_channel_offsets, Tensor output_channel_offsets, "
      "Tensor map_offsets) -> (Tensor, Tensor)");
  library.def(
      "carrier_channel_transform_double_backward("
      "Tensor output_adjoint, Tensor values, Tensor channel_maps, "
      "Tensor values_adjoint_tangent, "
      "Tensor channel_maps_adjoint_tangent, "
      "Tensor input_feature_offsets, Tensor output_feature_offsets, "
      "Tensor input_channel_offsets, Tensor output_channel_offsets, "
      "Tensor map_offsets) -> (Tensor, Tensor, Tensor)");
  library.def(
      "carrier_role_channel_map_adjoint("
      "Tensor edge_values, Tensor role_weights, "
      "Tensor atomic_output_adjoint, Tensor atom_centers, "
      "Tensor channel_maps, Tensor input_feature_offsets, "
      "Tensor output_feature_offsets, Tensor input_channel_offsets, "
      "Tensor output_channel_offsets, Tensor map_offsets) -> Tensor");
  library.def(
      "carrier_role_channel_map_adjoint_double_backward("
      "Tensor edge_values, Tensor role_weights, "
      "Tensor atomic_output_adjoint, Tensor atom_centers, "
      "Tensor channel_maps, Tensor channel_maps_adjoint_tangent, "
      "Tensor input_feature_offsets, Tensor output_feature_offsets, "
      "Tensor input_channel_offsets, Tensor output_channel_offsets, "
      "Tensor map_offsets) -> (Tensor, Tensor, Tensor)");
  library.def(
      "source_analysis(Tensor source, Tensor assembly_rows, "
      "Tensor assembly_columns, Tensor assembly_values, "
      "Tensor synthesis_rows, Tensor synthesis_columns, "
      "Tensor synthesis_values, int induced_dimension, "
      "int output_dimension) -> Tensor");
  library.def(
      "source_analysis_adjoint(Tensor output_adjoint, Tensor assembly_rows, "
      "Tensor assembly_columns, Tensor assembly_values, "
      "Tensor synthesis_rows, Tensor synthesis_columns, "
      "Tensor synthesis_values, int source_dimension, "
      "int induced_dimension) -> Tensor");
  library.def(
      "source_analysis_linear(Tensor source, Tensor assembly_rows, "
      "Tensor assembly_columns, Tensor assembly_values, "
      "Tensor synthesis_rows, Tensor synthesis_columns, "
      "Tensor synthesis_values, int induced_dimension, Tensor weight, "
      "Tensor bias) -> Tensor");
  library.def(
      "source_analysis_linear_adjoint(Tensor output_adjoint, Tensor source, "
      "Tensor assembly_rows, Tensor assembly_columns, "
      "Tensor assembly_values, Tensor synthesis_rows, "
      "Tensor synthesis_columns, Tensor synthesis_values, "
      "int induced_dimension, Tensor weight) -> (Tensor, Tensor, Tensor)");
  library.def(
      "compact_pair_product(Tensor left, Tensor right, "
      "bool antisymmetric) -> Tensor");
  library.def(
      "compact_pair_product_adjoint(Tensor output_adjoint, Tensor left, "
      "Tensor right, bool antisymmetric) -> (Tensor, Tensor)");
  library.def(
      "compact_exterior_power(Tensor factors) -> Tensor");
  library.def(
      "compact_exterior_power_adjoint(Tensor output_adjoint, "
      "Tensor factors) -> Tensor");
  library.def(
      "symmetric_power_monomial(Tensor input, Tensor monomial_counts, "
      "Tensor output_offsets, Tensor output_indices, "
      "Tensor monomial_values) -> Tensor");
  library.def(
      "symmetric_power_monomial_adjoint(Tensor output_adjoint, "
      "Tensor input, Tensor monomial_counts, Tensor output_offsets, "
      "Tensor output_indices, Tensor monomial_values) -> Tensor");
  library.def(
      "symmetric_power_monomial_double_backward("
      "Tensor input_adjoint_tangent, Tensor output_adjoint, Tensor input, "
      "Tensor monomial_counts, Tensor output_offsets, "
      "Tensor output_indices, Tensor monomial_values) -> (Tensor, Tensor)");
  library.def(
      "symmetric_power_shared_monomial(Tensor input, "
      "Tensor monomial_counts, Tensor output_offsets, "
      "Tensor coefficient_terms, Tensor coefficient_outputs, "
      "Tensor coefficient_values) -> Tensor");
  library.def(
      "symmetric_power_shared_monomial_adjoint(Tensor output_adjoint, "
      "Tensor input, Tensor monomial_counts, Tensor output_offsets, "
      "Tensor coefficient_terms, Tensor coefficient_outputs, "
      "Tensor coefficient_values) -> Tensor");
  library.def(
      "symmetric_power_shared_monomial_double_backward("
      "Tensor input_adjoint_tangent, Tensor output_adjoint, Tensor input, "
      "Tensor monomial_counts, Tensor output_offsets, "
      "Tensor coefficient_terms, Tensor coefficient_outputs, "
      "Tensor coefficient_values) -> (Tensor, Tensor)");
  library.def(
      "symmetric_power_shared_monomial_factored_double_backward("
      "Tensor input_adjoint_tangent, Tensor output_adjoint, Tensor input, "
      "Tensor monomial_counts, Tensor output_offsets, "
      "Tensor coefficient_terms, Tensor coefficient_outputs, "
      "Tensor coefficient_values) -> (Tensor, Tensor)");
  library.def(
      "symmetric_power_shared_sparse_monomial("
      "Tensor input, Tensor term_offsets, Tensor term_components, "
      "Tensor term_exponents, Tensor output_offsets, "
      "Tensor coefficient_terms, Tensor coefficient_outputs, "
      "Tensor coefficient_values) -> Tensor");
  library.def(
      "symmetric_power_shared_sparse_monomial_adjoint("
      "Tensor output_adjoint, Tensor input, Tensor term_offsets, "
      "Tensor term_components, Tensor term_exponents, "
      "Tensor output_offsets, Tensor coefficient_terms, "
      "Tensor coefficient_outputs, Tensor coefficient_values) -> Tensor");
  library.def(
      "symmetric_power_shared_sparse_monomial_double_backward("
      "Tensor input_adjoint_tangent, Tensor output_adjoint, Tensor input, "
      "Tensor term_offsets, Tensor term_components, Tensor term_exponents, "
      "Tensor output_offsets, Tensor coefficient_terms, "
      "Tensor coefficient_outputs, Tensor coefficient_values) "
      "-> (Tensor, Tensor)");
  library.def(
      "symmetric_power_shared_monomial_batched_adjoint("
      "Tensor output_adjoint, Tensor input, Tensor monomial_counts, "
      "Tensor output_offsets, Tensor coefficient_terms, "
      "Tensor coefficient_outputs, Tensor coefficient_values) -> Tensor");
  library.def(
      "factorized_angular(Tensor packed_slots, Tensor node_offsets, "
      "Tensor node_dimensions, Tensor node_leaf_offsets, Tensor node_left, "
      "Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor projection_values, int source_dimension, "
      "int workspace_dimension, "
      "int output_dimension) -> Tensor");
  library.def(
      "factorized_angular_adjoint(Tensor output_adjoint, "
      "Tensor packed_slots, Tensor node_offsets, Tensor node_dimensions, "
      "Tensor node_leaf_offsets, Tensor node_left, Tensor node_right, "
      "Tensor node_coefficient_offsets, Tensor coefficient_rows, "
      "Tensor coefficient_columns, Tensor coefficient_values, "
      "Tensor root_nodes, Tensor projection_values, "
      "int source_dimension, int workspace_dimension) -> Tensor");
  library.def(
      "factorized_angular_double_backward("
      "Tensor packed_adjoint_tangent, Tensor output_adjoint, "
      "Tensor packed_slots, Tensor node_offsets, Tensor node_dimensions, "
      "Tensor node_leaf_offsets, Tensor node_left, Tensor node_right, "
      "Tensor node_coefficient_offsets, Tensor coefficient_rows, "
      "Tensor coefficient_columns, Tensor coefficient_values, "
      "Tensor root_nodes, Tensor projection_values, "
      "int source_dimension, int workspace_dimension) -> (Tensor, Tensor)");
  library.def(
      "factorized_angular_heterogeneous("
      "Tensor packed_slots, Tensor node_offsets, Tensor node_dimensions, "
      "Tensor node_leaf_offsets, Tensor node_left, Tensor node_right, "
      "Tensor node_coefficient_offsets, Tensor coefficient_rows, "
      "Tensor coefficient_columns, Tensor coefficient_values, "
      "Tensor root_nodes, Tensor root_projection_starts, "
      "Tensor root_projection_dimensions, Tensor root_output_offsets, "
      "Tensor projection_values, int source_dimension, "
      "int workspace_dimension, int output_dimension) -> Tensor");
  library.def(
      "factorized_angular_heterogeneous_adjoint("
      "Tensor output_adjoint, Tensor packed_slots, Tensor node_offsets, "
      "Tensor node_dimensions, Tensor node_leaf_offsets, Tensor node_left, "
      "Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor root_projection_starts, Tensor root_projection_dimensions, "
      "Tensor root_output_offsets, Tensor projection_values, "
      "int source_dimension, int workspace_dimension) -> Tensor");
  library.def(
      "factorized_angular_heterogeneous_adjoint_with_workspace("
      "Tensor output_adjoint, Tensor packed_slots, Tensor node_offsets, "
      "Tensor node_dimensions, Tensor node_leaf_offsets, Tensor node_left, "
      "Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor root_projection_starts, Tensor root_projection_dimensions, "
      "Tensor root_output_offsets, Tensor projection_values, "
      "int source_dimension, int workspace_dimension) "
      "-> (Tensor, Tensor, Tensor)");
  library.def(
      "factorized_angular_heterogeneous_double_backward("
      "Tensor packed_adjoint_tangent, Tensor output_adjoint, "
      "Tensor packed_slots, Tensor node_offsets, Tensor node_dimensions, "
      "Tensor node_leaf_offsets, Tensor node_left, Tensor node_right, "
      "Tensor node_coefficient_offsets, Tensor coefficient_rows, "
      "Tensor coefficient_columns, Tensor coefficient_values, "
      "Tensor root_nodes, Tensor root_projection_starts, "
      "Tensor root_projection_dimensions, Tensor root_output_offsets, "
      "Tensor projection_values, int source_dimension, "
      "int workspace_dimension) -> (Tensor, Tensor)");
  library.def(
      "factorized_angular_heterogeneous_double_backward_from_workspace("
      "Tensor packed_adjoint_tangent, Tensor output_adjoint, "
      "Tensor workspace, Tensor workspace_adjoint, Tensor node_offsets, "
      "Tensor node_dimensions, Tensor node_leaf_offsets, Tensor node_left, "
      "Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor root_projection_starts, Tensor root_projection_dimensions, "
      "Tensor root_output_offsets, Tensor projection_values, "
      "int source_dimension, int workspace_dimension) -> (Tensor, Tensor)");
  library.def(
      "factorized_angular_segmented("
      "Tensor packed_slots, Tensor segment_source_offsets, "
      "Tensor segment_input_offsets, Tensor segment_input_dimensions, "
      "Tensor segment_workspace_offsets, "
      "Tensor segment_workspace_dimensions, Tensor segment_node_offsets, "
      "Tensor segment_root_offsets, Tensor segment_projection_offsets, "
      "Tensor segment_projection_dimensions, Tensor segment_output_offsets, "
      "Tensor node_offsets, Tensor node_dimensions, Tensor node_leaf_offsets, "
      "Tensor node_left, Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor root_projection_starts, Tensor root_projection_dimensions, "
      "Tensor root_output_offsets, Tensor projection_values, "
      "int total_source_count, int workspace_dimension, "
      "int output_dimension) -> Tensor");
  library.def(
      "factorized_angular_segmented_adjoint("
      "Tensor output_adjoint, Tensor packed_slots, "
      "Tensor segment_source_offsets, Tensor segment_input_offsets, "
      "Tensor segment_input_dimensions, Tensor segment_workspace_offsets, "
      "Tensor segment_workspace_dimensions, Tensor segment_node_offsets, "
      "Tensor segment_root_offsets, Tensor segment_projection_offsets, "
      "Tensor segment_projection_dimensions, Tensor segment_output_offsets, "
      "Tensor node_offsets, Tensor node_dimensions, Tensor node_leaf_offsets, "
      "Tensor node_left, Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor root_projection_starts, Tensor root_projection_dimensions, "
      "Tensor root_output_offsets, Tensor projection_values, "
      "int total_source_count, int workspace_dimension) -> Tensor");
  library.def(
      "factorized_angular_segmented_double_backward("
      "Tensor packed_adjoint_tangent, Tensor output_adjoint, "
      "Tensor packed_slots, Tensor segment_source_offsets, "
      "Tensor segment_input_offsets, Tensor segment_input_dimensions, "
      "Tensor segment_workspace_offsets, "
      "Tensor segment_workspace_dimensions, Tensor segment_node_offsets, "
      "Tensor segment_root_offsets, Tensor segment_projection_offsets, "
      "Tensor segment_projection_dimensions, Tensor segment_output_offsets, "
      "Tensor node_offsets, Tensor node_dimensions, Tensor node_leaf_offsets, "
      "Tensor node_left, Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor root_projection_starts, Tensor root_projection_dimensions, "
      "Tensor root_output_offsets, Tensor projection_values, "
      "int total_source_count, int workspace_dimension) -> (Tensor, Tensor)");
  library.def(
      "factorized_angular_linear(Tensor packed_slots, Tensor node_offsets, "
      "Tensor node_dimensions, Tensor node_leaf_offsets, Tensor node_left, "
      "Tensor node_right, Tensor node_coefficient_offsets, "
      "Tensor coefficient_rows, Tensor coefficient_columns, "
      "Tensor coefficient_values, Tensor root_nodes, "
      "Tensor projection_values, int source_dimension, "
      "int workspace_dimension, Tensor weight, "
      "Tensor bias) -> Tensor");
  library.def(
      "factorized_angular_linear_adjoint(Tensor output_adjoint, "
      "Tensor packed_slots, Tensor node_offsets, Tensor node_dimensions, "
      "Tensor node_leaf_offsets, Tensor node_left, Tensor node_right, "
      "Tensor node_coefficient_offsets, Tensor coefficient_rows, "
      "Tensor coefficient_columns, Tensor coefficient_values, "
      "Tensor root_nodes, Tensor projection_values, int source_dimension, "
      "int workspace_dimension, "
      "Tensor weight) -> (Tensor, Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(ye3t_runtime, CPU, library) {
  library.impl(
      "complex_spherical_harmonics_recurrence",
      &complex_spherical_harmonics_recurrence_cpu);
  library.impl(
      "cheb_exp_cos_radial_with_derivative",
      &cheb_exp_cos_radial_with_derivative_cpu);
  library.impl(
      "cheb_exp_cos_radial_table_with_derivative",
      &cheb_exp_cos_radial_table_with_derivative_cpu);
  library.impl(
      "cheb_exp_cos_radial_table_double_backward",
      &cheb_exp_cos_radial_table_double_backward_cpu);
  library.impl(
      "spherical_harmonics_with_derivative",
      &spherical_harmonics_with_derivative_cpu);
  library.impl(
      "spherical_harmonics_table_with_derivative",
      &spherical_harmonics_table_with_derivative_cpu);
  library.impl(
      "plain_site_basis_product_with_derivative",
      &plain_site_basis_product_with_derivative_cpu);
  library.impl(
      "scheduled_radial_angular_channels_with_derivative",
      &scheduled_radial_angular_channels_with_derivative_cpu);
  library.impl(
      "plain_site_basis_product_adjoint",
      &plain_site_basis_product_adjoint_cpu);
  library.impl("density_accumulate", &density_accumulate_cpu);
  library.impl(
      "density_accumulate_adjoint",
      &density_accumulate_adjoint_cpu);
  library.impl(
      "edge_outer_accumulate",
      &edge_outer_accumulate_cpu);
  library.impl(
      "edge_outer_accumulate_adjoint",
      &edge_outer_accumulate_adjoint_cpu);
  library.impl(
      "edge_outer_accumulate_double_backward",
      &edge_outer_accumulate_double_backward_cpu);
  library.impl(
      "softmax_gaussian_role_density",
      &softmax_gaussian_role_density_cpu);
  library.impl(
      "softmax_gaussian_role_density_adjoint",
      &softmax_gaussian_role_density_adjoint_cpu);
  library.impl(
      "softmax_gaussian_role_density_double_backward",
      &softmax_gaussian_role_density_double_backward_cpu);
  library.impl(
      "scheduled_softmax_gaussian_role_density",
      &scheduled_softmax_gaussian_role_density_cpu);
  library.impl(
      "scheduled_softmax_gaussian_role_density_adjoint",
      &scheduled_softmax_gaussian_role_density_adjoint_cpu);
  library.impl(
      "scheduled_softmax_gaussian_role_density_double_backward",
      &scheduled_softmax_gaussian_role_density_double_backward_cpu);
  library.impl(
      "carrier_gated_scatter",
      &carrier_gated_scatter_cpu);
  library.impl(
      "carrier_gated_scatter_adjoint",
      &carrier_gated_scatter_adjoint_cpu);
  library.impl(
      "carrier_gated_scatter_double_backward",
      &carrier_gated_scatter_double_backward_cpu);
  library.impl(
      "carrier_residual_gated_scatter",
      &carrier_residual_gated_scatter_cpu);
  library.impl(
      "carrier_residual_gated_scatter_adjoint",
      &carrier_residual_gated_scatter_adjoint_cpu);
  library.impl(
      "carrier_residual_gated_scatter_double_backward",
      &carrier_residual_gated_scatter_double_backward_cpu);
  library.impl(
      "carrier_segmented_residual_gated_scatter",
      &carrier_segmented_residual_gated_scatter_cpu);
  library.impl(
      "carrier_segmented_residual_gated_scatter_adjoint",
      &carrier_segmented_residual_gated_scatter_adjoint_cpu);
  library.impl(
      "carrier_segmented_residual_gated_scatter_double_backward",
      &carrier_segmented_residual_gated_scatter_double_backward_cpu);
  library.impl(
      "source_arena_gather",
      &source_arena_gather_cpu);
  library.impl(
      "source_arena_gather_adjoint",
      &source_arena_gather_adjoint_cpu);
  library.impl(
      "source_arena_gather_double_backward",
      &source_arena_gather_double_backward_cpu);
  library.impl(
      "source_arena_channel_transform",
      &source_arena_channel_transform_cpu);
  library.impl(
      "source_arena_channel_transform_adjoint",
      &source_arena_channel_transform_adjoint_cpu);
  library.impl(
      "source_arena_channel_transform_double_backward",
      &source_arena_channel_transform_double_backward_cpu);
  library.impl(
      "carrier_channel_update",
      &carrier_channel_update_cpu);
  library.impl(
      "carrier_channel_update_adjoint",
      &carrier_channel_update_adjoint_cpu);
  library.impl(
      "carrier_channel_update_double_backward",
      &carrier_channel_update_double_backward_cpu);
  library.impl(
      "carrier_channel_transform",
      &carrier_channel_transform_cpu);
  library.impl(
      "carrier_channel_transform_adjoint",
      &carrier_channel_transform_adjoint_cpu);
  library.impl(
      "carrier_channel_transform_double_backward",
      &carrier_channel_transform_double_backward_cpu);
  library.impl(
      "carrier_role_channel_map_adjoint",
      &carrier_role_channel_map_adjoint_cpu);
  library.impl(
      "carrier_role_channel_map_adjoint_double_backward",
      &carrier_role_channel_map_adjoint_double_backward_cpu);
  library.impl("source_analysis", &source_analysis_cpu);
  library.impl("source_analysis_adjoint", &source_analysis_adjoint_cpu);
  library.impl("source_analysis_linear", &source_analysis_linear_cpu);
  library.impl(
      "source_analysis_linear_adjoint",
      &source_analysis_linear_adjoint_cpu);
  library.impl("compact_pair_product", &compact_pair_product_cpu);
  library.impl(
      "compact_pair_product_adjoint",
      &compact_pair_product_adjoint_cpu);
  library.impl(
      "compact_exterior_power",
      &compact_exterior_power_cpu);
  library.impl(
      "compact_exterior_power_adjoint",
      &compact_exterior_power_adjoint_cpu);
  library.impl(
      "symmetric_power_monomial",
      &symmetric_power_monomial_cpu);
  library.impl(
      "symmetric_power_monomial_adjoint",
      &symmetric_power_monomial_adjoint_cpu);
  library.impl(
      "symmetric_power_shared_monomial",
      &symmetric_power_shared_monomial_cpu);
  library.impl(
      "symmetric_power_shared_monomial_adjoint",
      &symmetric_power_shared_monomial_adjoint_cpu);
  library.impl(
      "symmetric_power_shared_monomial_batched_adjoint",
      &symmetric_power_shared_monomial_batched_adjoint_cpu);
  library.impl("factorized_angular", &factorized_angular_cpu);
  library.impl(
      "factorized_angular_adjoint",
      &factorized_angular_adjoint_cpu);
  library.impl(
      "factorized_angular_double_backward",
      &factorized_angular_double_backward_cpu);
  library.impl(
      "factorized_angular_heterogeneous",
      &factorized_angular_heterogeneous_cpu);
  library.impl(
      "factorized_angular_heterogeneous_adjoint",
      &factorized_angular_heterogeneous_adjoint_cpu);
  library.impl(
      "factorized_angular_heterogeneous_double_backward",
      &factorized_angular_heterogeneous_double_backward_cpu);
  library.impl(
      "factorized_angular_segmented",
      &factorized_angular_segmented_cpu);
  library.impl(
      "factorized_angular_segmented_adjoint",
      &factorized_angular_segmented_adjoint_cpu);
  library.impl(
      "factorized_angular_segmented_double_backward",
      &factorized_angular_segmented_double_backward_cpu);
  library.impl(
      "factorized_angular_linear",
      &factorized_angular_linear_cpu);
  library.impl(
      "factorized_angular_linear_adjoint",
      &factorized_angular_linear_adjoint_cpu);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("core_abi_version", []() { return 40; });
  module.def("has_cpu", []() { return true; });
  module.def("has_cuda", []() {
#ifdef YE3T_HAS_CUDA
    return true;
#else
    return false;
#endif
  });
}
