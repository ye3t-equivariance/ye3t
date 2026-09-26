#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <cstdint>

namespace {

void check_generator_tensor(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
  TORCH_CHECK(tensor.dim() == 3, name, " must have shape (generator_count, dim, dim).");
  TORCH_CHECK(tensor.size(1) == tensor.size(2), name, " generator matrices must be square.");
  TORCH_CHECK(torch::is_floating_point(tensor), name, " must use a floating dtype.");
}

torch::Tensor assemble_subduction_constraint_matrix(
    torch::Tensor target_generators,
    torch::Tensor child_generators) {
  target_generators = target_generators.contiguous();
  child_generators = child_generators.contiguous();
  check_generator_tensor(target_generators, "target_generators");
  check_generator_tensor(child_generators, "child_generators");
  TORCH_CHECK(
      target_generators.scalar_type() == child_generators.scalar_type(),
      "target_generators and child_generators must have the same dtype.");
  const int64_t generator_count = target_generators.size(0);
  TORCH_CHECK(
      child_generators.size(0) == generator_count,
      "target_generators and child_generators must contain the same number of generators.");
  const int64_t target_dim = target_generators.size(1);
  const int64_t child_dim = child_generators.size(1);
  const int64_t vector_dim = target_dim * child_dim;
  auto out = torch::zeros(
      {generator_count * vector_dim, vector_dim},
      target_generators.options());

  AT_DISPATCH_FLOATING_TYPES(target_generators.scalar_type(), "assemble_subduction_constraint_matrix", [&] {
    const auto* target = target_generators.data_ptr<scalar_t>();
    const auto* child = child_generators.data_ptr<scalar_t>();
    auto* constraints = out.data_ptr<scalar_t>();
    at::parallel_for(0, generator_count * vector_dim, 0, [&](int64_t begin, int64_t end) {
      for (int64_t packed_row = begin; packed_row < end; ++packed_row) {
        const int64_t generator = packed_row / vector_dim;
        const int64_t local_row = packed_row % vector_dim;
        const int64_t child_row = local_row / target_dim;
        const int64_t target_row = local_row % target_dim;
        const int64_t global_row = generator * vector_dim + local_row;
        const int64_t target_offset = generator * target_dim * target_dim;
        const int64_t child_offset = generator * child_dim * child_dim;
        const int64_t out_offset = global_row * vector_dim;

        for (int64_t target_col = 0; target_col < target_dim; ++target_col) {
          const scalar_t value = target[target_offset + target_row * target_dim + target_col];
          if (value != static_cast<scalar_t>(0)) {
            constraints[out_offset + child_row * target_dim + target_col] += value;
          }
        }
        for (int64_t child_col = 0; child_col < child_dim; ++child_col) {
          const scalar_t value = child[child_offset + child_col * child_dim + child_row];
          if (value != static_cast<scalar_t>(0)) {
            constraints[out_offset + child_col * target_dim + target_row] -= value;
          }
        }
      }
    });
  });
  return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "assemble_subduction_constraint_matrix",
      &assemble_subduction_constraint_matrix,
      "Assemble Young-subgroup generator-equation constraints for numeric subduction.");
}
