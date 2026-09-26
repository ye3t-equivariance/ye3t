#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

struct ParsedKey {
  enum class Kind { Leaf, Sym, Node };

  Kind kind;
  int64_t L;
  int64_t total_L;
  int64_t block_count;
  std::shared_ptr<ParsedKey> left;
  std::shared_ptr<ParsedKey> right;
  std::string cache_key;
};

struct PathExpansion {
  int64_t block_count = 0;
  std::vector<int16_t> root_m;
  std::vector<int16_t> block_m;
  std::vector<double> coeffs;
};

void check_cpu_contiguous(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(!tensor.is_cuda(), name, " must be a CPU tensor.");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous.");
}

std::shared_ptr<ParsedKey> parse_factorized_key(py::handle object) {
  TORCH_CHECK(py::isinstance<py::tuple>(object), "factorized path key must be a tuple.");
  py::tuple tuple = py::reinterpret_borrow<py::tuple>(object);
  TORCH_CHECK(tuple.size() >= 2, "factorized path key tuple is too short.");
  const std::string kind = py::cast<std::string>(tuple[0]);
  auto parsed = std::make_shared<ParsedKey>();
  if (kind == "leaf" || kind == "sym") {
    parsed->kind = kind == "leaf" ? ParsedKey::Kind::Leaf : ParsedKey::Kind::Sym;
    parsed->L = py::cast<int64_t>(tuple[1]);
    parsed->total_L = parsed->L;
    parsed->block_count = 1;
    parsed->cache_key = (kind == "leaf" ? "l" : "s") + std::to_string(parsed->L);
    return parsed;
  }
  TORCH_CHECK(kind == "node", "Unsupported factorized path key kind: ", kind);
  TORCH_CHECK(tuple.size() == 4, "node factorized path key must have four entries.");
  parsed->kind = ParsedKey::Kind::Node;
  parsed->left = parse_factorized_key(tuple[1]);
  parsed->right = parse_factorized_key(tuple[2]);
  parsed->L = py::cast<int64_t>(tuple[3]);
  parsed->total_L = parsed->L;
  parsed->block_count = parsed->left->block_count + parsed->right->block_count;
  parsed->cache_key = "n(" + parsed->left->cache_key + "," + parsed->right->cache_key + "," +
      std::to_string(parsed->L) + ")";
  return parsed;
}

std::vector<long double> log_factorials_up_to(int64_t n) {
  std::vector<long double> out(static_cast<size_t>(n) + 1);
  for (int64_t i = 0; i <= n; ++i) {
    out[static_cast<size_t>(i)] = std::lgammal(static_cast<long double>(i) + 1.0L);
  }
  return out;
}

double cg_integer_log(
    int64_t j1,
    int64_t m1,
    int64_t j2,
    int64_t m2,
    int64_t j3,
    int64_t m3,
    const std::vector<long double>& logfacts) {
  if (m1 + m2 != m3 || std::llabs(m1) > j1 || std::llabs(m2) > j2 || std::llabs(m3) > j3) {
    return 0.0;
  }
  if (j3 < std::llabs(j1 - j2) || j3 > j1 + j2) {
    return 0.0;
  }
  const auto lf = [&](int64_t n) -> long double {
    TORCH_CHECK(n >= 0 && static_cast<size_t>(n) < logfacts.size(), "CG factorial argument out of range.");
    return logfacts[static_cast<size_t>(n)];
  };
  long double log_prefactor = std::log(static_cast<long double>(2 * j3 + 1)) +
      lf(j1 + j2 - j3) + lf(j1 - j2 + j3) + lf(-j1 + j2 + j3) + lf(j1 + m1) +
      lf(j1 - m1) + lf(j2 + m2) + lf(j2 - m2) + lf(j3 + m3) + lf(j3 - m3) -
      lf(j1 + j2 + j3 + 1);

  const int64_t z_min = std::max<int64_t>({0, j2 - j3 - m1, j1 - j3 + m2});
  const int64_t z_max = std::min<int64_t>({j1 + j2 - j3, j1 - m1, j2 + m2});
  long double total = 0.0L;
  for (int64_t z = z_min; z <= z_max; ++z) {
    const long double log_den = lf(z) + lf(j1 + j2 - j3 - z) + lf(j1 - m1 - z) +
        lf(j2 + m2 - z) + lf(j3 - j2 + m1 + z) + lf(j3 - j1 - m2 + z);
    const long double term = std::exp(-log_den);
    total += (z % 2) ? -term : term;
  }
  return static_cast<double>(std::sqrt(std::exp(log_prefactor)) * total);
}

uint64_t cg_cache_key(int64_t left_total, int64_t right_total, int64_t output_L) {
  return (static_cast<uint64_t>(left_total) << 42) ^ (static_cast<uint64_t>(right_total) << 21) ^
      static_cast<uint64_t>(output_L);
}

const std::vector<double>& cg_matrix_for_node_cpp(
    int64_t left_total,
    int64_t right_total,
    int64_t output_L,
    std::unordered_map<uint64_t, std::vector<double>>& cg_cache) {
  const uint64_t key = cg_cache_key(left_total, right_total, output_L);
  auto it = cg_cache.find(key);
  if (it != cg_cache.end()) {
    return it->second;
  }
  const int64_t left_dim = 2 * left_total + 1;
  const int64_t right_dim = 2 * right_total + 1;
  std::vector<double> matrix(static_cast<size_t>(left_dim * right_dim), 0.0);
  const auto logfacts = log_factorials_up_to(left_total + right_total + output_L + 1);
  for (int64_t m1 = -left_total; m1 <= left_total; ++m1) {
    for (int64_t m2 = -right_total; m2 <= right_total; ++m2) {
      const int64_t M = m1 + m2;
      if (std::llabs(M) > output_L) {
        continue;
      }
      const double value = cg_integer_log(left_total, m1, right_total, m2, output_L, M, logfacts);
      if (std::abs(value) > 1e-14) {
        matrix[static_cast<size_t>((m1 + left_total) * right_dim + (m2 + right_total))] = value;
      }
    }
  }
  auto inserted = cg_cache.emplace(key, std::move(matrix));
  return inserted.first->second;
}

std::shared_ptr<PathExpansion> expand_path_key_cpp(
    const std::shared_ptr<ParsedKey>& key,
    std::unordered_map<std::string, std::shared_ptr<PathExpansion>>& path_cache,
    std::unordered_map<uint64_t, std::vector<double>>& cg_cache) {
  auto cached = path_cache.find(key->cache_key);
  if (cached != path_cache.end()) {
    return cached->second;
  }
  auto out = std::make_shared<PathExpansion>();
  out->block_count = key->block_count;
  if (key->kind == ParsedKey::Kind::Leaf || key->kind == ParsedKey::Kind::Sym) {
    const int64_t L = key->L;
    const int64_t count = 2 * L + 1;
    out->root_m.reserve(static_cast<size_t>(count));
    out->block_m.reserve(static_cast<size_t>(count));
    out->coeffs.reserve(static_cast<size_t>(count));
    for (int64_t m = -L; m <= L; ++m) {
      out->root_m.push_back(static_cast<int16_t>(m));
      out->block_m.push_back(static_cast<int16_t>(m));
      out->coeffs.push_back(1.0);
    }
    path_cache.emplace(key->cache_key, out);
    return out;
  }

  auto left = expand_path_key_cpp(key->left, path_cache, cg_cache);
  auto right = expand_path_key_cpp(key->right, path_cache, cg_cache);
  const int64_t left_count = static_cast<int64_t>(left->root_m.size());
  const int64_t right_count = static_cast<int64_t>(right->root_m.size());
  const int64_t left_width = left->block_count;
  const int64_t right_width = right->block_count;
  const int64_t out_width = left_width + right_width;
  out->block_count = out_width;
  if (left_count == 0 || right_count == 0) {
    path_cache.emplace(key->cache_key, out);
    return out;
  }
  const int64_t estimated = left_count * right_count;
  if (estimated > 0 && estimated < 10000000) {
    out->root_m.reserve(static_cast<size_t>(estimated));
    out->block_m.reserve(static_cast<size_t>(estimated * out_width));
    out->coeffs.reserve(static_cast<size_t>(estimated));
  }
  const int64_t left_total = key->left->total_L;
  const int64_t right_total = key->right->total_L;
  const int64_t right_dim = 2 * right_total + 1;
  const auto& cg_matrix = cg_matrix_for_node_cpp(left_total, right_total, key->L, cg_cache);
  for (int64_t i = 0; i < left_count; ++i) {
    const int64_t M1 = left->root_m[static_cast<size_t>(i)];
    const double c1 = left->coeffs[static_cast<size_t>(i)];
    const int16_t* left_row = left->block_m.data() + i * left_width;
    for (int64_t j = 0; j < right_count; ++j) {
      const int64_t M2 = right->root_m[static_cast<size_t>(j)];
      const int64_t M = M1 + M2;
      if (std::llabs(M) > key->L) {
        continue;
      }
      const double cg = cg_matrix[static_cast<size_t>((M1 + left_total) * right_dim + (M2 + right_total))];
      if (std::abs(cg) <= 1e-14) {
        continue;
      }
      const double coeff = c1 * right->coeffs[static_cast<size_t>(j)] * cg;
      if (std::abs(coeff) <= 0.0) {
        continue;
      }
      const int16_t* right_row = right->block_m.data() + j * right_width;
      out->root_m.push_back(static_cast<int16_t>(M));
      out->block_m.insert(out->block_m.end(), left_row, left_row + left_width);
      out->block_m.insert(out->block_m.end(), right_row, right_row + right_width);
      out->coeffs.push_back(coeff);
    }
  }
  path_cache.emplace(key->cache_key, out);
  return out;
}

torch::Tensor int16_tensor_from_vector(const std::vector<int16_t>& values, std::vector<int64_t> shape) {
  auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt16).device(torch::kCPU));
  if (!values.empty()) {
    std::memcpy(tensor.data_ptr<int16_t>(), values.data(), values.size() * sizeof(int16_t));
  }
  return tensor;
}

torch::Tensor int64_tensor_from_vector(const std::vector<int64_t>& values, std::vector<int64_t> shape) {
  auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
  if (!values.empty()) {
    std::memcpy(tensor.data_ptr<int64_t>(), values.data(), values.size() * sizeof(int64_t));
  }
  return tensor;
}

torch::Tensor float64_tensor_from_vector(const std::vector<double>& values, std::vector<int64_t> shape) {
  auto tensor = torch::empty(shape, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU));
  if (!values.empty()) {
    std::memcpy(tensor.data_ptr<double>(), values.data(), values.size() * sizeof(double));
  }
  return tensor;
}

template <typename scalar_t>
void evaluate_componentwise(
    const scalar_t* block_values,
    const int64_t* component_offsets,
    const int64_t* component_label_index,
    const int64_t* block_m_indices,
    const scalar_t* coeffs,
    scalar_t* out,
    const int64_t batch_size,
    const int64_t basis_count,
    const int64_t magnetic_dim,
    const int64_t component_count,
    const int64_t block_count) {
  at::parallel_for(0, batch_size * component_count, 0, [&](int64_t begin, int64_t end) {
    for (int64_t linear = begin; linear < end; ++linear) {
      const int64_t batch = linear / component_count;
      const int64_t component = linear - batch * component_count;
      const int64_t label = component_label_index[component];
      const int64_t start = component_offsets[component];
      const int64_t stop = component_offsets[component + 1];
      scalar_t acc = scalar_t(0);
      for (int64_t term = start; term < stop; ++term) {
        scalar_t value = coeffs[term];
        const int64_t* term_m = block_m_indices + term * block_count;
        for (int64_t block = 0; block < block_count; ++block) {
          const int64_t offset =
              ((batch * basis_count + label) * block_count + block) * magnetic_dim + term_m[block];
          value *= block_values[offset];
        }
        acc += value;
      }
      out[batch * component_count + component] = acc;
    }
  });
}

}  // namespace

py::tuple expand_block_m_path_arrays_cpp(py::object key_object) {
  auto key = parse_factorized_key(key_object);
  std::unordered_map<std::string, std::shared_ptr<PathExpansion>> path_cache;
  std::unordered_map<uint64_t, std::vector<double>> cg_cache;
  auto expansion = expand_path_key_cpp(key, path_cache, cg_cache);
  const int64_t term_count = static_cast<int64_t>(expansion->root_m.size());
  return py::make_tuple(
      int16_tensor_from_vector(expansion->root_m, {term_count}),
      int16_tensor_from_vector(expansion->block_m, {term_count, expansion->block_count}),
      float64_tensor_from_vector(expansion->coeffs, {term_count}));
}

py::tuple assemble_factorized_schedule_from_keys_impl(
    py::sequence key_objects,
    py::sequence m_values_object,
    double coeff_tol,
    bool clear_path_cache_per_label) {
  const int64_t label_count = static_cast<int64_t>(py::len(key_objects));
  std::vector<int64_t> m_values;
  m_values.reserve(static_cast<size_t>(py::len(m_values_object)));
  for (py::handle item : m_values_object) {
    m_values.push_back(py::cast<int64_t>(item));
  }

  std::unordered_map<std::string, std::shared_ptr<PathExpansion>> path_cache;
  std::unordered_map<uint64_t, std::vector<double>> cg_cache;
  std::vector<int64_t> component_label_index;
  std::vector<int64_t> component_m;
  std::vector<int64_t> component_offsets;
  std::vector<int16_t> block_m_tuples;
  std::vector<double> coeffs;
  component_label_index.reserve(static_cast<size_t>(label_count * static_cast<int64_t>(m_values.size())));
  component_m.reserve(component_label_index.capacity());
  component_offsets.reserve(component_label_index.capacity() + 1);
  component_offsets.push_back(0);

  int64_t total_selected_terms = 0;
  int64_t block_count = 0;
  for (int64_t label_index = 0; label_index < label_count; ++label_index) {
    std::unordered_map<std::string, std::shared_ptr<PathExpansion>> local_path_cache;
    auto* active_path_cache = &path_cache;
    if (clear_path_cache_per_label) {
      active_path_cache = &local_path_cache;
    }
    auto key = parse_factorized_key(key_objects[static_cast<py::ssize_t>(label_index)]);
    auto expansion = expand_path_key_cpp(key, *active_path_cache, cg_cache);
    if (label_index == 0) {
      block_count = expansion->block_count;
    } else {
      TORCH_CHECK(expansion->block_count == block_count, "All factorized keys must have the same block count.");
    }
    const int64_t path_count = static_cast<int64_t>(expansion->root_m.size());
    for (int64_t M_R : m_values) {
      for (int64_t term = 0; term < path_count; ++term) {
        if (static_cast<int64_t>(expansion->root_m[static_cast<size_t>(term)]) != M_R) {
          continue;
        }
        const double coeff = expansion->coeffs[static_cast<size_t>(term)];
        if (coeff_tol > 0.0 && std::abs(coeff) <= coeff_tol) {
          continue;
        }
        ++total_selected_terms;
      }
    }
  }

  if (block_count > 0) {
    const int64_t max_size =
        static_cast<int64_t>(std::numeric_limits<size_t>::max() / sizeof(int16_t));
    TORCH_CHECK(
        total_selected_terms <= max_size / block_count,
        "Factorized schedule coefficient index payload is too large to allocate.");
  }
  block_m_tuples.reserve(static_cast<size_t>(total_selected_terms * block_count));
  coeffs.reserve(static_cast<size_t>(total_selected_terms));

  for (int64_t label_index = 0; label_index < label_count; ++label_index) {
    std::unordered_map<std::string, std::shared_ptr<PathExpansion>> local_path_cache;
    auto* active_path_cache = &path_cache;
    if (clear_path_cache_per_label) {
      active_path_cache = &local_path_cache;
    }
    auto key = parse_factorized_key(key_objects[static_cast<py::ssize_t>(label_index)]);
    auto expansion = expand_path_key_cpp(key, *active_path_cache, cg_cache);
    TORCH_CHECK(expansion->block_count == block_count, "All factorized keys must have the same block count.");
    const int64_t path_count = static_cast<int64_t>(expansion->root_m.size());
    for (int64_t M_R : m_values) {
      component_label_index.push_back(label_index);
      component_m.push_back(M_R);
      for (int64_t term = 0; term < path_count; ++term) {
        if (static_cast<int64_t>(expansion->root_m[static_cast<size_t>(term)]) != M_R) {
          continue;
        }
        const double coeff = expansion->coeffs[static_cast<size_t>(term)];
        if (coeff_tol > 0.0 && std::abs(coeff) <= coeff_tol) {
          continue;
        }
        const int16_t* row = expansion->block_m.data() + term * block_count;
        block_m_tuples.insert(block_m_tuples.end(), row, row + block_count);
        coeffs.push_back(coeff);
      }
      component_offsets.push_back(static_cast<int64_t>(coeffs.size()));
    }
  }

  const int64_t component_count = static_cast<int64_t>(component_label_index.size());
  const int64_t term_count = static_cast<int64_t>(coeffs.size());
  return py::make_tuple(
      int64_tensor_from_vector(component_label_index, {component_count}),
      int64_tensor_from_vector(component_m, {component_count}),
      int64_tensor_from_vector(component_offsets, {component_count + 1}),
      int16_tensor_from_vector(block_m_tuples, {term_count, block_count}),
      float64_tensor_from_vector(coeffs, {term_count}));
}

py::tuple assemble_factorized_schedule_from_keys_cpp(
    py::sequence key_objects,
    py::sequence m_values_object,
    double coeff_tol) {
  return assemble_factorized_schedule_from_keys_impl(
      key_objects,
      m_values_object,
      coeff_tol,
      false);
}

py::tuple assemble_factorized_schedule_from_keys_low_memory_cpp(
    py::sequence key_objects,
    py::sequence m_values_object,
    double coeff_tol) {
  return assemble_factorized_schedule_from_keys_impl(
      key_objects,
      m_values_object,
      coeff_tol,
      true);
}

torch::Tensor evaluate_factorized_schedule_cpu(
    torch::Tensor block_values,
    torch::Tensor component_offsets,
    torch::Tensor component_label_index,
    torch::Tensor block_m_indices,
    torch::Tensor coeffs,
    int64_t component_count,
    bool validate_indices) {
  TORCH_CHECK(block_values.dim() == 4, "block_values must have shape [batch, basis, block, m].");
  TORCH_CHECK(component_offsets.dim() == 1, "component_offsets must be one-dimensional.");
  TORCH_CHECK(component_label_index.dim() == 1, "component_label_index must be one-dimensional.");
  TORCH_CHECK(block_m_indices.dim() == 2, "block_m_indices must have shape [term, block].");
  TORCH_CHECK(coeffs.dim() == 1, "coeffs must be one-dimensional.");
  TORCH_CHECK(component_count >= 0, "component_count must be non-negative.");
  TORCH_CHECK(component_offsets.scalar_type() == torch::kInt64, "component_offsets must be int64.");
  TORCH_CHECK(component_label_index.scalar_type() == torch::kInt64, "component_label_index must be int64.");
  TORCH_CHECK(block_m_indices.scalar_type() == torch::kInt64, "block_m_indices must be int64.");
  TORCH_CHECK(block_values.scalar_type() == coeffs.scalar_type(), "block_values and coeffs must have the same dtype.");
  TORCH_CHECK(component_offsets.size(0) == component_count + 1, "component_offsets has wrong length.");
  TORCH_CHECK(component_label_index.size(0) == component_count, "component_label_index has wrong length.");
  TORCH_CHECK(coeffs.size(0) == block_m_indices.size(0), "coeffs and block_m_indices disagree on term_count.");
  TORCH_CHECK(block_values.size(2) == block_m_indices.size(1), "block_count mismatch.");

  block_values = block_values.contiguous();
  component_offsets = component_offsets.contiguous();
  component_label_index = component_label_index.contiguous();
  block_m_indices = block_m_indices.contiguous();
  coeffs = coeffs.contiguous();

  check_cpu_contiguous(block_values, "block_values");
  check_cpu_contiguous(component_offsets, "component_offsets");
  check_cpu_contiguous(component_label_index, "component_label_index");
  check_cpu_contiguous(block_m_indices, "block_m_indices");
  check_cpu_contiguous(coeffs, "coeffs");

  const int64_t batch_size = block_values.size(0);
  const int64_t basis_count = block_values.size(1);
  const int64_t block_count = block_values.size(2);
  const int64_t magnetic_dim = block_values.size(3);
  const int64_t term_count = coeffs.size(0);

  if (component_count == 0 || batch_size == 0) {
    return torch::zeros({batch_size, component_count}, block_values.options());
  }

  if (validate_indices) {
    auto label_acc = component_label_index.accessor<int64_t, 1>();
    for (int64_t component = 0; component < component_count; ++component) {
      TORCH_CHECK(label_acc[component] >= 0 && label_acc[component] < basis_count, "component_label_index out of range.");
    }
    auto offsets_acc = component_offsets.accessor<int64_t, 1>();
    TORCH_CHECK(offsets_acc[0] == 0, "component_offsets must start at zero.");
    for (int64_t component = 0; component < component_count; ++component) {
      TORCH_CHECK(offsets_acc[component] <= offsets_acc[component + 1], "component_offsets must be nondecreasing.");
    }
    TORCH_CHECK(offsets_acc[component_count] == term_count, "component_offsets final value must equal term_count.");
    auto m_acc = block_m_indices.accessor<int64_t, 2>();
    for (int64_t term = 0; term < term_count; ++term) {
      for (int64_t block = 0; block < block_count; ++block) {
        TORCH_CHECK(m_acc[term][block] >= 0 && m_acc[term][block] < magnetic_dim, "block_m_indices out of range.");
      }
    }
  }

  auto out = torch::zeros({batch_size, component_count}, block_values.options());
  const auto dtype = block_values.scalar_type();
  if (dtype == torch::kFloat32) {
    evaluate_componentwise<float>(
        block_values.data_ptr<float>(),
        component_offsets.data_ptr<int64_t>(),
        component_label_index.data_ptr<int64_t>(),
        block_m_indices.data_ptr<int64_t>(),
        coeffs.data_ptr<float>(),
        out.data_ptr<float>(),
        batch_size,
        basis_count,
        magnetic_dim,
        component_count,
        block_count);
  } else if (dtype == torch::kFloat64) {
    evaluate_componentwise<double>(
        block_values.data_ptr<double>(),
        component_offsets.data_ptr<int64_t>(),
        component_label_index.data_ptr<int64_t>(),
        block_m_indices.data_ptr<int64_t>(),
        coeffs.data_ptr<double>(),
        out.data_ptr<double>(),
        batch_size,
        basis_count,
        magnetic_dim,
        component_count,
        block_count);
  } else if (dtype == torch::kComplexFloat) {
    evaluate_componentwise<c10::complex<float>>(
        block_values.data_ptr<c10::complex<float>>(),
        component_offsets.data_ptr<int64_t>(),
        component_label_index.data_ptr<int64_t>(),
        block_m_indices.data_ptr<int64_t>(),
        coeffs.data_ptr<c10::complex<float>>(),
        out.data_ptr<c10::complex<float>>(),
        batch_size,
        basis_count,
        magnetic_dim,
        component_count,
        block_count);
  } else if (dtype == torch::kComplexDouble) {
    evaluate_componentwise<c10::complex<double>>(
        block_values.data_ptr<c10::complex<double>>(),
        component_offsets.data_ptr<int64_t>(),
        component_label_index.data_ptr<int64_t>(),
        block_m_indices.data_ptr<int64_t>(),
        coeffs.data_ptr<c10::complex<double>>(),
        out.data_ptr<c10::complex<double>>(),
        batch_size,
        basis_count,
        magnetic_dim,
        component_count,
        block_count);
  } else {
    TORCH_CHECK(false, "Unsupported dtype for evaluate_factorized_schedule_cpu.");
  }
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "evaluate_factorized_schedule_cpu",
      &evaluate_factorized_schedule_cpu,
      "Evaluate a YE3T factorized coefficient schedule on CPU.");
  m.def(
      "expand_block_m_path_arrays_cpp",
      &expand_block_m_path_arrays_cpp,
      "Expand one YE3T factorized path key to root-M/block-M/coefficient arrays.");
  m.def(
      "assemble_factorized_schedule_from_keys_cpp",
      &assemble_factorized_schedule_from_keys_cpp,
      "Assemble YE3T factorized schedule arrays from angular factorized path keys.");
  m.def(
      "assemble_factorized_schedule_from_keys_low_memory_cpp",
      &assemble_factorized_schedule_from_keys_low_memory_cpp,
      "Assemble YE3T factorized schedule arrays with path cache cleared per label.");
}
