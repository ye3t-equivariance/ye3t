#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace {

void check_cuda_data_tensor(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64 ||
          tensor.scalar_type() == torch::kComplexFloat ||
          tensor.scalar_type() == torch::kComplexDouble,
      name,
      " must use float32, float64, complex64, or complex128");
}

void check_cuda_factor_tensor(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(
      tensor.dim() == 3,
      name,
      " must be a three-dimensional tensor");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64 ||
          tensor.scalar_type() == torch::kComplexFloat ||
          tensor.scalar_type() == torch::kComplexDouble,
      name,
      " must use float32, float64, complex64, or complex128");
}

void check_cuda_int64_vector(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt64,
      name,
      " must use int64");
}

void check_cuda_int64_matrix(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kInt64,
      name,
      " must use int64");
}

void check_cuda_value_vector(
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

void check_cuda_indices(
    const torch::Tensor& rows,
    const torch::Tensor& columns,
    const torch::Tensor& values,
    const torch::Tensor& data,
    const char* prefix) {
  TORCH_CHECK(rows.device() == data.device(), prefix, " rows device mismatch");
  TORCH_CHECK(
      columns.device() == data.device(),
      prefix,
      " columns device mismatch");
  TORCH_CHECK(
      values.device() == data.device(),
      prefix,
      " values device mismatch");
  TORCH_CHECK(rows.is_contiguous(), prefix, " rows must be contiguous");
  TORCH_CHECK(columns.is_contiguous(), prefix, " columns must be contiguous");
  TORCH_CHECK(values.is_contiguous(), prefix, " values must be contiguous");
  TORCH_CHECK(rows.dim() == 1, prefix, " rows must be one-dimensional");
  TORCH_CHECK(columns.dim() == 1, prefix, " columns must be one-dimensional");
  TORCH_CHECK(values.dim() == 1, prefix, " values must be one-dimensional");
  TORCH_CHECK(rows.scalar_type() == torch::kInt64, prefix, " rows must be int64");
  TORCH_CHECK(
      columns.scalar_type() == torch::kInt64,
      prefix,
      " columns must be int64");
  TORCH_CHECK(
      values.scalar_type() == data.scalar_type(),
      prefix,
      " values must have the data dtype");
  TORCH_CHECK(
      rows.numel() == columns.numel() && rows.numel() == values.numel(),
      prefix,
      " COO arrays must have equal length");
}

void check_cuda_centers(
    const torch::Tensor& centers,
    const torch::Tensor& data,
    const char* prefix) {
  TORCH_CHECK(
      centers.device() == data.device(),
      prefix,
      " centers device mismatch");
  TORCH_CHECK(
      centers.is_contiguous(),
      prefix,
      " centers must be contiguous");
  TORCH_CHECK(
      centers.dim() == 1,
      prefix,
      " centers must be one-dimensional");
  TORCH_CHECK(
      centers.scalar_type() == torch::kInt64,
      prefix,
      " centers must be int64");
}

void check_cuda_real_vector(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 1, name, " must be one-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64,
      name,
      " must use float32 or float64");
}

void check_cuda_real_matrix(
    const torch::Tensor& tensor,
    const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be on CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be two-dimensional");
  TORCH_CHECK(
      tensor.scalar_type() == torch::kFloat32 ||
          tensor.scalar_type() == torch::kFloat64,
      name,
      " must use float32 or float64");
}

template <typename Scalar>
__device__ inline Scalar conjugate_value(Scalar value) {
  return value;
}

template <>
__device__ inline c10::complex<float> conjugate_value(
    c10::complex<float> value) {
  return c10::complex<float>(value.real(), -value.imag());
}

template <>
__device__ inline c10::complex<double> conjugate_value(
    c10::complex<double> value) {
  return c10::complex<double>(value.real(), -value.imag());
}

__device__ inline float real_component(c10::complex<float> value) {
  return value.real();
}

__device__ inline double real_component(c10::complex<double> value) {
  return value.real();
}

template <typename Control, typename Scalar>
__device__ inline Control project_control_gradient(Scalar value) {
  if constexpr (
      std::is_same_v<Control, float> &&
      std::is_same_v<Scalar, c10::complex<float>>) {
    return value.real();
  } else if constexpr (
      std::is_same_v<Control, double> &&
      std::is_same_v<Scalar, c10::complex<double>>) {
    return value.real();
  } else {
    return value;
  }
}

template <typename Scalar>
__device__ inline double squared_magnitude(Scalar value) {
  const double converted = static_cast<double>(value);
  return converted * converted;
}

template <>
__device__ inline double squared_magnitude(
    c10::complex<float> value) {
  return static_cast<double>(value.real()) * value.real() +
      static_cast<double>(value.imag()) * value.imag();
}

template <>
__device__ inline double squared_magnitude(
    c10::complex<double> value) {
  return value.real() * value.real() + value.imag() * value.imag();
}

template <typename Scalar>
__device__ inline void atomic_add_value(
    Scalar* destination,
    Scalar value) {
  atomicAdd(destination, value);
}

template <>
__device__ inline void atomic_add_value(
    c10::complex<float>* destination,
    c10::complex<float> value) {
  auto* components = reinterpret_cast<float*>(destination);
  atomicAdd(components, value.real());
  atomicAdd(components + 1, value.imag());
}

template <typename Scalar>
__global__ void density_accumulate_kernel(
    const Scalar* edge_values,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_values) {
  const std::int64_t total = edge_count * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / channel_count;
    const std::int64_t channel = index - edge * channel_count;
    const std::int64_t center = centers[edge];
    CUDA_KERNEL_ASSERT(center >= 0 && center < atom_count);
    atomic_add_value(
        atomic_values + center * channel_count + channel,
        edge_values[index]);
  }
}

template <typename Scalar>
__global__ void edge_outer_accumulate_kernel(
    const Scalar* left,
    const Scalar* right,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t left_dimension,
    std::int64_t right_dimension,
    std::int64_t atom_count,
    Scalar* atomic_values) {
  const std::int64_t edge_width =
      left_dimension * right_dimension;
  const std::int64_t total = edge_count * edge_width;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / edge_width;
    const std::int64_t local = index - edge * edge_width;
    const std::int64_t left_index = local / right_dimension;
    const std::int64_t right_index =
        local - left_index * right_dimension;
    const std::int64_t center = centers[edge];
    CUDA_KERNEL_ASSERT(center >= 0 && center < atom_count);
    atomicAdd(
        atomic_values +
            (center * left_dimension + left_index) *
                right_dimension +
            right_index,
        left[edge * left_dimension + left_index] *
            right[edge * right_dimension + right_index]);
  }
}

template <typename Scalar>
__global__ void edge_outer_accumulate_adjoint_kernel(
    const Scalar* atomic_adjoint,
    const Scalar* left,
    const Scalar* right,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t left_dimension,
    std::int64_t right_dimension,
    std::int64_t atom_count,
    Scalar* left_adjoint,
    Scalar* right_adjoint) {
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t center = centers[edge];
    CUDA_KERNEL_ASSERT(center >= 0 && center < atom_count);
    const Scalar* atom_row =
        atomic_adjoint +
        center * left_dimension * right_dimension;
    const Scalar* left_row = left + edge * left_dimension;
    const Scalar* right_row = right + edge * right_dimension;
    Scalar* left_output = left_adjoint + edge * left_dimension;
    Scalar* right_output = right_adjoint + edge * right_dimension;
    for (std::int64_t left_index = 0;
         left_index < left_dimension;
         ++left_index) {
      Scalar value = Scalar(0);
      const Scalar* adjoint_row =
          atom_row + left_index * right_dimension;
      for (std::int64_t right_index = 0;
           right_index < right_dimension;
           ++right_index) {
        value += adjoint_row[right_index] * right_row[right_index];
      }
      left_output[left_index] = value;
    }
    for (std::int64_t right_index = 0;
         right_index < right_dimension;
         ++right_index) {
      Scalar value = Scalar(0);
      for (std::int64_t left_index = 0;
           left_index < left_dimension;
           ++left_index) {
        value +=
            atom_row[left_index * right_dimension + right_index] *
            left_row[left_index];
      }
      right_output[right_index] = value;
    }
  }
}

template <typename Scalar>
__global__ void edge_outer_accumulate_double_backward_kernel(
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
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t center = centers[edge];
    CUDA_KERNEL_ASSERT(center >= 0 && center < atom_count);
    const Scalar* atom_row =
        atomic_adjoint +
        center * left_dimension * right_dimension;
    Scalar* atomic_tangent_row =
        atomic_adjoint_tangent +
        center * left_dimension * right_dimension;
    const Scalar* left_row = left + edge * left_dimension;
    const Scalar* right_row = right + edge * right_dimension;
    const Scalar* left_tangent_row =
        left_adjoint_tangent + edge * left_dimension;
    const Scalar* right_tangent_row =
        right_adjoint_tangent + edge * right_dimension;
    Scalar* left_second_row =
        left_second_adjoint + edge * left_dimension;
    Scalar* right_second_row =
        right_second_adjoint + edge * right_dimension;
    for (std::int64_t left_index = 0;
         left_index < left_dimension;
         ++left_index) {
      Scalar value = Scalar(0);
      for (std::int64_t right_index = 0;
           right_index < right_dimension;
           ++right_index) {
        const std::int64_t local =
            left_index * right_dimension + right_index;
        atomicAdd(
            atomic_tangent_row + local,
            left_tangent_row[left_index] * right_row[right_index] +
                left_row[left_index] *
                    right_tangent_row[right_index]);
        value += atom_row[local] * right_tangent_row[right_index];
      }
      left_second_row[left_index] = value;
    }
    for (std::int64_t right_index = 0;
         right_index < right_dimension;
         ++right_index) {
      Scalar value = Scalar(0);
      for (std::int64_t left_index = 0;
           left_index < left_dimension;
           ++left_index) {
        value +=
            atom_row[left_index * right_dimension + right_index] *
            left_tangent_row[left_index];
      }
      right_second_row[right_index] = value;
    }
  }
}

template <typename Scalar>
__device__ void softmax_gaussian_statistics(
    Scalar distance,
    Scalar cutoff,
    const Scalar* filter_centers,
    Scalar filter_width,
    std::int64_t role_count,
    Scalar* maximum,
    Scalar* denominator,
    Scalar* mean_log_derivative,
    Scalar* derivative_log_moment) {
  const Scalar scaled = distance / cutoff;
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  *maximum = -std::numeric_limits<Scalar>::infinity();
  for (std::int64_t role = 0; role < role_count; ++role) {
    const Scalar delta =
        (scaled - filter_centers[role]) / filter_width;
    *maximum = max(*maximum, Scalar(-0.5) * delta * delta);
  }
  *denominator = Scalar(0);
  for (std::int64_t role = 0; role < role_count; ++role) {
    const Scalar delta =
        (scaled - filter_centers[role]) / filter_width;
    *denominator += exp(Scalar(-0.5) * delta * delta - *maximum);
  }
  *mean_log_derivative = Scalar(0);
  for (std::int64_t role = 0; role < role_count; ++role) {
    const Scalar delta =
        (scaled - filter_centers[role]) / filter_width;
    const Scalar weight =
        exp(Scalar(-0.5) * delta * delta - *maximum) / *denominator;
    const Scalar log_derivative =
        -(scaled - filter_centers[role]) *
        inverse_width_squared / cutoff;
    *mean_log_derivative += weight * log_derivative;
  }
  if (derivative_log_moment == nullptr) {
    return;
  }
  *derivative_log_moment = Scalar(0);
  for (std::int64_t role = 0; role < role_count; ++role) {
    const Scalar delta =
        (scaled - filter_centers[role]) / filter_width;
    const Scalar weight =
        exp(Scalar(-0.5) * delta * delta - *maximum) / *denominator;
    const Scalar log_derivative =
        -(scaled - filter_centers[role]) *
        inverse_width_squared / cutoff;
    *derivative_log_moment +=
        weight * (log_derivative - *mean_log_derivative) *
        log_derivative;
  }
}

template <typename Scalar>
__global__ void softmax_gaussian_role_density_kernel(
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
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    CUDA_KERNEL_ASSERT(cutoffs[edge] > Scalar(0));
    CUDA_KERNEL_ASSERT(cutoffs[edge] > Scalar(0));
    Scalar maximum;
    Scalar denominator;
    Scalar mean_log_derivative;
    softmax_gaussian_statistics(
        distances[edge], cutoffs[edge], filter_centers, filter_width,
        role_count, &maximum, &denominator, &mean_log_derivative,
        static_cast<Scalar*>(nullptr));
    const Scalar scaled = distances[edge] / cutoffs[edge];
    const Scalar* edge_row = edge_values + edge * channel_count;
    Scalar* atom_row =
        atomic_values + atom * role_count * channel_count;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          exp(Scalar(-0.5) * delta * delta - maximum) / denominator;
      for (std::int64_t channel = 0; channel < channel_count; ++channel) {
        atomicAdd(
            atom_row + role * channel_count + channel,
            weight * edge_row[channel]);
      }
    }
  }
}

template <typename Scalar>
__global__ void softmax_gaussian_role_density_adjoint_kernel(
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
    std::int64_t atom_count,
    Scalar* distance_adjoint,
    Scalar* edge_adjoint) {
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    CUDA_KERNEL_ASSERT(cutoffs[edge] > Scalar(0));
    Scalar maximum;
    Scalar denominator;
    Scalar mean_log_derivative;
    softmax_gaussian_statistics(
        distances[edge], cutoffs[edge], filter_centers, filter_width,
        role_count, &maximum, &denominator, &mean_log_derivative,
        static_cast<Scalar*>(nullptr));
    const Scalar scaled = distances[edge] / cutoffs[edge];
    const Scalar inverse_width_squared =
        Scalar(1) / (filter_width * filter_width);
    const Scalar* atom_row =
        atomic_adjoint + atom * role_count * channel_count;
    const Scalar* edge_row = edge_values + edge * channel_count;
    Scalar* edge_output = edge_adjoint + edge * channel_count;
    for (std::int64_t channel = 0; channel < channel_count; ++channel) {
      edge_output[channel] = Scalar(0);
    }
    Scalar distance_output = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          exp(Scalar(-0.5) * delta * delta - maximum) / denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoffs[edge];
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
__global__ void softmax_gaussian_role_density_double_backward_kernel(
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
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    CUDA_KERNEL_ASSERT(cutoffs[edge] > Scalar(0));
    Scalar maximum;
    Scalar denominator;
    Scalar mean_log_derivative;
    Scalar derivative_log_moment;
    softmax_gaussian_statistics(
        distances[edge], cutoffs[edge], filter_centers, filter_width,
        role_count, &maximum, &denominator, &mean_log_derivative,
        &derivative_log_moment);
    const Scalar scaled = distances[edge] / cutoffs[edge];
    const Scalar inverse_width_squared =
        Scalar(1) / (filter_width * filter_width);
    const Scalar* atom_row =
        atomic_adjoint + atom * role_count * channel_count;
    Scalar* atom_tangent_row =
        atomic_adjoint_tangent + atom * role_count * channel_count;
    const Scalar* edge_row = edge_values + edge * channel_count;
    const Scalar* edge_tangent =
        edge_adjoint_tangent + edge * channel_count;
    Scalar* edge_second = edge_second_adjoint + edge * channel_count;
    for (std::int64_t channel = 0; channel < channel_count; ++channel) {
      edge_second[channel] = Scalar(0);
    }
    Scalar distance_second = Scalar(0);
    const Scalar distance_tangent = distance_adjoint_tangent[edge];
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          exp(Scalar(-0.5) * delta * delta - maximum) / denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoffs[edge];
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
        atomicAdd(
            tangent_row + channel,
            distance_tangent * derivative * edge_row[channel] +
                weight * edge_tangent[channel]);
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
__global__ void scheduled_role_density_kernel(
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
  for (std::int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    Scalar maximum;
    Scalar denominator;
    Scalar mean_log_derivative;
    softmax_gaussian_statistics(
        distances[edge], cutoffs[edge], filter_centers, filter_width,
        role_count, &maximum, &denominator, &mean_log_derivative,
        static_cast<Scalar*>(nullptr));
    const Scalar scaled = distances[edge] / cutoffs[edge];
    Scalar* atom_row =
        atomic_values + atom * role_count * output_width;
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          exp(Scalar(-0.5) * delta * delta - maximum) / denominator;
      Scalar* output_row = atom_row + role * output_width;
      for (std::int64_t channel = 0;
           channel < channel_count;
           ++channel) {
        if (channel_types[channel] >= 0 &&
            channel_types[channel] != edge_types[edge]) {
          continue;
        }
        CUDA_KERNEL_ASSERT(
            channel_radial_indices[channel] >= 0 &&
            channel_radial_indices[channel] < radial_width);
        CUDA_KERNEL_ASSERT(
            channel_angular_indices[channel] >= 0 &&
            channel_angular_indices[channel] < angular_width);
        const Scalar value = channel_scales[channel] *
            radial_values[
                edge * radial_width + channel_radial_indices[channel]] *
            angular_values[
                edge * angular_width + channel_angular_indices[channel]];
        atomicAdd(output_row + channel, weight * value);
      }
      atomicAdd(
          output_row + channel_count,
          weight * soft_weights[edge]);
    }
  }
}

template <typename Scalar>
__global__ void scheduled_role_density_adjoint_kernel(
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
    std::int64_t atom_count,
    Scalar* radial_adjoint,
    Scalar* angular_adjoint,
    Scalar* distance_adjoint,
    Scalar* soft_adjoint) {
  const std::int64_t output_width = channel_count + 1;
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  for (std::int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    CUDA_KERNEL_ASSERT(cutoffs[edge] > Scalar(0));
    Scalar maximum;
    Scalar denominator;
    Scalar mean_log_derivative;
    softmax_gaussian_statistics(
        distances[edge], cutoffs[edge], filter_centers, filter_width,
        role_count, &maximum, &denominator, &mean_log_derivative,
        static_cast<Scalar*>(nullptr));
    const Scalar scaled = distances[edge] / cutoffs[edge];
    const Scalar* atom_row =
        atomic_adjoint + atom * role_count * output_width;
    Scalar distance_output = Scalar(0);
    Scalar soft_output = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          exp(Scalar(-0.5) * delta * delta - maximum) / denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoffs[edge];
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
        CUDA_KERNEL_ASSERT(
            channel_radial_indices[channel] >= 0 &&
            channel_radial_indices[channel] < radial_width);
        CUDA_KERNEL_ASSERT(
            channel_angular_indices[channel] >= 0 &&
            channel_angular_indices[channel] < angular_width);
        const std::int64_t radial_offset =
            edge * radial_width + channel_radial_indices[channel];
        const std::int64_t angular_offset =
            edge * angular_width + channel_angular_indices[channel];
        const Scalar radial = radial_values[radial_offset];
        const Scalar angular = angular_values[angular_offset];
        const Scalar scale = channel_scales[channel];
        const Scalar adjoint = adjoint_row[channel];
        radial_adjoint[radial_offset] +=
            weight * adjoint * scale * angular;
        angular_adjoint[angular_offset] +=
            weight * adjoint * scale * radial;
        role_adjoint += adjoint * scale * radial * angular;
      }
      distance_output += derivative * role_adjoint;
    }
    distance_adjoint[edge] = distance_output;
    soft_adjoint[edge] = soft_output;
  }
}

template <typename Scalar>
__global__ void scheduled_role_density_double_backward_kernel(
    const Scalar* atomic_adjoint,
    const Scalar* radial_values,
    const Scalar* angular_values,
    const Scalar* distances,
    const Scalar* cutoffs,
    const Scalar* filter_centers,
    Scalar filter_width,
    const Scalar* soft_weights,
    const Scalar* radial_tangent,
    const Scalar* angular_tangent,
    const Scalar* distance_tangent,
    const Scalar* soft_tangent,
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
    Scalar* atomic_tangent,
    Scalar* radial_second,
    Scalar* angular_second,
    Scalar* distance_second,
    Scalar* soft_second) {
  const std::int64_t output_width = channel_count + 1;
  const Scalar inverse_width_squared =
      Scalar(1) / (filter_width * filter_width);
  for (std::int64_t edge = blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    CUDA_KERNEL_ASSERT(cutoffs[edge] > Scalar(0));
    Scalar maximum;
    Scalar denominator;
    Scalar mean_log_derivative;
    Scalar derivative_log_moment;
    softmax_gaussian_statistics(
        distances[edge], cutoffs[edge], filter_centers, filter_width,
        role_count, &maximum, &denominator, &mean_log_derivative,
        &derivative_log_moment);
    const Scalar scaled = distances[edge] / cutoffs[edge];
    const Scalar* atom_row =
        atomic_adjoint + atom * role_count * output_width;
    Scalar* atom_tangent_row =
        atomic_tangent + atom * role_count * output_width;
    Scalar distance_output = Scalar(0);
    Scalar soft_output = Scalar(0);
    const Scalar distance_direction = distance_tangent[edge];
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Scalar delta =
          (scaled - filter_centers[role]) / filter_width;
      const Scalar weight =
          exp(Scalar(-0.5) * delta * delta - maximum) / denominator;
      const Scalar log_derivative =
          -(scaled - filter_centers[role]) *
          inverse_width_squared / cutoffs[edge];
      const Scalar derivative =
          weight * (log_derivative - mean_log_derivative);
      const Scalar second_derivative = weight *
          ((log_derivative - mean_log_derivative) *
               (log_derivative - mean_log_derivative) -
           derivative_log_moment);
      const Scalar* adjoint_row = atom_row + role * output_width;
      Scalar* tangent_row = atom_tangent_row + role * output_width;
      atomicAdd(
          tangent_row + channel_count,
          weight * soft_tangent[edge] +
              derivative * distance_direction * soft_weights[edge]);
      Scalar role_primal =
          adjoint_row[channel_count] * soft_weights[edge];
      Scalar role_tangent =
          adjoint_row[channel_count] * soft_tangent[edge];
      soft_output += derivative * distance_direction *
          adjoint_row[channel_count];
      for (std::int64_t channel = 0;
           channel < channel_count;
           ++channel) {
        if (channel_types[channel] >= 0 &&
            channel_types[channel] != edge_types[edge]) {
          continue;
        }
        CUDA_KERNEL_ASSERT(
            channel_radial_indices[channel] >= 0 &&
            channel_radial_indices[channel] < radial_width);
        CUDA_KERNEL_ASSERT(
            channel_angular_indices[channel] >= 0 &&
            channel_angular_indices[channel] < angular_width);
        const std::int64_t radial_offset =
            edge * radial_width + channel_radial_indices[channel];
        const std::int64_t angular_offset =
            edge * angular_width + channel_angular_indices[channel];
        const Scalar radial = radial_values[radial_offset];
        const Scalar angular = angular_values[angular_offset];
        const Scalar radial_direction = radial_tangent[radial_offset];
        const Scalar angular_direction = angular_tangent[angular_offset];
        const Scalar scale = channel_scales[channel];
        const Scalar value = scale * radial * angular;
        const Scalar value_tangent = scale * (
            radial_direction * angular + radial * angular_direction);
        atomicAdd(
            tangent_row + channel,
            weight * value_tangent +
                derivative * distance_direction * value);
        const Scalar adjoint = adjoint_row[channel];
        radial_second[radial_offset] += adjoint * scale * (
            weight * angular_direction +
            derivative * distance_direction * angular);
        angular_second[angular_offset] += adjoint * scale * (
            weight * radial_direction +
            derivative * distance_direction * radial);
        role_primal += adjoint * value;
        role_tangent += adjoint * value_tangent;
      }
      distance_output +=
          second_derivative * distance_direction * role_primal +
          derivative * role_tangent;
    }
    distance_second[edge] = distance_output;
    soft_second[edge] = soft_output;
  }
}

template <typename Scalar>
__global__ void carrier_gated_scatter_kernel(
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t target_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  const std::int64_t total = edge_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / feature_count;
    const std::int64_t feature =
        index - edge * feature_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < target_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    atomic_add_value(
        target_values + target * feature_count + feature,
        node_values[source * feature_count + feature] *
            edge_gates[edge * channel_count + channel]);
  }
}

template <typename Scalar>
__global__ void carrier_gated_scatter_adjoint_kernel(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Scalar* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t target_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Scalar* gate_adjoint) {
  const std::int64_t total = edge_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / feature_count;
    const std::int64_t feature =
        index - edge * feature_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < target_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    const Scalar gradient =
        target_adjoint[target * feature_count + feature];
    atomic_add_value(
        node_adjoint + source * feature_count + feature,
        gradient *
            conjugate_value(
                edge_gates[edge * channel_count + channel]));
    atomic_add_value(
        gate_adjoint + edge * channel_count + channel,
        gradient *
            conjugate_value(
                node_values[source * feature_count + feature]));
  }
}

template <typename Scalar>
__global__ void carrier_gated_scatter_double_backward_kernel(
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
  const std::int64_t total = edge_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / feature_count;
    const std::int64_t feature =
        index - edge * feature_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < target_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    const Scalar target_gradient =
        target_adjoint[target * feature_count + feature];
    const Scalar node_tangent =
        node_adjoint_tangent[source * feature_count + feature];
    const Scalar gate_tangent =
        gate_adjoint_tangent[edge * channel_count + channel];
    atomic_add_value(
        target_adjoint_tangent +
            target * feature_count + feature,
        node_tangent *
                edge_gates[edge * channel_count + channel] +
            gate_tangent *
                node_values[source * feature_count + feature]);
    atomic_add_value(
        node_second_adjoint +
            source * feature_count + feature,
        conjugate_value(gate_tangent) * target_gradient);
    atomic_add_value(
        gate_second_adjoint +
            edge * channel_count + channel,
        conjugate_value(node_tangent) * target_gradient);
  }
}

template <typename Scalar, typename Real>
__global__ void carrier_gated_scatter_real_gates_kernel(
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t target_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  const std::int64_t total = edge_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / feature_count;
    const std::int64_t feature = index - edge * feature_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < target_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    atomic_add_value(
        target_values + target * feature_count + feature,
        node_values[source * feature_count + feature] *
            edge_gates[edge * channel_count + channel]);
  }
}

template <typename Scalar, typename Real>
__global__ void carrier_gated_scatter_real_gates_adjoint_kernel(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const Real* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* feature_channels,
    std::int64_t edge_count,
    std::int64_t node_count,
    std::int64_t target_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint,
    Real* gate_adjoint) {
  const std::int64_t total = edge_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / feature_count;
    const std::int64_t feature = index - edge * feature_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < target_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    const Scalar gradient =
        target_adjoint[target * feature_count + feature];
    atomic_add_value(
        node_adjoint + source * feature_count + feature,
        gradient * edge_gates[edge * channel_count + channel]);
    atomicAdd(
        gate_adjoint + edge * channel_count + channel,
        real_component(
            gradient * conjugate_value(
                node_values[source * feature_count + feature])));
  }
}

template <typename Scalar, typename Real>
__global__ void carrier_gated_scatter_real_gates_double_backward_kernel(
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
  const std::int64_t total = edge_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / feature_count;
    const std::int64_t feature = index - edge * feature_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < target_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    const Scalar target_gradient =
        target_adjoint[target * feature_count + feature];
    const Scalar node_tangent =
        node_adjoint_tangent[source * feature_count + feature];
    const Real gate_tangent =
        gate_adjoint_tangent[edge * channel_count + channel];
    atomic_add_value(
        target_adjoint_tangent + target * feature_count + feature,
        node_tangent * edge_gates[edge * channel_count + channel] +
            gate_tangent *
                node_values[source * feature_count + feature]);
    atomic_add_value(
        node_second_adjoint + source * feature_count + feature,
        gate_tangent * target_gradient);
    atomicAdd(
        gate_second_adjoint + edge * channel_count + channel,
        real_component(
            conjugate_value(node_tangent) * target_gradient));
  }
}

template <typename Scalar, typename Gate>
__device__ inline Scalar multiply_conjugate_gate(
    Scalar value,
    Gate gate) {
  if constexpr (std::is_same<Scalar, Gate>::value) {
    return value * conjugate_value(gate);
  } else {
    return value * gate;
  }
}

template <typename Scalar, typename Gate>
__device__ inline Gate carrier_gate_inner_product(
    Scalar left,
    Scalar right) {
  if constexpr (std::is_same<Scalar, Gate>::value) {
    return left * conjugate_value(right);
  } else {
    return real_component(left * conjugate_value(right));
  }
}

template <typename Scalar, typename Gate>
__global__ void carrier_segmented_residual_scatter_forward_kernel(
    const Scalar* node_values,
    const Gate* edge_gates,
    const std::int64_t* edge_sources,
    const std::int64_t* target_offsets,
    const std::int64_t* feature_channels,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_values) {
  const std::int64_t total = node_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t target = index / feature_count;
    const std::int64_t feature = index - target * feature_count;
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    Scalar value = node_values[index];
    const std::int64_t edge_start = target_offsets[target];
    const std::int64_t edge_stop = target_offsets[target + 1];
    for (std::int64_t edge = edge_start;
         edge < edge_stop;
         ++edge) {
      const std::int64_t source = edge_sources[edge];
      CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
      value +=
          node_values[source * feature_count + feature] *
          edge_gates[edge * channel_count + channel];
    }
    target_values[index] = value;
  }
}

template <typename Scalar, typename Gate>
__global__ void carrier_segmented_residual_scatter_node_adjoint_kernel(
    const Scalar* target_adjoint,
    const Gate* edge_gates,
    const std::int64_t* edge_targets,
    const std::int64_t* source_offsets,
    const std::int64_t* source_edges,
    const std::int64_t* feature_channels,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_adjoint) {
  const std::int64_t total = node_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t source = index / feature_count;
    const std::int64_t feature = index - source * feature_count;
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    Scalar value = target_adjoint[index];
    const std::int64_t edge_start = source_offsets[source];
    const std::int64_t edge_stop = source_offsets[source + 1];
    for (std::int64_t entry = edge_start;
         entry < edge_stop;
         ++entry) {
      const std::int64_t edge = source_edges[entry];
      CUDA_KERNEL_ASSERT(edge >= 0 && edge < edge_count);
      const std::int64_t target = edge_targets[edge];
      CUDA_KERNEL_ASSERT(target >= 0 && target < node_count);
      value += multiply_conjugate_gate<Scalar, Gate>(
          target_adjoint[target * feature_count + feature],
          edge_gates[edge * channel_count + channel]);
    }
    node_adjoint[index] = value;
  }
}

template <typename Scalar, typename Gate>
__global__ void carrier_segmented_residual_scatter_gate_adjoint_kernel(
    const Scalar* target_adjoint,
    const Scalar* node_values,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* channel_offsets,
    const std::int64_t* channel_features,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Gate* gate_adjoint) {
  const std::int64_t total = edge_count * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / channel_count;
    const std::int64_t channel = index - edge * channel_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < node_count);
    Gate value = Gate(0);
    for (std::int64_t entry = channel_offsets[channel];
         entry < channel_offsets[channel + 1];
         ++entry) {
      const std::int64_t feature = channel_features[entry];
      CUDA_KERNEL_ASSERT(feature >= 0 && feature < feature_count);
      value += carrier_gate_inner_product<Scalar, Gate>(
          target_adjoint[target * feature_count + feature],
          node_values[source * feature_count + feature]);
    }
    gate_adjoint[index] = value;
  }
}

template <typename Scalar, typename Gate>
__global__ void carrier_segmented_residual_scatter_target_tangent_kernel(
    const Scalar* node_values,
    const Gate* edge_gates,
    const Scalar* node_adjoint_tangent,
    const Gate* gate_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* target_offsets,
    const std::int64_t* feature_channels,
    std::int64_t node_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* target_adjoint_tangent) {
  const std::int64_t total = node_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t target = index / feature_count;
    const std::int64_t feature = index - target * feature_count;
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    Scalar value = node_adjoint_tangent[index];
    for (std::int64_t edge = target_offsets[target];
         edge < target_offsets[target + 1];
         ++edge) {
      const std::int64_t source = edge_sources[edge];
      CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
      value +=
          node_adjoint_tangent[source * feature_count + feature] *
              edge_gates[edge * channel_count + channel] +
          gate_adjoint_tangent[edge * channel_count + channel] *
              node_values[source * feature_count + feature];
    }
    target_adjoint_tangent[index] = value;
  }
}

template <typename Scalar, typename Gate>
__global__ void carrier_segmented_residual_scatter_node_second_kernel(
    const Scalar* target_adjoint,
    const Gate* gate_adjoint_tangent,
    const std::int64_t* edge_targets,
    const std::int64_t* source_offsets,
    const std::int64_t* source_edges,
    const std::int64_t* feature_channels,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Scalar* node_second_adjoint) {
  const std::int64_t total = node_count * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t source = index / feature_count;
    const std::int64_t feature = index - source * feature_count;
    const std::int64_t channel = feature_channels[feature];
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
    Scalar value = Scalar(0);
    for (std::int64_t entry = source_offsets[source];
         entry < source_offsets[source + 1];
         ++entry) {
      const std::int64_t edge = source_edges[entry];
      CUDA_KERNEL_ASSERT(edge >= 0 && edge < edge_count);
      const std::int64_t target = edge_targets[edge];
      CUDA_KERNEL_ASSERT(target >= 0 && target < node_count);
      value += multiply_conjugate_gate<Scalar, Gate>(
          target_adjoint[target * feature_count + feature],
          gate_adjoint_tangent[edge * channel_count + channel]);
    }
    node_second_adjoint[index] = value;
  }
}

template <typename Scalar, typename Gate>
__global__ void carrier_segmented_residual_scatter_gate_second_kernel(
    const Scalar* target_adjoint,
    const Scalar* node_adjoint_tangent,
    const std::int64_t* edge_sources,
    const std::int64_t* edge_targets,
    const std::int64_t* channel_offsets,
    const std::int64_t* channel_features,
    std::int64_t node_count,
    std::int64_t edge_count,
    std::int64_t feature_count,
    std::int64_t channel_count,
    Gate* gate_second_adjoint) {
  const std::int64_t total = edge_count * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / channel_count;
    const std::int64_t channel = index - edge * channel_count;
    const std::int64_t source = edge_sources[edge];
    const std::int64_t target = edge_targets[edge];
    CUDA_KERNEL_ASSERT(source >= 0 && source < node_count);
    CUDA_KERNEL_ASSERT(target >= 0 && target < node_count);
    Gate value = Gate(0);
    for (std::int64_t entry = channel_offsets[channel];
         entry < channel_offsets[channel + 1];
         ++entry) {
      const std::int64_t feature = channel_features[entry];
      CUDA_KERNEL_ASSERT(feature >= 0 && feature < feature_count);
      value += carrier_gate_inner_product<Scalar, Gate>(
          target_adjoint[target * feature_count + feature],
          node_adjoint_tangent[source * feature_count + feature]);
    }
    gate_second_adjoint[index] = value;
  }
}

__device__ __forceinline__ std::int64_t carrier_offset_block(
    const std::int64_t* offsets,
    std::int64_t block_count,
    std::int64_t index) {
  if (block_count <= 8) {
    for (std::int64_t block = 0; block < block_count; ++block) {
      if (index < offsets[block + 1]) {
        return block;
      }
    }
    return block_count;
  }
  std::int64_t lower = 0;
  std::int64_t upper = block_count;
  while (lower < upper) {
    const std::int64_t middle = lower + (upper - lower) / 2;
    if (index < offsets[middle + 1]) {
      upper = middle;
    } else {
      lower = middle + 1;
    }
  }
  return lower;
}

template <typename Scalar>
__global__ void source_arena_gather_kernel(
    const Scalar* producer,
    const std::int64_t* gather_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t output_width,
    Scalar* output) {
  const std::int64_t work = batch_size * output_width;
  for (std::int64_t linear =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < work;
       linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = linear / output_width;
    const std::int64_t output_index = linear % output_width;
    const std::int64_t center_type = center_types[output_index];
    if (center_type >= 0 && center_type != atom_types[batch]) {
      output[linear] = Scalar{};
    } else {
      output[linear] = producer[
          batch * producer_width + gather_indices[output_index]];
    }
  }
}

template <typename Scalar>
__global__ void source_arena_gather_adjoint_kernel(
    const Scalar* output_adjoint,
    const std::int64_t* reverse_offsets,
    const std::int64_t* reverse_output_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t output_width,
    Scalar* producer_adjoint) {
  const std::int64_t work = batch_size * producer_width;
  for (std::int64_t linear =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       linear < work;
       linear += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = linear / producer_width;
    const std::int64_t producer_index = linear % producer_width;
    const std::int64_t atom_type = atom_types[batch];
    Scalar value{};
    for (std::int64_t entry = reverse_offsets[producer_index];
         entry < reverse_offsets[producer_index + 1];
         ++entry) {
      const std::int64_t output_index = reverse_output_indices[entry];
      const std::int64_t center_type = center_types[output_index];
      if (center_type < 0 || center_type == atom_type) {
        value += output_adjoint[batch * output_width + output_index];
      }
    }
    producer_adjoint[linear] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void source_arena_channel_transform_kernel(
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
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t total = batch_size * output_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_width;
    const std::int64_t output_feature = index - batch * output_width;
    const std::int64_t block = carrier_offset_block(
        output_feature_offsets, block_count, output_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (output_feature_offsets[block + 1] - output_start) /
        output_channels;
    const std::int64_t local_output = output_feature - output_start;
    const std::int64_t output_channel = local_output / inner_width;
    const std::int64_t inner =
        local_output - output_channel * inner_width;
    const std::int64_t center_type = center_types[input_start];
    if (center_type >= 0 && center_type != atom_types[batch]) {
      output[index] = Scalar{};
      continue;
    }
    const std::int64_t map_start = map_offsets[block];
    if (map_offsets[block + 1] == map_start) {
      const std::int64_t logical_input =
          input_start + output_channel * inner_width + inner;
      output[index] = producer[
          batch * producer_width + gather_indices[logical_input]];
      continue;
    }
    Scalar value{};
    for (std::int64_t input_channel = 0;
         input_channel < input_channels;
         ++input_channel) {
      const std::int64_t logical_input =
          input_start + input_channel * inner_width + inner;
      value += channel_maps[
          map_start + output_channel * input_channels + input_channel] *
          producer[
              batch * producer_width + gather_indices[logical_input]];
    }
    output[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void source_arena_channel_transform_producer_adjoint_kernel(
    const Scalar* output_adjoint,
    const Control* channel_maps,
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
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* producer_adjoint) {
  const std::int64_t total = batch_size * producer_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / producer_width;
    const std::int64_t producer_index = index - batch * producer_width;
    const std::int64_t atom_type = atom_types[batch];
    Scalar value{};
    for (std::int64_t entry = reverse_offsets[producer_index];
         entry < reverse_offsets[producer_index + 1];
         ++entry) {
      const std::int64_t logical_input = reverse_output_indices[entry];
      const std::int64_t center_type = center_types[logical_input];
      if (center_type >= 0 && center_type != atom_type) {
        continue;
      }
      const std::int64_t block = carrier_offset_block(
          input_feature_offsets, block_count, logical_input);
      CUDA_KERNEL_ASSERT(block < block_count);
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) /
          input_channels;
      const std::int64_t local_input = logical_input - input_start;
      const std::int64_t input_channel = local_input / inner_width;
      const std::int64_t inner =
          local_input - input_channel * inner_width;
      const std::int64_t map_start = map_offsets[block];
      if (map_offsets[block + 1] == map_start) {
        value += output_adjoint[
            batch * output_width + output_start +
            input_channel * inner_width + inner];
        continue;
      }
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        value += output_adjoint[
            batch * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            channel_maps[
                map_start + output_channel * input_channels +
                input_channel]);
      }
    }
    producer_adjoint[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void source_arena_channel_transform_maps_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* producer,
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
    std::int64_t output_width,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* maps_adjoint) {
  for (std::int64_t map_index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) /
        input_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel = local_map / input_channels;
    const std::int64_t input_channel =
        local_map - output_channel * input_channels;
    CUDA_KERNEL_ASSERT(output_channel < output_channels);
    const std::int64_t center_type = center_types[input_start];
    Scalar value{};
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      if (center_type >= 0 && center_type != atom_types[batch]) {
        continue;
      }
      for (std::int64_t inner = 0; inner < inner_width; ++inner) {
        const std::int64_t logical_input =
            input_start + input_channel * inner_width + inner;
        value += output_adjoint[
            batch * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            producer[
                batch * producer_width + gather_indices[logical_input]]);
      }
    }
    maps_adjoint[map_index] = project_control_gradient<Control>(value);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void source_arena_channel_transform_output_tangent_kernel(
    const Scalar* producer,
    const Control* channel_maps,
    const Scalar* producer_adjoint_tangent,
    const Control* maps_adjoint_tangent,
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
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* output_tangent) {
  const std::int64_t total = batch_size * output_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_width;
    const std::int64_t output_feature = index - batch * output_width;
    const std::int64_t block = carrier_offset_block(
        output_feature_offsets, block_count, output_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (output_feature_offsets[block + 1] - output_start) /
        output_channels;
    const std::int64_t local_output = output_feature - output_start;
    const std::int64_t output_channel = local_output / inner_width;
    const std::int64_t inner =
        local_output - output_channel * inner_width;
    const std::int64_t center_type = center_types[input_start];
    if (center_type >= 0 && center_type != atom_types[batch]) {
      output_tangent[index] = Scalar{};
      continue;
    }
    const std::int64_t map_start = map_offsets[block];
    if (map_offsets[block + 1] == map_start) {
      const std::int64_t logical_input =
          input_start + output_channel * inner_width + inner;
      output_tangent[index] = producer_adjoint_tangent[
          batch * producer_width + gather_indices[logical_input]];
      continue;
    }
    Scalar value{};
    for (std::int64_t input_channel = 0;
         input_channel < input_channels;
         ++input_channel) {
      const std::int64_t logical_input =
          input_start + input_channel * inner_width + inner;
      const std::int64_t producer_index = gather_indices[logical_input];
      const std::int64_t map_index =
          map_start + output_channel * input_channels + input_channel;
      value += channel_maps[map_index] * producer_adjoint_tangent[
          batch * producer_width + producer_index] +
          maps_adjoint_tangent[map_index] * producer[
          batch * producer_width + producer_index];
    }
    output_tangent[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void source_arena_channel_transform_producer_second_kernel(
    const Scalar* output_adjoint,
    const Control* maps_adjoint_tangent,
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
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* producer_second) {
  const std::int64_t total = batch_size * producer_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / producer_width;
    const std::int64_t producer_index = index - batch * producer_width;
    const std::int64_t atom_type = atom_types[batch];
    Scalar value{};
    for (std::int64_t entry = reverse_offsets[producer_index];
         entry < reverse_offsets[producer_index + 1];
         ++entry) {
      const std::int64_t logical_input = reverse_output_indices[entry];
      const std::int64_t center_type = center_types[logical_input];
      if (center_type >= 0 && center_type != atom_type) {
        continue;
      }
      const std::int64_t block = carrier_offset_block(
          input_feature_offsets, block_count, logical_input);
      CUDA_KERNEL_ASSERT(block < block_count);
      const std::int64_t map_start = map_offsets[block];
      if (map_offsets[block + 1] == map_start) {
        continue;
      }
      const std::int64_t input_start = input_feature_offsets[block];
      const std::int64_t output_start = output_feature_offsets[block];
      const std::int64_t input_channels =
          input_channel_offsets[block + 1] - input_channel_offsets[block];
      const std::int64_t output_channels =
          output_channel_offsets[block + 1] - output_channel_offsets[block];
      const std::int64_t inner_width =
          (input_feature_offsets[block + 1] - input_start) /
          input_channels;
      const std::int64_t local_input = logical_input - input_start;
      const std::int64_t input_channel = local_input / inner_width;
      const std::int64_t inner =
          local_input - input_channel * inner_width;
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        value += output_adjoint[
            batch * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            maps_adjoint_tangent[
                map_start + output_channel * input_channels +
                input_channel]);
      }
    }
    producer_second[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void source_arena_channel_transform_maps_second_kernel(
    const Scalar* output_adjoint,
    const Scalar* producer_adjoint_tangent,
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
    std::int64_t output_width,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* maps_second) {
  for (std::int64_t map_index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) /
        input_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel = local_map / input_channels;
    const std::int64_t input_channel =
        local_map - output_channel * input_channels;
    CUDA_KERNEL_ASSERT(output_channel < output_channels);
    const std::int64_t center_type = center_types[input_start];
    Scalar value{};
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      if (center_type >= 0 && center_type != atom_types[batch]) {
        continue;
      }
      for (std::int64_t inner = 0; inner < inner_width; ++inner) {
        const std::int64_t logical_input =
            input_start + input_channel * inner_width + inner;
        value += output_adjoint[
            batch * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            producer_adjoint_tangent[
                batch * producer_width + gather_indices[logical_input]]);
      }
    }
    maps_second[map_index] = project_control_gradient<Control>(value);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_update_kernel(
    const Scalar* values,
    const Control* gates,
    const Control* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t total = batch_size * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / feature_count;
    const std::int64_t feature =
        index - batch * feature_count;
    const std::int64_t block =
        carrier_offset_block(feature_offsets, block_count, feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t local_feature = feature - feature_start;
    const std::int64_t output_channel = local_feature / inner_width;
    const std::int64_t inner =
        local_feature - output_channel * inner_width;
    Scalar mixed = Scalar(0);
    for (std::int64_t input_channel = 0;
         input_channel < local_channels;
         ++input_channel) {
      mixed +=
          channel_maps[
              map_offsets[block] +
              output_channel * local_channels +
              input_channel] *
          values[
              batch * feature_count +
              feature_start +
              input_channel * inner_width +
              inner];
    }
    output[index] =
        values[index] +
        gates[
            batch * channel_count +
            channel_start +
            output_channel] *
            mixed;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_transform_kernel(
    const Scalar* values,
    const Control* channel_maps,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* output) {
  const std::int64_t total = batch_size * output_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_width;
    const std::int64_t output_feature = index - batch * output_width;
    const std::int64_t block = carrier_offset_block(
        output_feature_offsets, block_count, output_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (output_feature_offsets[block + 1] - output_start) /
        output_channels;
    const std::int64_t local_output = output_feature - output_start;
    const std::int64_t output_channel = local_output / inner_width;
    const std::int64_t inner =
        local_output - output_channel * inner_width;
    Scalar value = Scalar(0);
    for (std::int64_t input_channel = 0;
         input_channel < input_channels;
         ++input_channel) {
      value += channel_maps[
          map_offsets[block] + output_channel * input_channels +
          input_channel] * values[
          batch * input_width + input_start +
          input_channel * inner_width + inner];
    }
    output[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_transform_values_adjoint_kernel(
    const Scalar* output_adjoint,
    const Control* channel_maps,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* values_adjoint) {
  const std::int64_t total = batch_size * input_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / input_width;
    const std::int64_t input_feature = index - batch * input_width;
    const std::int64_t block = carrier_offset_block(
        input_feature_offsets, block_count, input_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) / input_channels;
    const std::int64_t local_input = input_feature - input_start;
    const std::int64_t input_channel = local_input / inner_width;
    const std::int64_t inner = local_input - input_channel * inner_width;
    Scalar value = Scalar(0);
    for (std::int64_t output_channel = 0;
         output_channel < output_channels;
         ++output_channel) {
      value += output_adjoint[
          batch * output_width + output_start +
          output_channel * inner_width + inner] * conjugate_value(
          channel_maps[
              map_offsets[block] + output_channel * input_channels +
              input_channel]);
    }
    values_adjoint[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_transform_maps_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* maps_adjoint) {
  for (std::int64_t map_index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) / input_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel = local_map / input_channels;
    const std::int64_t input_channel =
        local_map - output_channel * input_channels;
    CUDA_KERNEL_ASSERT(output_channel < output_channels);
    Scalar value = Scalar(0);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      for (std::int64_t inner = 0; inner < inner_width; ++inner) {
        value += output_adjoint[
            batch * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(values[
            batch * input_width + input_start +
            input_channel * inner_width + inner]);
      }
    }
    maps_adjoint[map_index] = project_control_gradient<Control>(value);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_transform_output_tangent_kernel(
    const Scalar* values,
    const Control* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Control* maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* output_tangent) {
  const std::int64_t total = batch_size * output_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_width;
    const std::int64_t output_feature = index - batch * output_width;
    const std::int64_t block = carrier_offset_block(
        output_feature_offsets, block_count, output_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (output_feature_offsets[block + 1] - output_start) /
        output_channels;
    const std::int64_t local_output = output_feature - output_start;
    const std::int64_t output_channel = local_output / inner_width;
    const std::int64_t inner =
        local_output - output_channel * inner_width;
    Scalar value = Scalar(0);
    for (std::int64_t input_channel = 0;
         input_channel < input_channels;
         ++input_channel) {
      const std::int64_t map_index =
          map_offsets[block] + output_channel * input_channels + input_channel;
      const std::int64_t input_index =
          batch * input_width + input_start +
          input_channel * inner_width + inner;
      value += channel_maps[map_index] *
          values_adjoint_tangent[input_index] +
          maps_adjoint_tangent[map_index] * values[input_index];
    }
    output_tangent[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_transform_values_second_kernel(
    const Scalar* output_adjoint,
    const Control* maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* values_second) {
  const std::int64_t total = batch_size * input_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / input_width;
    const std::int64_t input_feature = index - batch * input_width;
    const std::int64_t block = carrier_offset_block(
        input_feature_offsets, block_count, input_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) / input_channels;
    const std::int64_t local_input = input_feature - input_start;
    const std::int64_t input_channel = local_input / inner_width;
    const std::int64_t inner = local_input - input_channel * inner_width;
    Scalar value = Scalar(0);
    for (std::int64_t output_channel = 0;
         output_channel < output_channels;
         ++output_channel) {
      value += output_adjoint[
          batch * output_width + output_start +
          output_channel * inner_width + inner] * conjugate_value(
          maps_adjoint_tangent[
              map_offsets[block] + output_channel * input_channels +
              input_channel]);
    }
    values_second[index] = value;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_transform_maps_second_kernel(
    const Scalar* output_adjoint,
    const Scalar* values_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* maps_second) {
  for (std::int64_t map_index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) / input_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel = local_map / input_channels;
    const std::int64_t input_channel =
        local_map - output_channel * input_channels;
    CUDA_KERNEL_ASSERT(output_channel < output_channels);
    Scalar value = Scalar(0);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      for (std::int64_t inner = 0; inner < inner_width; ++inner) {
        value += output_adjoint[
            batch * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            values_adjoint_tangent[
                batch * input_width + input_start +
                input_channel * inner_width + inner]);
      }
    }
    maps_second[map_index] = project_control_gradient<Control>(value);
  }
}

template <typename Scalar, typename Control, typename Real>
__global__ void carrier_role_channel_maps_adjoint_kernel(
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
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* maps_adjoint) {
  for (std::int64_t map_index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) / input_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel = local_map / input_channels;
    const std::int64_t input_channel =
        local_map - output_channel * input_channels;
    CUDA_KERNEL_ASSERT(output_channel < output_channels);
    Scalar value = Scalar(0);
    for (std::int64_t edge = 0; edge < edge_count; ++edge) {
      const std::int64_t atom = atom_centers[edge];
      CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
      for (std::int64_t role = 0; role < role_count; ++role) {
        const Real weight = role_weights[edge * role_count + role];
        for (std::int64_t inner = 0; inner < inner_width; ++inner) {
          value += Scalar(weight) * atomic_output_adjoint[
              (atom * role_count + role) * output_width + output_start +
              output_channel * inner_width + inner] * conjugate_value(
              edge_values[
                  edge * input_width + input_start +
                  input_channel * inner_width + inner]);
        }
      }
    }
    maps_adjoint[map_index] = project_control_gradient<Control>(value);
  }
}

template <typename Scalar, typename Control, typename Real>
__global__ void carrier_role_channel_maps_adjoint_parallel_kernel(
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
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* maps_adjoint) {
  const std::int64_t map_index = blockIdx.x;
  if (map_index >= map_count) {
    return;
  }
  const std::int64_t block =
      carrier_offset_block(map_offsets, block_count, map_index);
  CUDA_KERNEL_ASSERT(block < block_count);
  const std::int64_t input_start = input_feature_offsets[block];
  const std::int64_t output_start = output_feature_offsets[block];
  const std::int64_t input_channels =
      input_channel_offsets[block + 1] - input_channel_offsets[block];
  const std::int64_t output_channels =
      output_channel_offsets[block + 1] - output_channel_offsets[block];
  const std::int64_t inner_width =
      (input_feature_offsets[block + 1] - input_start) / input_channels;
  const std::int64_t local_map = map_index - map_offsets[block];
  const std::int64_t output_channel = local_map / input_channels;
  const std::int64_t input_channel =
      local_map - output_channel * input_channels;
  CUDA_KERNEL_ASSERT(output_channel < output_channels);
  Scalar value = Scalar(0);
  const std::int64_t reduction_size =
      edge_count * role_count * inner_width;
  for (std::int64_t flat = threadIdx.x;
       flat < reduction_size;
       flat += blockDim.x) {
    const std::int64_t edge_role = flat / inner_width;
    const std::int64_t inner = flat - edge_role * inner_width;
    const std::int64_t edge = edge_role / role_count;
    const std::int64_t role = edge_role - edge * role_count;
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    value += Scalar(role_weights[edge * role_count + role]) *
        atomic_output_adjoint[
            (atom * role_count + role) * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            edge_values[
                edge * input_width + input_start +
                input_channel * inner_width + inner]);
  }
  extern __shared__ __align__(16) unsigned char shared_storage[];
  Scalar* reductions = reinterpret_cast<Scalar*>(shared_storage);
  reductions[threadIdx.x] = value;
  __syncthreads();
  for (std::int64_t stride = blockDim.x / 2;
       stride > 0;
       stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    maps_adjoint[map_index] =
        project_control_gradient<Control>(reductions[0]);
  }
}

template <typename Scalar, typename Control, typename Real>
__global__ void carrier_role_channel_edge_second_kernel(
    const Real* role_weights,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const Control* maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* edge_values_second) {
  const std::int64_t total = edge_count * input_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / input_width;
    const std::int64_t input_feature = index - edge * input_width;
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    const std::int64_t block = carrier_offset_block(
        input_feature_offsets, block_count, input_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (input_feature_offsets[block + 1] - input_start) / input_channels;
    const std::int64_t local_input = input_feature - input_start;
    const std::int64_t input_channel = local_input / inner_width;
    const std::int64_t inner = local_input - input_channel * inner_width;
    Scalar value = Scalar(0);
    for (std::int64_t role = 0; role < role_count; ++role) {
      const Real weight = role_weights[edge * role_count + role];
      for (std::int64_t output_channel = 0;
           output_channel < output_channels;
           ++output_channel) {
        value += Scalar(weight) * atomic_output_adjoint[
            (atom * role_count + role) * output_width + output_start +
            output_channel * inner_width + inner] * conjugate_value(
            maps_adjoint_tangent[
                map_offsets[block] + output_channel * input_channels +
                input_channel]);
      }
    }
    edge_values_second[index] = value;
  }
}

template <typename Scalar, typename Control, typename Real>
__global__ void carrier_role_channel_role_second_kernel(
    const Scalar* edge_values,
    const Scalar* atomic_output_adjoint,
    const std::int64_t* atom_centers,
    const Control* maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Real* role_weights_second) {
  const std::int64_t total = edge_count * role_count;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / role_count;
    const std::int64_t role = index - edge * role_count;
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    Real value = Real(0);
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
          const Control map_tangent = maps_adjoint_tangent[
              map_offsets[block] + output_channel * input_channels +
              input_channel];
          for (std::int64_t inner = 0; inner < inner_width; ++inner) {
            const Scalar contribution = map_tangent * edge_values[
                edge * input_width + input_start +
                input_channel * inner_width + inner] * conjugate_value(
                atomic_output_adjoint[
                    (atom * role_count + role) * output_width +
                    output_start + output_channel * inner_width + inner]);
            value += project_control_gradient<Real>(contribution);
          }
        }
      }
    }
    role_weights_second[index] = value;
  }
}

template <typename Scalar, typename Control, typename Real>
__global__ void carrier_role_channel_output_tangent_kernel(
    const Scalar* edge_values,
    const Real* role_weights,
    const std::int64_t* atom_centers,
    const Control* maps_adjoint_tangent,
    const std::int64_t* input_feature_offsets,
    const std::int64_t* output_feature_offsets,
    const std::int64_t* input_channel_offsets,
    const std::int64_t* output_channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t edge_count,
    std::int64_t atom_count,
    std::int64_t role_count,
    std::int64_t input_width,
    std::int64_t output_width,
    std::int64_t block_count,
    Scalar* atomic_output_tangent) {
  const std::int64_t total = edge_count * role_count * output_width;
  for (std::int64_t index =
           static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge_role = index / output_width;
    const std::int64_t output_feature = index - edge_role * output_width;
    const std::int64_t edge = edge_role / role_count;
    const std::int64_t role = edge_role - edge * role_count;
    const std::int64_t atom = atom_centers[edge];
    CUDA_KERNEL_ASSERT(atom >= 0 && atom < atom_count);
    const std::int64_t block = carrier_offset_block(
        output_feature_offsets, block_count, output_feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t input_start = input_feature_offsets[block];
    const std::int64_t output_start = output_feature_offsets[block];
    const std::int64_t input_channels =
        input_channel_offsets[block + 1] - input_channel_offsets[block];
    const std::int64_t output_channels =
        output_channel_offsets[block + 1] - output_channel_offsets[block];
    const std::int64_t inner_width =
        (output_feature_offsets[block + 1] - output_start) /
        output_channels;
    const std::int64_t local_output = output_feature - output_start;
    const std::int64_t output_channel = local_output / inner_width;
    const std::int64_t inner = local_output - output_channel * inner_width;
    Scalar value = Scalar(0);
    for (std::int64_t input_channel = 0;
         input_channel < input_channels;
         ++input_channel) {
      value += maps_adjoint_tangent[
          map_offsets[block] + output_channel * input_channels +
          input_channel] * edge_values[
          edge * input_width + input_start +
          input_channel * inner_width + inner];
    }
    value *= Scalar(role_weights[edge * role_count + role]);
    atomic_add_value(
        atomic_output_tangent +
            (atom * role_count + role) * output_width + output_feature,
        value);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_values_adjoint_kernel(
    const Scalar* output_adjoint,
    const Control* gates,
    const Control* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t block_count,
    Scalar* values_adjoint) {
  const std::int64_t total = batch_size * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / feature_count;
    const std::int64_t feature =
        index - batch * feature_count;
    const std::int64_t block =
        carrier_offset_block(feature_offsets, block_count, feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t local_feature = feature - feature_start;
    const std::int64_t input_channel = local_feature / inner_width;
    const std::int64_t inner =
        local_feature - input_channel * inner_width;
    Scalar result = output_adjoint[index];
    for (std::int64_t output_channel = 0;
         output_channel < local_channels;
         ++output_channel) {
      const std::int64_t output_feature =
          feature_start +
          output_channel * inner_width +
          inner;
      result +=
          output_adjoint[
              batch * feature_count + output_feature] *
          conjugate_value(
              gates[
                  batch * channel_count +
                  channel_start +
                  output_channel] *
              channel_maps[
                  map_offsets[block] +
                  output_channel * local_channels +
                  input_channel]);
    }
    values_adjoint[index] = result;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_gates_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* channel_maps,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t block_count,
    Control* gates_adjoint) {
  const std::int64_t total = batch_size * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / channel_count;
    const std::int64_t channel =
        index - batch * channel_count;
    const std::int64_t block =
        carrier_offset_block(channel_offsets, block_count, channel);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t output_channel = channel - channel_start;
    Scalar result = Scalar(0);
    for (std::int64_t inner = 0; inner < inner_width; ++inner) {
      Scalar mixed = Scalar(0);
      for (std::int64_t input_channel = 0;
           input_channel < local_channels;
           ++input_channel) {
        mixed +=
            channel_maps[
                map_offsets[block] +
                output_channel * local_channels +
                input_channel] *
            values[
                batch * feature_count +
                feature_start +
                input_channel * inner_width +
                inner];
      }
      result +=
          output_adjoint[
              batch * feature_count +
              feature_start +
              output_channel * inner_width +
              inner] *
          conjugate_value(mixed);
    }
    gates_adjoint[index] =
        project_control_gradient<Control>(result);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_maps_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* gates,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* channel_maps_adjoint) {
  for (std::int64_t map_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index +=
           static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel =
        local_map / local_channels;
    const std::int64_t input_channel =
        local_map - output_channel * local_channels;
    Scalar result = Scalar(0);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const Control gate =
          gates[
              batch * channel_count +
              channel_start +
              output_channel];
      for (std::int64_t inner = 0; inner < inner_width; ++inner) {
        result +=
            output_adjoint[
                batch * feature_count +
                feature_start +
                output_channel * inner_width +
                inner] *
            conjugate_value(
                gate *
                values[
                    batch * feature_count +
                    feature_start +
                    input_channel * inner_width +
                    inner]);
      }
    }
    channel_maps_adjoint[map_index] =
        project_control_gradient<Control>(result);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_maps_adjoint_parallel_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* gates,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* channel_maps_adjoint) {
  const std::int64_t map_index = blockIdx.x;
  if (map_index >= map_count) {
    return;
  }
  const std::int64_t block =
      carrier_offset_block(map_offsets, block_count, map_index);
  CUDA_KERNEL_ASSERT(block < block_count);
  const std::int64_t feature_start = feature_offsets[block];
  const std::int64_t channel_start = channel_offsets[block];
  const std::int64_t local_channels =
      channel_offsets[block + 1] - channel_start;
  const std::int64_t inner_width =
      (feature_offsets[block + 1] - feature_start) /
      local_channels;
  const std::int64_t local_map = map_index - map_offsets[block];
  const std::int64_t output_channel = local_map / local_channels;
  const std::int64_t input_channel =
      local_map - output_channel * local_channels;
  Scalar result = Scalar(0);
  const std::int64_t reduction_size = batch_size * inner_width;
  for (std::int64_t flat = threadIdx.x;
       flat < reduction_size;
       flat += blockDim.x) {
    const std::int64_t batch = flat / inner_width;
    const std::int64_t inner = flat - batch * inner_width;
    const Control gate =
        gates[
            batch * channel_count +
            channel_start + output_channel];
    result +=
        output_adjoint[
            batch * feature_count +
            feature_start +
            output_channel * inner_width + inner] *
        conjugate_value(
            gate *
            values[
                batch * feature_count +
                feature_start +
                input_channel * inner_width + inner]);
  }
  extern __shared__ __align__(16) unsigned char shared_storage[];
  Scalar* reductions = reinterpret_cast<Scalar*>(shared_storage);
  reductions[threadIdx.x] = result;
  __syncthreads();
  for (std::int64_t stride = blockDim.x / 2;
       stride > 0;
       stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    channel_maps_adjoint[map_index] =
        project_control_gradient<Control>(reductions[0]);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_output_tangent_kernel(
    const Scalar* values,
    const Control* gates,
    const Control* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Control* gates_adjoint_tangent,
    const Control* channel_maps_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t block_count,
    Scalar* output_adjoint_tangent) {
  const std::int64_t total = batch_size * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / feature_count;
    const std::int64_t feature =
        index - batch * feature_count;
    const std::int64_t block =
        carrier_offset_block(feature_offsets, block_count, feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t local_feature = feature - feature_start;
    const std::int64_t output_channel = local_feature / inner_width;
    const std::int64_t inner =
        local_feature - output_channel * inner_width;
    Scalar mixed = Scalar(0);
    Scalar mixed_adjoint_tangent = Scalar(0);
    for (std::int64_t input_channel = 0;
         input_channel < local_channels;
         ++input_channel) {
      const std::int64_t input_feature =
          batch * feature_count +
          feature_start +
          input_channel * inner_width +
          inner;
      const std::int64_t map_index =
          map_offsets[block] +
          output_channel * local_channels +
          input_channel;
      mixed += channel_maps[map_index] * values[input_feature];
      mixed_adjoint_tangent +=
          channel_maps[map_index] *
              values_adjoint_tangent[input_feature] +
          values[input_feature] *
              channel_maps_adjoint_tangent[map_index];
    }
    const std::int64_t gate_index =
        batch * channel_count +
        channel_start +
        output_channel;
    output_adjoint_tangent[index] =
        values_adjoint_tangent[index] +
        gates[gate_index] * mixed_adjoint_tangent +
        gates_adjoint_tangent[gate_index] * mixed;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_values_second_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* gates,
    const Control* channel_maps,
    const Control* gates_adjoint_tangent,
    const Control* channel_maps_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t block_count,
    Scalar* values_second_adjoint) {
  const std::int64_t total = batch_size * feature_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / feature_count;
    const std::int64_t feature =
        index - batch * feature_count;
    const std::int64_t block =
        carrier_offset_block(feature_offsets, block_count, feature);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t local_feature = feature - feature_start;
    const std::int64_t input_channel = local_feature / inner_width;
    const std::int64_t inner =
        local_feature - input_channel * inner_width;
    Scalar result = Scalar(0);
    for (std::int64_t output_channel = 0;
         output_channel < local_channels;
         ++output_channel) {
      const std::int64_t gate_index =
          batch * channel_count +
          channel_start +
          output_channel;
      const Scalar gradient =
          output_adjoint[
              batch * feature_count +
              feature_start +
              output_channel * inner_width +
              inner];
      const Scalar mixed_adjoint =
          gradient * conjugate_value(gates[gate_index]);
      const Scalar mixed_second =
          gradient *
          conjugate_value(gates_adjoint_tangent[gate_index]);
      const std::int64_t map_index =
          map_offsets[block] +
          output_channel * local_channels +
          input_channel;
      result +=
          mixed_adjoint *
              conjugate_value(
                  channel_maps_adjoint_tangent[map_index]) +
          mixed_second *
              conjugate_value(channel_maps[map_index]);
    }
    values_second_adjoint[index] = result;
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_gates_second_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* channel_maps,
    const Scalar* values_adjoint_tangent,
    const Control* channel_maps_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t block_count,
    Control* gates_second_adjoint) {
  const std::int64_t total = batch_size * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / channel_count;
    const std::int64_t channel =
        index - batch * channel_count;
    const std::int64_t block =
        carrier_offset_block(channel_offsets, block_count, channel);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t output_channel = channel - channel_start;
    Scalar result = Scalar(0);
    for (std::int64_t inner = 0; inner < inner_width; ++inner) {
      Scalar mixed_adjoint_tangent = Scalar(0);
      for (std::int64_t input_channel = 0;
           input_channel < local_channels;
           ++input_channel) {
        const std::int64_t input_feature =
            batch * feature_count +
            feature_start +
            input_channel * inner_width +
            inner;
        const std::int64_t map_index =
            map_offsets[block] +
            output_channel * local_channels +
            input_channel;
        mixed_adjoint_tangent +=
            channel_maps[map_index] *
                values_adjoint_tangent[input_feature] +
            values[input_feature] *
                channel_maps_adjoint_tangent[map_index];
      }
      result +=
          output_adjoint[
              batch * feature_count +
              feature_start +
              output_channel * inner_width +
              inner] *
          conjugate_value(mixed_adjoint_tangent);
    }
    gates_second_adjoint[index] =
        project_control_gradient<Control>(result);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_maps_second_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* gates,
    const Scalar* values_adjoint_tangent,
    const Control* gates_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* channel_maps_second_adjoint) {
  for (std::int64_t map_index =
           blockIdx.x * blockDim.x + threadIdx.x;
       map_index < map_count;
       map_index +=
           static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t block =
        carrier_offset_block(map_offsets, block_count, map_index);
    CUDA_KERNEL_ASSERT(block < block_count);
    const std::int64_t feature_start = feature_offsets[block];
    const std::int64_t channel_start = channel_offsets[block];
    const std::int64_t local_channels =
        channel_offsets[block + 1] - channel_start;
    const std::int64_t inner_width =
        (feature_offsets[block + 1] - feature_start) /
        local_channels;
    const std::int64_t local_map = map_index - map_offsets[block];
    const std::int64_t output_channel =
        local_map / local_channels;
    const std::int64_t input_channel =
        local_map - output_channel * local_channels;
    Scalar result = Scalar(0);
    for (std::int64_t batch = 0; batch < batch_size; ++batch) {
      const std::int64_t gate_index =
          batch * channel_count +
          channel_start +
          output_channel;
      for (std::int64_t inner = 0; inner < inner_width; ++inner) {
        const std::int64_t output_feature =
            batch * feature_count +
            feature_start +
            output_channel * inner_width +
            inner;
        const std::int64_t input_feature =
            batch * feature_count +
            feature_start +
            input_channel * inner_width +
            inner;
        const Scalar mixed_adjoint =
            output_adjoint[output_feature] *
            conjugate_value(gates[gate_index]);
        const Scalar mixed_second =
            output_adjoint[output_feature] *
            conjugate_value(
                gates_adjoint_tangent[gate_index]);
        result +=
            mixed_adjoint *
                conjugate_value(
                    values_adjoint_tangent[input_feature]) +
            mixed_second *
                conjugate_value(values[input_feature]);
      }
    }
    channel_maps_second_adjoint[map_index] =
        project_control_gradient<Control>(result);
  }
}

template <typename Scalar, typename Control = Scalar>
__global__ void carrier_channel_maps_second_parallel_kernel(
    const Scalar* output_adjoint,
    const Scalar* values,
    const Control* gates,
    const Scalar* values_adjoint_tangent,
    const Control* gates_adjoint_tangent,
    const std::int64_t* feature_offsets,
    const std::int64_t* channel_offsets,
    const std::int64_t* map_offsets,
    std::int64_t batch_size,
    std::int64_t feature_count,
    std::int64_t channel_count,
    std::int64_t map_count,
    std::int64_t block_count,
    Control* channel_maps_second_adjoint) {
  const std::int64_t map_index = blockIdx.x;
  if (map_index >= map_count) {
    return;
  }
  const std::int64_t block =
      carrier_offset_block(map_offsets, block_count, map_index);
  CUDA_KERNEL_ASSERT(block < block_count);
  const std::int64_t feature_start = feature_offsets[block];
  const std::int64_t channel_start = channel_offsets[block];
  const std::int64_t local_channels =
      channel_offsets[block + 1] - channel_start;
  const std::int64_t inner_width =
      (feature_offsets[block + 1] - feature_start) /
      local_channels;
  const std::int64_t local_map = map_index - map_offsets[block];
  const std::int64_t output_channel = local_map / local_channels;
  const std::int64_t input_channel =
      local_map - output_channel * local_channels;
  Scalar result = Scalar(0);
  const std::int64_t reduction_size = batch_size * inner_width;
  for (std::int64_t flat = threadIdx.x;
       flat < reduction_size;
       flat += blockDim.x) {
    const std::int64_t batch = flat / inner_width;
    const std::int64_t inner = flat - batch * inner_width;
    const std::int64_t gate_index =
        batch * channel_count + channel_start + output_channel;
    const std::int64_t output_feature =
        batch * feature_count + feature_start +
        output_channel * inner_width + inner;
    const std::int64_t input_feature =
        batch * feature_count + feature_start +
        input_channel * inner_width + inner;
    const Scalar mixed_adjoint =
        output_adjoint[output_feature] *
        conjugate_value(gates[gate_index]);
    const Scalar mixed_second =
        output_adjoint[output_feature] *
        conjugate_value(gates_adjoint_tangent[gate_index]);
    result +=
        mixed_adjoint *
            conjugate_value(values_adjoint_tangent[input_feature]) +
        mixed_second * conjugate_value(values[input_feature]);
  }
  extern __shared__ __align__(16) unsigned char shared_storage[];
  Scalar* reductions = reinterpret_cast<Scalar*>(shared_storage);
  reductions[threadIdx.x] = result;
  __syncthreads();
  for (std::int64_t stride = blockDim.x / 2;
       stride > 0;
       stride /= 2) {
    if (threadIdx.x < stride) {
      reductions[threadIdx.x] += reductions[threadIdx.x + stride];
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    channel_maps_second_adjoint[map_index] =
        project_control_gradient<Control>(reductions[0]);
  }
}

template <typename Real>
__global__ void cheb_exp_cos_radial_kernel(
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t radial_index,
    Real* values,
    Real* derivatives) {
  const Real pi =
      static_cast<Real>(3.141592653589793238462643383279502884);
  const Real epsilon =
      sizeof(Real) == sizeof(float)
      ? static_cast<Real>(1.1920928955078125e-7)
      : static_cast<Real>(2.2204460492503131e-16);
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const Real radius = radii[edge];
    const Real cutoff =
        cutoffs[edge] > epsilon ? cutoffs[edge] : epsilon;
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
      values[edge] =
          Real(0.5) * (Real(1) + cos(pi * scaled));
      derivatives[edge] =
          -Real(0.5) * pi * sin(pi * scaled) / cutoff;
      continue;
    }

    const Real numerator_exp =
        exp(-lambda * (scaled - Real(1)));
    const Real denominator = expm1(lambda);
    const Real warped =
        Real(1) - Real(2) *
        expm1(-lambda * (scaled - Real(1))) / denominator;
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
    const Real chebyshev_derivative =
        Real(radial_index) * u_current;
    const Real warped_derivative =
        Real(2) * lambda * numerator_exp / (denominator * cutoff);
    const Real envelope = Real(1) + cos(pi * scaled);
    const Real envelope_derivative =
        -pi * sin(pi * scaled) / cutoff;
    values[edge] =
        Real(0.25) * (Real(1) - t_current) * envelope;
    derivatives[edge] = Real(0.25) * (
        -chebyshev_derivative * warped_derivative * envelope +
        (Real(1) - t_current) * envelope_derivative);
  }
}

template <typename Real>
__global__ void cheb_exp_cos_radial_table_kernel(
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t maximum_radial_index,
    Real* values,
    Real* derivatives) {
  const Real pi =
      static_cast<Real>(3.141592653589793238462643383279502884);
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const std::int64_t width = maximum_radial_index + 1;
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const Real radius = radii[edge];
    const Real cutoff = cutoffs[edge] > epsilon
        ? cutoffs[edge]
        : epsilon;
    const Real lambda = lambdas[edge];
    const Real scaled = radius / cutoff;
    Real* value_row = values + edge * width;
    Real* derivative_row = derivatives + edge * width;
    if (scaled > Real(1)) {
      for (std::int64_t radial_index = 0;
           radial_index <= maximum_radial_index;
           ++radial_index) {
        value_row[radial_index] = Real(0);
        derivative_row[radial_index] = Real(0);
      }
      continue;
    }
    value_row[0] = Real(1);
    derivative_row[0] = Real(0);
    if (maximum_radial_index == 0) {
      continue;
    }
    const Real cosine = cos(pi * scaled);
    const Real sine = sin(pi * scaled);
    value_row[1] = Real(0.5) * (Real(1) + cosine);
    derivative_row[1] =
        -Real(0.5) * pi * sine / cutoff;
    if (maximum_radial_index == 1) {
      continue;
    }

    const Real numerator_exp =
        exp(-lambda * (scaled - Real(1)));
    const Real denominator = expm1(lambda);
    const Real warped =
        Real(1) - Real(2) * expm1(
            -lambda * (scaled - Real(1))) / denominator;
    const Real warped_derivative =
        Real(2) * lambda * numerator_exp /
        (denominator * cutoff);
    const Real envelope = Real(1) + cosine;
    const Real envelope_derivative =
        -pi * sine / cutoff;
    Real t_previous = Real(1);
    Real t_current = warped;
    Real dt_previous = Real(0);
    Real dt_current = Real(1);
    for (std::int64_t radial_index = 2;
         radial_index <= maximum_radial_index;
         ++radial_index) {
      const Real t_next =
          Real(2) * warped * t_current - t_previous;
      const Real dt_next =
          Real(2) * t_current +
          Real(2) * warped * dt_current -
          dt_previous;
      value_row[radial_index] =
          Real(0.25) * (Real(1) - t_next) * envelope;
      derivative_row[radial_index] = Real(0.25) * (
          -dt_next * warped_derivative * envelope +
          (Real(1) - t_next) * envelope_derivative);
      t_previous = t_current;
      t_current = t_next;
      dt_previous = dt_current;
      dt_current = dt_next;
    }
  }
}

template <typename Real>
__global__ void cheb_exp_cos_radial_table_double_backward_kernel(
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
  const Real pi =
      static_cast<Real>(3.141592653589793238462643383279502884);
  const Real epsilon = std::numeric_limits<Real>::epsilon();
  const std::int64_t width = maximum_radial_index + 1;
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const Real tangent = grad_grad_radii[edge];
    for (std::int64_t radial_index = 0;
         radial_index <= maximum_radial_index;
         ++radial_index) {
      const std::int64_t index = edge * width + radial_index;
      values_adjoint_gradient[index] =
          tangent * radial_derivatives[index];
    }
    const Real cutoff = cutoffs[edge] > epsilon
        ? cutoffs[edge]
        : epsilon;
    const Real scaled = radii[edge] / cutoff;
    if (scaled > Real(1) || maximum_radial_index == 0) {
      radii_gradient[edge] = Real(0);
      continue;
    }
    const Real cosine = cos(pi * scaled);
    const Real sine = sin(pi * scaled);
    const Real inverse_cutoff = Real(1) / cutoff;
    const Real envelope = Real(1) + cosine;
    const Real envelope_derivative =
        -pi * sine * inverse_cutoff;
    const Real envelope_second =
        -pi * pi * cosine * inverse_cutoff * inverse_cutoff;
    Real second_contraction =
        values_adjoint[edge * width + 1] *
        Real(0.5) * envelope_second;
    if (maximum_radial_index >= 2) {
      const Real lambda = lambdas[edge];
      const Real numerator_exp =
          exp(-lambda * (scaled - Real(1)));
      const Real denominator = expm1(lambda);
      const Real warped =
          Real(1) - Real(2) * expm1(
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
__device__ inline Real spherical_normalization_device(
    std::int64_t angular_momentum,
    std::int64_t magnetic_abs) {
  const Real pi =
      static_cast<Real>(3.141592653589793238462643383279502884);
  const Real log_norm = Real(0.5) * (
      log(Real(2 * angular_momentum + 1)) -
      log(Real(4) * pi) +
      lgamma(Real(angular_momentum - magnetic_abs + 1)) -
      lgamma(Real(angular_momentum + magnetic_abs + 1)));
  return exp(log_norm);
}

template <typename Real>
struct SphericalDirectionalJet {
  Real value;
  Real gradient[3];
  Real directional_derivative;
  Real directional_gradient[3];
};

template <typename Real>
__device__ inline SphericalDirectionalJet<Real>
spherical_directional_jet_constant(
    Real value) {
  SphericalDirectionalJet<Real> output;
  output.value = value;
  output.directional_derivative = Real(0);
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    output.gradient[axis] = Real(0);
    output.directional_gradient[axis] = Real(0);
  }
  return output;
}

template <typename Real>
__device__ inline SphericalDirectionalJet<Real>
spherical_directional_jet_variable(
    Real value,
    std::int64_t variable_axis,
    Real direction) {
  SphericalDirectionalJet<Real> output =
      spherical_directional_jet_constant<Real>(value);
  output.gradient[variable_axis] = Real(1);
  output.directional_derivative = direction;
  return output;
}

template <typename Real>
__device__ inline SphericalDirectionalJet<Real>
spherical_directional_jet_add(
    const SphericalDirectionalJet<Real>& left,
    const SphericalDirectionalJet<Real>& right) {
  SphericalDirectionalJet<Real> output;
  output.value = left.value + right.value;
  output.directional_derivative =
      left.directional_derivative +
      right.directional_derivative;
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    output.gradient[axis] =
        left.gradient[axis] + right.gradient[axis];
    output.directional_gradient[axis] =
        left.directional_gradient[axis] +
        right.directional_gradient[axis];
  }
  return output;
}

template <typename Real>
__device__ inline SphericalDirectionalJet<Real>
spherical_directional_jet_scale(
    const SphericalDirectionalJet<Real>& input,
    Real scale) {
  SphericalDirectionalJet<Real> output;
  output.value = scale * input.value;
  output.directional_derivative =
      scale * input.directional_derivative;
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    output.gradient[axis] = scale * input.gradient[axis];
    output.directional_gradient[axis] =
        scale * input.directional_gradient[axis];
  }
  return output;
}

template <typename Real>
__device__ inline SphericalDirectionalJet<Real>
spherical_directional_jet_multiply(
    const SphericalDirectionalJet<Real>& left,
    const SphericalDirectionalJet<Real>& right) {
  SphericalDirectionalJet<Real> output;
  output.value = left.value * right.value;
  output.directional_derivative =
      left.directional_derivative * right.value +
      left.value * right.directional_derivative;
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    output.gradient[axis] =
        left.gradient[axis] * right.value +
        left.value * right.gradient[axis];
    output.directional_gradient[axis] =
        left.directional_gradient[axis] * right.value +
        left.directional_derivative * right.gradient[axis] +
        left.gradient[axis] * right.directional_derivative +
        left.value * right.directional_gradient[axis];
  }
  return output;
}

template <typename Real>
__device__ inline SphericalDirectionalJet<Real>
spherical_directional_jet_inverse_square_root(
    const SphericalDirectionalJet<Real>& input) {
  const Real inverse_root = rsqrt(input.value);
  const Real first =
      -Real(0.5) * inverse_root / input.value;
  const Real second =
      Real(0.75) * inverse_root /
      (input.value * input.value);
  SphericalDirectionalJet<Real> output;
  output.value = inverse_root;
  output.directional_derivative =
      first * input.directional_derivative;
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    output.gradient[axis] = first * input.gradient[axis];
    output.directional_gradient[axis] =
        second * input.directional_derivative *
            input.gradient[axis] +
        first * input.directional_gradient[axis];
  }
  return output;
}

template <typename Real>
__device__ inline void
accumulate_spherical_directional_jet_double_backward(
    const SphericalDirectionalJet<Real>& value,
    Real value_adjoint,
    Real* value_adjoint_gradient,
    Real* edge_gradient) {
  *value_adjoint_gradient = value.directional_derivative;
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    edge_gradient[axis] +=
        value_adjoint * value.directional_gradient[axis];
  }
}

template <typename Real>
__device__ inline void accumulate_real_spherical_degree_double_backward(
    const SphericalDirectionalJet<Real>* polynomial_row,
    std::int64_t angular_momentum,
    const SphericalDirectionalJet<Real>& unit_x,
    const SphericalDirectionalJet<Real>& unit_y,
    const Real* value_adjoint,
    Real* value_adjoint_gradient,
    Real* edge_gradient) {
  const Real norm_zero =
      spherical_normalization_device<Real>(angular_momentum, 0);
  const SphericalDirectionalJet<Real> zero_component =
      spherical_directional_jet_scale<Real>(
          polynomial_row[0],
          norm_zero);
  accumulate_spherical_directional_jet_double_backward<Real>(
      zero_component,
      value_adjoint[angular_momentum],
      value_adjoint_gradient + angular_momentum,
      edge_gradient);

  const Real sqrt_two = sqrt(Real(2));
  SphericalDirectionalJet<Real> power_real =
      spherical_directional_jet_constant<Real>(Real(1));
  SphericalDirectionalJet<Real> power_imag =
      spherical_directional_jet_constant<Real>(Real(0));
  for (std::int64_t magnetic = 1;
       magnetic <= angular_momentum;
       ++magnetic) {
    const SphericalDirectionalJet<Real> previous_power_real =
        power_real;
    const SphericalDirectionalJet<Real> previous_power_imag =
        power_imag;
    power_real = spherical_directional_jet_add<Real>(
        spherical_directional_jet_multiply<Real>(
            previous_power_real,
            unit_x),
        spherical_directional_jet_scale<Real>(
            spherical_directional_jet_multiply<Real>(
                previous_power_imag,
                unit_y),
            Real(-1)));
    power_imag = spherical_directional_jet_add<Real>(
        spherical_directional_jet_multiply<Real>(
            previous_power_real,
            unit_y),
        spherical_directional_jet_multiply<Real>(
            previous_power_imag,
            unit_x));
    const Real scale = sqrt_two *
        spherical_normalization_device<Real>(
            angular_momentum,
            magnetic);
    const SphericalDirectionalJet<Real> positive =
        spherical_directional_jet_scale<Real>(
            spherical_directional_jet_multiply<Real>(
                polynomial_row[magnetic],
                power_real),
            scale);
    const SphericalDirectionalJet<Real> negative =
        spherical_directional_jet_scale<Real>(
            spherical_directional_jet_multiply<Real>(
                polynomial_row[magnetic],
                power_imag),
            -scale);
    const std::int64_t positive_index =
        angular_momentum + magnetic;
    const std::int64_t negative_index =
        angular_momentum - magnetic;
    accumulate_spherical_directional_jet_double_backward<Real>(
        positive,
        value_adjoint[positive_index],
        value_adjoint_gradient + positive_index,
        edge_gradient);
    accumulate_spherical_directional_jet_double_backward<Real>(
        negative,
        value_adjoint[negative_index],
        value_adjoint_gradient + negative_index,
        edge_gradient);
  }
}

template <typename Real>
__global__ void real_spherical_harmonics_table_double_backward_kernel(
    const Real* edge_vectors,
    const Real* value_adjoint,
    const Real* edge_direction,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    SphericalDirectionalJet<Real>* scratch,
    Real* value_adjoint_gradient,
    Real* edge_gradient) {
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  const std::int64_t row_width =
      maximum_angular_momentum + 1;
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    SphericalDirectionalJet<Real>* row_zero =
        scratch + edge * 3 * row_width;
    SphericalDirectionalJet<Real>* row_one =
        row_zero + row_width;
    SphericalDirectionalJet<Real>* row_two =
        row_one + row_width;
    for (std::int64_t index = 0; index < row_width; ++index) {
      row_zero[index] =
          spherical_directional_jet_constant<Real>(Real(0));
      row_one[index] =
          spherical_directional_jet_constant<Real>(Real(0));
      row_two[index] =
          spherical_directional_jet_constant<Real>(Real(0));
    }
    SphericalDirectionalJet<Real>* previous_previous =
        row_zero;
    SphericalDirectionalJet<Real>* previous = row_one;
    SphericalDirectionalJet<Real>* current = row_two;
    previous[0] =
        spherical_directional_jet_constant<Real>(Real(1));

    const Real x_value = edge_vectors[edge * 3];
    const Real y_value = edge_vectors[edge * 3 + 1];
    const Real z_value = edge_vectors[edge * 3 + 2];
    const Real* edge_direction_row =
        edge_direction + edge * 3;
    const Real radius = sqrt(
        x_value * x_value +
        y_value * y_value +
        z_value * z_value);
    const SphericalDirectionalJet<Real> x =
        spherical_directional_jet_variable<Real>(
            x_value,
            0,
            edge_direction_row[0]);
    const SphericalDirectionalJet<Real> y =
        spherical_directional_jet_variable<Real>(
            y_value,
            1,
            edge_direction_row[1]);
    const SphericalDirectionalJet<Real> z =
        spherical_directional_jet_variable<Real>(
            z_value,
            2,
            edge_direction_row[2]);
    SphericalDirectionalJet<Real> unit_x;
    SphericalDirectionalJet<Real> unit_y;
    SphericalDirectionalJet<Real> unit_z;
    if (radius > epsilon) {
      const SphericalDirectionalJet<Real>
          radius_squared =
              spherical_directional_jet_add<Real>(
                  spherical_directional_jet_add<Real>(
                      spherical_directional_jet_multiply<Real>(
                          x,
                          x),
                      spherical_directional_jet_multiply<Real>(
                          y,
                          y)),
                  spherical_directional_jet_multiply<Real>(
                      z,
                      z));
      const SphericalDirectionalJet<Real> inverse_radius =
          spherical_directional_jet_inverse_square_root<Real>(
              radius_squared);
      unit_x =
          spherical_directional_jet_multiply<Real>(
              x,
              inverse_radius);
      unit_y =
          spherical_directional_jet_multiply<Real>(
              y,
              inverse_radius);
      unit_z =
          spherical_directional_jet_multiply<Real>(
              z,
              inverse_radius);
    } else {
      const Real inverse_epsilon = Real(1) / epsilon;
      unit_x = spherical_directional_jet_scale<Real>(
          x,
          inverse_epsilon);
      unit_y = spherical_directional_jet_scale<Real>(
          y,
          inverse_epsilon);
      unit_z = spherical_directional_jet_scale<Real>(
          z,
          inverse_epsilon);
    }
    Real local_edge_gradient[3] = {
        Real(0),
        Real(0),
        Real(0)};
    const Real* edge_adjoint_row =
        value_adjoint + edge * width;
    Real* value_adjoint_gradient_row =
        value_adjoint_gradient + edge * width;
    accumulate_real_spherical_degree_double_backward<Real>(
        previous,
        0,
        unit_x,
        unit_y,
        edge_adjoint_row,
        value_adjoint_gradient_row,
        local_edge_gradient);

    for (std::int64_t degree = 1;
         degree <= maximum_angular_momentum;
         ++degree) {
      for (std::int64_t order = 0;
           order < row_width;
           ++order) {
        current[order] =
            spherical_directional_jet_constant<Real>(Real(0));
      }
      const Real forward = Real(2 * degree - 1);
      const Real backward = Real(degree - 1);
      for (std::int64_t order = 0; order <= degree; ++order) {
        const SphericalDirectionalJet<Real> same_order =
            order < degree
            ? previous[order]
            : spherical_directional_jet_constant<Real>(
                Real(0));
        const SphericalDirectionalJet<Real> lower_order =
            order > 0
            ? previous[order - 1]
            : spherical_directional_jet_constant<Real>(
                Real(0));
        const SphericalDirectionalJet<Real> older =
            order < degree - 1
            ? previous_previous[order]
            : spherical_directional_jet_constant<Real>(
                Real(0));
        current[order] =
            spherical_directional_jet_scale<Real>(
                spherical_directional_jet_add<Real>(
                    spherical_directional_jet_add<Real>(
                        spherical_directional_jet_scale<Real>(
                            spherical_directional_jet_multiply<Real>(
                                unit_z,
                                same_order),
                            forward),
                        spherical_directional_jet_scale<Real>(
                            lower_order,
                            forward * Real(order))),
                    spherical_directional_jet_scale<Real>(
                        older,
                        -backward)),
                Real(1) / Real(degree));
      }
      SphericalDirectionalJet<Real>* recycled =
          previous_previous;
      previous_previous = previous;
      previous = current;
      current = recycled;
      const std::int64_t offset = degree * degree;
      accumulate_real_spherical_degree_double_backward<Real>(
          previous,
          degree,
          unit_x,
          unit_y,
          edge_adjoint_row + offset,
          value_adjoint_gradient_row + offset,
          local_edge_gradient);
    }
    for (std::int64_t axis = 0; axis < 3; ++axis) {
      edge_gradient[edge * 3 + axis] =
          local_edge_gradient[axis];
    }
  }
}

template <typename Real>
__device__ inline void store_cartesian_gradient(
    Real gradient_x,
    Real gradient_y,
    Real gradient_z,
    Real unit_x,
    Real unit_y,
    Real unit_z,
    Real radius,
    Real safe_radius,
    Real epsilon,
    Real* destination) {
  const Real unit[3] = {unit_x, unit_y, unit_z};
  const Real gradient[3] = {gradient_x, gradient_y, gradient_z};
  for (std::int64_t axis = 0; axis < 3; ++axis) {
    Real value = Real(0);
    for (std::int64_t unit_axis = 0; unit_axis < 3; ++unit_axis) {
      Real jacobian = Real(0);
      if (radius > epsilon) {
        jacobian =
            ((axis == unit_axis ? Real(1) : Real(0)) -
             unit[axis] * unit[unit_axis]) /
            safe_radius;
      } else if (axis == unit_axis) {
        jacobian = Real(1) / safe_radius;
      }
      value += gradient[unit_axis] * jacobian;
    }
    destination[axis] = value;
  }
}

template <typename Real>
__global__ void real_spherical_harmonics_kernel(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t angular_momentum,
    Real epsilon,
    Real* derivative_scratch,
    Real* values,
    Real* derivatives) {
  const std::int64_t width = 2 * angular_momentum + 1;
  const std::int64_t row_width = angular_momentum + 1;
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    Real* row_zero =
        derivative_scratch + edge * 3 * row_width;
    Real* row_one = row_zero + row_width;
    Real* row_two = row_one + row_width;
    for (std::int64_t index = 0; index < row_width; ++index) {
      row_zero[index] = Real(0);
      row_one[index] = Real(0);
      row_two[index] = Real(0);
    }
    Real* previous_previous = row_zero;
    Real* previous = row_one;
    Real* current = row_two;
    previous[0] = Real(1);

    const Real x = edge_vectors[edge * 3];
    const Real y = edge_vectors[edge * 3 + 1];
    const Real z = edge_vectors[edge * 3 + 2];
    const Real radius = sqrt(x * x + y * y + z * z);
    const Real safe_radius = radius > epsilon ? radius : epsilon;
    const Real unit_x = x / safe_radius;
    const Real unit_y = y / safe_radius;
    const Real unit_z = z / safe_radius;

    for (std::int64_t degree = 1;
         degree <= angular_momentum;
         ++degree) {
      for (std::int64_t order = 0;
           order < row_width;
           ++order) {
        current[order] = Real(0);
      }
      const Real forward = Real(2 * degree - 1);
      const Real backward = Real(degree - 1);
      for (std::int64_t order = 0; order <= degree; ++order) {
        const Real same_order =
            order < degree ? previous[order] : Real(0);
        const Real lower_order =
            order > 0 ? previous[order - 1] : Real(0);
        const Real older =
            order < degree - 1
            ? previous_previous[order]
            : Real(0);
        current[order] = (
            forward * (
                unit_z * same_order +
                Real(order) * lower_order) -
            backward * older) /
            Real(degree);
      }
      Real* recycled = previous_previous;
      previous_previous = previous;
      previous = current;
      current = recycled;
    }

    const std::int64_t center =
        edge * width + angular_momentum;
    const Real norm_zero =
        spherical_normalization_device<Real>(angular_momentum, 0);
    values[center] = norm_zero * previous[0];
    store_cartesian_gradient(
        Real(0),
        Real(0),
        norm_zero * (
            angular_momentum > 0 ? previous[1] : Real(0)),
        unit_x,
        unit_y,
        unit_z,
        radius,
        safe_radius,
        epsilon,
        derivatives + center * 3);

    const Real sqrt_two = sqrt(Real(2));
    Real power_real = Real(1);
    Real power_imag = Real(0);
    for (std::int64_t magnetic = 1;
         magnetic <= angular_momentum;
         ++magnetic) {
      const Real previous_power_real = power_real;
      const Real previous_power_imag = power_imag;
      power_real =
          previous_power_real * unit_x -
          previous_power_imag * unit_y;
      power_imag =
          previous_power_real * unit_y +
          previous_power_imag * unit_x;
      const Real polynomial = previous[magnetic];
      const Real polynomial_z =
          magnetic < angular_momentum
          ? previous[magnetic + 1]
          : Real(0);
      const Real scale = sqrt_two *
          spherical_normalization_device<Real>(
              angular_momentum,
              magnetic);
      const std::int64_t positive =
          edge * width + angular_momentum + magnetic;
      const std::int64_t negative =
          edge * width + angular_momentum - magnetic;

      values[positive] = scale * polynomial * power_real;
      store_cartesian_gradient(
          scale * polynomial * Real(magnetic) *
              previous_power_real,
          -scale * polynomial * Real(magnetic) *
              previous_power_imag,
          scale * polynomial_z * power_real,
          unit_x,
          unit_y,
          unit_z,
          radius,
          safe_radius,
          epsilon,
          derivatives + positive * 3);

      values[negative] = -scale * polynomial * power_imag;
      store_cartesian_gradient(
          -scale * polynomial * Real(magnetic) *
              previous_power_imag,
          -scale * polynomial * Real(magnetic) *
              previous_power_real,
          -scale * polynomial_z * power_imag,
          unit_x,
          unit_y,
          unit_z,
          radius,
          safe_radius,
          epsilon,
          derivatives + negative * 3);
    }
  }
}

template <typename Real>
__device__ inline void store_real_spherical_degree(
    const Real* polynomial_row,
    std::int64_t angular_momentum,
    Real unit_x,
    Real unit_y,
    Real unit_z,
    Real radius,
    Real safe_radius,
    Real epsilon,
    Real* values,
    Real* derivatives) {
  const Real norm_zero =
      spherical_normalization_device<Real>(angular_momentum, 0);
  values[angular_momentum] = norm_zero * polynomial_row[0];
  store_cartesian_gradient(
      Real(0),
      Real(0),
      norm_zero * (
          angular_momentum > 0 ? polynomial_row[1] : Real(0)),
      unit_x,
      unit_y,
      unit_z,
      radius,
      safe_radius,
      epsilon,
      derivatives + angular_momentum * 3);

  const Real sqrt_two = sqrt(Real(2));
  Real power_real = Real(1);
  Real power_imag = Real(0);
  for (std::int64_t magnetic = 1;
       magnetic <= angular_momentum;
       ++magnetic) {
    const Real previous_power_real = power_real;
    const Real previous_power_imag = power_imag;
    power_real =
        previous_power_real * unit_x -
        previous_power_imag * unit_y;
    power_imag =
        previous_power_real * unit_y +
        previous_power_imag * unit_x;
    const Real polynomial = polynomial_row[magnetic];
    const Real polynomial_z =
        magnetic < angular_momentum
        ? polynomial_row[magnetic + 1]
        : Real(0);
    const Real scale = sqrt_two *
        spherical_normalization_device<Real>(
            angular_momentum,
            magnetic);
    const std::int64_t positive =
        angular_momentum + magnetic;
    const std::int64_t negative =
        angular_momentum - magnetic;

    values[positive] = scale * polynomial * power_real;
    store_cartesian_gradient(
        scale * polynomial * Real(magnetic) *
            previous_power_real,
        -scale * polynomial * Real(magnetic) *
            previous_power_imag,
        scale * polynomial_z * power_real,
        unit_x,
        unit_y,
        unit_z,
        radius,
        safe_radius,
        epsilon,
        derivatives + positive * 3);

    values[negative] = -scale * polynomial * power_imag;
    store_cartesian_gradient(
        -scale * polynomial * Real(magnetic) *
            previous_power_imag,
        -scale * polynomial * Real(magnetic) *
            previous_power_real,
        -scale * polynomial_z * power_imag,
        unit_x,
        unit_y,
        unit_z,
        radius,
        safe_radius,
        epsilon,
        derivatives + negative * 3);
  }
}

template <typename Real>
__global__ void real_spherical_harmonics_table_kernel(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    Real* derivative_scratch,
    Real* values,
    Real* derivatives) {
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  const std::int64_t row_width =
      maximum_angular_momentum + 1;
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    Real* row_zero =
        derivative_scratch + edge * 3 * row_width;
    Real* row_one = row_zero + row_width;
    Real* row_two = row_one + row_width;
    for (std::int64_t index = 0; index < row_width; ++index) {
      row_zero[index] = Real(0);
      row_one[index] = Real(0);
      row_two[index] = Real(0);
    }
    Real* previous_previous = row_zero;
    Real* previous = row_one;
    Real* current = row_two;
    previous[0] = Real(1);

    const Real x = edge_vectors[edge * 3];
    const Real y = edge_vectors[edge * 3 + 1];
    const Real z = edge_vectors[edge * 3 + 2];
    const Real radius = sqrt(x * x + y * y + z * z);
    const Real safe_radius = radius > epsilon ? radius : epsilon;
    const Real unit_x = x / safe_radius;
    const Real unit_y = y / safe_radius;
    const Real unit_z = z / safe_radius;
    Real* value_row = values + edge * width;
    Real* derivative_row = derivatives + edge * width * 3;

    store_real_spherical_degree<Real>(
        previous,
        0,
        unit_x,
        unit_y,
        unit_z,
        radius,
        safe_radius,
        epsilon,
        value_row,
        derivative_row);

    for (std::int64_t degree = 1;
         degree <= maximum_angular_momentum;
         ++degree) {
      for (std::int64_t order = 0;
           order < row_width;
           ++order) {
        current[order] = Real(0);
      }
      const Real forward = Real(2 * degree - 1);
      const Real backward = Real(degree - 1);
      for (std::int64_t order = 0; order <= degree; ++order) {
        const Real same_order =
            order < degree ? previous[order] : Real(0);
        const Real lower_order =
            order > 0 ? previous[order - 1] : Real(0);
        const Real older =
            order < degree - 1
            ? previous_previous[order]
            : Real(0);
        current[order] = (
            forward * (
                unit_z * same_order +
                Real(order) * lower_order) -
            backward * older) /
            Real(degree);
      }
      Real* recycled = previous_previous;
      previous_previous = previous;
      previous = current;
      current = recycled;
      const std::int64_t offset = degree * degree;
      store_real_spherical_degree<Real>(
          previous,
          degree,
          unit_x,
          unit_y,
          unit_z,
          radius,
          safe_radius,
          epsilon,
          value_row + offset,
          derivative_row + offset * 3);
    }
  }
}

template <typename Real>
__global__ void real_to_complex_spherical_kernel(
    const Real* real_values,
    const Real* real_derivatives,
    std::int64_t edge_count,
    std::int64_t angular_momentum,
    c10::complex<Real>* values,
    c10::complex<Real>* derivatives) {
  const std::int64_t width = 2 * angular_momentum + 1;
  const std::int64_t total = edge_count * width;
  const Real inverse_sqrt_two = Real(1) / sqrt(Real(2));
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / width;
    const std::int64_t signed_m =
        index - edge * width - angular_momentum;
    if (signed_m == 0) {
      values[index] = c10::complex<Real>(
          real_values[index],
          Real(0));
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[index * 3 + axis] =
            c10::complex<Real>(
                real_derivatives[index * 3 + axis],
                Real(0));
      }
      continue;
    }
    const std::int64_t magnetic =
        signed_m < 0 ? -signed_m : signed_m;
    const std::int64_t negative =
        edge * width + angular_momentum - magnetic;
    const std::int64_t positive =
        edge * width + angular_momentum + magnetic;
    const Real sign = magnetic % 2 == 0 ? Real(1) : Real(-1);
    if (signed_m < 0) {
      values[index] = inverse_sqrt_two * c10::complex<Real>(
          real_values[positive],
          real_values[negative]);
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[index * 3 + axis] =
            inverse_sqrt_two * c10::complex<Real>(
                real_derivatives[positive * 3 + axis],
                real_derivatives[negative * 3 + axis]);
      }
    } else {
      values[index] = sign * inverse_sqrt_two *
          c10::complex<Real>(
              real_values[positive],
              -real_values[negative]);
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[index * 3 + axis] =
            sign * inverse_sqrt_two * c10::complex<Real>(
                real_derivatives[positive * 3 + axis],
                -real_derivatives[negative * 3 + axis]);
      }
    }
  }
}

template <typename Real>
__global__ void real_to_complex_spherical_table_kernel(
    const Real* real_values,
    const Real* real_derivatives,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    c10::complex<Real>* values,
    c10::complex<Real>* derivatives) {
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  const std::int64_t total = edge_count * width;
  const Real inverse_sqrt_two = Real(1) / sqrt(Real(2));
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / width;
    const std::int64_t component = index - edge * width;
    std::int64_t angular_momentum =
        static_cast<std::int64_t>(sqrt(static_cast<double>(component)));
    while (
        (angular_momentum + 1) * (angular_momentum + 1) <=
        component) {
      ++angular_momentum;
    }
    while (angular_momentum * angular_momentum > component) {
      --angular_momentum;
    }
    const std::int64_t degree_offset =
        angular_momentum * angular_momentum;
    const std::int64_t signed_m =
        component - degree_offset - angular_momentum;
    if (signed_m == 0) {
      values[index] = c10::complex<Real>(
          real_values[index],
          Real(0));
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[index * 3 + axis] =
            c10::complex<Real>(
                real_derivatives[index * 3 + axis],
                Real(0));
      }
      continue;
    }
    const std::int64_t magnetic =
        signed_m < 0 ? -signed_m : signed_m;
    const std::int64_t negative =
        edge * width + degree_offset +
        angular_momentum - magnetic;
    const std::int64_t positive =
        edge * width + degree_offset +
        angular_momentum + magnetic;
    const Real sign = magnetic % 2 == 0 ? Real(1) : Real(-1);
    if (signed_m < 0) {
      values[index] = inverse_sqrt_two * c10::complex<Real>(
          real_values[positive],
          real_values[negative]);
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[index * 3 + axis] =
            inverse_sqrt_two * c10::complex<Real>(
                real_derivatives[positive * 3 + axis],
                real_derivatives[negative * 3 + axis]);
      }
    } else {
      values[index] = sign * inverse_sqrt_two *
          c10::complex<Real>(
              real_values[positive],
              -real_values[negative]);
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        derivatives[index * 3 + axis] =
            sign * inverse_sqrt_two * c10::complex<Real>(
                real_derivatives[positive * 3 + axis],
                -real_derivatives[negative * 3 + axis]);
      }
    }
  }
}

template <typename Scalar>
__global__ void plain_site_basis_product_kernel(
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
    std::int64_t group_count,
    std::int64_t edge_count,
    std::int64_t term_count,
    std::int64_t channel_count,
    Scalar* edge_values,
    Scalar* edge_derivatives,
    Scalar* edge_charge_derivatives_center,
    Scalar* edge_charge_derivatives_neighbor) {
  const std::int64_t total = edge_count * term_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / term_count;
    const std::int64_t term = index - edge * term_count;
    const std::int64_t group = term_groups[term];
    const std::int64_t channel = term_channels[term];
    CUDA_KERNEL_ASSERT(group >= 0 && group < group_count);
    CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
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

template <typename Real>
__global__ void scheduled_radial_angular_channels_kernel(
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
  const std::int64_t total = edge_count * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / channel_count;
    const std::int64_t channel = index - edge * channel_count;
    const std::int64_t radial_index =
        channel_radial_indices[channel];
    const std::int64_t angular_index =
        channel_angular_indices[channel];
    CUDA_KERNEL_ASSERT(
        radial_index >= 0 && radial_index < radial_width);
    CUDA_KERNEL_ASSERT(
        angular_index >= 0 && angular_index < angular_width);
    const std::int64_t required_type = channel_types[channel];
    if (required_type >= 0 && edge_types[edge] != required_type) {
      edge_values[index] = Real(0);
      for (std::int64_t axis = 0; axis < 3; ++axis) {
        edge_derivatives[index * 3 + axis] = Real(0);
      }
      continue;
    }
    const std::int64_t radial_entry =
        edge * radial_width + radial_index;
    const std::int64_t angular_entry =
        edge * angular_width + angular_index;
    const Real radial = radial_values[radial_entry];
    const Real angular = angular_values[angular_entry];
    const Real scale = channel_scales[channel];
    edge_values[index] = scale * radial * angular;
    for (std::int64_t axis = 0; axis < 3; ++axis) {
      edge_derivatives[index * 3 + axis] = scale * (
          radial_derivatives[radial_entry] *
              radial_directions[edge * 3 + axis] *
              angular +
          radial *
              angular_derivatives[angular_entry * 3 + axis]);
    }
  }
}

template <typename Scalar>
__global__ void plain_site_basis_product_adjoint_kernel(
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
    std::int64_t group_count,
    std::int64_t edge_count,
    std::int64_t term_count,
    std::int64_t channel_count,
    Scalar* edge_position_adjoint,
    Scalar* edge_charge_adjoint_center,
    Scalar* edge_charge_adjoint_neighbor) {
  for (std::int64_t edge =
           blockIdx.x * blockDim.x + threadIdx.x;
       edge < edge_count;
       edge += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    Scalar position_x = Scalar(0);
    Scalar position_y = Scalar(0);
    Scalar position_z = Scalar(0);
    Scalar charge_center = Scalar(0);
    Scalar charge_neighbor = Scalar(0);
    for (std::int64_t term = 0; term < term_count; ++term) {
      const std::int64_t group = term_groups[term];
      const std::int64_t channel = term_channels[term];
      CUDA_KERNEL_ASSERT(group >= 0 && group < group_count);
      CUDA_KERNEL_ASSERT(channel >= 0 && channel < channel_count);
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
      const Scalar radial_derivative =
          radial_derivatives[group_entry];
      const Scalar weight = edge_weights[edge];
      const Scalar common =
          adjoint * prefactor;
      position_x +=
          common * (
              radial_derivative * radial_directions[edge * 3] *
                  angular +
              radial * angular_derivatives[term_entry * 3]) *
              weight +
          adjoint * unweighted_value *
              edge_weight_derivatives[edge * 3];
      position_y +=
          common * (
              radial_derivative *
                  radial_directions[edge * 3 + 1] * angular +
              radial *
                  angular_derivatives[term_entry * 3 + 1]) *
              weight +
          adjoint * unweighted_value *
              edge_weight_derivatives[edge * 3 + 1];
      position_z +=
          common * (
              radial_derivative *
                  radial_directions[edge * 3 + 2] * angular +
              radial *
                  angular_derivatives[term_entry * 3 + 2]) *
              weight +
          adjoint * unweighted_value *
              edge_weight_derivatives[edge * 3 + 2];
      const Scalar charge_common =
          adjoint * radial * angular * weight;
      charge_center +=
          charge_common *
          prefactor_derivatives_center[group_entry];
      charge_neighbor +=
          charge_common *
          prefactor_derivatives_neighbor[group_entry];
    }
    edge_position_adjoint[edge * 3] = position_x;
    edge_position_adjoint[edge * 3 + 1] = position_y;
    edge_position_adjoint[edge * 3 + 2] = position_z;
    edge_charge_adjoint_center[edge] = charge_center;
    edge_charge_adjoint_neighbor[edge] = charge_neighbor;
  }
}

__device__ inline std::int64_t compact_pair_index(
    std::int64_t first,
    std::int64_t second,
    std::int64_t dimension,
    bool antisymmetric) {
  if (antisymmetric) {
    return first * (dimension - 1) -
        first * (first - 1) / 2 +
        second - first - 1;
  }
  return first * dimension -
      first * (first - 1) / 2 +
      second - first;
}

__device__ inline void compact_pair_coordinates(
    std::int64_t output_index,
    std::int64_t dimension,
    bool antisymmetric,
    std::int64_t* first,
    std::int64_t* second) {
  std::int64_t remaining = output_index;
  for (std::int64_t row = 0; row < dimension; ++row) {
    const std::int64_t row_length =
        antisymmetric ? dimension - row - 1 : dimension - row;
    if (remaining < row_length) {
      *first = row;
      *second = row + remaining + (antisymmetric ? 1 : 0);
      return;
    }
    remaining -= row_length;
  }
  *first = 0;
  *second = 0;
}

template <typename Scalar>
__global__ void compact_pair_product_kernel(
    const Scalar* left,
    const Scalar* right,
    std::int64_t batch_size,
    std::int64_t dimension,
    bool antisymmetric,
    std::int64_t output_dimension,
    Scalar* output) {
  const std::int64_t total = batch_size * output_dimension;
  const Scalar inverse_sqrt_two =
      Scalar(0.707106781186547524400844362104849039);
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_dimension;
    const std::int64_t output_index =
        index - batch * output_dimension;
    std::int64_t first = 0;
    std::int64_t second = 0;
    compact_pair_coordinates(
        output_index,
        dimension,
        antisymmetric,
        &first,
        &second);
    const Scalar* left_row = left + batch * dimension;
    const Scalar* right_row = right + batch * dimension;
    if (!antisymmetric && first == second) {
      output[index] = left_row[first] * right_row[first];
    } else if (antisymmetric) {
      output[index] = inverse_sqrt_two * (
          left_row[first] * right_row[second] -
          left_row[second] * right_row[first]);
    } else {
      output[index] = inverse_sqrt_two * (
          left_row[first] * right_row[second] +
          left_row[second] * right_row[first]);
    }
  }
}

template <typename Scalar>
__global__ void compact_pair_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* left,
    const Scalar* right,
    std::int64_t batch_size,
    std::int64_t dimension,
    bool antisymmetric,
    std::int64_t output_dimension,
    Scalar* left_adjoint,
    Scalar* right_adjoint) {
  const std::int64_t total = batch_size * dimension;
  const Scalar inverse_sqrt_two =
      Scalar(0.707106781186547524400844362104849039);
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / dimension;
    const std::int64_t coordinate = index - batch * dimension;
    const Scalar* output_row =
        output_adjoint + batch * output_dimension;
    const Scalar* left_row = left + batch * dimension;
    const Scalar* right_row = right + batch * dimension;
    Scalar left_value = Scalar(0);
    Scalar right_value = Scalar(0);

    if (!antisymmetric) {
      const std::int64_t diagonal = compact_pair_index(
          coordinate,
          coordinate,
          dimension,
          false);
      const Scalar gradient = output_row[diagonal];
      left_value +=
          gradient * conjugate_value(right_row[coordinate]);
      right_value +=
          gradient * conjugate_value(left_row[coordinate]);
    }

    for (std::int64_t other = coordinate + 1;
         other < dimension;
         ++other) {
      const std::int64_t output_index = compact_pair_index(
          coordinate,
          other,
          dimension,
          antisymmetric);
      const Scalar gradient = output_row[output_index];
      left_value += inverse_sqrt_two * gradient *
          conjugate_value(right_row[other]);
      right_value += (antisymmetric ? -inverse_sqrt_two : inverse_sqrt_two) *
          gradient * conjugate_value(left_row[other]);
    }
    for (std::int64_t other = 0;
         other < coordinate;
         ++other) {
      const std::int64_t output_index = compact_pair_index(
          other,
          coordinate,
          dimension,
          antisymmetric);
      const Scalar gradient = output_row[output_index];
      left_value += (antisymmetric ? -inverse_sqrt_two : inverse_sqrt_two) *
          gradient * conjugate_value(right_row[other]);
      right_value += inverse_sqrt_two * gradient *
          conjugate_value(left_row[other]);
    }
    left_adjoint[index] = left_value;
    right_adjoint[index] = right_value;
  }
}

__device__ inline std::int64_t binomial_coefficient_device(
    std::int64_t dimension,
    std::int64_t order) {
  if (order < 0 || order > dimension) {
    return 0;
  }
  order = order < dimension - order ? order : dimension - order;
  std::int64_t value = 1;
  for (std::int64_t index = 1; index <= order; ++index) {
    value = value * (dimension - order + index) / index;
  }
  return value;
}

__device__ inline void exterior_combination_from_index(
    std::int64_t combination_index,
    std::int64_t dimension,
    std::int64_t order,
    std::int64_t* combination) {
  std::int64_t minimum = 0;
  for (std::int64_t position = 0; position < order; ++position) {
    const std::int64_t remaining = order - position - 1;
    const std::int64_t last = dimension - remaining;
    for (std::int64_t candidate = minimum;
         candidate < last;
         ++candidate) {
      const std::int64_t count = binomial_coefficient_device(
          dimension - candidate - 1,
          remaining);
      if (combination_index < count) {
        combination[position] = candidate;
        minimum = candidate + 1;
        break;
      }
      combination_index -= count;
    }
  }
}

template <typename Scalar>
__device__ inline Scalar exterior_determinant_subset(
    const Scalar* matrix,
    std::int64_t order) {
  if (order == 0) {
    return Scalar(1);
  }
  Scalar states[256];
  const std::int64_t state_count =
      std::int64_t(1) << order;
  for (std::int64_t state = 0; state < state_count; ++state) {
    states[state] = Scalar(0);
  }
  states[0] = Scalar(1);
  for (std::int64_t mask = 0; mask < state_count; ++mask) {
    const std::int64_t row = static_cast<std::int64_t>(
        __popcll(static_cast<unsigned long long>(mask)));
    if (row >= order) {
      continue;
    }
    for (std::int64_t column = 0; column < order; ++column) {
      const std::int64_t bit = std::int64_t(1) << column;
      if ((mask & bit) != 0) {
        continue;
      }
      const std::int64_t larger_mask =
          mask & ~((bit << 1) - 1);
      const bool negative = (
          __popcll(static_cast<unsigned long long>(larger_mask)) & 1
      ) != 0;
      const Scalar term =
          states[mask] * matrix[row * order + column];
      states[mask | bit] += negative ? -term : term;
    }
  }
  return states[state_count - 1];
}

template <typename Scalar>
__device__ inline Scalar exterior_determinant_gaussian(
    const Scalar* matrix,
    std::int64_t order) {
  if (order == 0) {
    return Scalar(1);
  }
  Scalar working[64];
  for (std::int64_t index = 0; index < order * order; ++index) {
    working[index] = matrix[index];
  }
  Scalar determinant = Scalar(1);
  bool negative = false;
  for (std::int64_t pivot = 0; pivot < order; ++pivot) {
    std::int64_t pivot_row = pivot;
    double pivot_magnitude =
        squared_magnitude(working[pivot * order + pivot]);
    for (std::int64_t row = pivot + 1; row < order; ++row) {
      const double candidate =
          squared_magnitude(working[row * order + pivot]);
      if (candidate > pivot_magnitude) {
        pivot_magnitude = candidate;
        pivot_row = row;
      }
    }
    if (pivot_magnitude == 0.0) {
      return Scalar(0);
    }
    if (pivot_row != pivot) {
      for (std::int64_t column = pivot; column < order; ++column) {
        const Scalar temporary =
            working[pivot * order + column];
        working[pivot * order + column] =
            working[pivot_row * order + column];
        working[pivot_row * order + column] = temporary;
      }
      negative = !negative;
    }
    const Scalar pivot_value =
        working[pivot * order + pivot];
    determinant *= pivot_value;
    for (std::int64_t row = pivot + 1; row < order; ++row) {
      const Scalar scale =
          working[row * order + pivot] / pivot_value;
      for (std::int64_t column = pivot + 1;
           column < order;
           ++column) {
        working[row * order + column] -=
            scale * working[pivot * order + column];
      }
    }
  }
  return negative ? -determinant : determinant;
}

template <typename Scalar>
__device__ inline void exterior_cofactor_matrix(
    const Scalar* matrix,
    std::int64_t order,
    Scalar* cofactors) {
  if (order == 1) {
    cofactors[0] = Scalar(1);
    return;
  }
  Scalar working[64];
  Scalar inverse[64];
  for (std::int64_t row = 0; row < order; ++row) {
    for (std::int64_t column = 0; column < order; ++column) {
      working[row * order + column] =
          matrix[row * order + column];
      inverse[row * order + column] =
          row == column ? Scalar(1) : Scalar(0);
    }
  }
  Scalar determinant = Scalar(1);
  bool negative = false;
  bool singular = false;
  for (std::int64_t pivot = 0; pivot < order; ++pivot) {
    std::int64_t pivot_row = pivot;
    double pivot_magnitude =
        squared_magnitude(working[pivot * order + pivot]);
    for (std::int64_t row = pivot + 1; row < order; ++row) {
      const double candidate =
          squared_magnitude(working[row * order + pivot]);
      if (candidate > pivot_magnitude) {
        pivot_magnitude = candidate;
        pivot_row = row;
      }
    }
    if (pivot_magnitude == 0.0) {
      singular = true;
      break;
    }
    if (pivot_row != pivot) {
      for (std::int64_t column = 0; column < order; ++column) {
        Scalar temporary = working[pivot * order + column];
        working[pivot * order + column] =
            working[pivot_row * order + column];
        working[pivot_row * order + column] = temporary;
        temporary = inverse[pivot * order + column];
        inverse[pivot * order + column] =
            inverse[pivot_row * order + column];
        inverse[pivot_row * order + column] = temporary;
      }
      negative = !negative;
    }
    const Scalar pivot_value =
        working[pivot * order + pivot];
    determinant *= pivot_value;
    for (std::int64_t column = 0; column < order; ++column) {
      working[pivot * order + column] /= pivot_value;
      inverse[pivot * order + column] /= pivot_value;
    }
    for (std::int64_t row = 0; row < order; ++row) {
      if (row == pivot) {
        continue;
      }
      const Scalar scale =
          working[row * order + pivot];
      for (std::int64_t column = 0; column < order; ++column) {
        working[row * order + column] -=
            scale * working[pivot * order + column];
        inverse[row * order + column] -=
            scale * inverse[pivot * order + column];
      }
    }
  }
  if (!singular) {
    if (negative) {
      determinant = -determinant;
    }
    for (std::int64_t row = 0; row < order; ++row) {
      for (std::int64_t column = 0; column < order; ++column) {
        cofactors[row * order + column] =
            determinant * inverse[column * order + row];
      }
    }
    return;
  }
  for (std::int64_t removed_row = 0;
       removed_row < order;
       ++removed_row) {
    for (std::int64_t removed_column = 0;
         removed_column < order;
         ++removed_column) {
      Scalar minor[49];
      std::int64_t minor_index = 0;
      for (std::int64_t row = 0; row < order; ++row) {
        if (row == removed_row) {
          continue;
        }
        for (std::int64_t column = 0; column < order; ++column) {
          if (column == removed_column) {
            continue;
          }
          minor[minor_index] = matrix[row * order + column];
          ++minor_index;
        }
      }
      Scalar cofactor = exterior_determinant_gaussian(
          minor,
          order - 1);
      if (((removed_row + removed_column) & 1) != 0) {
        cofactor = -cofactor;
      }
      cofactors[removed_row * order + removed_column] =
          cofactor;
    }
  }
}

template <typename Scalar>
__global__ void compact_exterior_power_kernel(
    const Scalar* factors,
    std::int64_t batch_size,
    std::int64_t order,
    std::int64_t dimension,
    std::int64_t output_dimension,
    Scalar normalization,
    Scalar* output) {
  const std::int64_t total = batch_size * output_dimension;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_dimension;
    const std::int64_t coordinate = index - batch * output_dimension;
    std::int64_t combination[8];
    Scalar matrix[64];
    exterior_combination_from_index(
        coordinate,
        dimension,
        order,
        combination);
    const Scalar* factor_batch =
        factors + batch * order * dimension;
    for (std::int64_t row = 0; row < order; ++row) {
      for (std::int64_t column = 0; column < order; ++column) {
        matrix[row * order + column] =
            factor_batch[
                row * dimension + combination[column]];
      }
    }
    output[index] =
        normalization * exterior_determinant_gaussian(matrix, order);
  }
}

template <typename Scalar>
__global__ void compact_exterior_power_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* factors,
    std::int64_t batch_size,
    std::int64_t order,
    std::int64_t dimension,
    std::int64_t output_dimension,
    Scalar normalization,
    Scalar* factors_adjoint) {
  const std::int64_t total = batch_size * output_dimension;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_dimension;
    const std::int64_t output_index =
        index - batch * output_dimension;
    const Scalar* factor_batch =
        factors + batch * order * dimension;
    std::int64_t combination[8];
    Scalar matrix[64];
    Scalar cofactors[64];
    exterior_combination_from_index(
        output_index,
        dimension,
        order,
        combination);
    for (std::int64_t row = 0; row < order; ++row) {
      for (std::int64_t column = 0; column < order; ++column) {
        matrix[row * order + column] =
            factor_batch[
                row * dimension + combination[column]];
      }
    }
    exterior_cofactor_matrix(matrix, order, cofactors);
    const Scalar gradient = output_adjoint[index];
    for (std::int64_t row = 0; row < order; ++row) {
      for (std::int64_t column = 0; column < order; ++column) {
        atomic_add_value(
            factors_adjoint +
                batch * order * dimension +
                row * dimension +
                combination[column],
            gradient * conjugate_value(
                normalization *
                cofactors[row * order + column]));
      }
    }
  }
}

template <typename Scalar>
__device__ inline Scalar integer_power_device(
    Scalar base,
    std::int64_t exponent) {
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

template <typename Scalar>
__global__ void symmetric_power_monomial_kernel(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    const std::int64_t* output_offsets,
    const Scalar* monomial_values,
    std::int64_t term_count,
    std::int64_t output_dimension,
    Scalar* output) {
  const std::int64_t total = batch_size * output_dimension;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_dimension;
    const std::int64_t output_index =
        index - batch * output_dimension;
    const std::int64_t start = output_offsets[output_index];
    const std::int64_t finish = output_offsets[output_index + 1];
    CUDA_KERNEL_ASSERT(
        start >= 0 && start <= finish && finish <= term_count);
    const Scalar* input_row = input + batch * input_dimension;
    Scalar value = Scalar(0);
    for (std::int64_t term = start; term < finish; ++term) {
      Scalar monomial = monomial_values[term];
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component];
        CUDA_KERNEL_ASSERT(exponent >= 0);
        monomial *= integer_power_device(
            input_row[component],
            exponent);
      }
      value += monomial;
    }
    output[index] = value;
  }
}

template <typename Scalar>
__global__ void symmetric_power_monomial_adjoint_kernel(
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
  const std::int64_t total = batch_size * term_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / term_count;
    const std::int64_t term = index - batch * term_count;
    const Scalar* input_row = input + batch * input_dimension;
    Scalar monomial = monomial_values[term];
    std::int64_t zero_count = 0;
    std::int64_t zero_component = -1;
    for (std::int64_t component = 0;
         component < input_dimension;
         ++component) {
      const std::int64_t exponent =
          monomial_counts[term * input_dimension + component];
      CUDA_KERNEL_ASSERT(exponent >= 0);
      if (exponent == 0) {
        continue;
      }
      if (input_row[component] == Scalar(0)) {
        ++zero_count;
        zero_component = component;
        continue;
      }
      monomial *= integer_power_device(
          input_row[component],
          exponent);
    }
    if (zero_count >= 2) {
      continue;
    }
    const std::int64_t output_index = output_indices[term];
    CUDA_KERNEL_ASSERT(
        output_index >= 0 && output_index < output_dimension);
    const Scalar output_gradient =
        output_adjoint[
            batch * output_dimension + output_index];
    if (zero_count == 1) {
      const std::int64_t exponent =
          monomial_counts[
              term * input_dimension + zero_component];
      if (exponent == 1) {
        atomic_add_value(
            input_adjoint +
                batch * input_dimension + zero_component,
            output_gradient * conjugate_value(monomial));
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
      atomic_add_value(
          input_adjoint + batch * input_dimension + component,
          output_gradient * conjugate_value(derivative));
    }
  }
}

template <typename Scalar>
__global__ void symmetric_power_monomial_double_backward_kernel(
    const Scalar* input_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    const std::int64_t* output_indices,
    const Scalar* monomial_values,
    std::int64_t term_count,
    std::int64_t output_dimension,
    Scalar* output_tangent,
    Scalar* input_tangent) {
  const std::int64_t total = batch_size * term_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / term_count;
    const std::int64_t term = index - batch * term_count;
    const std::int64_t output_index = output_indices[term];
    CUDA_KERNEL_ASSERT(
        output_index >= 0 && output_index < output_dimension);
    const Scalar* input_row = input + batch * input_dimension;
    const Scalar* tangent_row =
        input_adjoint_tangent + batch * input_dimension;
    const Scalar output_gradient =
        output_adjoint[batch * output_dimension + output_index];
    const Scalar coefficient = monomial_values[term];

    for (std::int64_t first = 0;
         first < input_dimension;
         ++first) {
      const std::int64_t first_exponent =
          monomial_counts[term * input_dimension + first];
      CUDA_KERNEL_ASSERT(first_exponent >= 0);
      if (first_exponent == 0) {
        continue;
      }
      Scalar first_derivative =
          coefficient * Scalar(first_exponent);
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component] -
            (component == first ? 1 : 0);
        CUDA_KERNEL_ASSERT(exponent >= 0);
        first_derivative *= integer_power_device(
            input_row[component], exponent);
      }
      atomic_add_value(
          output_tangent + batch * output_dimension + output_index,
          tangent_row[first] * first_derivative);

      for (std::int64_t second = 0;
           second < input_dimension;
           ++second) {
        const std::int64_t second_factor =
            monomial_counts[term * input_dimension + second] -
            (second == first ? 1 : 0);
        if (second_factor <= 0) {
          continue;
        }
        Scalar second_derivative =
            coefficient * Scalar(first_exponent * second_factor);
        for (std::int64_t component = 0;
             component < input_dimension;
             ++component) {
          const std::int64_t exponent =
              monomial_counts[term * input_dimension + component] -
              (component == first ? 1 : 0) -
              (component == second ? 1 : 0);
          CUDA_KERNEL_ASSERT(exponent >= 0);
          second_derivative *= integer_power_device(
              input_row[component], exponent);
        }
        atomic_add_value(
            input_tangent + batch * input_dimension + second,
            output_gradient * conjugate_value(tangent_row[first]) *
                conjugate_value(second_derivative));
      }
    }
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_monomial_double_backward_kernel(
    const Scalar* input_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    const std::int64_t* coefficient_terms,
    const std::int64_t* coefficient_outputs,
    const Scalar* coefficient_values,
    std::int64_t monomial_count,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* output_tangent,
    Scalar* input_tangent) {
  const std::int64_t total = batch_size * coefficient_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / coefficient_count;
    const std::int64_t coefficient_index =
        index - batch * coefficient_count;
    const std::int64_t term =
        coefficient_terms[coefficient_index];
    const std::int64_t output_index =
        coefficient_outputs[coefficient_index];
    CUDA_KERNEL_ASSERT(term >= 0 && term < monomial_count);
    CUDA_KERNEL_ASSERT(
        output_index >= 0 && output_index < output_dimension);
    const Scalar* input_row = input + batch * input_dimension;
    const Scalar* tangent_row =
        input_adjoint_tangent + batch * input_dimension;
    const Scalar output_gradient =
        output_adjoint[batch * output_dimension + output_index];
    const Scalar coefficient = coefficient_values[coefficient_index];
    Scalar monomial = coefficient;
    std::int64_t zero_degree = 0;
    std::int64_t first_zero_component = -1;
    std::int64_t second_zero_component = -1;
    for (std::int64_t component = 0;
         component < input_dimension;
         ++component) {
      const std::int64_t exponent =
          monomial_counts[term * input_dimension + component];
      CUDA_KERNEL_ASSERT(exponent >= 0);
      if (exponent == 0) {
        continue;
      }
      if (input_row[component] == Scalar(0)) {
        zero_degree += exponent;
        if (first_zero_component < 0) {
          first_zero_component = component;
        } else if (second_zero_component < 0) {
          second_zero_component = component;
        }
      } else {
        monomial *= integer_power_device(
            input_row[component], exponent);
      }
    }

    Scalar output_directional = Scalar(0);
    if (zero_degree <= 1) {
      for (std::int64_t first = 0;
           first < input_dimension;
           ++first) {
        const std::int64_t first_exponent =
            monomial_counts[term * input_dimension + first];
        if (first_exponent == 0) {
          continue;
        }
        Scalar first_derivative = Scalar(0);
        if (zero_degree == 0) {
          first_derivative =
              Scalar(first_exponent) * monomial / input_row[first];
        } else if (
            first == first_zero_component && first_exponent == 1) {
          first_derivative = monomial;
        }
        output_directional += tangent_row[first] * first_derivative;
      }
    }
    atomic_add_value(
        output_tangent + batch * output_dimension + output_index,
        output_directional);

    if (zero_degree > 2) {
      continue;
    }
    for (std::int64_t second = 0;
         second < input_dimension;
         ++second) {
      const std::int64_t second_exponent =
          monomial_counts[term * input_dimension + second];
      if (second_exponent == 0) {
        continue;
      }
      Scalar directional_second = Scalar(0);
      for (std::int64_t first = 0;
           first < input_dimension;
           ++first) {
        const std::int64_t first_exponent =
            monomial_counts[term * input_dimension + first];
        if (first_exponent == 0) {
          continue;
        }
        Scalar second_derivative = Scalar(0);
        if (zero_degree == 0) {
          if (first == second) {
            if (first_exponent >= 2) {
              second_derivative =
                  Scalar(first_exponent * (first_exponent - 1)) *
                  monomial /
                  (input_row[first] * input_row[first]);
            }
          } else {
            second_derivative =
                Scalar(first_exponent * second_exponent) * monomial /
                (input_row[first] * input_row[second]);
          }
        } else if (zero_degree == 1) {
          if (first == first_zero_component &&
              second != first_zero_component) {
            second_derivative =
                Scalar(second_exponent) * monomial / input_row[second];
          } else if (
              second == first_zero_component &&
              first != first_zero_component) {
            second_derivative =
                Scalar(first_exponent) * monomial / input_row[first];
          }
        } else if (second_zero_component < 0) {
          if (first == first_zero_component && first == second) {
            second_derivative = Scalar(2) * monomial;
          }
        } else if (
            (first == first_zero_component &&
             second == second_zero_component) ||
            (first == second_zero_component &&
             second == first_zero_component)) {
          second_derivative = monomial;
        }
        directional_second +=
            conjugate_value(tangent_row[first]) *
            conjugate_value(second_derivative);
      }
      atomic_add_value(
          input_tangent + batch * input_dimension + second,
          output_gradient * directional_second);
    }
  }
}

template <typename Scalar>
__global__ void
symmetric_power_shared_monomial_factored_double_backward_kernel(
    const Scalar* input_adjoint_tangent,
    const Scalar* input,
    const std::int64_t* monomial_counts,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    std::int64_t monomial_count,
    Scalar* monomial_adjoint_and_directional,
    Scalar* input_tangent) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const Scalar term_gradient = monomial_adjoint_and_directional[index];
    const Scalar* input_row = input + batch * input_dimension;
    const Scalar* tangent_row =
        input_adjoint_tangent + batch * input_dimension;
    Scalar monomial = Scalar(1);
    std::int64_t zero_degree = 0;
    std::int64_t first_zero_component = -1;
    std::int64_t second_zero_component = -1;
    for (std::int64_t component = 0;
         component < input_dimension;
         ++component) {
      const std::int64_t exponent =
          monomial_counts[term * input_dimension + component];
      CUDA_KERNEL_ASSERT(exponent >= 0);
      if (exponent == 0) {
        continue;
      }
      if (input_row[component] == Scalar(0)) {
        zero_degree += exponent;
        if (first_zero_component < 0) {
          first_zero_component = component;
        } else if (second_zero_component < 0) {
          second_zero_component = component;
        }
      } else {
        monomial *= integer_power_device(
            input_row[component], exponent);
      }
    }

    Scalar directional = Scalar(0);
    if (zero_degree == 0) {
      Scalar logarithmic_directional = Scalar(0);
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        logarithmic_directional +=
            Scalar(exponent) * tangent_row[component] /
            input_row[component];
      }
      directional = monomial * logarithmic_directional;
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        const Scalar hessian_direction =
            Scalar(exponent) * monomial / input_row[component] *
            (logarithmic_directional -
             tangent_row[component] / input_row[component]);
        atomic_add_value(
            input_tangent + batch * input_dimension + component,
            term_gradient * conjugate_value(hessian_direction));
      }
    } else if (zero_degree == 1) {
      directional = tangent_row[first_zero_component] * monomial;
      Scalar zero_component_direction = Scalar(0);
      for (std::int64_t component = 0;
           component < input_dimension;
           ++component) {
        if (component == first_zero_component) {
          continue;
        }
        const std::int64_t exponent =
            monomial_counts[term * input_dimension + component];
        if (exponent == 0) {
          continue;
        }
        const Scalar derivative =
            Scalar(exponent) * monomial / input_row[component];
        zero_component_direction +=
            tangent_row[component] * derivative;
        atomic_add_value(
            input_tangent + batch * input_dimension + component,
            term_gradient * conjugate_value(
                tangent_row[first_zero_component] * derivative));
      }
      atomic_add_value(
          input_tangent + batch * input_dimension + first_zero_component,
          term_gradient * conjugate_value(zero_component_direction));
    } else if (zero_degree == 2) {
      if (second_zero_component < 0) {
        const Scalar hessian_direction =
            Scalar(2) * monomial * tangent_row[first_zero_component];
        atomic_add_value(
            input_tangent + batch * input_dimension + first_zero_component,
            term_gradient * conjugate_value(hessian_direction));
      } else {
        atomic_add_value(
            input_tangent + batch * input_dimension + first_zero_component,
            term_gradient * conjugate_value(
                monomial * tangent_row[second_zero_component]));
        atomic_add_value(
            input_tangent + batch * input_dimension + second_zero_component,
            term_gradient * conjugate_value(
                monomial * tangent_row[first_zero_component]));
      }
    }
    monomial_adjoint_and_directional[index] = directional;
  }
}

// Sparse support encodes m_t(x) = product_q x[c_q]^a_q without scanning
// zero exponents across the full physical channel dimension.
template <typename Scalar>
__global__ void symmetric_power_shared_sparse_values_kernel(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* term_offsets,
    const std::int64_t* term_components,
    const std::int64_t* term_exponents,
    std::int64_t monomial_count,
    std::int64_t support_count,
    Scalar* monomial_values) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const std::int64_t start = term_offsets[term];
    const std::int64_t stop = term_offsets[term + 1];
    CUDA_KERNEL_ASSERT(start >= 0 && start <= stop && stop <= support_count);
    const Scalar* input_row = input + batch * input_dimension;
    Scalar value = Scalar(1);
    for (std::int64_t support = start; support < stop; ++support) {
      const std::int64_t component = term_components[support];
      const std::int64_t exponent = term_exponents[support];
      CUDA_KERNEL_ASSERT(component >= 0 && component < input_dimension);
      CUDA_KERNEL_ASSERT(exponent > 0);
      value *= integer_power_device(input_row[component], exponent);
    }
    monomial_values[index] = value;
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_sparse_input_adjoint_kernel(
    const Scalar* monomial_adjoint,
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* term_offsets,
    const std::int64_t* term_components,
    const std::int64_t* term_exponents,
    std::int64_t monomial_count,
    std::int64_t support_count,
    Scalar* input_adjoint) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const std::int64_t start = term_offsets[term];
    const std::int64_t stop = term_offsets[term + 1];
    CUDA_KERNEL_ASSERT(start >= 0 && start <= stop && stop <= support_count);
    const Scalar* input_row = input + batch * input_dimension;
    const Scalar term_gradient = monomial_adjoint[index];
    Scalar monomial = Scalar(1);
    std::int64_t zero_degree = 0;
    std::int64_t zero_component = -1;
    for (std::int64_t support = start; support < stop; ++support) {
      const std::int64_t component = term_components[support];
      const std::int64_t exponent = term_exponents[support];
      CUDA_KERNEL_ASSERT(component >= 0 && component < input_dimension);
      CUDA_KERNEL_ASSERT(exponent > 0);
      if (input_row[component] == Scalar(0)) {
        zero_degree += exponent;
        zero_component = component;
      } else {
        monomial *= integer_power_device(input_row[component], exponent);
      }
    }
    if (zero_degree == 1) {
      atomic_add_value(
          input_adjoint + batch * input_dimension + zero_component,
          term_gradient * conjugate_value(monomial));
      continue;
    }
    if (zero_degree != 0) {
      continue;
    }
    for (std::int64_t support = start; support < stop; ++support) {
      const std::int64_t component = term_components[support];
      const std::int64_t exponent = term_exponents[support];
      const Scalar derivative =
          Scalar(exponent) * monomial / input_row[component];
      atomic_add_value(
          input_adjoint + batch * input_dimension + component,
          term_gradient * conjugate_value(derivative));
    }
  }
}

template <typename Scalar>
__global__ void
symmetric_power_shared_sparse_factored_double_backward_kernel(
    const Scalar* input_adjoint_tangent,
    const Scalar* input,
    const std::int64_t* term_offsets,
    const std::int64_t* term_components,
    const std::int64_t* term_exponents,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    std::int64_t monomial_count,
    std::int64_t support_count,
    Scalar* monomial_adjoint_and_directional,
    Scalar* input_tangent) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const std::int64_t start = term_offsets[term];
    const std::int64_t stop = term_offsets[term + 1];
    CUDA_KERNEL_ASSERT(start >= 0 && start <= stop && stop <= support_count);
    const Scalar term_gradient = monomial_adjoint_and_directional[index];
    const Scalar* input_row = input + batch * input_dimension;
    const Scalar* tangent_row =
        input_adjoint_tangent + batch * input_dimension;
    Scalar monomial = Scalar(1);
    std::int64_t zero_degree = 0;
    std::int64_t first_zero_component = -1;
    std::int64_t second_zero_component = -1;
    for (std::int64_t support = start; support < stop; ++support) {
      const std::int64_t component = term_components[support];
      const std::int64_t exponent = term_exponents[support];
      CUDA_KERNEL_ASSERT(component >= 0 && component < input_dimension);
      CUDA_KERNEL_ASSERT(exponent > 0);
      if (input_row[component] == Scalar(0)) {
        zero_degree += exponent;
        if (first_zero_component < 0) {
          first_zero_component = component;
        } else if (second_zero_component < 0) {
          second_zero_component = component;
        }
      } else {
        monomial *= integer_power_device(input_row[component], exponent);
      }
    }

    Scalar directional = Scalar(0);
    if (zero_degree == 0) {
      Scalar logarithmic_directional = Scalar(0);
      for (std::int64_t support = start; support < stop; ++support) {
        const std::int64_t component = term_components[support];
        const std::int64_t exponent = term_exponents[support];
        logarithmic_directional +=
            Scalar(exponent) * tangent_row[component] /
            input_row[component];
      }
      directional = monomial * logarithmic_directional;
      for (std::int64_t support = start; support < stop; ++support) {
        const std::int64_t component = term_components[support];
        const std::int64_t exponent = term_exponents[support];
        const Scalar hessian_direction =
            Scalar(exponent) * monomial / input_row[component] *
            (logarithmic_directional -
             tangent_row[component] / input_row[component]);
        atomic_add_value(
            input_tangent + batch * input_dimension + component,
            term_gradient * conjugate_value(hessian_direction));
      }
    } else if (zero_degree == 1) {
      directional = tangent_row[first_zero_component] * monomial;
      Scalar zero_component_direction = Scalar(0);
      for (std::int64_t support = start; support < stop; ++support) {
        const std::int64_t component = term_components[support];
        if (component == first_zero_component) {
          continue;
        }
        const std::int64_t exponent = term_exponents[support];
        const Scalar derivative =
            Scalar(exponent) * monomial / input_row[component];
        zero_component_direction +=
            tangent_row[component] * derivative;
        atomic_add_value(
            input_tangent + batch * input_dimension + component,
            term_gradient * conjugate_value(
                tangent_row[first_zero_component] * derivative));
      }
      atomic_add_value(
          input_tangent + batch * input_dimension + first_zero_component,
          term_gradient * conjugate_value(zero_component_direction));
    } else if (zero_degree == 2) {
      if (second_zero_component < 0) {
        atomic_add_value(
            input_tangent + batch * input_dimension + first_zero_component,
            term_gradient * conjugate_value(
                Scalar(2) * monomial *
                tangent_row[first_zero_component]));
      } else {
        atomic_add_value(
            input_tangent + batch * input_dimension + first_zero_component,
            term_gradient * conjugate_value(
                monomial * tangent_row[second_zero_component]));
        atomic_add_value(
            input_tangent + batch * input_dimension + second_zero_component,
            term_gradient * conjugate_value(
                monomial * tangent_row[first_zero_component]));
      }
    }
    monomial_adjoint_and_directional[index] = directional;
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_values_kernel(
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    std::int64_t monomial_count,
    Scalar* monomial_values) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const Scalar* input_row = input + batch * input_dimension;
    Scalar value = Scalar(1);
    for (std::int64_t component = 0;
         component < input_dimension;
         ++component) {
      const std::int64_t exponent =
          monomial_counts[term * input_dimension + component];
      CUDA_KERNEL_ASSERT(exponent >= 0);
      value *= integer_power_device(
          input_row[component],
          exponent);
    }
    monomial_values[index] = value;
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_output_kernel(
    const Scalar* monomial_values,
    std::int64_t batch_size,
    std::int64_t monomial_count,
    const std::int64_t* output_offsets,
    const std::int64_t* coefficient_terms,
    const Scalar* coefficient_values,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* output) {
  const std::int64_t total = batch_size * output_dimension;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / output_dimension;
    const std::int64_t output_index =
        index - batch * output_dimension;
    const std::int64_t start = output_offsets[output_index];
    const std::int64_t finish = output_offsets[output_index + 1];
    CUDA_KERNEL_ASSERT(
        start >= 0 &&
        start <= finish &&
        finish <= coefficient_count);
    Scalar value = Scalar(0);
    for (std::int64_t coefficient = start;
         coefficient < finish;
         ++coefficient) {
      const std::int64_t term = coefficient_terms[coefficient];
      CUDA_KERNEL_ASSERT(
          term >= 0 && term < monomial_count);
      value +=
          coefficient_values[coefficient] *
          monomial_values[batch * monomial_count + term];
    }
    output[index] = value;
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_term_adjoint_kernel(
    const Scalar* output_adjoint,
    const std::int64_t* coefficient_terms,
    const std::int64_t* coefficient_outputs,
    const Scalar* coefficient_values,
    std::int64_t batch_size,
    std::int64_t monomial_count,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* monomial_adjoint) {
  const std::int64_t total = batch_size * coefficient_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / coefficient_count;
    const std::int64_t coefficient =
        index - batch * coefficient_count;
    const std::int64_t term = coefficient_terms[coefficient];
    const std::int64_t output_index =
        coefficient_outputs[coefficient];
    CUDA_KERNEL_ASSERT(
        term >= 0 && term < monomial_count);
    CUDA_KERNEL_ASSERT(
        output_index >= 0 && output_index < output_dimension);
    atomic_add_value(
        monomial_adjoint + batch * monomial_count + term,
        output_adjoint[
            batch * output_dimension + output_index] *
            conjugate_value(coefficient_values[coefficient]));
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_input_adjoint_kernel(
    const Scalar* monomial_adjoint,
    const Scalar* input,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    std::int64_t monomial_count,
    Scalar* input_adjoint) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const Scalar term_gradient =
        monomial_adjoint[index];
    const Scalar* input_row = input + batch * input_dimension;
    Scalar monomial = Scalar(1);
    std::int64_t zero_count = 0;
    std::int64_t zero_component = -1;
    for (std::int64_t component = 0;
         component < input_dimension;
         ++component) {
      const std::int64_t exponent =
          monomial_counts[term * input_dimension + component];
      CUDA_KERNEL_ASSERT(exponent >= 0);
      if (exponent == 0) {
        continue;
      }
      if (input_row[component] == Scalar(0)) {
        ++zero_count;
        zero_component = component;
        continue;
      }
      monomial *= integer_power_device(
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
        atomic_add_value(
            input_adjoint +
                batch * input_dimension + zero_component,
            term_gradient * conjugate_value(monomial));
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
      atomic_add_value(
          input_adjoint + batch * input_dimension + component,
          term_gradient * conjugate_value(derivative));
    }
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_batched_term_adjoint_kernel(
    const Scalar* output_adjoint,
    const std::int64_t* coefficient_terms,
    const std::int64_t* coefficient_outputs,
    const Scalar* coefficient_values,
    std::int64_t seed_count,
    std::int64_t batch_size,
    std::int64_t monomial_count,
    std::int64_t coefficient_count,
    std::int64_t output_dimension,
    Scalar* monomial_adjoint) {
  const std::int64_t sample_count = seed_count * batch_size;
  const std::int64_t total = sample_count * coefficient_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t sample = index / coefficient_count;
    const std::int64_t coefficient =
        index - sample * coefficient_count;
    const std::int64_t term = coefficient_terms[coefficient];
    const std::int64_t output_index =
        coefficient_outputs[coefficient];
    CUDA_KERNEL_ASSERT(
        term >= 0 && term < monomial_count);
    CUDA_KERNEL_ASSERT(
        output_index >= 0 && output_index < output_dimension);
    atomic_add_value(
        monomial_adjoint + sample * monomial_count + term,
        output_adjoint[
            sample * output_dimension + output_index] *
            conjugate_value(coefficient_values[coefficient]));
  }
}

template <typename Scalar>
__global__ void symmetric_power_shared_batched_input_adjoint_kernel(
    const Scalar* monomial_adjoint,
    const Scalar* input,
    std::int64_t seed_count,
    std::int64_t batch_size,
    std::int64_t input_dimension,
    const std::int64_t* monomial_counts,
    std::int64_t monomial_count,
    Scalar* input_adjoint) {
  const std::int64_t total = batch_size * monomial_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / monomial_count;
    const std::int64_t term = index - batch * monomial_count;
    const Scalar* input_row = input + batch * input_dimension;
    Scalar monomial = Scalar(1);
    std::int64_t zero_count = 0;
    std::int64_t zero_component = -1;
    for (std::int64_t component = 0;
         component < input_dimension;
         ++component) {
      const std::int64_t exponent =
          monomial_counts[term * input_dimension + component];
      CUDA_KERNEL_ASSERT(exponent >= 0);
      if (exponent == 0) {
        continue;
      }
      if (input_row[component] == Scalar(0)) {
        ++zero_count;
        zero_component = component;
        continue;
      }
      monomial *= integer_power_device(
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
      const Scalar derivative = conjugate_value(monomial);
      for (std::int64_t seed = 0; seed < seed_count; ++seed) {
        const std::int64_t sample = seed * batch_size + batch;
        atomic_add_value(
            input_adjoint +
                sample * input_dimension + zero_component,
            monomial_adjoint[
                sample * monomial_count + term] *
                derivative);
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
      const Scalar derivative = conjugate_value(
          Scalar(exponent) * monomial / input_row[component]);
      for (std::int64_t seed = 0; seed < seed_count; ++seed) {
        const std::int64_t sample = seed * batch_size + batch;
        atomic_add_value(
            input_adjoint +
                sample * input_dimension + component,
            monomial_adjoint[
                sample * monomial_count + term] *
                derivative);
      }
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_forward_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    Scalar* workspace,
    Scalar* output) {
  const std::int64_t sample_count =
      batch_size * source_dimension;
  for (std::int64_t sample =
           blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count;
       sample += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = sample / source_dimension;
    const std::int64_t source = sample % source_dimension;
    Scalar* batch_workspace =
        workspace + sample * workspace_dimension;
    Scalar* output_row = output + batch * output_dimension;
    for (std::int64_t index = 0; index < workspace_dimension; ++index) {
      batch_workspace[index] = Scalar(0);
    }
    const Scalar* input_row =
        packed_slots + sample * input_dimension;
    for (std::int64_t node = 0; node < node_count; ++node) {
      Scalar* node_output =
          batch_workspace + node_offsets[node];
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset >= 0) {
        for (
            std::int64_t component = 0;
            component < node_dimensions[node];
            ++component) {
          node_output[component] = input_row[leaf_offset + component];
        }
        continue;
      }
      const std::int64_t left_node = node_left[node];
      const std::int64_t right_node = node_right[node];
      const Scalar* left =
          batch_workspace + node_offsets[left_node];
      const Scalar* right =
          batch_workspace + node_offsets[right_node];
      const std::int64_t right_dimension =
          node_dimensions[right_node];
      for (
          std::int64_t entry = node_coefficient_offsets[node];
          entry < node_coefficient_offsets[node + 1];
          ++entry) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        node_output[column] +=
            conjugate_value(coefficient_values[entry]) *
            left[row / right_dimension] *
            right[row % right_dimension];
      }
    }
    for (std::int64_t root = 0; root < root_count; ++root) {
      const bool heterogeneous = root_projection_starts != nullptr;
      const std::int64_t root_node = root_nodes[root];
      CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
      const std::int64_t projection_start =
          heterogeneous ? root_projection_starts[root] : 0;
      const std::int64_t projection_dimension =
          heterogeneous
              ? root_projection_dimensions[root]
              : total_projection_dimension;
      const std::int64_t root_dimension =
          heterogeneous
              ? node_dimensions[root_node]
              : uniform_root_dimension;
      const std::int64_t output_offset =
          heterogeneous
              ? root_output_offsets[root]
              : root * projection_dimension * root_dimension;
      CUDA_KERNEL_ASSERT(
          projection_start >= 0 && projection_dimension > 0 &&
          projection_start + projection_dimension <=
              total_projection_dimension);
      CUDA_KERNEL_ASSERT(
          output_offset >= 0 &&
          output_offset + projection_dimension * root_dimension <=
              output_dimension);
      const Scalar* root_value =
          batch_workspace + node_offsets[root_node];
      for (
          std::int64_t projection = 0;
          projection < projection_dimension;
          ++projection) {
        Scalar* output_block =
            output_row +
            output_offset + projection * root_dimension;
        const Scalar projection_value =
            projection_values[
                source * total_projection_dimension +
                projection_start + projection];
        for (
            std::int64_t magnetic = 0;
            magnetic < root_dimension;
            ++magnetic) {
          const Scalar contribution =
              root_value[magnetic] * projection_value;
          if (source_dimension == 1) {
            output_block[magnetic] += contribution;
          } else {
            atomic_add_value(output_block + magnetic, contribution);
          }
        }
      }
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_adjoint_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* packed_slots_adjoint) {
  const std::int64_t sample_count =
      batch_size * source_dimension;
  for (std::int64_t sample =
           blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count;
       sample += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = sample / source_dimension;
    const std::int64_t source = sample % source_dimension;
    Scalar* batch_workspace =
        workspace + sample * workspace_dimension;
    Scalar* batch_workspace_adjoint =
        workspace_adjoint + sample * workspace_dimension;
    Scalar* input_adjoint =
        packed_slots_adjoint + sample * input_dimension;
    const Scalar* output_row =
        output_adjoint + batch * output_dimension;
    for (std::int64_t index = 0; index < input_dimension; ++index) {
      input_adjoint[index] = Scalar(0);
    }
    for (std::int64_t index = 0; index < workspace_dimension; ++index) {
      batch_workspace[index] = Scalar(0);
      batch_workspace_adjoint[index] = Scalar(0);
    }
    const Scalar* input_source =
        packed_slots + sample * input_dimension;
    for (std::int64_t node = 0; node < node_count; ++node) {
      Scalar* node_output =
          batch_workspace + node_offsets[node];
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset >= 0) {
        for (
            std::int64_t component = 0;
            component < node_dimensions[node];
            ++component) {
          node_output[component] =
              input_source[leaf_offset + component];
        }
        continue;
      }
      const std::int64_t left_node = node_left[node];
      const std::int64_t right_node = node_right[node];
      const Scalar* left =
          batch_workspace + node_offsets[left_node];
      const Scalar* right =
          batch_workspace + node_offsets[right_node];
      const std::int64_t right_dimension =
          node_dimensions[right_node];
      for (
          std::int64_t entry = node_coefficient_offsets[node];
          entry < node_coefficient_offsets[node + 1];
          ++entry) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        node_output[column] +=
            conjugate_value(coefficient_values[entry]) *
            left[row / right_dimension] *
            right[row % right_dimension];
      }
    }
    for (std::int64_t root = 0; root < root_count; ++root) {
      const bool heterogeneous = root_projection_starts != nullptr;
      const std::int64_t root_node = root_nodes[root];
      CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
      const std::int64_t projection_start =
          heterogeneous ? root_projection_starts[root] : 0;
      const std::int64_t projection_dimension =
          heterogeneous
              ? root_projection_dimensions[root]
              : total_projection_dimension;
      const std::int64_t root_dimension =
          heterogeneous
              ? node_dimensions[root_node]
              : uniform_root_dimension;
      const std::int64_t output_offset =
          heterogeneous
              ? root_output_offsets[root]
              : root * projection_dimension * root_dimension;
      CUDA_KERNEL_ASSERT(
          projection_start >= 0 && projection_dimension > 0 &&
          projection_start + projection_dimension <=
              total_projection_dimension);
      CUDA_KERNEL_ASSERT(
          output_offset >= 0 &&
          output_offset + projection_dimension * root_dimension <=
              output_dimension);
      Scalar* root_adjoint =
          batch_workspace_adjoint +
          node_offsets[root_node];
      for (
          std::int64_t projection = 0;
          projection < projection_dimension;
          ++projection) {
        const Scalar* output_block =
            output_row +
            output_offset + projection * root_dimension;
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
              conjugate_value(projection_value);
        }
      }
    }
    for (std::int64_t node = node_count; node-- > 0;) {
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset >= 0) {
        continue;
      }
      Scalar* node_adjoint =
          batch_workspace_adjoint + node_offsets[node];
      const std::int64_t left_node = node_left[node];
      const std::int64_t right_node = node_right[node];
      const Scalar* left =
          batch_workspace + node_offsets[left_node];
      const Scalar* right =
          batch_workspace + node_offsets[right_node];
      Scalar* left_adjoint =
          batch_workspace_adjoint + node_offsets[left_node];
      Scalar* right_adjoint =
          batch_workspace_adjoint + node_offsets[right_node];
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
            conjugate_value(right[right_index]);
        right_adjoint[right_index] +=
            gradient * coefficient_values[entry] *
            conjugate_value(left[left_index]);
      }
    }
    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      const Scalar* leaf_adjoint =
          batch_workspace_adjoint + node_offsets[node];
      for (
          std::int64_t component = 0;
          component < node_dimensions[node];
          ++component) {
        input_adjoint[leaf_offset + component] +=
            leaf_adjoint[component];
      }
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_double_backward_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    bool workspaces_ready,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* adjoint_tangent,
    Scalar* primal_tangent,
    Scalar* output_tangent,
    Scalar* packed_slots_tangent) {
  const std::int64_t sample_count =
      batch_size * source_dimension;
  for (std::int64_t sample =
           blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count;
       sample += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = sample / source_dimension;
    const std::int64_t source = sample % source_dimension;
    Scalar* batch_workspace =
        workspace + sample * workspace_dimension;
    Scalar* batch_workspace_adjoint =
        workspace_adjoint + sample * workspace_dimension;
    Scalar* batch_adjoint_tangent =
        adjoint_tangent + sample * workspace_dimension;
    Scalar* batch_primal_tangent =
        primal_tangent + sample * workspace_dimension;
    Scalar* output_tangent_row =
        output_tangent + batch * output_dimension;
    Scalar* packed_tangent_row =
        packed_slots_tangent + sample * input_dimension;
    const Scalar* output_adjoint_row =
        output_adjoint + batch * output_dimension;
    for (std::int64_t index = 0; index < input_dimension; ++index) {
      packed_tangent_row[index] = Scalar(0);
    }
    for (std::int64_t index = 0; index < workspace_dimension; ++index) {
      if (!workspaces_ready) {
        batch_workspace[index] = Scalar(0);
        batch_workspace_adjoint[index] = Scalar(0);
      }
      batch_adjoint_tangent[index] = Scalar(0);
      batch_primal_tangent[index] = Scalar(0);
    }
    const Scalar* input_source =
        packed_slots + sample * input_dimension;
    const Scalar* input_adjoint_tangent =
        packed_adjoint_tangent + sample * input_dimension;

    if (!workspaces_ready) {
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output =
            batch_workspace + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          for (
              std::int64_t component = 0;
              component < node_dimensions[node];
              ++component) {
            node_output[component] =
                input_source[leaf_offset + component];
          }
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          node_output[column] +=
              conjugate_value(coefficient_values[entry]) *
              left[row / right_dimension] *
              right[row % right_dimension];
        }
      }

      for (std::int64_t root = 0; root < root_count; ++root) {
        const bool heterogeneous = root_projection_starts != nullptr;
        const std::int64_t root_node = root_nodes[root];
        CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
        const std::int64_t projection_start =
            heterogeneous ? root_projection_starts[root] : 0;
        const std::int64_t projection_dimension =
            heterogeneous
                ? root_projection_dimensions[root]
                : total_projection_dimension;
        const std::int64_t root_dimension =
            heterogeneous
                ? node_dimensions[root_node]
                : uniform_root_dimension;
        const std::int64_t output_offset =
            heterogeneous
                ? root_output_offsets[root]
                : root * projection_dimension * root_dimension;
        CUDA_KERNEL_ASSERT(
            projection_start >= 0 && projection_dimension > 0 &&
            projection_start + projection_dimension <=
                total_projection_dimension);
        CUDA_KERNEL_ASSERT(
            output_offset >= 0 &&
            output_offset + projection_dimension * root_dimension <=
                output_dimension);
        Scalar* root_adjoint =
            batch_workspace_adjoint + node_offsets[root_node];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          const Scalar* output_block =
              output_adjoint_row +
              output_offset + projection * root_dimension;
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
                conjugate_value(projection_value);
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
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        const Scalar* node_adjoint =
            batch_workspace_adjoint + node_offsets[node];
        Scalar* left_adjoint =
            batch_workspace_adjoint + node_offsets[left_node];
        Scalar* right_adjoint =
            batch_workspace_adjoint + node_offsets[right_node];
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
              conjugate_value(right[right_index]);
          right_adjoint[right_index] +=
              gradient * coefficient_values[entry] *
              conjugate_value(left[left_index]);
        }
      }
    }

      for (std::int64_t node = 0; node < node_count; ++node) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset < 0) {
          continue;
        }
        Scalar* leaf_tangent =
            batch_adjoint_tangent + node_offsets[node];
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
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        const Scalar* node_adjoint =
            batch_workspace_adjoint + node_offsets[node];
        Scalar* node_adjoint_tangent =
            batch_adjoint_tangent + node_offsets[node];
        const Scalar* left_adjoint_tangent =
            batch_adjoint_tangent + node_offsets[left_node];
        const Scalar* right_adjoint_tangent =
            batch_adjoint_tangent + node_offsets[right_node];
        Scalar* left_primal_tangent =
            batch_primal_tangent + node_offsets[left_node];
        Scalar* right_primal_tangent =
            batch_primal_tangent + node_offsets[right_node];
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
              conjugate_value(coefficient) *
              (
                  left_adjoint_tangent[left_index] *
                      right[right_index] +
                  right_adjoint_tangent[right_index] *
                      left[left_index]);
          left_primal_tangent[left_index] +=
              conjugate_value(right_adjoint_tangent[right_index]) *
              node_adjoint[column] * coefficient;
          right_primal_tangent[right_index] +=
              conjugate_value(left_adjoint_tangent[left_index]) *
              node_adjoint[column] * coefficient;
        }
      }

      for (std::int64_t root = 0; root < root_count; ++root) {
        const bool heterogeneous = root_projection_starts != nullptr;
        const std::int64_t root_node = root_nodes[root];
        CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
        const std::int64_t projection_start =
            heterogeneous ? root_projection_starts[root] : 0;
        const std::int64_t projection_dimension =
            heterogeneous
                ? root_projection_dimensions[root]
                : total_projection_dimension;
        const std::int64_t root_dimension =
            heterogeneous
                ? node_dimensions[root_node]
                : uniform_root_dimension;
        const std::int64_t output_offset =
            heterogeneous
                ? root_output_offsets[root]
                : root * projection_dimension * root_dimension;
        CUDA_KERNEL_ASSERT(
            projection_start >= 0 && projection_dimension > 0 &&
            projection_start + projection_dimension <=
                total_projection_dimension);
        CUDA_KERNEL_ASSERT(
            output_offset >= 0 &&
            output_offset + projection_dimension * root_dimension <=
                output_dimension);
        const Scalar* root_adjoint_tangent =
            batch_adjoint_tangent + node_offsets[root_node];
        for (
            std::int64_t projection = 0;
            projection < projection_dimension;
            ++projection) {
          Scalar* output_block =
              output_tangent_row +
              output_offset + projection * root_dimension;
          const Scalar projection_value =
              projection_values[
                  source * total_projection_dimension +
                  projection_start + projection];
          for (
              std::int64_t magnetic = 0;
              magnetic < root_dimension;
              ++magnetic) {
            const Scalar contribution =
                root_adjoint_tangent[magnetic] * projection_value;
            if (source_dimension == 1) {
              output_block[magnetic] += contribution;
            } else {
              atomic_add_value(output_block + magnetic, contribution);
            }
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
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        const Scalar* node_tangent =
            batch_primal_tangent + node_offsets[node];
        Scalar* left_tangent =
            batch_primal_tangent + node_offsets[left_node];
        Scalar* right_tangent =
            batch_primal_tangent + node_offsets[right_node];
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
              conjugate_value(right[right_index]);
          right_tangent[right_index] +=
              gradient * coefficient_values[entry] *
              conjugate_value(left[left_index]);
        }
      }

      Scalar* input_tangent = packed_tangent_row;
      for (std::int64_t node = 0; node < node_count; ++node) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset < 0) {
          continue;
        }
        const Scalar* leaf_tangent =
            batch_primal_tangent + node_offsets[node];
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

__device__ inline std::int64_t factorized_segment_for_source(
    std::int64_t source,
    const std::int64_t* segment_source_offsets,
    std::int64_t segment_count) {
  std::int64_t lower = 0;
  std::int64_t upper = segment_count;
  while (lower + 1 < upper) {
    const std::int64_t middle = lower + (upper - lower) / 2;
    if (segment_source_offsets[middle] <= source) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

constexpr std::int64_t kFactorizedSegmentExecutionAll = -1;
constexpr std::int64_t kFactorizedSegmentExecutionSerial = 0;
constexpr std::int64_t kFactorizedSegmentExecutionWarp = 1;
constexpr std::int64_t kFactorizedSegmentExecutionHybrid = 2;
constexpr std::int64_t kFactorizedSegmentWarpMinCoefficientCount = 128;

__device__ inline bool factorized_segment_execution_enabled(
    std::int64_t segment,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t execution_class) {
  if (execution_class == kFactorizedSegmentExecutionAll) {
    return true;
  }
  const std::int64_t node_start = segment_node_offsets[segment];
  const std::int64_t node_stop = segment_node_offsets[segment + 1];
  const std::int64_t coefficient_count =
      node_coefficient_offsets[node_stop] -
      node_coefficient_offsets[node_start];
  const std::int64_t selected_class =
      coefficient_count >= kFactorizedSegmentWarpMinCoefficientCount
      ? kFactorizedSegmentExecutionWarp
      : kFactorizedSegmentExecutionSerial;
  return execution_class == selected_class;
}

template <typename Scalar>
__device__ inline void factorized_segment_primal(
    const Scalar* input,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t node_start,
    std::int64_t node_count,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    Scalar* workspace) {
  for (std::int64_t local_node = 0;
       local_node < node_count;
       ++local_node) {
    const std::int64_t node = node_start + local_node;
    Scalar* node_output = workspace + node_offsets[node];
    const std::int64_t leaf_offset = node_leaf_offsets[node];
    if (leaf_offset >= 0) {
      for (std::int64_t component = 0;
           component < node_dimensions[node];
           ++component) {
        node_output[component] = input[leaf_offset + component];
      }
      continue;
    }
    const std::int64_t left_node =
        node_start + node_left[node];
    const std::int64_t right_node =
        node_start + node_right[node];
    const Scalar* left = workspace + node_offsets[left_node];
    const Scalar* right = workspace + node_offsets[right_node];
    const std::int64_t right_dimension = node_dimensions[right_node];
    for (std::int64_t entry = node_coefficient_offsets[node];
         entry < node_coefficient_offsets[node + 1];
         ++entry) {
      const std::int64_t row = coefficient_rows[entry];
      const std::int64_t column = coefficient_columns[entry];
      node_output[column] +=
          conjugate_value(coefficient_values[entry]) *
          left[row / right_dimension] *
          right[row % right_dimension];
    }
  }
}

template <typename Scalar>
__device__ inline void factorized_segment_seed_adjoint(
    const Scalar* output_adjoint,
    std::int64_t source,
    std::int64_t projection_base,
    std::int64_t total_projection_dimension,
    std::int64_t output_base,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    std::int64_t node_start,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    std::int64_t root_start,
    std::int64_t root_count,
    const Scalar* projection_values,
    Scalar* workspace_adjoint) {
  for (std::int64_t local_root = 0;
       local_root < root_count;
       ++local_root) {
    const std::int64_t root = root_start + local_root;
    const std::int64_t root_node =
        node_start + root_nodes[root];
    const std::int64_t root_dimension = node_dimensions[root_node];
    Scalar* root_adjoint =
        workspace_adjoint + node_offsets[root_node];
    for (std::int64_t projection = 0;
         projection < root_projection_dimensions[root];
         ++projection) {
      const Scalar* output_block =
          output_adjoint + output_base + root_output_offsets[root] +
          projection * root_dimension;
      const Scalar projection_value = projection_values[
          projection_base +
          source * total_projection_dimension +
          root_projection_starts[root] + projection];
      for (std::int64_t magnetic = 0;
           magnetic < root_dimension;
           ++magnetic) {
        root_adjoint[magnetic] +=
            output_block[magnetic] *
            conjugate_value(projection_value);
      }
    }
  }
}

template <typename Scalar>
__device__ inline void factorized_segment_reverse(
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    std::int64_t node_start,
    std::int64_t node_count,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const Scalar* workspace,
    Scalar* workspace_adjoint) {
  for (std::int64_t local_node = node_count;
       local_node-- > 0;) {
    const std::int64_t node = node_start + local_node;
    if (node_leaf_offsets[node] >= 0) {
      continue;
    }
    const std::int64_t left_node =
        node_start + node_left[node];
    const std::int64_t right_node =
        node_start + node_right[node];
    const Scalar* left = workspace + node_offsets[left_node];
    const Scalar* right = workspace + node_offsets[right_node];
    const Scalar* node_adjoint =
        workspace_adjoint + node_offsets[node];
    Scalar* left_adjoint =
        workspace_adjoint + node_offsets[left_node];
    Scalar* right_adjoint =
        workspace_adjoint + node_offsets[right_node];
    const std::int64_t right_dimension = node_dimensions[right_node];
    for (std::int64_t entry = node_coefficient_offsets[node];
         entry < node_coefficient_offsets[node + 1];
         ++entry) {
      const std::int64_t row = coefficient_rows[entry];
      const std::int64_t column = coefficient_columns[entry];
      const std::int64_t left_index = row / right_dimension;
      const std::int64_t right_index = row % right_dimension;
      const Scalar gradient = node_adjoint[column];
      left_adjoint[left_index] +=
          gradient * coefficient_values[entry] *
          conjugate_value(right[right_index]);
      right_adjoint[right_index] +=
          gradient * coefficient_values[entry] *
          conjugate_value(left[left_index]);
    }
  }
}

template <typename Scalar>
__device__ inline void factorized_segment_gather_leaves(
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    std::int64_t node_start,
    std::int64_t node_count,
    const Scalar* workspace_values,
    Scalar* input_values) {
  for (std::int64_t local_node = 0;
       local_node < node_count;
       ++local_node) {
    const std::int64_t node = node_start + local_node;
    const std::int64_t leaf_offset = node_leaf_offsets[node];
    if (leaf_offset < 0) {
      continue;
    }
    const Scalar* leaf = workspace_values + node_offsets[node];
    for (std::int64_t component = 0;
         component < node_dimensions[node];
         ++component) {
      input_values[leaf_offset + component] += leaf[component];
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_segmented_forward_kernel(
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t packed_input_dimension,
    const std::int64_t* segment_source_offsets,
    const std::int64_t* segment_input_offsets,
    const std::int64_t* segment_input_dimensions,
    const std::int64_t* segment_workspace_offsets,
    const std::int64_t* segment_workspace_dimensions,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* segment_root_offsets,
    const std::int64_t* segment_projection_offsets,
    const std::int64_t* segment_projection_dimensions,
    const std::int64_t* segment_output_offsets,
    std::int64_t segment_count,
    std::int64_t total_source_count,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension,
    std::int64_t execution_class,
    Scalar* workspace,
    Scalar* output) {
  const std::int64_t sample_count = batch_size * total_source_count;
  for (std::int64_t sample =
           blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count;
       sample += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = sample / total_source_count;
    const std::int64_t global_source = sample % total_source_count;
    const std::int64_t segment = factorized_segment_for_source(
        global_source,
        segment_source_offsets,
        segment_count);
    if (!factorized_segment_execution_enabled(
            segment,
            segment_node_offsets,
            node_coefficient_offsets,
            execution_class)) {
      continue;
    }
    const std::int64_t source =
        global_source - segment_source_offsets[segment];
    const std::int64_t source_dimension =
        segment_source_offsets[segment + 1] -
        segment_source_offsets[segment];
    const std::int64_t input_dimension =
        segment_input_dimensions[segment];
    const std::int64_t local_workspace_dimension =
        segment_workspace_dimensions[segment];
    const std::int64_t node_start = segment_node_offsets[segment];
    const std::int64_t node_count =
        segment_node_offsets[segment + 1] - node_start;
    const std::int64_t root_start = segment_root_offsets[segment];
    const std::int64_t root_count =
        segment_root_offsets[segment + 1] - root_start;
    Scalar* local_workspace =
        workspace + batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    for (std::int64_t index = 0;
         index < local_workspace_dimension;
         ++index) {
      local_workspace[index] = Scalar(0);
    }
    const Scalar* input =
        packed_slots + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    factorized_segment_primal(
        input,
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace);
    Scalar* output_row = output + batch * output_dimension;
    for (std::int64_t local_root = 0;
         local_root < root_count;
         ++local_root) {
      const std::int64_t root = root_start + local_root;
      const std::int64_t root_node =
          node_start + root_nodes[root];
      const std::int64_t root_dimension = node_dimensions[root_node];
      const Scalar* root_value =
          local_workspace + node_offsets[root_node];
      for (std::int64_t projection = 0;
           projection < root_projection_dimensions[root];
           ++projection) {
        Scalar* output_block =
            output_row + segment_output_offsets[segment] +
            root_output_offsets[root] +
            projection * root_dimension;
        const Scalar projection_value = projection_values[
            segment_projection_offsets[segment] +
            source * segment_projection_dimensions[segment] +
            root_projection_starts[root] + projection];
        for (std::int64_t magnetic = 0;
             magnetic < root_dimension;
             ++magnetic) {
          const Scalar contribution =
              root_value[magnetic] * projection_value;
          if (source_dimension == 1) {
            output_block[magnetic] += contribution;
          } else {
            atomic_add_value(output_block + magnetic, contribution);
          }
        }
      }
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_segmented_adjoint_kernel(
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t packed_input_dimension,
    const std::int64_t* segment_source_offsets,
    const std::int64_t* segment_input_offsets,
    const std::int64_t* segment_input_dimensions,
    const std::int64_t* segment_workspace_offsets,
    const std::int64_t* segment_workspace_dimensions,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* segment_root_offsets,
    const std::int64_t* segment_projection_offsets,
    const std::int64_t* segment_projection_dimensions,
    const std::int64_t* segment_output_offsets,
    std::int64_t segment_count,
    std::int64_t total_source_count,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension,
    std::int64_t execution_class,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* packed_adjoint) {
  const std::int64_t sample_count = batch_size * total_source_count;
  for (std::int64_t sample =
           blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count;
       sample += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = sample / total_source_count;
    const std::int64_t global_source = sample % total_source_count;
    const std::int64_t segment = factorized_segment_for_source(
        global_source,
        segment_source_offsets,
        segment_count);
    if (!factorized_segment_execution_enabled(
            segment,
            segment_node_offsets,
            node_coefficient_offsets,
            execution_class)) {
      continue;
    }
    const std::int64_t source =
        global_source - segment_source_offsets[segment];
    const std::int64_t input_dimension =
        segment_input_dimensions[segment];
    const std::int64_t local_workspace_dimension =
        segment_workspace_dimensions[segment];
    const std::int64_t node_start = segment_node_offsets[segment];
    const std::int64_t node_count =
        segment_node_offsets[segment + 1] - node_start;
    const std::int64_t root_start = segment_root_offsets[segment];
    const std::int64_t root_count =
        segment_root_offsets[segment + 1] - root_start;
    Scalar* local_workspace =
        workspace + batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    Scalar* local_workspace_adjoint =
        workspace_adjoint + batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    Scalar* input_adjoint =
        packed_adjoint + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    for (std::int64_t index = 0; index < input_dimension; ++index) {
      input_adjoint[index] = Scalar(0);
    }
    for (std::int64_t index = 0;
         index < local_workspace_dimension;
         ++index) {
      local_workspace[index] = Scalar(0);
      local_workspace_adjoint[index] = Scalar(0);
    }
    const Scalar* input =
        packed_slots + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    factorized_segment_primal(
        input,
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace);
    factorized_segment_seed_adjoint(
        output_adjoint + batch * output_dimension,
        source,
        segment_projection_offsets[segment],
        segment_projection_dimensions[segment],
        segment_output_offsets[segment],
        node_offsets,
        node_dimensions,
        node_start,
        root_nodes,
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        root_start,
        root_count,
        projection_values,
        local_workspace_adjoint);
    factorized_segment_reverse(
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace,
        local_workspace_adjoint);
    factorized_segment_gather_leaves(
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_start,
        node_count,
        local_workspace_adjoint,
        input_adjoint);
  }
}

template <typename Scalar>
__global__ void factorized_angular_segmented_double_backward_kernel(
    const Scalar* packed_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t packed_input_dimension,
    const std::int64_t* segment_source_offsets,
    const std::int64_t* segment_input_offsets,
    const std::int64_t* segment_input_dimensions,
    const std::int64_t* segment_workspace_offsets,
    const std::int64_t* segment_workspace_dimensions,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* segment_root_offsets,
    const std::int64_t* segment_projection_offsets,
    const std::int64_t* segment_projection_dimensions,
    const std::int64_t* segment_output_offsets,
    std::int64_t segment_count,
    std::int64_t total_source_count,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension,
    std::int64_t execution_class,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* adjoint_tangent,
    Scalar* primal_tangent,
    Scalar* output_tangent,
    Scalar* packed_tangent) {
  const std::int64_t sample_count = batch_size * total_source_count;
  for (std::int64_t sample =
           blockIdx.x * blockDim.x + threadIdx.x;
       sample < sample_count;
       sample += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = sample / total_source_count;
    const std::int64_t global_source = sample % total_source_count;
    const std::int64_t segment = factorized_segment_for_source(
        global_source,
        segment_source_offsets,
        segment_count);
    if (!factorized_segment_execution_enabled(
            segment,
            segment_node_offsets,
            node_coefficient_offsets,
            execution_class)) {
      continue;
    }
    const std::int64_t source =
        global_source - segment_source_offsets[segment];
    const std::int64_t source_dimension =
        segment_source_offsets[segment + 1] -
        segment_source_offsets[segment];
    const std::int64_t input_dimension =
        segment_input_dimensions[segment];
    const std::int64_t local_workspace_dimension =
        segment_workspace_dimensions[segment];
    const std::int64_t node_start = segment_node_offsets[segment];
    const std::int64_t node_count =
        segment_node_offsets[segment + 1] - node_start;
    const std::int64_t root_start = segment_root_offsets[segment];
    const std::int64_t root_count =
        segment_root_offsets[segment + 1] - root_start;
    const std::int64_t workspace_base =
        batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    Scalar* local_workspace = workspace + workspace_base;
    Scalar* local_workspace_adjoint =
        workspace_adjoint + workspace_base;
    Scalar* local_adjoint_tangent =
        adjoint_tangent + workspace_base;
    Scalar* local_primal_tangent =
        primal_tangent + workspace_base;
    Scalar* input_tangent =
        packed_tangent + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    const Scalar* input =
        packed_slots + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    const Scalar* input_adjoint_tangent =
        packed_adjoint_tangent + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    for (std::int64_t index = 0; index < input_dimension; ++index) {
      input_tangent[index] = Scalar(0);
    }
    for (std::int64_t index = 0;
         index < local_workspace_dimension;
         ++index) {
      local_workspace[index] = Scalar(0);
      local_workspace_adjoint[index] = Scalar(0);
      local_adjoint_tangent[index] = Scalar(0);
      local_primal_tangent[index] = Scalar(0);
    }
    factorized_segment_primal(
        input,
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace);
    factorized_segment_seed_adjoint(
        output_adjoint + batch * output_dimension,
        source,
        segment_projection_offsets[segment],
        segment_projection_dimensions[segment],
        segment_output_offsets[segment],
        node_offsets,
        node_dimensions,
        node_start,
        root_nodes,
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        root_start,
        root_count,
        projection_values,
        local_workspace_adjoint);
    factorized_segment_reverse(
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace,
        local_workspace_adjoint);
    for (std::int64_t local_node = 0;
         local_node < node_count;
         ++local_node) {
      const std::int64_t node = node_start + local_node;
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      Scalar* leaf_tangent =
          local_adjoint_tangent + node_offsets[node];
      for (std::int64_t component = 0;
           component < node_dimensions[node];
           ++component) {
        leaf_tangent[component] +=
            input_adjoint_tangent[leaf_offset + component];
      }
    }
    for (std::int64_t local_node = 0;
         local_node < node_count;
         ++local_node) {
      const std::int64_t node = node_start + local_node;
      if (node_leaf_offsets[node] >= 0) {
        continue;
      }
      const std::int64_t left_node =
          node_start + node_left[node];
      const std::int64_t right_node =
          node_start + node_right[node];
      const Scalar* left = local_workspace + node_offsets[left_node];
      const Scalar* right = local_workspace + node_offsets[right_node];
      const Scalar* node_adjoint =
          local_workspace_adjoint + node_offsets[node];
      Scalar* node_adjoint_tangent =
          local_adjoint_tangent + node_offsets[node];
      const Scalar* left_adjoint_tangent =
          local_adjoint_tangent + node_offsets[left_node];
      const Scalar* right_adjoint_tangent =
          local_adjoint_tangent + node_offsets[right_node];
      Scalar* left_primal_tangent =
          local_primal_tangent + node_offsets[left_node];
      Scalar* right_primal_tangent =
          local_primal_tangent + node_offsets[right_node];
      const std::int64_t right_dimension = node_dimensions[right_node];
      for (std::int64_t entry = node_coefficient_offsets[node];
           entry < node_coefficient_offsets[node + 1];
           ++entry) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        const std::int64_t left_index = row / right_dimension;
        const std::int64_t right_index = row % right_dimension;
        const Scalar coefficient = coefficient_values[entry];
        node_adjoint_tangent[column] +=
            conjugate_value(coefficient) *
            (left_adjoint_tangent[left_index] * right[right_index] +
             right_adjoint_tangent[right_index] * left[left_index]);
        left_primal_tangent[left_index] +=
            conjugate_value(right_adjoint_tangent[right_index]) *
            node_adjoint[column] * coefficient;
        right_primal_tangent[right_index] +=
            conjugate_value(left_adjoint_tangent[left_index]) *
            node_adjoint[column] * coefficient;
      }
    }
    Scalar* output_tangent_row =
        output_tangent + batch * output_dimension;
    for (std::int64_t local_root = 0;
         local_root < root_count;
         ++local_root) {
      const std::int64_t root = root_start + local_root;
      const std::int64_t root_node =
          node_start + root_nodes[root];
      const std::int64_t root_dimension = node_dimensions[root_node];
      const Scalar* root_adjoint_tangent =
          local_adjoint_tangent + node_offsets[root_node];
      for (std::int64_t projection = 0;
           projection < root_projection_dimensions[root];
           ++projection) {
        Scalar* output_block =
            output_tangent_row + segment_output_offsets[segment] +
            root_output_offsets[root] +
            projection * root_dimension;
        const Scalar projection_value = projection_values[
            segment_projection_offsets[segment] +
            source * segment_projection_dimensions[segment] +
            root_projection_starts[root] + projection];
        for (std::int64_t magnetic = 0;
             magnetic < root_dimension;
             ++magnetic) {
          const Scalar contribution =
              root_adjoint_tangent[magnetic] * projection_value;
          if (source_dimension == 1) {
            output_block[magnetic] += contribution;
          } else {
            atomic_add_value(output_block + magnetic, contribution);
          }
        }
      }
    }
    factorized_segment_reverse(
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace,
        local_primal_tangent);
    factorized_segment_gather_leaves(
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_start,
        node_count,
        local_primal_tangent,
        input_tangent);
  }
}

template <typename Scalar>
__device__ inline void factorized_angular_forward_warp_body(
    const Scalar* input_row,
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
    std::int64_t workspace_dimension,
    Scalar* workspace) {
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t index = lane;
       index < workspace_dimension;
       index += warpSize) {
    workspace[index] = Scalar(0);
  }
  __syncwarp();
  for (std::int64_t node = 0; node < node_count; ++node) {
    Scalar* node_output = workspace + node_offsets[node];
    const std::int64_t leaf_offset = node_leaf_offsets[node];
    if (leaf_offset >= 0) {
      for (std::int64_t component = lane;
           component < node_dimensions[node];
           component += warpSize) {
        node_output[component] = input_row[leaf_offset + component];
      }
    } else {
      const std::int64_t left_node = node_left[node];
      const std::int64_t right_node = node_right[node];
      const Scalar* left = workspace + node_offsets[left_node];
      const Scalar* right = workspace + node_offsets[right_node];
      const std::int64_t right_dimension =
          node_dimensions[right_node];
      for (std::int64_t entry =
               node_coefficient_offsets[node] + lane;
           entry < node_coefficient_offsets[node + 1];
           entry += warpSize) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        atomic_add_value(
            node_output + column,
            conjugate_value(coefficient_values[entry]) *
                left[row / right_dimension] *
                right[row % right_dimension]);
      }
    }
    __syncwarp();
  }
}

template <typename Scalar>
__device__ inline void factorized_angular_adjoint_warp_body(
    const Scalar* output_adjoint_row,
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
    std::int64_t source,
    std::int64_t total_projection_dimension,
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    const Scalar* workspace,
    Scalar* workspace_adjoint) {
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t index = lane;
       index < workspace_dimension;
       index += warpSize) {
    workspace_adjoint[index] = Scalar(0);
  }
  __syncwarp();
  for (std::int64_t root = 0; root < root_count; ++root) {
    const bool heterogeneous = root_projection_starts != nullptr;
    const std::int64_t root_node = root_nodes[root];
    CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
    const std::int64_t projection_start =
        heterogeneous ? root_projection_starts[root] : 0;
    const std::int64_t projection_dimension =
        heterogeneous
            ? root_projection_dimensions[root]
            : total_projection_dimension;
    const std::int64_t root_dimension =
        heterogeneous
            ? node_dimensions[root_node]
            : uniform_root_dimension;
    const std::int64_t output_offset =
        heterogeneous
            ? root_output_offsets[root]
            : root * projection_dimension * root_dimension;
    CUDA_KERNEL_ASSERT(
        projection_start >= 0 && projection_dimension > 0 &&
        projection_start + projection_dimension <=
            total_projection_dimension);
    CUDA_KERNEL_ASSERT(
        output_offset >= 0 &&
        output_offset + projection_dimension * root_dimension <=
            output_dimension);
    Scalar* root_adjoint =
        workspace_adjoint + node_offsets[root_node];
    const std::int64_t root_work =
        projection_dimension * root_dimension;
    for (std::int64_t index = lane;
         index < root_work;
         index += warpSize) {
      const std::int64_t projection = index / root_dimension;
      const std::int64_t magnetic =
          index - projection * root_dimension;
      const Scalar projection_value =
          projection_values[
              source * total_projection_dimension +
              projection_start + projection];
      atomic_add_value(
          root_adjoint + magnetic,
          output_adjoint_row[
              output_offset + projection * root_dimension + magnetic] *
              conjugate_value(projection_value));
    }
    __syncwarp();
  }
  for (std::int64_t node = node_count; node-- > 0;) {
    if (node_leaf_offsets[node] >= 0) {
      continue;
    }
    const std::int64_t left_node = node_left[node];
    const std::int64_t right_node = node_right[node];
    const Scalar* left = workspace + node_offsets[left_node];
    const Scalar* right = workspace + node_offsets[right_node];
    const Scalar* node_adjoint =
        workspace_adjoint + node_offsets[node];
    Scalar* left_adjoint =
        workspace_adjoint + node_offsets[left_node];
    Scalar* right_adjoint =
        workspace_adjoint + node_offsets[right_node];
    const std::int64_t right_dimension = node_dimensions[right_node];
    for (std::int64_t entry =
             node_coefficient_offsets[node] + lane;
         entry < node_coefficient_offsets[node + 1];
         entry += warpSize) {
      const std::int64_t row = coefficient_rows[entry];
      const std::int64_t column = coefficient_columns[entry];
      const std::int64_t left_index = row / right_dimension;
      const std::int64_t right_index = row % right_dimension;
      const Scalar gradient = node_adjoint[column];
      atomic_add_value(
          left_adjoint + left_index,
          gradient * coefficient_values[entry] *
              conjugate_value(right[right_index]));
      atomic_add_value(
          right_adjoint + right_index,
          gradient * coefficient_values[entry] *
              conjugate_value(left[left_index]));
    }
    __syncwarp();
  }
}

template <typename Scalar>
__global__ void factorized_angular_forward_warp_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    Scalar* workspace,
    Scalar* output) {
  const std::int64_t sample_count = batch_size * source_dimension;
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t sample = blockIdx.x;
       sample < sample_count;
       sample += gridDim.x) {
    const std::int64_t batch = sample / source_dimension;
    const std::int64_t source = sample % source_dimension;
    Scalar* batch_workspace =
        workspace + sample * workspace_dimension;
    const Scalar* input_row =
        packed_slots + sample * input_dimension;
    factorized_angular_forward_warp_body(
        input_row,
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        workspace_dimension,
        batch_workspace);
    Scalar* output_row = output + batch * output_dimension;
    for (std::int64_t root = 0; root < root_count; ++root) {
      const bool heterogeneous = root_projection_starts != nullptr;
      const std::int64_t root_node = root_nodes[root];
      CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
      const std::int64_t projection_start =
          heterogeneous ? root_projection_starts[root] : 0;
      const std::int64_t projection_dimension =
          heterogeneous
              ? root_projection_dimensions[root]
              : total_projection_dimension;
      const std::int64_t root_dimension =
          heterogeneous
              ? node_dimensions[root_node]
              : uniform_root_dimension;
      const std::int64_t output_offset =
          heterogeneous
              ? root_output_offsets[root]
              : root * projection_dimension * root_dimension;
      const Scalar* root_value =
          batch_workspace + node_offsets[root_node];
      const std::int64_t root_work =
          projection_dimension * root_dimension;
      for (std::int64_t index = lane;
           index < root_work;
           index += warpSize) {
        const std::int64_t projection = index / root_dimension;
        const std::int64_t magnetic =
            index - projection * root_dimension;
        const Scalar contribution =
            root_value[magnetic] *
            projection_values[
                source * total_projection_dimension +
                projection_start + projection];
        Scalar* destination =
            output_row + output_offset +
            projection * root_dimension + magnetic;
        if (source_dimension == 1) {
          *destination += contribution;
        } else {
          atomic_add_value(destination, contribution);
        }
      }
      __syncwarp();
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_adjoint_warp_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* packed_slots_adjoint) {
  const std::int64_t sample_count = batch_size * source_dimension;
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t sample = blockIdx.x;
       sample < sample_count;
       sample += gridDim.x) {
    const std::int64_t batch = sample / source_dimension;
    const std::int64_t source = sample % source_dimension;
    Scalar* batch_workspace =
        workspace + sample * workspace_dimension;
    Scalar* batch_workspace_adjoint =
        workspace_adjoint + sample * workspace_dimension;
    Scalar* input_adjoint =
        packed_slots_adjoint + sample * input_dimension;
    for (std::int64_t index = lane;
         index < input_dimension;
         index += warpSize) {
      input_adjoint[index] = Scalar(0);
    }
    __syncwarp();
    factorized_angular_forward_warp_body(
        packed_slots + sample * input_dimension,
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        workspace_dimension,
        batch_workspace);
    factorized_angular_adjoint_warp_body(
        output_adjoint + batch * output_dimension,
        node_offsets,
        node_dimensions,
        node_leaf_offsets,
        node_left,
        node_right,
        node_coefficient_offsets,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        root_nodes,
        root_count,
        root_projection_starts,
        root_projection_dimensions,
        root_output_offsets,
        projection_values,
        source,
        total_projection_dimension,
        workspace_dimension,
        uniform_root_dimension,
        output_dimension,
        batch_workspace,
        batch_workspace_adjoint);
    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      const Scalar* leaf_adjoint =
          batch_workspace_adjoint + node_offsets[node];
      for (std::int64_t component = lane;
           component < node_dimensions[node];
           component += warpSize) {
        atomic_add_value(
            input_adjoint + leaf_offset + component,
            leaf_adjoint[component]);
      }
      __syncwarp();
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_double_backward_warp_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t uniform_root_dimension,
    std::int64_t output_dimension,
    bool workspaces_ready,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* adjoint_tangent,
    Scalar* primal_tangent,
    Scalar* output_tangent,
    Scalar* packed_slots_tangent) {
  extern __shared__ __align__(16) unsigned char shared_storage[];
  Scalar* shared_adjoint_tangent =
      reinterpret_cast<Scalar*>(shared_storage);
  Scalar* shared_primal_tangent =
      shared_adjoint_tangent + workspace_dimension;
  (void)adjoint_tangent;
  (void)primal_tangent;
  const std::int64_t sample_count = batch_size * source_dimension;
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t sample = blockIdx.x;
       sample < sample_count;
       sample += gridDim.x) {
    const std::int64_t batch = sample / source_dimension;
    const std::int64_t source = sample % source_dimension;
    Scalar* batch_workspace =
        workspace + sample * workspace_dimension;
    Scalar* batch_workspace_adjoint =
        workspace_adjoint + sample * workspace_dimension;
    Scalar* batch_adjoint_tangent = shared_adjoint_tangent;
    Scalar* batch_primal_tangent = shared_primal_tangent;
    Scalar* packed_tangent_row =
        packed_slots_tangent + sample * input_dimension;
    for (std::int64_t index = lane;
         index < input_dimension;
         index += warpSize) {
      packed_tangent_row[index] = Scalar(0);
    }
    for (std::int64_t index = lane;
         index < workspace_dimension;
         index += warpSize) {
      batch_adjoint_tangent[index] = Scalar(0);
      batch_primal_tangent[index] = Scalar(0);
    }
    __syncwarp();
    if (!workspaces_ready) {
      factorized_angular_forward_warp_body(
          packed_slots + sample * input_dimension,
          node_offsets,
          node_dimensions,
          node_leaf_offsets,
          node_left,
          node_right,
          node_coefficient_offsets,
          node_count,
          coefficient_rows,
          coefficient_columns,
          coefficient_values,
          workspace_dimension,
          batch_workspace);
      factorized_angular_adjoint_warp_body(
          output_adjoint + batch * output_dimension,
          node_offsets,
          node_dimensions,
          node_leaf_offsets,
          node_left,
          node_right,
          node_coefficient_offsets,
          node_count,
          coefficient_rows,
          coefficient_columns,
          coefficient_values,
          root_nodes,
          root_count,
          root_projection_starts,
          root_projection_dimensions,
          root_output_offsets,
          projection_values,
          source,
          total_projection_dimension,
          workspace_dimension,
          uniform_root_dimension,
          output_dimension,
          batch_workspace,
          batch_workspace_adjoint);
    }

    const Scalar* input_adjoint_tangent =
        packed_adjoint_tangent + sample * input_dimension;
    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      Scalar* leaf_tangent =
          batch_adjoint_tangent + node_offsets[node];
      for (std::int64_t component = lane;
           component < node_dimensions[node];
           component += warpSize) {
        atomic_add_value(
            leaf_tangent + component,
            input_adjoint_tangent[leaf_offset + component]);
      }
      __syncwarp();
    }

    for (std::int64_t node = 0; node < node_count; ++node) {
      if (node_leaf_offsets[node] >= 0) {
        continue;
      }
      const std::int64_t left_node = node_left[node];
      const std::int64_t right_node = node_right[node];
      const Scalar* left = batch_workspace + node_offsets[left_node];
      const Scalar* right = batch_workspace + node_offsets[right_node];
      const Scalar* node_adjoint =
          batch_workspace_adjoint + node_offsets[node];
      Scalar* node_adjoint_tangent =
          batch_adjoint_tangent + node_offsets[node];
      const Scalar* left_adjoint_tangent =
          batch_adjoint_tangent + node_offsets[left_node];
      const Scalar* right_adjoint_tangent =
          batch_adjoint_tangent + node_offsets[right_node];
      Scalar* left_primal_tangent =
          batch_primal_tangent + node_offsets[left_node];
      Scalar* right_primal_tangent =
          batch_primal_tangent + node_offsets[right_node];
      const std::int64_t right_dimension = node_dimensions[right_node];
      for (std::int64_t entry =
               node_coefficient_offsets[node] + lane;
           entry < node_coefficient_offsets[node + 1];
           entry += warpSize) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        const std::int64_t left_index = row / right_dimension;
        const std::int64_t right_index = row % right_dimension;
        const Scalar coefficient = coefficient_values[entry];
        atomic_add_value(
            node_adjoint_tangent + column,
            conjugate_value(coefficient) *
                (
                    left_adjoint_tangent[left_index] *
                        right[right_index] +
                    right_adjoint_tangent[right_index] *
                        left[left_index]));
        atomic_add_value(
            left_primal_tangent + left_index,
            conjugate_value(right_adjoint_tangent[right_index]) *
                node_adjoint[column] * coefficient);
        atomic_add_value(
            right_primal_tangent + right_index,
            conjugate_value(left_adjoint_tangent[left_index]) *
                node_adjoint[column] * coefficient);
      }
      __syncwarp();
    }

    for (std::int64_t root = 0; root < root_count; ++root) {
      const bool heterogeneous = root_projection_starts != nullptr;
      const std::int64_t root_node = root_nodes[root];
      const std::int64_t projection_start =
          heterogeneous ? root_projection_starts[root] : 0;
      const std::int64_t projection_dimension =
          heterogeneous
              ? root_projection_dimensions[root]
              : total_projection_dimension;
      const std::int64_t root_dimension =
          heterogeneous
              ? node_dimensions[root_node]
              : uniform_root_dimension;
      const std::int64_t output_offset =
          heterogeneous
              ? root_output_offsets[root]
              : root * projection_dimension * root_dimension;
      const Scalar* root_adjoint_tangent =
          batch_adjoint_tangent + node_offsets[root_node];
      const std::int64_t root_work =
          projection_dimension * root_dimension;
      for (std::int64_t index = lane;
           index < root_work;
           index += warpSize) {
        const std::int64_t projection = index / root_dimension;
        const std::int64_t magnetic =
            index - projection * root_dimension;
        const Scalar contribution =
            root_adjoint_tangent[magnetic] *
            projection_values[
                source * total_projection_dimension +
                projection_start + projection];
        Scalar* destination =
            output_tangent + batch * output_dimension + output_offset +
            projection * root_dimension + magnetic;
        if (source_dimension == 1) {
          *destination += contribution;
        } else {
          atomic_add_value(destination, contribution);
        }
      }
      __syncwarp();
    }

    for (std::int64_t node = node_count; node-- > 0;) {
      if (node_leaf_offsets[node] >= 0) {
        continue;
      }
      const std::int64_t left_node = node_left[node];
      const std::int64_t right_node = node_right[node];
      const Scalar* left = batch_workspace + node_offsets[left_node];
      const Scalar* right = batch_workspace + node_offsets[right_node];
      const Scalar* node_tangent =
          batch_primal_tangent + node_offsets[node];
      Scalar* left_tangent =
          batch_primal_tangent + node_offsets[left_node];
      Scalar* right_tangent =
          batch_primal_tangent + node_offsets[right_node];
      const std::int64_t right_dimension = node_dimensions[right_node];
      for (std::int64_t entry =
               node_coefficient_offsets[node] + lane;
           entry < node_coefficient_offsets[node + 1];
           entry += warpSize) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        const std::int64_t left_index = row / right_dimension;
        const std::int64_t right_index = row % right_dimension;
        const Scalar gradient = node_tangent[column];
        atomic_add_value(
            left_tangent + left_index,
            gradient * coefficient_values[entry] *
                conjugate_value(right[right_index]));
        atomic_add_value(
            right_tangent + right_index,
            gradient * coefficient_values[entry] *
                conjugate_value(left[left_index]));
      }
      __syncwarp();
    }

    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      const Scalar* leaf_tangent =
          batch_primal_tangent + node_offsets[node];
      for (std::int64_t component = lane;
           component < node_dimensions[node];
           component += warpSize) {
        atomic_add_value(
            packed_tangent_row + leaf_offset + component,
            leaf_tangent[component]);
      }
      __syncwarp();
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_segmented_forward_warp_kernel(
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t packed_input_dimension,
    const std::int64_t* segment_source_offsets,
    const std::int64_t* segment_input_offsets,
    const std::int64_t* segment_input_dimensions,
    const std::int64_t* segment_workspace_offsets,
    const std::int64_t* segment_workspace_dimensions,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* segment_root_offsets,
    const std::int64_t* segment_projection_offsets,
    const std::int64_t* segment_projection_dimensions,
    const std::int64_t* segment_output_offsets,
    std::int64_t segment_count,
    std::int64_t total_source_count,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension,
    std::int64_t execution_class,
    Scalar* workspace,
    Scalar* output) {
  const std::int64_t sample_count = batch_size * total_source_count;
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t sample = blockIdx.x;
       sample < sample_count;
       sample += gridDim.x) {
    const std::int64_t batch = sample / total_source_count;
    const std::int64_t global_source = sample % total_source_count;
    const std::int64_t segment = factorized_segment_for_source(
        global_source,
        segment_source_offsets,
        segment_count);
    if (!factorized_segment_execution_enabled(
            segment,
            segment_node_offsets,
            node_coefficient_offsets,
            execution_class)) {
      continue;
    }
    const std::int64_t source =
        global_source - segment_source_offsets[segment];
    const std::int64_t source_dimension =
        segment_source_offsets[segment + 1] -
        segment_source_offsets[segment];
    const std::int64_t input_dimension =
        segment_input_dimensions[segment];
    const std::int64_t local_workspace_dimension =
        segment_workspace_dimensions[segment];
    const std::int64_t node_start = segment_node_offsets[segment];
    const std::int64_t node_count =
        segment_node_offsets[segment + 1] - node_start;
    const std::int64_t root_start = segment_root_offsets[segment];
    const std::int64_t root_count =
        segment_root_offsets[segment + 1] - root_start;
    const std::int64_t projection_dimension =
        segment_projection_dimensions[segment];
    Scalar* local_workspace =
        workspace + batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    const Scalar* input =
        packed_slots + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    factorized_angular_forward_warp_body(
        input,
        node_offsets + node_start,
        node_dimensions + node_start,
        node_leaf_offsets + node_start,
        node_left + node_start,
        node_right + node_start,
        node_coefficient_offsets + node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace_dimension,
        local_workspace);
    Scalar* output_row =
        output + batch * output_dimension +
        segment_output_offsets[segment];
    for (std::int64_t local_root = 0;
         local_root < root_count;
         ++local_root) {
      const std::int64_t root = root_start + local_root;
      const std::int64_t root_node = root_nodes[root];
      CUDA_KERNEL_ASSERT(root_node >= 0 && root_node < node_count);
      const std::int64_t root_dimension =
          node_dimensions[node_start + root_node];
      const Scalar* root_value =
          local_workspace + node_offsets[node_start + root_node];
      const std::int64_t root_work =
          root_projection_dimensions[root] * root_dimension;
      for (std::int64_t index = lane;
           index < root_work;
           index += warpSize) {
        const std::int64_t projection = index / root_dimension;
        const std::int64_t magnetic =
            index - projection * root_dimension;
        const Scalar contribution =
            root_value[magnetic] * projection_values[
                segment_projection_offsets[segment] +
                source * projection_dimension +
                root_projection_starts[root] + projection];
        Scalar* destination =
            output_row + root_output_offsets[root] +
            projection * root_dimension + magnetic;
        if (source_dimension == 1) {
          *destination += contribution;
        } else {
          atomic_add_value(destination, contribution);
        }
      }
      __syncwarp();
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_segmented_adjoint_warp_kernel(
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t packed_input_dimension,
    const std::int64_t* segment_source_offsets,
    const std::int64_t* segment_input_offsets,
    const std::int64_t* segment_input_dimensions,
    const std::int64_t* segment_workspace_offsets,
    const std::int64_t* segment_workspace_dimensions,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* segment_root_offsets,
    const std::int64_t* segment_projection_offsets,
    const std::int64_t* segment_projection_dimensions,
    const std::int64_t* segment_output_offsets,
    std::int64_t segment_count,
    std::int64_t total_source_count,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension,
    std::int64_t execution_class,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* packed_adjoint) {
  const std::int64_t sample_count = batch_size * total_source_count;
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t sample = blockIdx.x;
       sample < sample_count;
       sample += gridDim.x) {
    const std::int64_t batch = sample / total_source_count;
    const std::int64_t global_source = sample % total_source_count;
    const std::int64_t segment = factorized_segment_for_source(
        global_source,
        segment_source_offsets,
        segment_count);
    if (!factorized_segment_execution_enabled(
            segment,
            segment_node_offsets,
            node_coefficient_offsets,
            execution_class)) {
      continue;
    }
    const std::int64_t source =
        global_source - segment_source_offsets[segment];
    const std::int64_t input_dimension =
        segment_input_dimensions[segment];
    const std::int64_t local_workspace_dimension =
        segment_workspace_dimensions[segment];
    const std::int64_t node_start = segment_node_offsets[segment];
    const std::int64_t node_count =
        segment_node_offsets[segment + 1] - node_start;
    const std::int64_t root_start = segment_root_offsets[segment];
    const std::int64_t root_count =
        segment_root_offsets[segment + 1] - root_start;
    const std::int64_t local_output_dimension =
        segment_output_offsets[segment + 1] -
        segment_output_offsets[segment];
    const std::int64_t workspace_base =
        batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    Scalar* local_workspace = workspace + workspace_base;
    Scalar* local_workspace_adjoint =
        workspace_adjoint + workspace_base;
    Scalar* input_adjoint =
        packed_adjoint + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    for (std::int64_t index = lane;
         index < input_dimension;
         index += warpSize) {
      input_adjoint[index] = Scalar(0);
    }
    __syncwarp();
    const Scalar* input =
        packed_slots + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    factorized_angular_forward_warp_body(
        input,
        node_offsets + node_start,
        node_dimensions + node_start,
        node_leaf_offsets + node_start,
        node_left + node_start,
        node_right + node_start,
        node_coefficient_offsets + node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace_dimension,
        local_workspace);
    factorized_angular_adjoint_warp_body(
        output_adjoint + batch * output_dimension +
            segment_output_offsets[segment],
        node_offsets + node_start,
        node_dimensions + node_start,
        node_leaf_offsets + node_start,
        node_left + node_start,
        node_right + node_start,
        node_coefficient_offsets + node_start,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        root_nodes + root_start,
        root_count,
        root_projection_starts + root_start,
        root_projection_dimensions + root_start,
        root_output_offsets + root_start,
        projection_values + segment_projection_offsets[segment],
        source,
        segment_projection_dimensions[segment],
        local_workspace_dimension,
        0,
        local_output_dimension,
        local_workspace,
        local_workspace_adjoint);
    for (std::int64_t local_node = 0;
         local_node < node_count;
         ++local_node) {
      const std::int64_t node = node_start + local_node;
      const std::int64_t leaf_offset = node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      const Scalar* leaf_adjoint =
          local_workspace_adjoint + node_offsets[node];
      for (std::int64_t component = lane;
           component < node_dimensions[node];
           component += warpSize) {
        atomic_add_value(
            input_adjoint + leaf_offset + component,
            leaf_adjoint[component]);
      }
      __syncwarp();
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_segmented_double_backward_warp_kernel(
    const Scalar* packed_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    std::int64_t packed_input_dimension,
    const std::int64_t* segment_source_offsets,
    const std::int64_t* segment_input_offsets,
    const std::int64_t* segment_input_dimensions,
    const std::int64_t* segment_workspace_offsets,
    const std::int64_t* segment_workspace_dimensions,
    const std::int64_t* segment_node_offsets,
    const std::int64_t* segment_root_offsets,
    const std::int64_t* segment_projection_offsets,
    const std::int64_t* segment_projection_dimensions,
    const std::int64_t* segment_output_offsets,
    std::int64_t segment_count,
    std::int64_t total_source_count,
    const std::int64_t* node_offsets,
    const std::int64_t* node_dimensions,
    const std::int64_t* node_leaf_offsets,
    const std::int64_t* node_left,
    const std::int64_t* node_right,
    const std::int64_t* node_coefficient_offsets,
    const std::int64_t* coefficient_rows,
    const std::int64_t* coefficient_columns,
    const Scalar* coefficient_values,
    const std::int64_t* root_nodes,
    const std::int64_t* root_projection_starts,
    const std::int64_t* root_projection_dimensions,
    const std::int64_t* root_output_offsets,
    const Scalar* projection_values,
    std::int64_t workspace_dimension,
    std::int64_t output_dimension,
    std::int64_t execution_class,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* adjoint_tangent,
    Scalar* primal_tangent,
    Scalar* output_tangent,
    Scalar* packed_tangent) {
  const std::int64_t sample_count = batch_size * total_source_count;
  const std::int64_t lane = threadIdx.x;
  for (std::int64_t sample = blockIdx.x;
       sample < sample_count;
       sample += gridDim.x) {
    const std::int64_t batch = sample / total_source_count;
    const std::int64_t global_source = sample % total_source_count;
    const std::int64_t segment = factorized_segment_for_source(
        global_source,
        segment_source_offsets,
        segment_count);
    if (!factorized_segment_execution_enabled(
            segment,
            segment_node_offsets,
            node_coefficient_offsets,
            execution_class)) {
      continue;
    }
    const std::int64_t source =
        global_source - segment_source_offsets[segment];
    const std::int64_t source_dimension =
        segment_source_offsets[segment + 1] -
        segment_source_offsets[segment];
    const std::int64_t input_dimension =
        segment_input_dimensions[segment];
    const std::int64_t local_workspace_dimension =
        segment_workspace_dimensions[segment];
    const std::int64_t node_start = segment_node_offsets[segment];
    const std::int64_t node_count =
        segment_node_offsets[segment + 1] - node_start;
    const std::int64_t root_start = segment_root_offsets[segment];
    const std::int64_t root_count =
        segment_root_offsets[segment + 1] - root_start;
    const std::int64_t local_output_dimension =
        segment_output_offsets[segment + 1] -
        segment_output_offsets[segment];
    const std::int64_t workspace_base =
        batch * workspace_dimension +
        segment_workspace_offsets[segment] +
        source * local_workspace_dimension;
    Scalar* local_workspace = workspace + workspace_base;
    Scalar* local_workspace_adjoint =
        workspace_adjoint + workspace_base;
    Scalar* local_adjoint_tangent =
        adjoint_tangent + workspace_base;
    Scalar* local_primal_tangent =
        primal_tangent + workspace_base;
    Scalar* input_tangent =
        packed_tangent + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    const Scalar* input =
        packed_slots + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    const Scalar* input_adjoint_tangent =
        packed_adjoint_tangent + batch * packed_input_dimension +
        segment_input_offsets[segment] + source * input_dimension;
    const std::int64_t* local_node_offsets = node_offsets + node_start;
    const std::int64_t* local_node_dimensions =
        node_dimensions + node_start;
    const std::int64_t* local_node_leaf_offsets =
        node_leaf_offsets + node_start;
    const std::int64_t* local_node_left = node_left + node_start;
    const std::int64_t* local_node_right = node_right + node_start;
    const std::int64_t* local_node_coefficient_offsets =
        node_coefficient_offsets + node_start;
    for (std::int64_t index = lane;
         index < input_dimension;
         index += warpSize) {
      input_tangent[index] = Scalar(0);
    }
    for (std::int64_t index = lane;
         index < local_workspace_dimension;
         index += warpSize) {
      local_adjoint_tangent[index] = Scalar(0);
      local_primal_tangent[index] = Scalar(0);
    }
    __syncwarp();
    factorized_angular_forward_warp_body(
        input,
        local_node_offsets,
        local_node_dimensions,
        local_node_leaf_offsets,
        local_node_left,
        local_node_right,
        local_node_coefficient_offsets,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        local_workspace_dimension,
        local_workspace);
    factorized_angular_adjoint_warp_body(
        output_adjoint + batch * output_dimension +
            segment_output_offsets[segment],
        local_node_offsets,
        local_node_dimensions,
        local_node_leaf_offsets,
        local_node_left,
        local_node_right,
        local_node_coefficient_offsets,
        node_count,
        coefficient_rows,
        coefficient_columns,
        coefficient_values,
        root_nodes + root_start,
        root_count,
        root_projection_starts + root_start,
        root_projection_dimensions + root_start,
        root_output_offsets + root_start,
        projection_values + segment_projection_offsets[segment],
        source,
        segment_projection_dimensions[segment],
        local_workspace_dimension,
        0,
        local_output_dimension,
        local_workspace,
        local_workspace_adjoint);

    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = local_node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      Scalar* leaf_tangent =
          local_adjoint_tangent + local_node_offsets[node];
      for (std::int64_t component = lane;
           component < local_node_dimensions[node];
           component += warpSize) {
        atomic_add_value(
            leaf_tangent + component,
            input_adjoint_tangent[leaf_offset + component]);
      }
      __syncwarp();
    }

    for (std::int64_t node = 0; node < node_count; ++node) {
      if (local_node_leaf_offsets[node] >= 0) {
        continue;
      }
      const std::int64_t left_node = local_node_left[node];
      const std::int64_t right_node = local_node_right[node];
      const Scalar* left = local_workspace + local_node_offsets[left_node];
      const Scalar* right = local_workspace + local_node_offsets[right_node];
      const Scalar* node_adjoint =
          local_workspace_adjoint + local_node_offsets[node];
      Scalar* node_adjoint_tangent =
          local_adjoint_tangent + local_node_offsets[node];
      const Scalar* left_adjoint_tangent =
          local_adjoint_tangent + local_node_offsets[left_node];
      const Scalar* right_adjoint_tangent =
          local_adjoint_tangent + local_node_offsets[right_node];
      Scalar* left_primal_tangent =
          local_primal_tangent + local_node_offsets[left_node];
      Scalar* right_primal_tangent =
          local_primal_tangent + local_node_offsets[right_node];
      const std::int64_t right_dimension =
          local_node_dimensions[right_node];
      for (std::int64_t entry =
               local_node_coefficient_offsets[node] + lane;
           entry < local_node_coefficient_offsets[node + 1];
           entry += warpSize) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        const std::int64_t left_index = row / right_dimension;
        const std::int64_t right_index = row % right_dimension;
        const Scalar coefficient = coefficient_values[entry];
        atomic_add_value(
            node_adjoint_tangent + column,
            conjugate_value(coefficient) *
                (left_adjoint_tangent[left_index] * right[right_index] +
                 right_adjoint_tangent[right_index] * left[left_index]));
        atomic_add_value(
            left_primal_tangent + left_index,
            conjugate_value(right_adjoint_tangent[right_index]) *
                node_adjoint[column] * coefficient);
        atomic_add_value(
            right_primal_tangent + right_index,
            conjugate_value(left_adjoint_tangent[left_index]) *
                node_adjoint[column] * coefficient);
      }
      __syncwarp();
    }

    Scalar* output_tangent_row =
        output_tangent + batch * output_dimension +
        segment_output_offsets[segment];
    for (std::int64_t local_root = 0;
         local_root < root_count;
         ++local_root) {
      const std::int64_t root = root_start + local_root;
      const std::int64_t root_node = root_nodes[root];
      const std::int64_t root_dimension =
          local_node_dimensions[root_node];
      const Scalar* root_adjoint_tangent =
          local_adjoint_tangent + local_node_offsets[root_node];
      const std::int64_t root_work =
          root_projection_dimensions[root] * root_dimension;
      for (std::int64_t index = lane;
           index < root_work;
           index += warpSize) {
        const std::int64_t projection = index / root_dimension;
        const std::int64_t magnetic =
            index - projection * root_dimension;
        const Scalar contribution =
            root_adjoint_tangent[magnetic] * projection_values[
                segment_projection_offsets[segment] +
                source * segment_projection_dimensions[segment] +
                root_projection_starts[root] + projection];
        Scalar* destination =
            output_tangent_row + root_output_offsets[root] +
            projection * root_dimension + magnetic;
        if (source_dimension == 1) {
          *destination += contribution;
        } else {
          atomic_add_value(destination, contribution);
        }
      }
      __syncwarp();
    }

    for (std::int64_t node = node_count; node-- > 0;) {
      if (local_node_leaf_offsets[node] >= 0) {
        continue;
      }
      const std::int64_t left_node = local_node_left[node];
      const std::int64_t right_node = local_node_right[node];
      const Scalar* left = local_workspace + local_node_offsets[left_node];
      const Scalar* right = local_workspace + local_node_offsets[right_node];
      const Scalar* node_tangent =
          local_primal_tangent + local_node_offsets[node];
      Scalar* left_tangent =
          local_primal_tangent + local_node_offsets[left_node];
      Scalar* right_tangent =
          local_primal_tangent + local_node_offsets[right_node];
      const std::int64_t right_dimension =
          local_node_dimensions[right_node];
      for (std::int64_t entry =
               local_node_coefficient_offsets[node] + lane;
           entry < local_node_coefficient_offsets[node + 1];
           entry += warpSize) {
        const std::int64_t row = coefficient_rows[entry];
        const std::int64_t column = coefficient_columns[entry];
        const std::int64_t left_index = row / right_dimension;
        const std::int64_t right_index = row % right_dimension;
        const Scalar gradient = node_tangent[column];
        atomic_add_value(
            left_tangent + left_index,
            gradient * coefficient_values[entry] *
                conjugate_value(right[right_index]));
        atomic_add_value(
            right_tangent + right_index,
            gradient * coefficient_values[entry] *
                conjugate_value(left[left_index]));
      }
      __syncwarp();
    }

    for (std::int64_t node = 0; node < node_count; ++node) {
      const std::int64_t leaf_offset = local_node_leaf_offsets[node];
      if (leaf_offset < 0) {
        continue;
      }
      const Scalar* leaf_tangent =
          local_primal_tangent + local_node_offsets[node];
      for (std::int64_t component = lane;
           component < local_node_dimensions[node];
           component += warpSize) {
        atomic_add_value(
            input_tangent + leaf_offset + component,
            leaf_tangent[component]);
      }
      __syncwarp();
    }
  }
}

template <typename Scalar>
__global__ void factorized_angular_linear_forward_kernel(
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
    const Scalar* bias,
    std::int64_t workspace_dimension,
    std::int64_t root_dimension,
    Scalar* workspace,
    Scalar* output) {
  for (std::int64_t batch =
           blockIdx.x * blockDim.x + threadIdx.x;
       batch < batch_size;
       batch += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    Scalar* batch_workspace =
        workspace + batch * workspace_dimension;
    Scalar value = bias[0];
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      for (std::int64_t index = 0; index < workspace_dimension; ++index) {
        batch_workspace[index] = Scalar(0);
      }
      const Scalar* input_row =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output =
            batch_workspace + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          for (
              std::int64_t component = 0;
              component < node_dimensions[node];
              ++component) {
            node_output[component] = input_row[leaf_offset + component];
          }
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          node_output[column] +=
              conjugate_value(coefficient_values[entry]) *
              left[row / right_dimension] *
              right[row % right_dimension];
        }
      }
      for (std::int64_t root = 0; root < root_count; ++root) {
        const Scalar* root_value =
            batch_workspace + node_offsets[root_nodes[root]];
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
__global__ void factorized_angular_linear_adjoint_kernel(
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
    std::int64_t workspace_dimension,
    std::int64_t root_dimension,
    Scalar* workspace,
    Scalar* workspace_adjoint,
    Scalar* packed_slots_adjoint,
    Scalar* weight_adjoint,
    Scalar* bias_adjoint) {
  for (std::int64_t batch =
           blockIdx.x * blockDim.x + threadIdx.x;
       batch < batch_size;
       batch += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    Scalar* batch_workspace =
        workspace + batch * workspace_dimension;
    Scalar* batch_workspace_adjoint =
        workspace_adjoint + batch * workspace_dimension;
    Scalar* input_adjoint =
        packed_slots_adjoint +
        batch * source_dimension * input_dimension;
    const Scalar gradient = output_adjoint[batch];
    atomic_add_value(bias_adjoint, gradient);
    for (
        std::int64_t index = 0;
        index < source_dimension * input_dimension;
        ++index) {
      input_adjoint[index] = Scalar(0);
    }
    for (std::int64_t source = 0; source < source_dimension; ++source) {
      for (std::int64_t index = 0; index < workspace_dimension; ++index) {
        batch_workspace[index] = Scalar(0);
        batch_workspace_adjoint[index] = Scalar(0);
      }
      const Scalar* input_source =
          packed_slots +
          (batch * source_dimension + source) * input_dimension;
      for (std::int64_t node = 0; node < node_count; ++node) {
        Scalar* node_output =
            batch_workspace + node_offsets[node];
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          for (
              std::int64_t component = 0;
              component < node_dimensions[node];
              ++component) {
            node_output[component] =
                input_source[leaf_offset + component];
          }
          continue;
        }
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        const std::int64_t right_dimension =
            node_dimensions[right_node];
        for (
            std::int64_t entry = node_coefficient_offsets[node];
            entry < node_coefficient_offsets[node + 1];
            ++entry) {
          const std::int64_t row = coefficient_rows[entry];
          const std::int64_t column = coefficient_columns[entry];
          node_output[column] +=
              conjugate_value(coefficient_values[entry]) *
              left[row / right_dimension] *
              right[row % right_dimension];
        }
      }
      for (std::int64_t root = 0; root < root_count; ++root) {
        const Scalar* root_value =
            batch_workspace + node_offsets[root_nodes[root]];
        Scalar* root_adjoint =
            batch_workspace_adjoint +
            node_offsets[root_nodes[root]];
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
            atomic_add_value(
                weight_adjoint + feature,
                gradient *
                    conjugate_value(
                        root_value[magnetic] * projection_value));
            root_adjoint[magnetic] +=
                gradient *
                conjugate_value(weight[feature]) *
                conjugate_value(projection_value);
          }
        }
      }
      for (std::int64_t node = node_count; node-- > 0;) {
        const std::int64_t leaf_offset = node_leaf_offsets[node];
        if (leaf_offset >= 0) {
          continue;
        }
        Scalar* node_adjoint =
            batch_workspace_adjoint + node_offsets[node];
        const std::int64_t left_node = node_left[node];
        const std::int64_t right_node = node_right[node];
        const Scalar* left =
            batch_workspace + node_offsets[left_node];
        const Scalar* right =
            batch_workspace + node_offsets[right_node];
        Scalar* left_adjoint =
            batch_workspace_adjoint + node_offsets[left_node];
        Scalar* right_adjoint =
            batch_workspace_adjoint + node_offsets[right_node];
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
              conjugate_value(right[right_index]);
          right_adjoint[right_index] +=
              node_gradient *
              coefficient_values[entry] *
              conjugate_value(left[left_index]);
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
            batch_workspace_adjoint + node_offsets[node];
        for (
            std::int64_t component = 0;
            component < node_dimensions[node];
            ++component) {
          input_source_adjoint[leaf_offset + component] +=
              leaf_adjoint[component];
        }
      }
    }
  }
}

template <typename Scalar>
__global__ void density_accumulate_adjoint_kernel(
    const Scalar* atomic_adjoint,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* edge_adjoint) {
  const std::int64_t total = edge_count * channel_count;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t edge = index / channel_count;
    const std::int64_t channel = index - edge * channel_count;
    const std::int64_t center = centers[edge];
    CUDA_KERNEL_ASSERT(center >= 0 && center < atom_count);
    edge_adjoint[index] =
        atomic_adjoint[center * channel_count + channel];
  }
}

template <>
__device__ inline void atomic_add_value(
    c10::complex<double>* destination,
    c10::complex<double> value) {
  auto* components = reinterpret_cast<double*>(destination);
  atomicAdd(components, value.real());
  atomicAdd(components + 1, value.imag());
}

template <typename Scalar>
__global__ void source_assembly_kernel(
    const Scalar* source,
    std::int64_t batch_size,
    std::int64_t source_dimension,
    const std::int64_t* rows,
    const std::int64_t* columns,
    const Scalar* values,
    std::int64_t nnz,
    std::int64_t induced_dimension,
    Scalar* ambient) {
  const std::int64_t total = batch_size * nnz;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / nnz;
    const std::int64_t entry = index - batch * nnz;
    atomic_add_value(
        ambient + batch * induced_dimension + rows[entry],
        values[entry] *
            source[batch * source_dimension + columns[entry]]);
  }
}

template <typename Scalar>
__global__ void synthesis_analysis_kernel(
    const Scalar* ambient,
    std::int64_t batch_size,
    std::int64_t induced_dimension,
    const std::int64_t* rows,
    const std::int64_t* columns,
    const Scalar* values,
    std::int64_t nnz,
    std::int64_t output_dimension,
    Scalar* output) {
  const std::int64_t total = batch_size * nnz;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / nnz;
    const std::int64_t entry = index - batch * nnz;
    atomic_add_value(
        output + batch * output_dimension + columns[entry],
        conjugate_value(values[entry]) *
            ambient[batch * induced_dimension + rows[entry]]);
  }
}

template <typename Scalar>
__global__ void synthesis_adjoint_kernel(
    const Scalar* output_adjoint,
    std::int64_t batch_size,
    std::int64_t output_dimension,
    const std::int64_t* rows,
    const std::int64_t* columns,
    const Scalar* values,
    std::int64_t nnz,
    std::int64_t induced_dimension,
    Scalar* ambient_adjoint) {
  const std::int64_t total = batch_size * nnz;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / nnz;
    const std::int64_t entry = index - batch * nnz;
    atomic_add_value(
        ambient_adjoint + batch * induced_dimension + rows[entry],
        values[entry] *
            output_adjoint[
                batch * output_dimension + columns[entry]]);
  }
}

template <typename Scalar>
__global__ void source_assembly_adjoint_kernel(
    const Scalar* ambient_adjoint,
    std::int64_t batch_size,
    std::int64_t induced_dimension,
    const std::int64_t* rows,
    const std::int64_t* columns,
    const Scalar* values,
    std::int64_t nnz,
    std::int64_t source_dimension,
    Scalar* source_adjoint) {
  const std::int64_t total = batch_size * nnz;
  for (std::int64_t index =
           blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<std::int64_t>(blockDim.x) * gridDim.x) {
    const std::int64_t batch = index / nnz;
    const std::int64_t entry = index - batch * nnz;
    atomic_add_value(
        source_adjoint + batch * source_dimension + columns[entry],
        conjugate_value(values[entry]) *
            ambient_adjoint[
                batch * induced_dimension + rows[entry]]);
  }
}

int launch_blocks(std::int64_t work_items, int threads = 256) {
  return static_cast<int>(
      std::min<std::int64_t>(
          (work_items + threads - 1) / threads,
      65535));
}

std::tuple<torch::Tensor, torch::Tensor>
cheb_exp_cos_radial_with_derivative_cuda(
    const torch::Tensor& radii,
    const torch::Tensor& cutoffs,
    const torch::Tensor& lambdas,
    std::int64_t radial_index) {
  check_cuda_real_vector(radii, "radii");
  check_cuda_real_vector(cutoffs, "cutoffs");
  check_cuda_real_vector(lambdas, "lambdas");
  TORCH_CHECK(
      radii.sizes() == cutoffs.sizes() &&
          radii.sizes() == lambdas.sizes(),
      "radii, cutoffs, and lambdas must have identical shapes");
  TORCH_CHECK(
      radii.scalar_type() == cutoffs.scalar_type() &&
          radii.scalar_type() == lambdas.scalar_type(),
      "radii, cutoffs, and lambdas must have identical dtypes");
  TORCH_CHECK(radial_index >= 0, "radial_index must be non-negative");
  c10::cuda::CUDAGuard device_guard(radii.device());
  auto values = torch::empty_like(radii);
  auto derivatives = torch::empty_like(radii);
  if (radii.numel() == 0) {
    return std::make_tuple(values, derivatives);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      radii.scalar_type(),
      "ye3t_cheb_exp_cos_radial_with_derivative_cuda",
      [&] {
        cheb_exp_cos_radial_kernel<scalar_t>
            <<<launch_blocks(radii.numel()), threads, 0, stream>>>(
                radii.data_ptr<scalar_t>(),
                cutoffs.data_ptr<scalar_t>(),
                lambdas.data_ptr<scalar_t>(),
                radii.numel(),
                radial_index,
                values.data_ptr<scalar_t>(),
                derivatives.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
cheb_exp_cos_radial_table_with_derivative_cuda(
    const torch::Tensor& radii,
    const torch::Tensor& cutoffs,
    const torch::Tensor& lambdas,
    std::int64_t maximum_radial_index) {
  check_cuda_real_vector(radii, "radii");
  check_cuda_real_vector(cutoffs, "cutoffs");
  check_cuda_real_vector(lambdas, "lambdas");
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
  c10::cuda::CUDAGuard device_guard(radii.device());
  const std::int64_t width = maximum_radial_index + 1;
  auto values = torch::empty(
      {radii.numel(), width},
      radii.options());
  auto derivatives = torch::empty_like(values);
  if (radii.numel() == 0) {
    return std::make_tuple(values, derivatives);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      radii.scalar_type(),
      "ye3t_cheb_exp_cos_radial_table_with_derivative_cuda",
      [&] {
        cheb_exp_cos_radial_table_kernel<scalar_t>
            <<<launch_blocks(radii.numel()), threads, 0, stream>>>(
                radii.data_ptr<scalar_t>(),
                cutoffs.data_ptr<scalar_t>(),
                lambdas.data_ptr<scalar_t>(),
                radii.numel(),
                maximum_radial_index,
                values.data_ptr<scalar_t>(),
                derivatives.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
cheb_exp_cos_radial_table_double_backward_cuda(
    const torch::Tensor& values_adjoint,
    const torch::Tensor& radial_derivatives,
    const torch::Tensor& radii,
    const torch::Tensor& cutoffs,
    const torch::Tensor& lambdas,
    const torch::Tensor& grad_grad_radii,
    std::int64_t maximum_radial_index) {
  check_cuda_real_vector(radii, "radii");
  check_cuda_real_vector(cutoffs, "cutoffs");
  check_cuda_real_vector(lambdas, "lambdas");
  check_cuda_real_vector(grad_grad_radii, "grad_grad_radii");
  check_cuda_real_matrix(values_adjoint, "values_adjoint");
  check_cuda_real_matrix(radial_derivatives, "radial_derivatives");
  const std::int64_t width = maximum_radial_index + 1;
  TORCH_CHECK(
      maximum_radial_index >= 0,
      "maximum_radial_index must be non-negative");
  TORCH_CHECK(
      values_adjoint.size(0) == radii.numel() &&
          values_adjoint.size(1) == width &&
          radial_derivatives.sizes() == values_adjoint.sizes(),
      "radial table shapes do not match maximum_radial_index");
  TORCH_CHECK(
      values_adjoint.device() == radii.device() &&
          radial_derivatives.device() == radii.device() &&
          cutoffs.device() == radii.device() &&
          lambdas.device() == radii.device() &&
          grad_grad_radii.device() == radii.device(),
      "radial double-backward tensors must use one CUDA device");
  TORCH_CHECK(
      values_adjoint.scalar_type() == radii.scalar_type() &&
          radial_derivatives.scalar_type() == radii.scalar_type() &&
          cutoffs.scalar_type() == radii.scalar_type() &&
          lambdas.scalar_type() == radii.scalar_type() &&
          grad_grad_radii.scalar_type() == radii.scalar_type(),
      "radial double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      radii.sizes() == cutoffs.sizes() &&
          radii.sizes() == lambdas.sizes() &&
          radii.sizes() == grad_grad_radii.sizes(),
      "radial vectors must have identical shapes");
  c10::cuda::CUDAGuard device_guard(radii.device());
  auto values_adjoint_gradient = torch::empty_like(values_adjoint);
  auto radii_gradient = torch::empty_like(radii);
  if (radii.numel() == 0) {
    return std::make_tuple(
        values_adjoint_gradient,
        radii_gradient);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      radii.scalar_type(),
      "ye3t_cheb_exp_cos_radial_table_double_backward_cuda",
      [&] {
        cheb_exp_cos_radial_table_double_backward_kernel<scalar_t>
            <<<launch_blocks(radii.numel()), threads, 0, stream>>>(
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      values_adjoint_gradient,
      radii_gradient);
}

std::tuple<torch::Tensor, torch::Tensor>
spherical_harmonics_with_derivative_cuda(
    const torch::Tensor& edge_vectors,
    std::int64_t angular_momentum,
    bool real_output,
    double epsilon) {
  TORCH_CHECK(edge_vectors.is_cuda(), "edge_vectors must be on CUDA");
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
  c10::cuda::CUDAGuard device_guard(edge_vectors.device());
  const std::int64_t width = 2 * angular_momentum + 1;
  auto real_values = torch::empty(
      {edge_vectors.size(0), width},
      edge_vectors.options());
  auto real_derivatives = torch::empty(
      {edge_vectors.size(0), width, 3},
      edge_vectors.options());
  auto scratch = torch::empty(
      {edge_vectors.size(0), 3 * (angular_momentum + 1)},
      edge_vectors.options());
  if (edge_vectors.size(0) > 0) {
    constexpr int threads = 128;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_real_spherical_harmonics_with_derivative_cuda",
        [&] {
          real_spherical_harmonics_kernel<scalar_t>
              <<<launch_blocks(edge_vectors.size(0)), threads, 0, stream>>>(
                  edge_vectors.data_ptr<scalar_t>(),
                  edge_vectors.size(0),
                  angular_momentum,
                  static_cast<scalar_t>(epsilon),
                  scratch.data_ptr<scalar_t>(),
                  real_values.data_ptr<scalar_t>(),
                  real_derivatives.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  if (real_output) {
    return std::make_tuple(real_values, real_derivatives);
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
  const std::int64_t conversion_work =
      edge_vectors.size(0) * width;
  if (conversion_work > 0) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_complex_spherical_harmonics_with_derivative_cuda",
        [&] {
          real_to_complex_spherical_kernel<scalar_t>
              <<<launch_blocks(conversion_work), threads, 0, stream>>>(
                  real_values.data_ptr<scalar_t>(),
                  real_derivatives.data_ptr<scalar_t>(),
                  edge_vectors.size(0),
                  angular_momentum,
                  values.data_ptr<c10::complex<scalar_t>>(),
                  derivatives.data_ptr<c10::complex<scalar_t>>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
spherical_harmonics_table_with_derivative_cuda(
    const torch::Tensor& edge_vectors,
    std::int64_t maximum_angular_momentum,
    bool real_output,
    double epsilon) {
  TORCH_CHECK(edge_vectors.is_cuda(), "edge_vectors must be on CUDA");
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
  c10::cuda::CUDAGuard device_guard(edge_vectors.device());
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  auto real_values = torch::empty(
      {edge_vectors.size(0), width},
      edge_vectors.options());
  auto real_derivatives = torch::empty(
      {edge_vectors.size(0), width, 3},
      edge_vectors.options());
  auto scratch = torch::empty(
      {
          edge_vectors.size(0),
          3 * (maximum_angular_momentum + 1),
      },
      edge_vectors.options());
  if (edge_vectors.size(0) > 0) {
    constexpr int threads = 128;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_real_spherical_harmonics_table_with_derivative_cuda",
        [&] {
          real_spherical_harmonics_table_kernel<scalar_t>
              <<<launch_blocks(edge_vectors.size(0)), threads, 0, stream>>>(
                  edge_vectors.data_ptr<scalar_t>(),
                  edge_vectors.size(0),
                  maximum_angular_momentum,
                  static_cast<scalar_t>(epsilon),
                  scratch.data_ptr<scalar_t>(),
                  real_values.data_ptr<scalar_t>(),
                  real_derivatives.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  if (real_output) {
    return std::make_tuple(real_values, real_derivatives);
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
  const std::int64_t conversion_work =
      edge_vectors.size(0) * width;
  if (conversion_work > 0) {
    constexpr int threads = 256;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_complex_spherical_harmonics_table_with_derivative_cuda",
        [&] {
          real_to_complex_spherical_table_kernel<scalar_t>
              <<<launch_blocks(conversion_work), threads, 0, stream>>>(
                  real_values.data_ptr<scalar_t>(),
                  real_derivatives.data_ptr<scalar_t>(),
                  edge_vectors.size(0),
                  maximum_angular_momentum,
                  values.data_ptr<c10::complex<scalar_t>>(),
                  derivatives.data_ptr<c10::complex<scalar_t>>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return std::make_tuple(values, derivatives);
}

std::tuple<torch::Tensor, torch::Tensor>
spherical_harmonics_table_double_backward_cuda(
    const torch::Tensor& edge_vectors,
    const torch::Tensor& value_adjoint,
    const torch::Tensor& edge_direction,
    std::int64_t maximum_angular_momentum,
    double epsilon) {
  check_cuda_real_matrix(edge_vectors, "edge_vectors");
  check_cuda_real_matrix(value_adjoint, "value_adjoint");
  check_cuda_real_matrix(edge_direction, "edge_direction");
  TORCH_CHECK(
      edge_vectors.size(1) == 3,
      "edge_vectors must have shape [edge_count, 3]");
  TORCH_CHECK(
      edge_direction.sizes() == edge_vectors.sizes(),
      "edge_direction must match edge_vectors");
  TORCH_CHECK(
      maximum_angular_momentum >= 0,
      "maximum_angular_momentum must be non-negative");
  TORCH_CHECK(epsilon > 0.0, "epsilon must be positive");
  const std::int64_t width =
      (maximum_angular_momentum + 1) *
      (maximum_angular_momentum + 1);
  TORCH_CHECK(
      value_adjoint.size(0) == edge_vectors.size(0) &&
          value_adjoint.size(1) == width,
      "value_adjoint must have shape [edge_count, (maximum_L + 1)^2]");
  TORCH_CHECK(
      edge_vectors.scalar_type() == value_adjoint.scalar_type() &&
          edge_vectors.scalar_type() == edge_direction.scalar_type(),
      "edge vectors, value adjoints, and directions must share dtype");
  TORCH_CHECK(
      edge_vectors.device() == value_adjoint.device() &&
          edge_vectors.device() == edge_direction.device(),
      "edge vectors, value adjoints, and directions must share device");
  c10::cuda::CUDAGuard device_guard(edge_vectors.device());
  auto value_adjoint_gradient =
      torch::empty_like(value_adjoint);
  auto edge_gradient = torch::empty_like(edge_vectors);
  auto scratch = torch::empty(
      {
          edge_vectors.size(0),
          3,
          maximum_angular_momentum + 1,
          8,
      },
      edge_vectors.options());
  if (edge_vectors.size(0) > 0) {
    constexpr int threads = 128;
    const auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES(
        edge_vectors.scalar_type(),
        "ye3t_real_spherical_harmonics_table_double_backward_cuda",
        [&] {
          static_assert(
              sizeof(SphericalDirectionalJet<scalar_t>) ==
                  8 * sizeof(scalar_t),
              "SphericalDirectionalJet must have packed scalar storage");
          real_spherical_harmonics_table_double_backward_kernel<scalar_t>
              <<<launch_blocks(edge_vectors.size(0)), threads, 0, stream>>>(
                  edge_vectors.data_ptr<scalar_t>(),
                  value_adjoint.data_ptr<scalar_t>(),
                  edge_direction.data_ptr<scalar_t>(),
                  edge_vectors.size(0),
                  maximum_angular_momentum,
                  static_cast<scalar_t>(epsilon),
                  reinterpret_cast<
                      SphericalDirectionalJet<scalar_t>*>(
                      scratch.data_ptr<scalar_t>()),
                  value_adjoint_gradient.data_ptr<scalar_t>(),
                  edge_gradient.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
  }
  return std::make_tuple(
      value_adjoint_gradient,
      edge_gradient);
}

std::tuple<
    torch::Tensor,
    torch::Tensor,
    torch::Tensor,
    torch::Tensor>
plain_site_basis_product_with_derivative_cuda(
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
  check_cuda_data_tensor(radial_values, "radial_values");
  check_cuda_data_tensor(radial_derivatives, "radial_derivatives");
  check_cuda_data_tensor(angular_values, "angular_values");
  check_cuda_factor_tensor(angular_derivatives, "angular_derivatives");
  check_cuda_data_tensor(prefactors, "prefactors");
  check_cuda_data_tensor(
      prefactor_derivatives_center,
      "prefactor_derivatives_center");
  check_cuda_data_tensor(
      prefactor_derivatives_neighbor,
      "prefactor_derivatives_neighbor");
  check_cuda_data_tensor(radial_directions, "radial_directions");
  check_cuda_int64_vector(term_groups, "term_groups");
  check_cuda_int64_vector(term_channels, "term_channels");
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
  TORCH_CHECK(
      term_groups.device() == device &&
          term_channels.device() == device,
      "source-product term indices must be on the data device");
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
  c10::cuda::CUDAGuard device_guard(device);
  auto edge_values = torch::zeros(
      {edge_count, channel_count},
      radial_values.options());
  auto edge_derivatives = torch::zeros(
      {edge_count, channel_count, 3},
      radial_values.options());
  auto charge_center = torch::zeros_like(edge_values);
  auto charge_neighbor = torch::zeros_like(edge_values);
  if (edge_count == 0 || term_count == 0) {
    return std::make_tuple(
        edge_values,
        edge_derivatives,
        charge_center,
        charge_neighbor);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const std::int64_t work_items = edge_count * term_count;
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dtype,
      "ye3t_plain_site_basis_product_with_derivative_cuda",
      [&] {
        plain_site_basis_product_kernel<scalar_t>
            <<<launch_blocks(work_items), threads, 0, stream>>>(
                radial_values.data_ptr<scalar_t>(),
                radial_derivatives.data_ptr<scalar_t>(),
                angular_values.data_ptr<scalar_t>(),
                angular_derivatives.data_ptr<scalar_t>(),
                prefactors.data_ptr<scalar_t>(),
                prefactor_derivatives_center.data_ptr<scalar_t>(),
                prefactor_derivatives_neighbor.data_ptr<scalar_t>(),
                radial_directions.data_ptr<scalar_t>(),
                term_groups.data_ptr<std::int64_t>(),
                term_channels.data_ptr<std::int64_t>(),
                group_count,
                edge_count,
                term_count,
                channel_count,
                edge_values.data_ptr<scalar_t>(),
                edge_derivatives.data_ptr<scalar_t>(),
                charge_center.data_ptr<scalar_t>(),
                charge_neighbor.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      edge_values,
      edge_derivatives,
      charge_center,
      charge_neighbor);
}

std::tuple<torch::Tensor, torch::Tensor>
scheduled_radial_angular_channels_with_derivative_cuda(
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
  check_cuda_real_matrix(radial_values, "radial_values");
  check_cuda_real_matrix(radial_derivatives, "radial_derivatives");
  check_cuda_real_matrix(angular_values, "angular_values");
  check_cuda_factor_tensor(angular_derivatives, "angular_derivatives");
  check_cuda_real_matrix(radial_directions, "radial_directions");
  check_cuda_int64_vector(edge_types, "edge_types");
  check_cuda_int64_vector(
      channel_radial_indices,
      "channel_radial_indices");
  check_cuda_int64_vector(
      channel_angular_indices,
      "channel_angular_indices");
  check_cuda_int64_vector(channel_types, "channel_types");
  check_cuda_real_vector(channel_scales, "channel_scales");
  const auto dtype = radial_values.scalar_type();
  const auto device = radial_values.device();
  for (const auto& tensor : {
           radial_derivatives,
           angular_values,
           angular_derivatives,
           radial_directions,
           channel_scales}) {
    TORCH_CHECK(
        tensor.scalar_type() == dtype,
        "scheduled radial-angular data tensors must share one dtype");
    TORCH_CHECK(
        tensor.device() == device,
        "scheduled radial-angular tensors must share one device");
  }
  for (const auto& tensor : {
           edge_types,
           channel_radial_indices,
           channel_angular_indices,
           channel_types}) {
    TORCH_CHECK(
        tensor.device() == device,
        "scheduled radial-angular indices must share the data device");
  }
  const auto edge_count = radial_values.size(0);
  const auto radial_width = radial_values.size(1);
  const auto angular_width = angular_values.size(1);
  const auto channel_count = channel_scales.numel();
  TORCH_CHECK(
      radial_derivatives.sizes() == radial_values.sizes(),
      "radial derivatives must match radial values");
  TORCH_CHECK(
      angular_values.size(0) == edge_count,
      "radial and angular tables must share one edge axis");
  TORCH_CHECK(
      angular_derivatives.size(0) == edge_count &&
          angular_derivatives.size(1) == angular_width &&
          angular_derivatives.size(2) == 3,
      "angular derivatives must have shape [edge_count, angular_width, 3]");
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
  TORCH_CHECK(
      radial_width > 0 && angular_width > 0,
      "radial and angular tables must have positive width");
  c10::cuda::CUDAGuard device_guard(device);
  auto edge_values = torch::empty(
      {edge_count, channel_count},
      radial_values.options());
  auto edge_derivatives = torch::empty(
      {edge_count, channel_count, 3},
      radial_values.options());
  if (edge_count == 0 || channel_count == 0) {
    return std::make_tuple(edge_values, edge_derivatives);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  const std::int64_t work_items = edge_count * channel_count;
  AT_DISPATCH_FLOATING_TYPES(
      dtype,
      "ye3t_scheduled_radial_angular_channels_with_derivative_cuda",
      [&] {
        scheduled_radial_angular_channels_kernel<scalar_t>
            <<<launch_blocks(work_items), threads, 0, stream>>>(
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(edge_values, edge_derivatives);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
plain_site_basis_product_adjoint_cuda(
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
  check_cuda_data_tensor(radial_values, "radial_values");
  check_cuda_data_tensor(radial_derivatives, "radial_derivatives");
  check_cuda_data_tensor(angular_values, "angular_values");
  check_cuda_factor_tensor(angular_derivatives, "angular_derivatives");
  check_cuda_data_tensor(prefactors, "prefactors");
  check_cuda_data_tensor(
      prefactor_derivatives_center,
      "prefactor_derivatives_center");
  check_cuda_data_tensor(
      prefactor_derivatives_neighbor,
      "prefactor_derivatives_neighbor");
  check_cuda_data_tensor(radial_directions, "radial_directions");
  check_cuda_value_vector(edge_weights, radial_values, "edge_weights");
  check_cuda_data_tensor(
      edge_weight_derivatives,
      "edge_weight_derivatives");
  check_cuda_int64_vector(term_groups, "term_groups");
  check_cuda_int64_vector(term_channels, "term_channels");
  check_cuda_data_tensor(edge_adjoint, "edge_adjoint");
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
  TORCH_CHECK(
      term_groups.device() == device &&
          term_channels.device() == device,
      "source-adjoint term indices must be on the data device");
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
  c10::cuda::CUDAGuard device_guard(device);
  auto edge_position_adjoint = torch::empty(
      {edge_count, 3},
      radial_values.options());
  auto edge_charge_center = torch::empty(
      {edge_count},
      radial_values.options());
  auto edge_charge_neighbor = torch::empty_like(edge_charge_center);
  if (edge_count == 0) {
    return std::make_tuple(
        edge_position_adjoint,
        edge_charge_center,
        edge_charge_neighbor);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      dtype,
      "ye3t_plain_site_basis_product_adjoint_cuda",
      [&] {
        plain_site_basis_product_adjoint_kernel<scalar_t>
            <<<launch_blocks(edge_count), threads, 0, stream>>>(
                radial_values.data_ptr<scalar_t>(),
                radial_derivatives.data_ptr<scalar_t>(),
                angular_values.data_ptr<scalar_t>(),
                angular_derivatives.data_ptr<scalar_t>(),
                prefactors.data_ptr<scalar_t>(),
                prefactor_derivatives_center.data_ptr<scalar_t>(),
                prefactor_derivatives_neighbor.data_ptr<scalar_t>(),
                radial_directions.data_ptr<scalar_t>(),
                edge_weights.data_ptr<scalar_t>(),
                edge_weight_derivatives.data_ptr<scalar_t>(),
                term_groups.data_ptr<std::int64_t>(),
                term_channels.data_ptr<std::int64_t>(),
                edge_adjoint.data_ptr<scalar_t>(),
                group_count,
                edge_count,
                term_count,
                channel_count,
                edge_position_adjoint.data_ptr<scalar_t>(),
                edge_charge_center.data_ptr<scalar_t>(),
                edge_charge_neighbor.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      edge_position_adjoint,
      edge_charge_center,
      edge_charge_neighbor);
}

torch::Tensor compact_pair_product_cuda(
    const torch::Tensor& left,
    const torch::Tensor& right,
    bool antisymmetric) {
  check_cuda_data_tensor(left, "left");
  check_cuda_data_tensor(right, "right");
  TORCH_CHECK(
      left.device() == right.device(),
      "left and right must be on the same CUDA device");
  TORCH_CHECK(
      left.sizes() == right.sizes(),
      "left and right must have identical shapes");
  TORCH_CHECK(
      left.scalar_type() == right.scalar_type(),
      "left and right must have identical dtypes");
  c10::cuda::CUDAGuard device_guard(left.device());
  const std::int64_t dimension = left.size(1);
  const std::int64_t output_dimension = antisymmetric
      ? dimension * (dimension - 1) / 2
      : dimension * (dimension + 1) / 2;
  auto output = torch::empty(
      {left.size(0), output_dimension},
      left.options());
  const std::int64_t work = left.size(0) * output_dimension;
  if (work == 0) {
    return output;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      left.scalar_type(),
      "ye3t_compact_pair_product_cuda",
      [&] {
        compact_pair_product_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                left.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                left.size(0),
                dimension,
                antisymmetric,
                output_dimension,
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
compact_pair_product_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& left,
    const torch::Tensor& right,
    bool antisymmetric) {
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(left, "left");
  check_cuda_data_tensor(right, "right");
  TORCH_CHECK(
      output_adjoint.device() == left.device() &&
          left.device() == right.device(),
      "adjoint and inputs must be on the same CUDA device");
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
  c10::cuda::CUDAGuard device_guard(left.device());
  const std::int64_t dimension = left.size(1);
  const std::int64_t output_dimension = antisymmetric
      ? dimension * (dimension - 1) / 2
      : dimension * (dimension + 1) / 2;
  TORCH_CHECK(
      output_adjoint.size(1) == output_dimension,
      "output_adjoint width does not match compact product");
  auto left_adjoint = torch::empty_like(left);
  auto right_adjoint = torch::empty_like(right);
  const std::int64_t work = left.size(0) * dimension;
  if (work == 0) {
    return std::make_tuple(left_adjoint, right_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      left.scalar_type(),
      "ye3t_compact_pair_product_adjoint_cuda",
      [&] {
        compact_pair_adjoint_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                left.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                left.size(0),
                dimension,
                antisymmetric,
                output_dimension,
                left_adjoint.data_ptr<scalar_t>(),
                right_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(left_adjoint, right_adjoint);
}

torch::Tensor compact_exterior_power_cuda(
    const torch::Tensor& factors) {
  check_cuda_factor_tensor(factors, "factors");
  const std::int64_t order = factors.size(1);
  const std::int64_t dimension = factors.size(2);
  TORCH_CHECK(order >= 1, "compact exterior power order must be positive");
  TORCH_CHECK(
      order <= 8,
      "native compact exterior power currently supports order <= 8");
  c10::cuda::CUDAGuard device_guard(factors.device());
  const std::int64_t output_dimension =
      binomial_coefficient(dimension, order);
  auto output = torch::empty(
      {factors.size(0), output_dimension},
      factors.options());
  const std::int64_t work =
      factors.size(0) * output_dimension;
  if (work == 0) {
    return output;
  }
  double factorial = 1.0;
  for (std::int64_t value = 2; value <= order; ++value) {
    factorial *= static_cast<double>(value);
  }
  const double normalization = 1.0 / std::sqrt(factorial);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      factors.scalar_type(),
      "ye3t_compact_exterior_power_cuda",
      [&] {
        compact_exterior_power_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                factors.data_ptr<scalar_t>(),
                factors.size(0),
                order,
                dimension,
                output_dimension,
                scalar_t(normalization),
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

torch::Tensor compact_exterior_power_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& factors) {
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_factor_tensor(factors, "factors");
  TORCH_CHECK(
      output_adjoint.device() == factors.device(),
      "output_adjoint and factors must be on the same CUDA device");
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
  const std::int64_t output_dimension =
      binomial_coefficient(dimension, order);
  TORCH_CHECK(
      output_adjoint.size(1) == output_dimension,
      "output_adjoint width does not match compact exterior power");
  c10::cuda::CUDAGuard device_guard(factors.device());
  auto factors_adjoint = torch::zeros_like(factors);
  const std::int64_t work =
      factors.size(0) * output_dimension;
  if (work == 0) {
    return factors_adjoint;
  }
  double factorial = 1.0;
  for (std::int64_t value = 2; value <= order; ++value) {
    factorial *= static_cast<double>(value);
  }
  const double normalization = 1.0 / std::sqrt(factorial);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      factors.scalar_type(),
      "ye3t_compact_exterior_power_adjoint_cuda",
      [&] {
        compact_exterior_power_adjoint_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                factors.data_ptr<scalar_t>(),
                factors.size(0),
                order,
                dimension,
                output_dimension,
                scalar_t(normalization),
                factors_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return factors_adjoint;
}

torch::Tensor symmetric_power_monomial_cuda(
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& output_indices,
    const torch::Tensor& monomial_values) {
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(output_indices, "output_indices");
  check_cuda_value_vector(
      monomial_values,
      input,
      "monomial_values");
  TORCH_CHECK(
      monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          output_indices.device() == input.device(),
      "symmetric-power table tensors must use the input CUDA device");
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
  c10::cuda::CUDAGuard device_guard(input.device());
  auto output = torch::empty(
      {input.size(0), output_dimension},
      input.options());
  const std::int64_t work =
      input.size(0) * output_dimension;
  if (work == 0) {
    return output;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_monomial_cuda",
      [&] {
        symmetric_power_monomial_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                input.data_ptr<scalar_t>(),
                input.size(0),
                input.size(1),
                monomial_counts.data_ptr<std::int64_t>(),
                output_offsets.data_ptr<std::int64_t>(),
                monomial_values.data_ptr<scalar_t>(),
                term_count,
                output_dimension,
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

torch::Tensor symmetric_power_monomial_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& output_indices,
    const torch::Tensor& monomial_values) {
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(output_indices, "output_indices");
  check_cuda_value_vector(
      monomial_values,
      input,
      "monomial_values");
  TORCH_CHECK(
      output_adjoint.device() == input.device() &&
          monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          output_indices.device() == input.device(),
      "adjoint, input, and table tensors must use the same CUDA device");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type(),
      "output_adjoint and input must have identical dtypes");
  TORCH_CHECK(
      output_adjoint.size(0) == input.size(0),
      "output_adjoint batch dimension must match input");
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
      "output_offsets length must match output_adjoint");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto input_adjoint = torch::zeros_like(input);
  const std::int64_t work =
      input.size(0) * term_count;
  if (work == 0) {
    return input_adjoint;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_monomial_adjoint_cuda",
      [&] {
        symmetric_power_monomial_adjoint_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.size(0),
                input.size(1),
                monomial_counts.data_ptr<std::int64_t>(),
                output_indices.data_ptr<std::int64_t>(),
                monomial_values.data_ptr<scalar_t>(),
                term_count,
                output_adjoint.size(1),
                input_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return input_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
symmetric_power_monomial_double_backward_cuda(
    const torch::Tensor& input_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& output_indices,
    const torch::Tensor& monomial_values) {
  check_cuda_data_tensor(
      input_adjoint_tangent,
      "input_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(output_indices, "output_indices");
  check_cuda_value_vector(
      monomial_values,
      input,
      "monomial_values");
  TORCH_CHECK(
      input_adjoint_tangent.device() == input.device() &&
          output_adjoint.device() == input.device() &&
          monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          output_indices.device() == input.device(),
      "double-backward and table tensors must use one CUDA device");
  TORCH_CHECK(
      input_adjoint_tangent.scalar_type() == input.scalar_type() &&
          output_adjoint.scalar_type() == input.scalar_type(),
      "double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      input_adjoint_tangent.sizes() == input.sizes(),
      "input_adjoint_tangent shape must match input");
  TORCH_CHECK(
      output_adjoint.size(0) == input.size(0),
      "output_adjoint batch dimension must match input");
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
      "output_offsets length must match output_adjoint");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto input_tangent = torch::zeros_like(input);
  const std::int64_t work = input.size(0) * term_count;
  if (work == 0) {
    return std::make_tuple(output_tangent, input_tangent);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_monomial_double_backward_cuda",
      [&] {
        symmetric_power_monomial_double_backward_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                input_adjoint_tangent.data_ptr<scalar_t>(),
                output_adjoint.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.size(0),
                input.size(1),
                monomial_counts.data_ptr<std::int64_t>(),
                output_indices.data_ptr<std::int64_t>(),
                monomial_values.data_ptr<scalar_t>(),
                term_count,
                output_adjoint.size(1),
                output_tangent.data_ptr<scalar_t>(),
                input_tangent.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(output_tangent, input_tangent);
}

torch::Tensor symmetric_power_shared_monomial_cuda(
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "shared symmetric-power tables must use the input CUDA device");
  const std::int64_t monomial_count =
      monomial_counts.size(0);
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
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_values = torch::empty(
      {input.size(0), monomial_count},
      input.options());
  auto output = torch::empty(
      {input.size(0), output_dimension},
      input.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_cuda",
      [&] {
        const std::int64_t monomial_work =
            input.size(0) * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_values_kernel<scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  input.data_ptr<scalar_t>(),
                  input.size(0),
                  input.size(1),
                  monomial_counts.data_ptr<std::int64_t>(),
                  monomial_count,
                  monomial_values.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t output_work =
            input.size(0) * output_dimension;
        if (output_work > 0) {
          symmetric_power_shared_output_kernel<scalar_t>
              <<<launch_blocks(output_work), threads, 0, stream>>>(
                  monomial_values.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  output_offsets.data_ptr<std::int64_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  coefficient_count,
                  output_dimension,
                  output.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return output;
}

torch::Tensor symmetric_power_shared_monomial_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      output_adjoint.device() == input.device() &&
          monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "shared symmetric-power adjoint tensors must use one CUDA device");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type() &&
          output_adjoint.size(0) == input.size(0),
      "output_adjoint dtype and batch must match input");
  const std::int64_t monomial_count =
      monomial_counts.size(0);
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
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_adjoint = torch::zeros(
      {input.size(0), monomial_count},
      input.options());
  auto input_adjoint = torch::zeros_like(input);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_adjoint_cuda",
      [&] {
        const std::int64_t coefficient_work =
            input.size(0) * coefficient_count;
        if (coefficient_work > 0) {
          symmetric_power_shared_term_adjoint_kernel<scalar_t>
              <<<launch_blocks(coefficient_work), threads, 0, stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_outputs.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  coefficient_count,
                  output_adjoint.size(1),
                  monomial_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t monomial_work =
            input.size(0) * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_input_adjoint_kernel<scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  monomial_adjoint.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  input.size(0),
                  input.size(1),
                  monomial_counts.data_ptr<std::int64_t>(),
                  monomial_count,
                  input_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return input_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
symmetric_power_shared_monomial_double_backward_cuda(
    const torch::Tensor& input_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(
      input_adjoint_tangent,
      "input_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      input_adjoint_tangent.device() == input.device() &&
          output_adjoint.device() == input.device() &&
          monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "shared double-backward tensors must use one CUDA device");
  TORCH_CHECK(
      input_adjoint_tangent.scalar_type() == input.scalar_type() &&
          output_adjoint.scalar_type() == input.scalar_type(),
      "shared double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      input_adjoint_tangent.sizes() == input.sizes(),
      "input_adjoint_tangent shape must match input");
  TORCH_CHECK(
      output_adjoint.size(0) == input.size(0),
      "output_adjoint batch dimension must match input");
  const std::int64_t monomial_count = monomial_counts.size(0);
  const std::int64_t coefficient_count = coefficient_terms.numel();
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
  c10::cuda::CUDAGuard device_guard(input.device());
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto input_tangent = torch::zeros_like(input);
  const std::int64_t work = input.size(0) * coefficient_count;
  if (work == 0) {
    return std::make_tuple(output_tangent, input_tangent);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_double_backward_cuda",
      [&] {
        symmetric_power_shared_monomial_double_backward_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                input_adjoint_tangent.data_ptr<scalar_t>(),
                output_adjoint.data_ptr<scalar_t>(),
                input.data_ptr<scalar_t>(),
                input.size(0),
                input.size(1),
                monomial_counts.data_ptr<std::int64_t>(),
                coefficient_terms.data_ptr<std::int64_t>(),
                coefficient_outputs.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                monomial_count,
                coefficient_count,
                output_adjoint.size(1),
                output_tangent.data_ptr<scalar_t>(),
                input_tangent.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(output_tangent, input_tangent);
}

std::tuple<torch::Tensor, torch::Tensor>
symmetric_power_shared_monomial_factored_double_backward_cuda(
    const torch::Tensor& input_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(
      input_adjoint_tangent,
      "input_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      input_adjoint_tangent.device() == input.device() &&
          output_adjoint.device() == input.device() &&
          monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "factored shared double-backward tensors must use one CUDA device");
  TORCH_CHECK(
      input_adjoint_tangent.scalar_type() == input.scalar_type() &&
          output_adjoint.scalar_type() == input.scalar_type(),
      "factored shared double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      input_adjoint_tangent.sizes() == input.sizes(),
      "input_adjoint_tangent shape must match input");
  TORCH_CHECK(
      output_adjoint.size(0) == input.size(0),
      "output_adjoint batch dimension must match input");
  const std::int64_t monomial_count = monomial_counts.size(0);
  const std::int64_t coefficient_count = coefficient_terms.numel();
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
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_workspace = torch::zeros(
      {input.size(0), monomial_count},
      input.options());
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto input_tangent = torch::zeros_like(input);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_factored_double_backward_cuda",
      [&] {
        const std::int64_t coefficient_work =
            input.size(0) * coefficient_count;
        if (coefficient_work > 0) {
          symmetric_power_shared_term_adjoint_kernel<scalar_t>
              <<<launch_blocks(coefficient_work), threads, 0, stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_outputs.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  coefficient_count,
                  output_adjoint.size(1),
                  monomial_workspace.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t monomial_work =
            input.size(0) * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_monomial_factored_double_backward_kernel<
              scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  input_adjoint_tangent.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  monomial_counts.data_ptr<std::int64_t>(),
                  input.size(0),
                  input.size(1),
                  monomial_count,
                  monomial_workspace.data_ptr<scalar_t>(),
                  input_tangent.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t output_work =
            input.size(0) * output_adjoint.size(1);
        if (output_work > 0) {
          symmetric_power_shared_output_kernel<scalar_t>
              <<<launch_blocks(output_work), threads, 0, stream>>>(
                  monomial_workspace.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  output_offsets.data_ptr<std::int64_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  coefficient_count,
                  output_adjoint.size(1),
                  output_tangent.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return std::make_tuple(output_tangent, input_tangent);
}

torch::Tensor symmetric_power_shared_sparse_monomial_cuda(
    const torch::Tensor& input,
    const torch::Tensor& term_offsets,
    const torch::Tensor& term_components,
    const torch::Tensor& term_exponents,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_vector(term_offsets, "term_offsets");
  check_cuda_int64_vector(term_components, "term_components");
  check_cuda_int64_vector(term_exponents, "term_exponents");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      term_offsets.device() == input.device() &&
          term_components.device() == input.device() &&
          term_exponents.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "sparse shared-power tables must use the input CUDA device");
  TORCH_CHECK(
      term_offsets.numel() >= 1,
      "term_offsets must contain at least one entry");
  TORCH_CHECK(
      term_components.numel() == term_exponents.numel(),
      "sparse support component and exponent arrays must have equal length");
  const std::int64_t monomial_count = term_offsets.numel() - 1;
  const std::int64_t support_count = term_components.numel();
  const std::int64_t coefficient_count = coefficient_terms.numel();
  TORCH_CHECK(
      coefficient_outputs.numel() == coefficient_count &&
          coefficient_values.numel() == coefficient_count,
      "shared symmetric-power coefficient arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() >= 1,
      "output_offsets must contain at least one entry");
  const std::int64_t output_dimension = output_offsets.numel() - 1;
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_values = torch::empty(
      {input.size(0), monomial_count},
      input.options());
  auto output = torch::empty(
      {input.size(0), output_dimension},
      input.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_sparse_monomial_cuda",
      [&] {
        const std::int64_t monomial_work =
            input.size(0) * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_sparse_values_kernel<scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  input.data_ptr<scalar_t>(),
                  input.size(0),
                  input.size(1),
                  term_offsets.data_ptr<std::int64_t>(),
                  term_components.data_ptr<std::int64_t>(),
                  term_exponents.data_ptr<std::int64_t>(),
                  monomial_count,
                  support_count,
                  monomial_values.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t output_work =
            input.size(0) * output_dimension;
        if (output_work > 0) {
          symmetric_power_shared_output_kernel<scalar_t>
              <<<launch_blocks(output_work), threads, 0, stream>>>(
                  monomial_values.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  output_offsets.data_ptr<std::int64_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  coefficient_count,
                  output_dimension,
                  output.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return output;
}

torch::Tensor symmetric_power_shared_sparse_monomial_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& term_offsets,
    const torch::Tensor& term_components,
    const torch::Tensor& term_exponents,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_vector(term_offsets, "term_offsets");
  check_cuda_int64_vector(term_components, "term_components");
  check_cuda_int64_vector(term_exponents, "term_exponents");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      output_adjoint.device() == input.device() &&
          term_offsets.device() == input.device() &&
          term_components.device() == input.device() &&
          term_exponents.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "sparse shared-power adjoint tensors must use one CUDA device");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type() &&
          output_adjoint.size(0) == input.size(0),
      "output_adjoint dtype and batch must match input");
  TORCH_CHECK(
      term_offsets.numel() >= 1,
      "term_offsets must contain at least one entry");
  TORCH_CHECK(
      term_components.numel() == term_exponents.numel(),
      "sparse support component and exponent arrays must have equal length");
  const std::int64_t monomial_count = term_offsets.numel() - 1;
  const std::int64_t support_count = term_components.numel();
  const std::int64_t coefficient_count = coefficient_terms.numel();
  TORCH_CHECK(
      coefficient_outputs.numel() == coefficient_count &&
          coefficient_values.numel() == coefficient_count,
      "shared symmetric-power coefficient arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() == output_adjoint.size(1) + 1,
      "output_offsets length must match output_adjoint");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_adjoint = torch::zeros(
      {input.size(0), monomial_count},
      input.options());
  auto input_adjoint = torch::zeros_like(input);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_sparse_monomial_adjoint_cuda",
      [&] {
        const std::int64_t coefficient_work =
            input.size(0) * coefficient_count;
        if (coefficient_work > 0) {
          symmetric_power_shared_term_adjoint_kernel<scalar_t>
              <<<launch_blocks(coefficient_work), threads, 0, stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_outputs.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  coefficient_count,
                  output_adjoint.size(1),
                  monomial_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t monomial_work =
            input.size(0) * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_sparse_input_adjoint_kernel<scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  monomial_adjoint.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  input.size(0),
                  input.size(1),
                  term_offsets.data_ptr<std::int64_t>(),
                  term_components.data_ptr<std::int64_t>(),
                  term_exponents.data_ptr<std::int64_t>(),
                  monomial_count,
                  support_count,
                  input_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return input_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
symmetric_power_shared_sparse_monomial_double_backward_cuda(
    const torch::Tensor& input_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& term_offsets,
    const torch::Tensor& term_components,
    const torch::Tensor& term_exponents,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_data_tensor(
      input_adjoint_tangent,
      "input_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_vector(term_offsets, "term_offsets");
  check_cuda_int64_vector(term_components, "term_components");
  check_cuda_int64_vector(term_exponents, "term_exponents");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      input_adjoint_tangent.device() == input.device() &&
          output_adjoint.device() == input.device() &&
          term_offsets.device() == input.device() &&
          term_components.device() == input.device() &&
          term_exponents.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "sparse shared double-backward tensors must use one CUDA device");
  TORCH_CHECK(
      input_adjoint_tangent.scalar_type() == input.scalar_type() &&
          output_adjoint.scalar_type() == input.scalar_type(),
      "sparse shared double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      input_adjoint_tangent.sizes() == input.sizes(),
      "input_adjoint_tangent shape must match input");
  TORCH_CHECK(
      output_adjoint.size(0) == input.size(0),
      "output_adjoint batch dimension must match input");
  TORCH_CHECK(
      term_offsets.numel() >= 1,
      "term_offsets must contain at least one entry");
  TORCH_CHECK(
      term_components.numel() == term_exponents.numel(),
      "sparse support component and exponent arrays must have equal length");
  const std::int64_t monomial_count = term_offsets.numel() - 1;
  const std::int64_t support_count = term_components.numel();
  const std::int64_t coefficient_count = coefficient_terms.numel();
  TORCH_CHECK(
      coefficient_outputs.numel() == coefficient_count &&
          coefficient_values.numel() == coefficient_count,
      "shared symmetric-power coefficient arrays must have equal length");
  TORCH_CHECK(
      output_offsets.numel() == output_adjoint.size(1) + 1,
      "output_offsets length must match output_adjoint");
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_workspace = torch::zeros(
      {input.size(0), monomial_count},
      input.options());
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto input_tangent = torch::zeros_like(input);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_sparse_monomial_double_backward_cuda",
      [&] {
        const std::int64_t coefficient_work =
            input.size(0) * coefficient_count;
        if (coefficient_work > 0) {
          symmetric_power_shared_term_adjoint_kernel<scalar_t>
              <<<launch_blocks(coefficient_work), threads, 0, stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_outputs.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  coefficient_count,
                  output_adjoint.size(1),
                  monomial_workspace.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t monomial_work =
            input.size(0) * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_sparse_factored_double_backward_kernel<
              scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  input_adjoint_tangent.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  term_offsets.data_ptr<std::int64_t>(),
                  term_components.data_ptr<std::int64_t>(),
                  term_exponents.data_ptr<std::int64_t>(),
                  input.size(0),
                  input.size(1),
                  monomial_count,
                  support_count,
                  monomial_workspace.data_ptr<scalar_t>(),
                  input_tangent.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t output_work =
            input.size(0) * output_adjoint.size(1);
        if (output_work > 0) {
          symmetric_power_shared_output_kernel<scalar_t>
              <<<launch_blocks(output_work), threads, 0, stream>>>(
                  monomial_workspace.data_ptr<scalar_t>(),
                  input.size(0),
                  monomial_count,
                  output_offsets.data_ptr<std::int64_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  coefficient_count,
                  output_adjoint.size(1),
                  output_tangent.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return std::make_tuple(output_tangent, input_tangent);
}

torch::Tensor symmetric_power_shared_monomial_batched_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& input,
    const torch::Tensor& monomial_counts,
    const torch::Tensor& output_offsets,
    const torch::Tensor& coefficient_terms,
    const torch::Tensor& coefficient_outputs,
    const torch::Tensor& coefficient_values) {
  check_cuda_factor_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(input, "input");
  check_cuda_int64_matrix(monomial_counts, "monomial_counts");
  check_cuda_int64_vector(output_offsets, "output_offsets");
  check_cuda_int64_vector(coefficient_terms, "coefficient_terms");
  check_cuda_int64_vector(coefficient_outputs, "coefficient_outputs");
  check_cuda_value_vector(
      coefficient_values,
      input,
      "coefficient_values");
  TORCH_CHECK(
      output_adjoint.device() == input.device() &&
          monomial_counts.device() == input.device() &&
          output_offsets.device() == input.device() &&
          coefficient_terms.device() == input.device() &&
          coefficient_outputs.device() == input.device(),
      "shared batched adjoint tensors must use one CUDA device");
  TORCH_CHECK(
      output_adjoint.scalar_type() == input.scalar_type() &&
          output_adjoint.size(1) == input.size(0),
      "output_adjoint dtype and batch must match input");
  const std::int64_t seed_count = output_adjoint.size(0);
  const std::int64_t batch_size = input.size(0);
  const std::int64_t monomial_count =
      monomial_counts.size(0);
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
  c10::cuda::CUDAGuard device_guard(input.device());
  auto monomial_adjoint = torch::zeros(
      {seed_count, batch_size, monomial_count},
      input.options());
  auto input_adjoint = torch::zeros(
      {seed_count, batch_size, input.size(1)},
      input.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(),
      "ye3t_symmetric_power_shared_monomial_batched_adjoint_cuda",
      [&] {
        const std::int64_t coefficient_work =
            seed_count * batch_size * coefficient_count;
        if (coefficient_work > 0) {
          symmetric_power_shared_batched_term_adjoint_kernel<scalar_t>
              <<<launch_blocks(coefficient_work), threads, 0, stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  coefficient_terms.data_ptr<std::int64_t>(),
                  coefficient_outputs.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  seed_count,
                  batch_size,
                  monomial_count,
                  coefficient_count,
                  output_dimension,
                  monomial_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t monomial_work =
            batch_size * monomial_count;
        if (monomial_work > 0) {
          symmetric_power_shared_batched_input_adjoint_kernel<scalar_t>
              <<<launch_blocks(monomial_work), threads, 0, stream>>>(
                  monomial_adjoint.data_ptr<scalar_t>(),
                  input.data_ptr<scalar_t>(),
                  seed_count,
                  batch_size,
                  input.size(1),
                  monomial_counts.data_ptr<std::int64_t>(),
                  monomial_count,
                  input_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return input_adjoint;
}

bool use_factorized_angular_warp_kernel(
    std::int64_t sample_count,
    std::int64_t workspace_dimension,
    std::int64_t node_count,
    std::int64_t coefficient_count) {
  const char* requested_policy =
      std::getenv("YE3T_FACTORIZED_ANGULAR_CUDA_POLICY");
  const std::string policy =
      requested_policy == nullptr ? "auto" : requested_policy;
  TORCH_CHECK(
      policy == "auto" || policy == "serial" || policy == "warp",
      "YE3T_FACTORIZED_ANGULAR_CUDA_POLICY must be auto, serial, or warp");
  if (policy == "serial") {
    return false;
  }
  if (policy == "warp") {
    return sample_count > 0;
  }
  (void)workspace_dimension;
  (void)node_count;
  (void)coefficient_count;
  return false;
}

std::int64_t factorized_angular_segmented_execution_policy(
    std::int64_t sample_count,
    std::int64_t segment_count,
    std::int64_t coefficient_count) {
  const char* requested_policy =
      std::getenv("YE3T_FACTORIZED_ANGULAR_SEGMENTED_CUDA_POLICY");
  const std::string policy =
      requested_policy == nullptr ? "auto" : requested_policy;
  TORCH_CHECK(
      policy == "auto" || policy == "serial" || policy == "warp",
      "YE3T_FACTORIZED_ANGULAR_SEGMENTED_CUDA_POLICY must be auto, serial, or warp");
  if (policy == "serial") {
    return kFactorizedSegmentExecutionSerial;
  }
  if (policy == "warp") {
    return sample_count > 0
        ? kFactorizedSegmentExecutionWarp
        : kFactorizedSegmentExecutionSerial;
  }
  if (sample_count <= 0 ||
      coefficient_count < kFactorizedSegmentWarpMinCoefficientCount) {
    return kFactorizedSegmentExecutionSerial;
  }
  return segment_count == 1
      ? kFactorizedSegmentExecutionWarp
      : kFactorizedSegmentExecutionHybrid;
}

int factorized_angular_warp_blocks(std::int64_t sample_count) {
  return static_cast<int>(
      std::min<std::int64_t>(sample_count, 65535));
}

torch::Tensor factorized_angular_cuda(
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
  check_cuda_data_tensor(packed_slots, "packed_slots");
  check_cuda_int64_vector(node_offsets, "node_offsets");
  check_cuda_int64_vector(node_dimensions, "node_dimensions");
  check_cuda_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_cuda_int64_vector(node_left, "node_left");
  check_cuda_int64_vector(node_right, "node_right");
  check_cuda_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_cuda_int64_vector(coefficient_rows, "coefficient_rows");
  check_cuda_int64_vector(coefficient_columns, "coefficient_columns");
  check_cuda_int64_vector(root_nodes, "root_nodes");
  check_cuda_value_vector(
      coefficient_values,
      packed_slots,
      "coefficient_values");
  check_cuda_value_vector(
      projection_values,
      packed_slots,
      "projection_values");
  TORCH_CHECK(
      node_offsets.device() == packed_slots.device() &&
          node_dimensions.device() == packed_slots.device() &&
          node_leaf_offsets.device() == packed_slots.device() &&
          node_left.device() == packed_slots.device() &&
          node_right.device() == packed_slots.device() &&
          node_coefficient_offsets.device() == packed_slots.device() &&
          coefficient_rows.device() == packed_slots.device() &&
          coefficient_columns.device() == packed_slots.device() &&
          root_nodes.device() == packed_slots.device(),
      "factorized angular tensors must use one CUDA device");
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
  TORCH_CHECK(output_dimension > 0, "output_dimension must be positive");
  const std::int64_t input_dimension =
      packed_slots.size(1) / source_dimension;
  const std::int64_t projection_dimension =
      projection_values.numel() / source_dimension;
  TORCH_CHECK(
      workspace_dimension > 0 &&
          workspace_dimension <= node_count * input_dimension,
      "factorized workspace_dimension is outside its certified bound");
  const std::int64_t root_block_count =
      root_nodes.numel() * projection_dimension;
  TORCH_CHECK(
      output_dimension % root_block_count == 0,
      "output_dimension does not contain complete factorized root blocks");
  const std::int64_t root_dimension =
      output_dimension / root_block_count;
  TORCH_CHECK(
      output_dimension ==
          root_nodes.numel() * projection_dimension * root_dimension,
      "output_dimension does not match factorized roots and projection");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {
          packed_slots.size(0) * source_dimension,
          workspace_dimension,
      },
      packed_slots.options());
  auto output = torch::zeros(
      {packed_slots.size(0), output_dimension},
      packed_slots.options());
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * source_dimension;
        if (use_factorized_angular_warp_kernel(
                sample_count,
                workspace_dimension,
                node_count,
                coefficient_values.numel())) {
          factorized_angular_forward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  packed_slots.data_ptr<scalar_t>(),
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
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  nullptr,
                  nullptr,
                  nullptr,
                  projection_values.data_ptr<scalar_t>(),
                  projection_dimension,
                  workspace_dimension,
                  root_dimension,
                  output_dimension,
                  workspace.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>());
        } else {
          factorized_angular_forward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_slots.data_ptr<scalar_t>(),
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
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                nullptr,
                nullptr,
                nullptr,
                projection_values.data_ptr<scalar_t>(),
                projection_dimension,
                workspace_dimension,
                root_dimension,
                output_dimension,
                workspace.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

torch::Tensor factorized_angular_adjoint_cuda(
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
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(packed_slots, "packed_slots");
  check_cuda_int64_vector(node_offsets, "node_offsets");
  check_cuda_int64_vector(node_dimensions, "node_dimensions");
  check_cuda_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_cuda_int64_vector(node_left, "node_left");
  check_cuda_int64_vector(node_right, "node_right");
  check_cuda_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_cuda_int64_vector(coefficient_rows, "coefficient_rows");
  check_cuda_int64_vector(coefficient_columns, "coefficient_columns");
  check_cuda_int64_vector(root_nodes, "root_nodes");
  check_cuda_value_vector(
      coefficient_values,
      packed_slots,
      "coefficient_values");
  check_cuda_value_vector(
      projection_values,
      packed_slots,
      "projection_values");
  TORCH_CHECK(
      output_adjoint.device() == packed_slots.device() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint device, dtype, and batch must match packed_slots");
  TORCH_CHECK(
      node_offsets.device() == packed_slots.device() &&
          node_dimensions.device() == packed_slots.device() &&
          node_leaf_offsets.device() == packed_slots.device() &&
          node_left.device() == packed_slots.device() &&
          node_right.device() == packed_slots.device() &&
          node_coefficient_offsets.device() == packed_slots.device() &&
          coefficient_rows.device() == packed_slots.device() &&
          coefficient_columns.device() == packed_slots.device() &&
          root_nodes.device() == packed_slots.device(),
      "factorized angular tensors must use one CUDA device");
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
  TORCH_CHECK(
      workspace_dimension > 0 &&
          workspace_dimension <= node_count * input_dimension,
      "factorized workspace_dimension is outside its certified bound");
  const std::int64_t output_dimension =
      output_adjoint.size(1);
  const std::int64_t root_block_count =
      root_nodes.numel() * projection_dimension;
  TORCH_CHECK(
      output_dimension % root_block_count == 0,
      "output_adjoint does not contain complete factorized root blocks");
  const std::int64_t root_dimension =
      output_dimension / root_block_count;
  TORCH_CHECK(
      output_adjoint.size(1) == output_dimension,
      "output_adjoint width does not match factorized output");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {
          packed_slots.size(0) * source_dimension,
          workspace_dimension,
      },
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  auto packed_adjoint = torch::empty_like(packed_slots);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_adjoint_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * source_dimension;
        if (use_factorized_angular_warp_kernel(
                sample_count,
                workspace_dimension,
                node_count,
                coefficient_values.numel())) {
          factorized_angular_adjoint_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_slots.data_ptr<scalar_t>(),
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
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  nullptr,
                  nullptr,
                  nullptr,
                  projection_values.data_ptr<scalar_t>(),
                  projection_dimension,
                  workspace_dimension,
                  root_dimension,
                  output_dimension,
                  workspace.data_ptr<scalar_t>(),
                  workspace_adjoint.data_ptr<scalar_t>(),
                  packed_adjoint.data_ptr<scalar_t>());
        } else {
          factorized_angular_adjoint_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
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
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                nullptr,
                nullptr,
                nullptr,
                projection_values.data_ptr<scalar_t>(),
                projection_dimension,
                workspace_dimension,
                root_dimension,
                output_dimension,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                packed_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return packed_adjoint;
}

struct FactorizedCudaDimensions {
  std::int64_t node_count;
  std::int64_t input_dimension;
  std::int64_t projection_dimension;
  std::int64_t workspace_dimension;
  std::int64_t root_dimension;
  std::int64_t output_dimension;
};

FactorizedCudaDimensions factorized_cuda_dimensions(
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
  check_cuda_data_tensor(packed_slots, "packed_slots");
  check_cuda_int64_vector(node_offsets, "node_offsets");
  check_cuda_int64_vector(node_dimensions, "node_dimensions");
  check_cuda_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_cuda_int64_vector(node_left, "node_left");
  check_cuda_int64_vector(node_right, "node_right");
  check_cuda_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_cuda_int64_vector(coefficient_rows, "coefficient_rows");
  check_cuda_int64_vector(coefficient_columns, "coefficient_columns");
  check_cuda_int64_vector(root_nodes, "root_nodes");
  check_cuda_value_vector(
      coefficient_values,
      packed_slots,
      "coefficient_values");
  check_cuda_value_vector(
      projection_values,
      packed_slots,
      "projection_values");
  TORCH_CHECK(
      node_offsets.device() == packed_slots.device() &&
          node_dimensions.device() == packed_slots.device() &&
          node_leaf_offsets.device() == packed_slots.device() &&
          node_left.device() == packed_slots.device() &&
          node_right.device() == packed_slots.device() &&
          node_coefficient_offsets.device() == packed_slots.device() &&
          coefficient_rows.device() == packed_slots.device() &&
          coefficient_columns.device() == packed_slots.device() &&
          root_nodes.device() == packed_slots.device(),
      "factorized angular tensors must use one CUDA device");
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
  TORCH_CHECK(
      workspace_dimension > 0 &&
          workspace_dimension <= node_count * input_dimension,
      "factorized workspace_dimension is outside its certified bound");
  TORCH_CHECK(
      output_dimension > 0,
      "factorized output_dimension must be positive");
  const std::int64_t root_block_count =
      root_nodes.numel() * projection_dimension;
  TORCH_CHECK(
      output_dimension % root_block_count == 0,
      "factorized output does not contain complete root blocks");
  const std::int64_t root_dimension =
      output_dimension / root_block_count;
  return {
      node_count,
      input_dimension,
      projection_dimension,
      workspace_dimension,
      root_dimension,
      output_dimension,
  };
}

struct HeterogeneousFactorizedCudaDimensions {
  std::int64_t node_count;
  std::int64_t input_dimension;
  std::int64_t total_projection_dimension;
  std::int64_t workspace_dimension;
  std::int64_t output_dimension;
};

HeterogeneousFactorizedCudaDimensions
heterogeneous_factorized_cuda_dimensions(
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
  check_cuda_data_tensor(packed_slots, "packed_slots");
  check_cuda_int64_vector(node_offsets, "node_offsets");
  check_cuda_int64_vector(node_dimensions, "node_dimensions");
  check_cuda_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_cuda_int64_vector(node_left, "node_left");
  check_cuda_int64_vector(node_right, "node_right");
  check_cuda_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_cuda_int64_vector(coefficient_rows, "coefficient_rows");
  check_cuda_int64_vector(coefficient_columns, "coefficient_columns");
  check_cuda_int64_vector(root_nodes, "root_nodes");
  check_cuda_int64_vector(
      root_projection_starts,
      "root_projection_starts");
  check_cuda_int64_vector(
      root_projection_dimensions,
      "root_projection_dimensions");
  check_cuda_int64_vector(root_output_offsets, "root_output_offsets");
  check_cuda_value_vector(
      coefficient_values,
      packed_slots,
      "coefficient_values");
  check_cuda_value_vector(
      projection_values,
      packed_slots,
      "projection_values");
  TORCH_CHECK(
      node_offsets.device() == packed_slots.device() &&
          node_dimensions.device() == packed_slots.device() &&
          node_leaf_offsets.device() == packed_slots.device() &&
          node_left.device() == packed_slots.device() &&
          node_right.device() == packed_slots.device() &&
          node_coefficient_offsets.device() == packed_slots.device() &&
          coefficient_rows.device() == packed_slots.device() &&
          coefficient_columns.device() == packed_slots.device() &&
          root_nodes.device() == packed_slots.device() &&
          root_projection_starts.device() == packed_slots.device() &&
          root_projection_dimensions.device() == packed_slots.device() &&
          root_output_offsets.device() == packed_slots.device(),
      "heterogeneous factorized tensors must use one CUDA device");
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
  const std::int64_t root_count = root_nodes.numel();
  TORCH_CHECK(root_count > 0, "factorized plan requires roots");
  TORCH_CHECK(
      root_projection_starts.numel() == root_count &&
          root_projection_dimensions.numel() == root_count &&
          root_output_offsets.numel() == root_count,
      "heterogeneous factorized root metadata must match root count");
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
  const std::int64_t total_projection_dimension =
      projection_values.numel() / source_dimension;
  TORCH_CHECK(
      workspace_dimension > 0 &&
          workspace_dimension <= node_count * input_dimension,
      "factorized workspace_dimension is outside its certified bound");
  TORCH_CHECK(
      output_dimension > 0,
      "heterogeneous factorized output_dimension must be positive");
  return {
      node_count,
      input_dimension,
      total_projection_dimension,
      workspace_dimension,
      output_dimension,
  };
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_double_backward_cuda(
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
  check_cuda_data_tensor(
      packed_adjoint_tangent,
      "packed_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  const FactorizedCudaDimensions dimensions =
      factorized_cuda_dimensions(
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
          projection_values,
          source_dimension,
          workspace_dimension,
          output_adjoint.size(1));
  TORCH_CHECK(
      packed_adjoint_tangent.device() == packed_slots.device() &&
          output_adjoint.device() == packed_slots.device(),
      "factorized double-backward tensors must use one CUDA device");
  TORCH_CHECK(
      packed_adjoint_tangent.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "factorized double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      packed_adjoint_tangent.sizes() == packed_slots.sizes(),
      "packed_adjoint_tangent shape must match packed_slots");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0) &&
          output_adjoint.size(1) == dimensions.output_dimension,
      "output_adjoint shape does not match factorized output");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {
          packed_slots.size(0) * source_dimension,
          dimensions.workspace_dimension,
      },
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  const std::int64_t sample_count =
      packed_slots.size(0) * source_dimension;
  const bool use_warp = use_factorized_angular_warp_kernel(
      sample_count,
      dimensions.workspace_dimension,
      dimensions.node_count,
      coefficient_values.numel()) &&
      2 * dimensions.workspace_dimension * packed_slots.element_size() <=
          32 * 1024;
  auto adjoint_tangent = use_warp
      ? torch::empty({0}, workspace.options())
      : torch::empty_like(workspace);
  auto primal_tangent = use_warp
      ? torch::empty({0}, workspace.options())
      : torch::empty_like(workspace);
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_slots);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_double_backward_cuda",
      [&] {
        if (use_warp) {
          factorized_angular_double_backward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 2 * dimensions.workspace_dimension * sizeof(scalar_t),
                 stream>>>(
                  packed_adjoint_tangent.data_ptr<scalar_t>(),
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  dimensions.input_dimension,
                  source_dimension,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  dimensions.node_count,
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  nullptr,
                  nullptr,
                  nullptr,
                  projection_values.data_ptr<scalar_t>(),
                  dimensions.projection_dimension,
                  dimensions.workspace_dimension,
                  dimensions.root_dimension,
                  dimensions.output_dimension,
                  false,
                  workspace.data_ptr<scalar_t>(),
                  workspace_adjoint.data_ptr<scalar_t>(),
                  adjoint_tangent.data_ptr<scalar_t>(),
                  primal_tangent.data_ptr<scalar_t>(),
                  output_tangent.data_ptr<scalar_t>(),
                  packed_tangent.data_ptr<scalar_t>());
        } else {
          factorized_angular_double_backward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_adjoint_tangent.data_ptr<scalar_t>(),
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                nullptr,
                nullptr,
                nullptr,
                projection_values.data_ptr<scalar_t>(),
                dimensions.projection_dimension,
                dimensions.workspace_dimension,
                dimensions.root_dimension,
                dimensions.output_dimension,
                false,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                adjoint_tangent.data_ptr<scalar_t>(),
                primal_tangent.data_ptr<scalar_t>(),
                output_tangent.data_ptr<scalar_t>(),
                packed_tangent.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

torch::Tensor factorized_angular_heterogeneous_cuda(
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
  const auto dimensions = heterogeneous_factorized_cuda_dimensions(
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
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {
          packed_slots.size(0) * source_dimension,
          dimensions.workspace_dimension,
      },
      packed_slots.options());
  auto output = torch::zeros(
      {packed_slots.size(0), dimensions.output_dimension},
      packed_slots.options());
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * source_dimension;
        if (use_factorized_angular_warp_kernel(
                sample_count,
                dimensions.workspace_dimension,
                dimensions.node_count,
                coefficient_values.numel())) {
          factorized_angular_forward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  dimensions.input_dimension,
                  source_dimension,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  dimensions.node_count,
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  dimensions.total_projection_dimension,
                  dimensions.workspace_dimension,
                  0,
                  dimensions.output_dimension,
                  workspace.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>());
        } else {
          factorized_angular_forward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                dimensions.total_projection_dimension,
                dimensions.workspace_dimension,
                0,
                dimensions.output_dimension,
                workspace.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
factorized_angular_heterogeneous_adjoint_with_workspace_cuda(
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
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  const auto dimensions = heterogeneous_factorized_cuda_dimensions(
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
  TORCH_CHECK(
      output_adjoint.device() == packed_slots.device() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint device, dtype, and batch must match packed_slots");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {
          packed_slots.size(0) * source_dimension,
          dimensions.workspace_dimension,
      },
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  auto packed_adjoint = torch::empty_like(packed_slots);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_adjoint_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * source_dimension;
        if (use_factorized_angular_warp_kernel(
                sample_count,
                dimensions.workspace_dimension,
                dimensions.node_count,
                coefficient_values.numel())) {
          factorized_angular_adjoint_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  dimensions.input_dimension,
                  source_dimension,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  dimensions.node_count,
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  dimensions.total_projection_dimension,
                  dimensions.workspace_dimension,
                  0,
                  dimensions.output_dimension,
                  workspace.data_ptr<scalar_t>(),
                  workspace_adjoint.data_ptr<scalar_t>(),
                  packed_adjoint.data_ptr<scalar_t>());
        } else {
          factorized_angular_adjoint_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                dimensions.total_projection_dimension,
                dimensions.workspace_dimension,
                0,
                dimensions.output_dimension,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                packed_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      packed_adjoint,
      workspace,
      workspace_adjoint);
}

torch::Tensor factorized_angular_heterogeneous_adjoint_cuda(
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
  return std::get<0>(
      factorized_angular_heterogeneous_adjoint_with_workspace_cuda(
          output_adjoint,
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
          workspace_dimension));
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_heterogeneous_double_backward_cuda(
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
  check_cuda_data_tensor(
      packed_adjoint_tangent,
      "packed_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  const auto dimensions = heterogeneous_factorized_cuda_dimensions(
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
  TORCH_CHECK(
      packed_adjoint_tangent.device() == packed_slots.device() &&
          output_adjoint.device() == packed_slots.device(),
      "factorized double-backward tensors must use one CUDA device");
  TORCH_CHECK(
      packed_adjoint_tangent.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type(),
      "factorized double-backward tensors must have identical dtypes");
  TORCH_CHECK(
      packed_adjoint_tangent.sizes() == packed_slots.sizes(),
      "packed_adjoint_tangent shape must match packed_slots");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_slots.size(0),
      "output_adjoint batch must match packed_slots");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {
          packed_slots.size(0) * source_dimension,
          dimensions.workspace_dimension,
      },
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  const std::int64_t sample_count =
      packed_slots.size(0) * source_dimension;
  const bool use_warp = use_factorized_angular_warp_kernel(
      sample_count,
      dimensions.workspace_dimension,
      dimensions.node_count,
      coefficient_values.numel()) &&
      2 * dimensions.workspace_dimension * packed_slots.element_size() <=
          32 * 1024;
  auto adjoint_tangent = use_warp
      ? torch::empty({0}, workspace.options())
      : torch::empty_like(workspace);
  auto primal_tangent = use_warp
      ? torch::empty({0}, workspace.options())
      : torch::empty_like(workspace);
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_slots);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_double_backward_cuda",
      [&] {
        if (use_warp) {
          factorized_angular_double_backward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 2 * dimensions.workspace_dimension * sizeof(scalar_t),
                 stream>>>(
                  packed_adjoint_tangent.data_ptr<scalar_t>(),
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  dimensions.input_dimension,
                  source_dimension,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  dimensions.node_count,
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  dimensions.total_projection_dimension,
                  dimensions.workspace_dimension,
                  0,
                  dimensions.output_dimension,
                  false,
                  workspace.data_ptr<scalar_t>(),
                  workspace_adjoint.data_ptr<scalar_t>(),
                  adjoint_tangent.data_ptr<scalar_t>(),
                  primal_tangent.data_ptr<scalar_t>(),
                  output_tangent.data_ptr<scalar_t>(),
                  packed_tangent.data_ptr<scalar_t>());
        } else {
          factorized_angular_double_backward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_adjoint_tangent.data_ptr<scalar_t>(),
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                dimensions.total_projection_dimension,
                dimensions.workspace_dimension,
                0,
                dimensions.output_dimension,
                false,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                adjoint_tangent.data_ptr<scalar_t>(),
                primal_tangent.data_ptr<scalar_t>(),
                output_tangent.data_ptr<scalar_t>(),
                packed_tangent.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_heterogeneous_double_backward_from_workspace_cuda(
    const torch::Tensor& packed_adjoint_tangent,
    const torch::Tensor& output_adjoint,
    const torch::Tensor& workspace,
    const torch::Tensor& workspace_adjoint,
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
  check_cuda_data_tensor(
      packed_adjoint_tangent,
      "packed_adjoint_tangent");
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  const auto dimensions = heterogeneous_factorized_cuda_dimensions(
      packed_adjoint_tangent,
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
  TORCH_CHECK(
      output_adjoint.device() == packed_adjoint_tangent.device() &&
          workspace.device() == packed_adjoint_tangent.device() &&
          workspace_adjoint.device() == packed_adjoint_tangent.device(),
      "cached factorized tensors must use one CUDA device");
  TORCH_CHECK(
      output_adjoint.scalar_type() ==
              packed_adjoint_tangent.scalar_type() &&
          workspace.scalar_type() ==
              packed_adjoint_tangent.scalar_type() &&
          workspace_adjoint.scalar_type() ==
              packed_adjoint_tangent.scalar_type(),
      "cached factorized tensors must have identical dtypes");
  TORCH_CHECK(
      output_adjoint.size(0) == packed_adjoint_tangent.size(0),
      "output_adjoint batch must match packed_adjoint_tangent");
  TORCH_CHECK(
      workspace.is_contiguous() && workspace_adjoint.is_contiguous() &&
          workspace.sizes() == workspace_adjoint.sizes() &&
          workspace.dim() == 2 &&
          workspace.size(0) ==
              packed_adjoint_tangent.size(0) * source_dimension &&
          workspace.size(1) == dimensions.workspace_dimension,
      "cached factorized workspaces must have shape "
      "[batch * source_dimension, workspace_dimension]");
  c10::cuda::CUDAGuard device_guard(packed_adjoint_tangent.device());
  const std::int64_t sample_count =
      packed_adjoint_tangent.size(0) * source_dimension;
  const bool use_warp = use_factorized_angular_warp_kernel(
      sample_count,
      dimensions.workspace_dimension,
      dimensions.node_count,
      coefficient_values.numel()) &&
      2 * dimensions.workspace_dimension *
              packed_adjoint_tangent.element_size() <=
          32 * 1024;
  auto adjoint_tangent = use_warp
      ? torch::empty({0}, workspace.options())
      : torch::empty_like(workspace);
  auto primal_tangent = use_warp
      ? torch::empty({0}, workspace.options())
      : torch::empty_like(workspace);
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_adjoint_tangent);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_adjoint_tangent.scalar_type(),
      "ye3t_factorized_angular_heterogeneous_"
      "double_backward_from_workspace_cuda",
      [&] {
        if (use_warp) {
          factorized_angular_double_backward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 2 * dimensions.workspace_dimension * sizeof(scalar_t),
                 stream>>>(
                  packed_adjoint_tangent.data_ptr<scalar_t>(),
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_adjoint_tangent.data_ptr<scalar_t>(),
                  packed_adjoint_tangent.size(0),
                  dimensions.input_dimension,
                  source_dimension,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  dimensions.node_count,
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_nodes.numel(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  dimensions.total_projection_dimension,
                  dimensions.workspace_dimension,
                  0,
                  dimensions.output_dimension,
                  true,
                  const_cast<scalar_t*>(workspace.data_ptr<scalar_t>()),
                  const_cast<scalar_t*>(
                      workspace_adjoint.data_ptr<scalar_t>()),
                  adjoint_tangent.data_ptr<scalar_t>(),
                  primal_tangent.data_ptr<scalar_t>(),
                  output_tangent.data_ptr<scalar_t>(),
                  packed_tangent.data_ptr<scalar_t>());
        } else {
          factorized_angular_double_backward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_adjoint_tangent.data_ptr<scalar_t>(),
                output_adjoint.data_ptr<scalar_t>(),
                packed_adjoint_tangent.data_ptr<scalar_t>(),
                packed_adjoint_tangent.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                dimensions.total_projection_dimension,
                dimensions.workspace_dimension,
                0,
                dimensions.output_dimension,
                true,
                const_cast<scalar_t*>(workspace.data_ptr<scalar_t>()),
                const_cast<scalar_t*>(
                    workspace_adjoint.data_ptr<scalar_t>()),
                adjoint_tangent.data_ptr<scalar_t>(),
                primal_tangent.data_ptr<scalar_t>(),
                output_tangent.data_ptr<scalar_t>(),
                packed_tangent.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

std::int64_t check_segmented_factorized_cuda_inputs(
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
  check_cuda_data_tensor(packed_slots, "packed_slots");
  check_cuda_int64_vector(
      segment_source_offsets,
      "segment_source_offsets");
  check_cuda_int64_vector(
      segment_input_offsets,
      "segment_input_offsets");
  check_cuda_int64_vector(
      segment_input_dimensions,
      "segment_input_dimensions");
  check_cuda_int64_vector(
      segment_workspace_offsets,
      "segment_workspace_offsets");
  check_cuda_int64_vector(
      segment_workspace_dimensions,
      "segment_workspace_dimensions");
  check_cuda_int64_vector(
      segment_node_offsets,
      "segment_node_offsets");
  check_cuda_int64_vector(
      segment_root_offsets,
      "segment_root_offsets");
  check_cuda_int64_vector(
      segment_projection_offsets,
      "segment_projection_offsets");
  check_cuda_int64_vector(
      segment_projection_dimensions,
      "segment_projection_dimensions");
  check_cuda_int64_vector(
      segment_output_offsets,
      "segment_output_offsets");
  check_cuda_int64_vector(node_offsets, "node_offsets");
  check_cuda_int64_vector(node_dimensions, "node_dimensions");
  check_cuda_int64_vector(node_leaf_offsets, "node_leaf_offsets");
  check_cuda_int64_vector(node_left, "node_left");
  check_cuda_int64_vector(node_right, "node_right");
  check_cuda_int64_vector(
      node_coefficient_offsets,
      "node_coefficient_offsets");
  check_cuda_int64_vector(coefficient_rows, "coefficient_rows");
  check_cuda_int64_vector(coefficient_columns, "coefficient_columns");
  check_cuda_int64_vector(root_nodes, "root_nodes");
  check_cuda_int64_vector(
      root_projection_starts,
      "root_projection_starts");
  check_cuda_int64_vector(
      root_projection_dimensions,
      "root_projection_dimensions");
  check_cuda_int64_vector(root_output_offsets, "root_output_offsets");
  check_cuda_value_vector(
      coefficient_values,
      packed_slots,
      "coefficient_values");
  check_cuda_value_vector(
      projection_values,
      packed_slots,
      "projection_values");
  TORCH_CHECK(
      segment_source_offsets.device() == packed_slots.device() &&
          segment_input_offsets.device() == packed_slots.device() &&
          segment_input_dimensions.device() == packed_slots.device() &&
          segment_workspace_offsets.device() == packed_slots.device() &&
          segment_workspace_dimensions.device() == packed_slots.device() &&
          segment_node_offsets.device() == packed_slots.device() &&
          segment_root_offsets.device() == packed_slots.device() &&
          segment_projection_offsets.device() == packed_slots.device() &&
          segment_projection_dimensions.device() == packed_slots.device() &&
          segment_output_offsets.device() == packed_slots.device() &&
          node_offsets.device() == packed_slots.device() &&
          node_dimensions.device() == packed_slots.device() &&
          node_leaf_offsets.device() == packed_slots.device() &&
          node_left.device() == packed_slots.device() &&
          node_right.device() == packed_slots.device() &&
          node_coefficient_offsets.device() == packed_slots.device() &&
          coefficient_rows.device() == packed_slots.device() &&
          coefficient_columns.device() == packed_slots.device() &&
          root_nodes.device() == packed_slots.device() &&
          root_projection_starts.device() == packed_slots.device() &&
          root_projection_dimensions.device() == packed_slots.device() &&
          root_output_offsets.device() == packed_slots.device(),
      "segmented factorized metadata must use one CUDA device");
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
  TORCH_CHECK(node_count > 0 && root_count > 0,
      "segmented plan requires nodes and roots");
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
  TORCH_CHECK(
      total_source_count > 0 && workspace_dimension > 0 &&
          output_dimension > 0 && packed_slots.size(1) > 0,
      "segmented runtime dimensions must be positive");
  return segment_count;
}

torch::Tensor factorized_angular_segmented_cuda(
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
  const std::int64_t segment_count =
      check_segmented_factorized_cuda_inputs(
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
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {packed_slots.size(0), workspace_dimension},
      packed_slots.options());
  auto output = torch::zeros(
      {packed_slots.size(0), output_dimension},
      packed_slots.options());
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_segmented_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * total_source_count;
        const std::int64_t execution_policy =
            factorized_angular_segmented_execution_policy(
            sample_count,
            segment_count,
            coefficient_values.numel());
        if (execution_policy != kFactorizedSegmentExecutionSerial) {
          factorized_angular_segmented_forward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  packed_slots.size(1),
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
                  segment_count,
                  total_source_count,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  workspace_dimension,
                  output_dimension,
                  execution_policy == kFactorizedSegmentExecutionHybrid
                      ? kFactorizedSegmentExecutionWarp
                      : kFactorizedSegmentExecutionAll,
                  workspace.data_ptr<scalar_t>(),
                  output.data_ptr<scalar_t>());
        }
        if (execution_policy != kFactorizedSegmentExecutionWarp) {
          factorized_angular_segmented_forward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                packed_slots.size(1),
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
                segment_count,
                total_source_count,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                workspace_dimension,
                output_dimension,
                execution_policy == kFactorizedSegmentExecutionHybrid
                    ? kFactorizedSegmentExecutionSerial
                    : kFactorizedSegmentExecutionAll,
                workspace.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

torch::Tensor factorized_angular_segmented_adjoint_cuda(
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
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      output_adjoint.device() == packed_slots.device() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.size(0) == packed_slots.size(0),
      "segmented output adjoint must match packed input");
  const std::int64_t output_dimension = output_adjoint.size(1);
  const std::int64_t segment_count =
      check_segmented_factorized_cuda_inputs(
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
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {packed_slots.size(0), workspace_dimension},
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  auto packed_adjoint = torch::empty_like(packed_slots);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_segmented_adjoint_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * total_source_count;
        const std::int64_t execution_policy =
            factorized_angular_segmented_execution_policy(
            sample_count,
            segment_count,
            coefficient_values.numel());
        if (execution_policy != kFactorizedSegmentExecutionSerial) {
          factorized_angular_segmented_adjoint_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  packed_slots.size(1),
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
                  segment_count,
                  total_source_count,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  workspace_dimension,
                  output_dimension,
                  execution_policy == kFactorizedSegmentExecutionHybrid
                      ? kFactorizedSegmentExecutionWarp
                      : kFactorizedSegmentExecutionAll,
                  workspace.data_ptr<scalar_t>(),
                  workspace_adjoint.data_ptr<scalar_t>(),
                  packed_adjoint.data_ptr<scalar_t>());
        }
        if (execution_policy != kFactorizedSegmentExecutionWarp) {
          factorized_angular_segmented_adjoint_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                packed_slots.size(1),
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
                segment_count,
                total_source_count,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                workspace_dimension,
                output_dimension,
                execution_policy == kFactorizedSegmentExecutionHybrid
                    ? kFactorizedSegmentExecutionSerial
                    : kFactorizedSegmentExecutionAll,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                packed_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return packed_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor>
factorized_angular_segmented_double_backward_cuda(
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
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(
      packed_adjoint_tangent,
      "packed_adjoint_tangent");
  TORCH_CHECK(
      packed_adjoint_tangent.device() == packed_slots.device() &&
          output_adjoint.device() == packed_slots.device() &&
          packed_adjoint_tangent.scalar_type() ==
              packed_slots.scalar_type() &&
          output_adjoint.scalar_type() == packed_slots.scalar_type() &&
          packed_adjoint_tangent.sizes() == packed_slots.sizes() &&
          output_adjoint.size(0) == packed_slots.size(0),
      "segmented double-backward tensors must match packed input");
  const std::int64_t output_dimension = output_adjoint.size(1);
  const std::int64_t segment_count =
      check_segmented_factorized_cuda_inputs(
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
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {packed_slots.size(0), workspace_dimension},
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  auto adjoint_tangent = torch::empty_like(workspace);
  auto primal_tangent = torch::empty_like(workspace);
  auto output_tangent = torch::zeros_like(output_adjoint);
  auto packed_tangent = torch::empty_like(packed_slots);
  constexpr int threads = 32;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_segmented_double_backward_cuda",
      [&] {
        const std::int64_t sample_count =
            packed_slots.size(0) * total_source_count;
        const std::int64_t execution_policy =
            factorized_angular_segmented_execution_policy(
            sample_count,
            segment_count,
            coefficient_values.numel());
        if (execution_policy != kFactorizedSegmentExecutionSerial) {
          factorized_angular_segmented_double_backward_warp_kernel<scalar_t>
              <<<factorized_angular_warp_blocks(sample_count),
                 threads,
                 0,
                 stream>>>(
                  packed_adjoint_tangent.data_ptr<scalar_t>(),
                  output_adjoint.data_ptr<scalar_t>(),
                  packed_slots.data_ptr<scalar_t>(),
                  packed_slots.size(0),
                  packed_slots.size(1),
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
                  segment_count,
                  total_source_count,
                  node_offsets.data_ptr<std::int64_t>(),
                  node_dimensions.data_ptr<std::int64_t>(),
                  node_leaf_offsets.data_ptr<std::int64_t>(),
                  node_left.data_ptr<std::int64_t>(),
                  node_right.data_ptr<std::int64_t>(),
                  node_coefficient_offsets.data_ptr<std::int64_t>(),
                  coefficient_rows.data_ptr<std::int64_t>(),
                  coefficient_columns.data_ptr<std::int64_t>(),
                  coefficient_values.data_ptr<scalar_t>(),
                  root_nodes.data_ptr<std::int64_t>(),
                  root_projection_starts.data_ptr<std::int64_t>(),
                  root_projection_dimensions.data_ptr<std::int64_t>(),
                  root_output_offsets.data_ptr<std::int64_t>(),
                  projection_values.data_ptr<scalar_t>(),
                  workspace_dimension,
                  output_dimension,
                  execution_policy == kFactorizedSegmentExecutionHybrid
                      ? kFactorizedSegmentExecutionWarp
                      : kFactorizedSegmentExecutionAll,
                  workspace.data_ptr<scalar_t>(),
                  workspace_adjoint.data_ptr<scalar_t>(),
                  adjoint_tangent.data_ptr<scalar_t>(),
                  primal_tangent.data_ptr<scalar_t>(),
                  output_tangent.data_ptr<scalar_t>(),
                  packed_tangent.data_ptr<scalar_t>());
        }
        if (execution_policy != kFactorizedSegmentExecutionWarp) {
          factorized_angular_segmented_double_backward_kernel<scalar_t>
              <<<launch_blocks(sample_count, threads),
                 threads,
                 0,
                 stream>>>(
                packed_adjoint_tangent.data_ptr<scalar_t>(),
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                packed_slots.size(1),
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
                segment_count,
                total_source_count,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_projection_starts.data_ptr<std::int64_t>(),
                root_projection_dimensions.data_ptr<std::int64_t>(),
                root_output_offsets.data_ptr<std::int64_t>(),
                projection_values.data_ptr<scalar_t>(),
                workspace_dimension,
                output_dimension,
                execution_policy == kFactorizedSegmentExecutionHybrid
                    ? kFactorizedSegmentExecutionSerial
                    : kFactorizedSegmentExecutionAll,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                adjoint_tangent.data_ptr<scalar_t>(),
                primal_tangent.data_ptr<scalar_t>(),
                output_tangent.data_ptr<scalar_t>(),
                packed_tangent.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(output_tangent, packed_tangent);
}

torch::Tensor factorized_angular_linear_cuda(
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
  const FactorizedCudaDimensions dimensions =
      factorized_cuda_dimensions(
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
          projection_values,
          source_dimension,
          workspace_dimension,
          weight.numel());
  check_cuda_value_vector(weight, packed_slots, "weight");
  TORCH_CHECK(
      bias.is_cuda() && bias.device() == packed_slots.device() &&
          bias.is_contiguous() && bias.dim() == 0 &&
          bias.scalar_type() == packed_slots.scalar_type(),
      "bias must be a contiguous CUDA scalar with data dtype");
  TORCH_CHECK(
      weight.numel() == dimensions.output_dimension,
      "weight length must match factorized output dimension");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {packed_slots.size(0), dimensions.workspace_dimension},
      packed_slots.options());
  auto output = torch::empty(
      {packed_slots.size(0)},
      packed_slots.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_linear_cuda",
      [&] {
        factorized_angular_linear_forward_kernel<scalar_t>
            <<<launch_blocks(packed_slots.size(0)),
               threads,
               0,
               stream>>>(
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                projection_values.data_ptr<scalar_t>(),
                dimensions.projection_dimension,
                weight.data_ptr<scalar_t>(),
                bias.data_ptr<scalar_t>(),
                dimensions.workspace_dimension,
                dimensions.root_dimension,
                workspace.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
factorized_angular_linear_adjoint_cuda(
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
  const FactorizedCudaDimensions dimensions =
      factorized_cuda_dimensions(
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
          projection_values,
          source_dimension,
          workspace_dimension,
          weight.numel());
  TORCH_CHECK(
      output_adjoint.is_cuda() &&
          output_adjoint.device() == packed_slots.device() &&
          output_adjoint.is_contiguous() &&
          output_adjoint.dim() == 1 &&
          output_adjoint.scalar_type() == packed_slots.scalar_type() &&
          output_adjoint.numel() == packed_slots.size(0),
      "output_adjoint must be a contiguous CUDA batch vector");
  check_cuda_value_vector(weight, packed_slots, "weight");
  TORCH_CHECK(
      weight.numel() == dimensions.output_dimension,
      "weight length must match factorized output dimension");
  c10::cuda::CUDAGuard device_guard(packed_slots.device());
  auto workspace = torch::empty(
      {packed_slots.size(0), dimensions.workspace_dimension},
      packed_slots.options());
  auto workspace_adjoint = torch::empty_like(workspace);
  auto packed_adjoint = torch::empty_like(packed_slots);
  auto weight_adjoint = torch::zeros_like(weight);
  auto bias_adjoint = torch::zeros({}, weight.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      packed_slots.scalar_type(),
      "ye3t_factorized_angular_linear_adjoint_cuda",
      [&] {
        factorized_angular_linear_adjoint_kernel<scalar_t>
            <<<launch_blocks(packed_slots.size(0)),
               threads,
               0,
               stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                packed_slots.data_ptr<scalar_t>(),
                packed_slots.size(0),
                dimensions.input_dimension,
                source_dimension,
                node_offsets.data_ptr<std::int64_t>(),
                node_dimensions.data_ptr<std::int64_t>(),
                node_leaf_offsets.data_ptr<std::int64_t>(),
                node_left.data_ptr<std::int64_t>(),
                node_right.data_ptr<std::int64_t>(),
                node_coefficient_offsets.data_ptr<std::int64_t>(),
                dimensions.node_count,
                coefficient_rows.data_ptr<std::int64_t>(),
                coefficient_columns.data_ptr<std::int64_t>(),
                coefficient_values.data_ptr<scalar_t>(),
                root_nodes.data_ptr<std::int64_t>(),
                root_nodes.numel(),
                projection_values.data_ptr<scalar_t>(),
                dimensions.projection_dimension,
                weight.data_ptr<scalar_t>(),
                dimensions.workspace_dimension,
                dimensions.root_dimension,
                workspace.data_ptr<scalar_t>(),
                workspace_adjoint.data_ptr<scalar_t>(),
                packed_adjoint.data_ptr<scalar_t>(),
                weight_adjoint.data_ptr<scalar_t>(),
                bias_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      packed_adjoint,
      weight_adjoint,
      bias_adjoint);
}

torch::Tensor density_accumulate_cuda(
    const torch::Tensor& edge_values,
    const torch::Tensor& centers,
    std::int64_t atom_count) {
  check_cuda_data_tensor(edge_values, "edge_values");
  check_cuda_centers(centers, edge_values, "density");
  TORCH_CHECK(
      centers.numel() == edge_values.size(0),
      "centers length must match the edge count");
  TORCH_CHECK(atom_count > 0, "atom_count must be positive");
  c10::cuda::CUDAGuard device_guard(edge_values.device());
  auto atomic_values = torch::zeros(
      {atom_count, edge_values.size(1)},
      edge_values.options());
  const std::int64_t work =
      edge_values.size(0) * edge_values.size(1);
  if (work == 0) {
    return atomic_values;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      edge_values.scalar_type(),
      "ye3t_density_accumulate_cuda",
      [&] {
        density_accumulate_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                edge_values.data_ptr<scalar_t>(),
                centers.data_ptr<std::int64_t>(),
                edge_values.size(0),
                edge_values.size(1),
                atom_count,
                atomic_values.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return atomic_values;
}

torch::Tensor density_accumulate_adjoint_cuda(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& centers) {
  check_cuda_data_tensor(atomic_adjoint, "atomic_adjoint");
  check_cuda_centers(centers, atomic_adjoint, "density adjoint");
  TORCH_CHECK(
      atomic_adjoint.size(0) > 0,
      "atomic_adjoint must have at least one atom row");
  c10::cuda::CUDAGuard device_guard(atomic_adjoint.device());
  auto edge_adjoint = torch::empty(
      {centers.numel(), atomic_adjoint.size(1)},
      atomic_adjoint.options());
  const std::int64_t work =
      centers.numel() * atomic_adjoint.size(1);
  if (work == 0) {
    return edge_adjoint;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      atomic_adjoint.scalar_type(),
      "ye3t_density_accumulate_adjoint_cuda",
      [&] {
        density_accumulate_adjoint_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                atomic_adjoint.data_ptr<scalar_t>(),
                centers.data_ptr<std::int64_t>(),
                centers.numel(),
                atomic_adjoint.size(1),
                atomic_adjoint.size(0),
                edge_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return edge_adjoint;
}

void check_edge_outer_cuda_inputs(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& centers,
    std::int64_t atom_count) {
  check_cuda_data_tensor(left, "left");
  check_cuda_data_tensor(right, "right");
  check_cuda_centers(centers, left, "edge outer");
  TORCH_CHECK(
      left.scalar_type() == torch::kFloat32 ||
          left.scalar_type() == torch::kFloat64,
      "edge outer accumulation requires float32 or float64 inputs");
  TORCH_CHECK(
      right.device() == left.device() &&
          right.scalar_type() == left.scalar_type(),
      "left and right must have the same device and dtype");
  TORCH_CHECK(
      right.size(0) == left.size(0),
      "left and right edge counts must match");
  TORCH_CHECK(
      centers.numel() == left.size(0),
      "centers length must match the edge count");
  TORCH_CHECK(atom_count > 0, "atom_count must be positive");
}

torch::Tensor edge_outer_accumulate_cuda(
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& centers,
    std::int64_t atom_count) {
  check_edge_outer_cuda_inputs(left, right, centers, atom_count);
  c10::cuda::CUDAGuard device_guard(left.device());
  auto atomic_values = torch::zeros(
      {atom_count, left.size(1), right.size(1)},
      left.options());
  const std::int64_t work =
      left.size(0) * left.size(1) * right.size(1);
  if (work == 0) {
    return atomic_values;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "ye3t_edge_outer_accumulate_cuda",
      [&] {
        edge_outer_accumulate_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                left.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                centers.data_ptr<std::int64_t>(),
                left.size(0),
                left.size(1),
                right.size(1),
                atom_count,
                atomic_values.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return atomic_values;
}

std::tuple<torch::Tensor, torch::Tensor>
edge_outer_accumulate_adjoint_cuda(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& centers) {
  TORCH_CHECK(
      atomic_adjoint.dim() == 3 && atomic_adjoint.size(0) > 0,
      "atomic_adjoint must be three-dimensional with an atom axis");
  check_edge_outer_cuda_inputs(
      left,
      right,
      centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.is_cuda() &&
          atomic_adjoint.is_contiguous() &&
          atomic_adjoint.device() == left.device() &&
          atomic_adjoint.scalar_type() == left.scalar_type() &&
          atomic_adjoint.size(1) == left.size(1) &&
          atomic_adjoint.size(2) == right.size(1),
      "atomic_adjoint must match left and right device, dtype, and dimensions");
  c10::cuda::CUDAGuard device_guard(left.device());
  auto left_adjoint = torch::empty_like(left);
  auto right_adjoint = torch::empty_like(right);
  if (left.size(0) == 0) {
    return std::make_tuple(left_adjoint, right_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "ye3t_edge_outer_accumulate_adjoint_cuda",
      [&] {
        edge_outer_accumulate_adjoint_kernel<scalar_t>
            <<<launch_blocks(left.size(0)), threads, 0, stream>>>(
                atomic_adjoint.data_ptr<scalar_t>(),
                left.data_ptr<scalar_t>(),
                right.data_ptr<scalar_t>(),
                centers.data_ptr<std::int64_t>(),
                left.size(0),
                left.size(1),
                right.size(1),
                atomic_adjoint.size(0),
                left_adjoint.data_ptr<scalar_t>(),
                right_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(left_adjoint, right_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
edge_outer_accumulate_double_backward_cuda(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& left,
    const torch::Tensor& right,
    const torch::Tensor& left_adjoint_tangent,
    const torch::Tensor& right_adjoint_tangent,
    const torch::Tensor& centers) {
  TORCH_CHECK(
      atomic_adjoint.dim() == 3 && atomic_adjoint.size(0) > 0,
      "atomic_adjoint must be three-dimensional with an atom axis");
  check_edge_outer_cuda_inputs(
      left,
      right,
      centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.is_cuda() &&
          atomic_adjoint.is_contiguous() &&
          atomic_adjoint.device() == left.device() &&
          atomic_adjoint.scalar_type() == left.scalar_type() &&
          atomic_adjoint.size(1) == left.size(1) &&
          atomic_adjoint.size(2) == right.size(1),
      "atomic_adjoint must match left and right device, dtype, and dimensions");
  TORCH_CHECK(
      left_adjoint_tangent.is_cuda() &&
          left_adjoint_tangent.is_contiguous() &&
          left_adjoint_tangent.device() == left.device() &&
          left_adjoint_tangent.scalar_type() == left.scalar_type() &&
          left_adjoint_tangent.sizes() == left.sizes(),
      "left_adjoint_tangent must match left");
  TORCH_CHECK(
      right_adjoint_tangent.is_cuda() &&
          right_adjoint_tangent.is_contiguous() &&
          right_adjoint_tangent.device() == right.device() &&
          right_adjoint_tangent.scalar_type() == right.scalar_type() &&
          right_adjoint_tangent.sizes() == right.sizes(),
      "right_adjoint_tangent must match right");
  c10::cuda::CUDAGuard device_guard(left.device());
  auto atomic_adjoint_tangent = torch::zeros_like(atomic_adjoint);
  auto left_second_adjoint = torch::empty_like(left);
  auto right_second_adjoint = torch::empty_like(right);
  if (left.size(0) == 0) {
    return std::make_tuple(
        atomic_adjoint_tangent,
        left_second_adjoint,
        right_second_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      left.scalar_type(),
      "ye3t_edge_outer_accumulate_double_backward_cuda",
      [&] {
        edge_outer_accumulate_double_backward_kernel<scalar_t>
            <<<launch_blocks(left.size(0)), threads, 0, stream>>>(
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      atomic_adjoint_tangent,
      left_second_adjoint,
      right_second_adjoint);
}

void check_softmax_gaussian_role_density_cuda_inputs(
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& atom_centers,
    std::int64_t atom_count) {
  check_cuda_real_vector(distances, "distances");
  check_cuda_real_vector(cutoffs, "cutoffs");
  check_cuda_real_vector(filter_centers, "filter_centers");
  check_cuda_real_matrix(edge_values, "edge_values");
  check_cuda_centers(atom_centers, distances, "role density");
  TORCH_CHECK(
      cutoffs.device() == distances.device() &&
          filter_centers.device() == distances.device() &&
          edge_values.device() == distances.device() &&
          cutoffs.scalar_type() == distances.scalar_type() &&
          filter_centers.scalar_type() == distances.scalar_type() &&
          edge_values.scalar_type() == distances.scalar_type(),
      "softmax-Gaussian role-density floating inputs must share device and dtype");
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
}

torch::Tensor softmax_gaussian_role_density_cuda(
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& atom_centers,
    std::int64_t atom_count) {
  check_softmax_gaussian_role_density_cuda_inputs(
      distances, cutoffs, filter_centers, filter_width,
      edge_values, atom_centers, atom_count);
  c10::cuda::CUDAGuard device_guard(distances.device());
  auto atomic_values = torch::zeros(
      {atom_count, filter_centers.numel(), edge_values.size(1)},
      edge_values.options());
  if (distances.numel() == 0) {
    return atomic_values;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_softmax_gaussian_role_density_cuda",
      [&] {
        softmax_gaussian_role_density_kernel<scalar_t>
            <<<launch_blocks(distances.numel()), threads, 0, stream>>>(
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return atomic_values;
}

std::tuple<torch::Tensor, torch::Tensor>
softmax_gaussian_role_density_adjoint_cuda(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& atom_centers) {
  TORCH_CHECK(
      atomic_adjoint.dim() == 3 && atomic_adjoint.size(0) > 0,
      "atomic_adjoint must be three-dimensional with an atom axis");
  check_softmax_gaussian_role_density_cuda_inputs(
      distances, cutoffs, filter_centers, filter_width,
      edge_values, atom_centers, atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.is_cuda() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.device() == distances.device() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == edge_values.size(1),
      "atomic_adjoint must match role-density output layout and dtype");
  c10::cuda::CUDAGuard device_guard(distances.device());
  auto distance_adjoint = torch::empty_like(distances);
  auto edge_adjoint = torch::empty_like(edge_values);
  if (distances.numel() == 0) {
    return std::make_tuple(distance_adjoint, edge_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_softmax_gaussian_role_density_adjoint_cuda",
      [&] {
        softmax_gaussian_role_density_adjoint_kernel<scalar_t>
            <<<launch_blocks(distances.numel()), threads, 0, stream>>>(
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
                atomic_adjoint.size(0),
                distance_adjoint.data_ptr<scalar_t>(),
                edge_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(distance_adjoint, edge_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
softmax_gaussian_role_density_double_backward_cuda(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& edge_values,
    const torch::Tensor& distance_adjoint_tangent,
    const torch::Tensor& edge_adjoint_tangent,
    const torch::Tensor& atom_centers) {
  TORCH_CHECK(
      atomic_adjoint.dim() == 3 && atomic_adjoint.size(0) > 0,
      "atomic_adjoint must be three-dimensional with an atom axis");
  check_softmax_gaussian_role_density_cuda_inputs(
      distances, cutoffs, filter_centers, filter_width,
      edge_values, atom_centers, atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.is_cuda() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.device() == distances.device() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == edge_values.size(1),
      "atomic_adjoint must match role-density output layout and dtype");
  TORCH_CHECK(
      distance_adjoint_tangent.is_cuda() &&
          distance_adjoint_tangent.is_contiguous() &&
          distance_adjoint_tangent.device() == distances.device() &&
          distance_adjoint_tangent.scalar_type() == distances.scalar_type() &&
          distance_adjoint_tangent.sizes() == distances.sizes(),
      "distance_adjoint_tangent must match distances");
  TORCH_CHECK(
      edge_adjoint_tangent.is_cuda() &&
          edge_adjoint_tangent.is_contiguous() &&
          edge_adjoint_tangent.device() == edge_values.device() &&
          edge_adjoint_tangent.scalar_type() == edge_values.scalar_type() &&
          edge_adjoint_tangent.sizes() == edge_values.sizes(),
      "edge_adjoint_tangent must match edge_values");
  c10::cuda::CUDAGuard device_guard(distances.device());
  auto atomic_adjoint_tangent = torch::zeros_like(atomic_adjoint);
  auto distance_second_adjoint = torch::empty_like(distances);
  auto edge_second_adjoint = torch::empty_like(edge_values);
  if (distances.numel() == 0) {
    return std::make_tuple(
        atomic_adjoint_tangent,
        distance_second_adjoint,
        edge_second_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(),
      "ye3t_softmax_gaussian_role_density_double_backward_cuda",
      [&] {
        softmax_gaussian_role_density_double_backward_kernel<scalar_t>
            <<<launch_blocks(distances.numel()), threads, 0, stream>>>(
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      atomic_adjoint_tangent,
      distance_second_adjoint,
      edge_second_adjoint);
}

void check_scheduled_role_density_cuda_inputs(
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
  check_cuda_real_matrix(radial_values, "radial_values");
  check_cuda_real_matrix(angular_values, "angular_values");
  check_cuda_real_vector(distances, "distances");
  check_cuda_real_vector(cutoffs, "cutoffs");
  check_cuda_real_vector(filter_centers, "filter_centers");
  check_cuda_real_vector(soft_weights, "soft_weights");
  check_cuda_real_vector(channel_scales, "channel_scales");
  check_cuda_int64_vector(edge_types, "edge_types");
  check_cuda_int64_vector(channel_radial_indices, "channel_radial_indices");
  check_cuda_int64_vector(channel_angular_indices, "channel_angular_indices");
  check_cuda_int64_vector(channel_types, "channel_types");
  check_cuda_int64_vector(atom_centers, "atom_centers");
  const auto device = distances.device();
  const auto dtype = distances.scalar_type();
  TORCH_CHECK(
      radial_values.device() == device && angular_values.device() == device &&
          cutoffs.device() == device && filter_centers.device() == device &&
          soft_weights.device() == device && channel_scales.device() == device &&
          edge_types.device() == device &&
          channel_radial_indices.device() == device &&
          channel_angular_indices.device() == device &&
          channel_types.device() == device && atom_centers.device() == device,
      "scheduled role-density inputs must share one CUDA device");
  TORCH_CHECK(
      radial_values.scalar_type() == dtype && angular_values.scalar_type() == dtype &&
          cutoffs.scalar_type() == dtype && filter_centers.scalar_type() == dtype &&
          soft_weights.scalar_type() == dtype && channel_scales.scalar_type() == dtype,
      "scheduled role-density floating inputs must share one dtype");
  const std::int64_t edge_count = distances.numel();
  TORCH_CHECK(
      radial_values.size(0) == edge_count && angular_values.size(0) == edge_count &&
          cutoffs.numel() == edge_count && soft_weights.numel() == edge_count &&
          edge_types.numel() == edge_count && atom_centers.numel() == edge_count,
      "scheduled role-density edge counts must match");
  TORCH_CHECK(
      radial_values.size(1) > 0 && angular_values.size(1) > 0,
      "scheduled role-density radial and angular tables must be nonempty");
  const std::int64_t channel_count = channel_scales.numel();
  TORCH_CHECK(
      channel_count > 0 && channel_radial_indices.numel() == channel_count &&
          channel_angular_indices.numel() == channel_count &&
          channel_types.numel() == channel_count,
      "scheduled role-density channel schedules must match");
  TORCH_CHECK(
      filter_centers.numel() > 0 && filter_width > 0.0 && atom_count > 0,
      "scheduled role density requires roles, positive width, and atoms");
}

torch::Tensor scheduled_role_density_cuda(
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
  check_scheduled_role_density_cuda_inputs(
      radial_values, angular_values, distances, cutoffs, filter_centers,
      filter_width, soft_weights, edge_types, channel_radial_indices,
      channel_angular_indices, channel_types, channel_scales, atom_centers,
      atom_count);
  c10::cuda::CUDAGuard device_guard(distances.device());
  auto output = torch::zeros(
      {atom_count, filter_centers.numel(), channel_scales.numel() + 1},
      distances.options());
  if (distances.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(), "ye3t_scheduled_role_density_cuda", [&] {
        scheduled_role_density_kernel<scalar_t>
            <<<launch_blocks(distances.numel()), threads, 0, stream>>>(
                radial_values.data_ptr<scalar_t>(), angular_values.data_ptr<scalar_t>(),
                distances.data_ptr<scalar_t>(), cutoffs.data_ptr<scalar_t>(),
                filter_centers.data_ptr<scalar_t>(), static_cast<scalar_t>(filter_width),
                soft_weights.data_ptr<scalar_t>(), edge_types.data_ptr<std::int64_t>(),
                channel_radial_indices.data_ptr<std::int64_t>(),
                channel_angular_indices.data_ptr<std::int64_t>(),
                channel_types.data_ptr<std::int64_t>(), channel_scales.data_ptr<scalar_t>(),
                atom_centers.data_ptr<std::int64_t>(), distances.numel(),
                radial_values.size(1), angular_values.size(1), filter_centers.numel(),
                channel_scales.numel(), atom_count, output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
scheduled_role_density_adjoint_cuda(
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
  check_scheduled_role_density_cuda_inputs(
      radial_values, angular_values, distances, cutoffs, filter_centers,
      filter_width, soft_weights, edge_types, channel_radial_indices,
      channel_angular_indices, channel_types, channel_scales, atom_centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.is_cuda() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.device() == distances.device() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.dim() == 3 &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == channel_scales.numel() + 1,
      "atomic_adjoint must match scheduled role-density output");
  c10::cuda::CUDAGuard device_guard(distances.device());
  auto radial_adjoint = torch::zeros_like(radial_values);
  auto angular_adjoint = torch::zeros_like(angular_values);
  auto distance_adjoint = torch::empty_like(distances);
  auto soft_adjoint = torch::empty_like(soft_weights);
  if (distances.numel() == 0) {
    return std::make_tuple(radial_adjoint, angular_adjoint, distance_adjoint, soft_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(), "ye3t_scheduled_role_density_adjoint_cuda", [&] {
        scheduled_role_density_adjoint_kernel<scalar_t>
            <<<launch_blocks(distances.numel()), threads, 0, stream>>>(
                atomic_adjoint.data_ptr<scalar_t>(), radial_values.data_ptr<scalar_t>(),
                angular_values.data_ptr<scalar_t>(), distances.data_ptr<scalar_t>(),
                cutoffs.data_ptr<scalar_t>(), filter_centers.data_ptr<scalar_t>(),
                static_cast<scalar_t>(filter_width), soft_weights.data_ptr<scalar_t>(),
                edge_types.data_ptr<std::int64_t>(),
                channel_radial_indices.data_ptr<std::int64_t>(),
                channel_angular_indices.data_ptr<std::int64_t>(),
                channel_types.data_ptr<std::int64_t>(), channel_scales.data_ptr<scalar_t>(),
                atom_centers.data_ptr<std::int64_t>(), distances.numel(),
                radial_values.size(1), angular_values.size(1), filter_centers.numel(),
                channel_scales.numel(), atomic_adjoint.size(0),
                radial_adjoint.data_ptr<scalar_t>(), angular_adjoint.data_ptr<scalar_t>(),
                distance_adjoint.data_ptr<scalar_t>(), soft_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(radial_adjoint, angular_adjoint, distance_adjoint, soft_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
scheduled_role_density_double_backward_cuda(
    const torch::Tensor& atomic_adjoint,
    const torch::Tensor& radial_values,
    const torch::Tensor& angular_values,
    const torch::Tensor& distances,
    const torch::Tensor& cutoffs,
    const torch::Tensor& filter_centers,
    double filter_width,
    const torch::Tensor& soft_weights,
    const torch::Tensor& radial_tangent,
    const torch::Tensor& angular_tangent,
    const torch::Tensor& distance_tangent,
    const torch::Tensor& soft_tangent,
    const torch::Tensor& edge_types,
    const torch::Tensor& channel_radial_indices,
    const torch::Tensor& channel_angular_indices,
    const torch::Tensor& channel_types,
    const torch::Tensor& channel_scales,
    const torch::Tensor& atom_centers) {
  check_scheduled_role_density_cuda_inputs(
      radial_values, angular_values, distances, cutoffs, filter_centers,
      filter_width, soft_weights, edge_types, channel_radial_indices,
      channel_angular_indices, channel_types, channel_scales, atom_centers,
      atomic_adjoint.size(0));
  TORCH_CHECK(
      atomic_adjoint.is_cuda() && atomic_adjoint.is_contiguous() &&
          atomic_adjoint.device() == distances.device() &&
          atomic_adjoint.scalar_type() == distances.scalar_type() &&
          atomic_adjoint.dim() == 3 &&
          atomic_adjoint.size(1) == filter_centers.numel() &&
          atomic_adjoint.size(2) == channel_scales.numel() + 1 &&
          radial_tangent.is_cuda() && radial_tangent.is_contiguous() &&
          radial_tangent.device() == radial_values.device() &&
          radial_tangent.scalar_type() == radial_values.scalar_type() &&
          radial_tangent.sizes() == radial_values.sizes() &&
          angular_tangent.is_cuda() && angular_tangent.is_contiguous() &&
          angular_tangent.device() == angular_values.device() &&
          angular_tangent.scalar_type() == angular_values.scalar_type() &&
          angular_tangent.sizes() == angular_values.sizes() &&
          distance_tangent.is_cuda() && distance_tangent.is_contiguous() &&
          distance_tangent.device() == distances.device() &&
          distance_tangent.scalar_type() == distances.scalar_type() &&
          distance_tangent.sizes() == distances.sizes() &&
          soft_tangent.is_cuda() && soft_tangent.is_contiguous() &&
          soft_tangent.device() == soft_weights.device() &&
          soft_tangent.scalar_type() == soft_weights.scalar_type() &&
          soft_tangent.sizes() == soft_weights.sizes(),
      "scheduled role-density double-backward tensors must match primals");
  c10::cuda::CUDAGuard device_guard(distances.device());
  auto atomic_tangent = torch::zeros_like(atomic_adjoint);
  auto radial_second = torch::zeros_like(radial_values);
  auto angular_second = torch::zeros_like(angular_values);
  auto distance_second = torch::empty_like(distances);
  auto soft_second = torch::empty_like(soft_weights);
  if (distances.numel() == 0) {
    return std::make_tuple(
        atomic_tangent, radial_second, angular_second, distance_second, soft_second);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES(
      distances.scalar_type(), "ye3t_scheduled_role_density_double_backward_cuda", [&] {
        scheduled_role_density_double_backward_kernel<scalar_t>
            <<<launch_blocks(distances.numel()), threads, 0, stream>>>(
                atomic_adjoint.data_ptr<scalar_t>(), radial_values.data_ptr<scalar_t>(),
                angular_values.data_ptr<scalar_t>(), distances.data_ptr<scalar_t>(),
                cutoffs.data_ptr<scalar_t>(), filter_centers.data_ptr<scalar_t>(),
                static_cast<scalar_t>(filter_width), soft_weights.data_ptr<scalar_t>(),
                radial_tangent.data_ptr<scalar_t>(), angular_tangent.data_ptr<scalar_t>(),
                distance_tangent.data_ptr<scalar_t>(), soft_tangent.data_ptr<scalar_t>(),
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
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      atomic_tangent, radial_second, angular_second, distance_second, soft_second);
}

void check_carrier_gated_scatter_cuda_inputs(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    std::int64_t target_count) {
  check_cuda_data_tensor(node_values, "node_values");
  check_cuda_data_tensor(edge_gates, "edge_gates");
  check_cuda_int64_vector(edge_sources, "edge_sources");
  check_cuda_int64_vector(edge_targets, "edge_targets");
  check_cuda_int64_vector(feature_channels, "feature_channels");
  const bool real_gates_for_complex_values =
      (node_values.scalar_type() == torch::kComplexFloat &&
       edge_gates.scalar_type() == torch::kFloat) ||
      (node_values.scalar_type() == torch::kComplexDouble &&
       edge_gates.scalar_type() == torch::kDouble);
  TORCH_CHECK(
      edge_gates.device() == node_values.device() &&
          (edge_gates.scalar_type() == node_values.scalar_type() ||
           real_gates_for_complex_values),
      "edge_gates must share node_values device and either match its dtype "
      "or use its real component dtype for complex carriers");
  TORCH_CHECK(
      edge_sources.device() == node_values.device() &&
          edge_targets.device() == node_values.device() &&
          feature_channels.device() == node_values.device(),
      "carrier scatter indices must share node_values device");
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
}

bool carrier_gated_scatter_uses_real_gates_cuda(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates) {
  return
      (node_values.scalar_type() == torch::kComplexFloat &&
       edge_gates.scalar_type() == torch::kFloat) ||
      (node_values.scalar_type() == torch::kComplexDouble &&
       edge_gates.scalar_type() == torch::kDouble);
}

torch::Tensor carrier_gated_scatter_cuda_impl(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    std::int64_t target_count,
    bool include_residual) {
  check_carrier_gated_scatter_cuda_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_count);
  c10::cuda::CUDAGuard device_guard(node_values.device());
  TORCH_CHECK(
      !include_residual || target_count == node_values.size(0),
      "residual carrier scatter requires one target row per node");
  auto target_values = include_residual
      ? node_values.clone()
      : torch::zeros(
            {target_count, node_values.size(1)},
            node_values.options());
  const std::int64_t work =
      edge_gates.size(0) * node_values.size(1);
  if (work == 0) {
    return target_values;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_gated_scatter_uses_real_gates_cuda(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      carrier_gated_scatter_real_gates_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(work), threads, 0, stream>>>(
              node_values.data_ptr<c10::complex<float>>(),
              edge_gates.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              edge_gates.size(0),
              node_values.size(0),
              target_count,
              node_values.size(1),
              edge_gates.size(1),
              target_values.data_ptr<c10::complex<float>>());
    } else {
      carrier_gated_scatter_real_gates_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(work), threads, 0, stream>>>(
              node_values.data_ptr<c10::complex<double>>(),
              edge_gates.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              edge_gates.size(0),
              node_values.size(0),
              target_count,
              node_values.size(1),
              edge_gates.size(1),
              target_values.data_ptr<c10::complex<double>>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return target_values;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_gated_scatter_cuda",
      [&] {
        carrier_gated_scatter_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                node_values.data_ptr<scalar_t>(),
                edge_gates.data_ptr<scalar_t>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                edge_gates.size(0),
                node_values.size(0),
                target_count,
                node_values.size(1),
                edge_gates.size(1),
                target_values.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return target_values;
}

torch::Tensor carrier_gated_scatter_cuda(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    std::int64_t target_count) {
  return carrier_gated_scatter_cuda_impl(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_count,
      false);
}

torch::Tensor carrier_residual_gated_scatter_cuda(
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  return carrier_gated_scatter_cuda_impl(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      node_values.size(0),
      true);
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_gated_scatter_adjoint_cuda_impl(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    bool include_residual) {
  TORCH_CHECK(
      target_adjoint.dim() == 2 && target_adjoint.size(0) > 0,
      "target_adjoint must be two-dimensional with a target axis");
  check_carrier_gated_scatter_cuda_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_adjoint.size(0));
  TORCH_CHECK(
      target_adjoint.is_cuda() &&
          target_adjoint.is_contiguous() &&
          target_adjoint.device() == node_values.device() &&
          target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.size(1) == node_values.size(1),
      "target_adjoint must match node_values device, dtype, and width");
  c10::cuda::CUDAGuard device_guard(node_values.device());
  TORCH_CHECK(
      !include_residual ||
          target_adjoint.size(0) == node_values.size(0),
      "residual carrier scatter adjoint requires one target row per node");
  auto node_adjoint = include_residual
      ? target_adjoint.clone()
      : torch::zeros_like(node_values);
  auto gate_adjoint = torch::zeros_like(edge_gates);
  const std::int64_t work =
      edge_gates.size(0) * node_values.size(1);
  if (work == 0) {
    return std::make_tuple(node_adjoint, gate_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_gated_scatter_uses_real_gates_cuda(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      carrier_gated_scatter_real_gates_adjoint_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<float>>(),
              node_values.data_ptr<c10::complex<float>>(),
              edge_gates.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              edge_gates.size(0),
              node_values.size(0),
              target_adjoint.size(0),
              node_values.size(1),
              edge_gates.size(1),
              node_adjoint.data_ptr<c10::complex<float>>(),
              gate_adjoint.data_ptr<float>());
    } else {
      carrier_gated_scatter_real_gates_adjoint_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<double>>(),
              node_values.data_ptr<c10::complex<double>>(),
              edge_gates.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              edge_gates.size(0),
              node_values.size(0),
              target_adjoint.size(0),
              node_values.size(1),
              edge_gates.size(1),
              node_adjoint.data_ptr<c10::complex<double>>(),
              gate_adjoint.data_ptr<double>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(node_adjoint, gate_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_gated_scatter_adjoint_cuda",
      [&] {
        carrier_gated_scatter_adjoint_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                target_adjoint.data_ptr<scalar_t>(),
                node_values.data_ptr<scalar_t>(),
                edge_gates.data_ptr<scalar_t>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                edge_gates.size(0),
                node_values.size(0),
                target_adjoint.size(0),
                node_values.size(1),
                edge_gates.size(1),
                node_adjoint.data_ptr<scalar_t>(),
                gate_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(node_adjoint, gate_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_gated_scatter_adjoint_cuda(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  return carrier_gated_scatter_adjoint_cuda_impl(
      target_adjoint,
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      false);
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_residual_gated_scatter_adjoint_cuda(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  return carrier_gated_scatter_adjoint_cuda_impl(
      target_adjoint,
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      true);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_gated_scatter_double_backward_cuda_impl(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& node_adjoint_tangent,
    const torch::Tensor& gate_adjoint_tangent,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels,
    bool include_residual) {
  TORCH_CHECK(
      target_adjoint.dim() == 2 && target_adjoint.size(0) > 0,
      "target_adjoint must be two-dimensional with a target axis");
  check_carrier_gated_scatter_cuda_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      target_adjoint.size(0));
  TORCH_CHECK(
      target_adjoint.is_cuda() &&
          target_adjoint.is_contiguous() &&
          target_adjoint.device() == node_values.device() &&
          target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.size(1) == node_values.size(1),
      "target_adjoint must match node_values");
  TORCH_CHECK(
      node_adjoint_tangent.is_cuda() &&
          node_adjoint_tangent.is_contiguous() &&
          node_adjoint_tangent.device() == node_values.device() &&
          node_adjoint_tangent.scalar_type() ==
              node_values.scalar_type() &&
          node_adjoint_tangent.sizes() == node_values.sizes(),
      "node_adjoint_tangent must match node_values");
  TORCH_CHECK(
      gate_adjoint_tangent.is_cuda() &&
          gate_adjoint_tangent.is_contiguous() &&
          gate_adjoint_tangent.device() == edge_gates.device() &&
          gate_adjoint_tangent.scalar_type() ==
              edge_gates.scalar_type() &&
          gate_adjoint_tangent.sizes() == edge_gates.sizes(),
      "gate_adjoint_tangent must match edge_gates");
  c10::cuda::CUDAGuard device_guard(node_values.device());
  TORCH_CHECK(
      !include_residual ||
          target_adjoint.size(0) == node_values.size(0),
      "residual carrier scatter double backward requires one target row per node");
  auto target_adjoint_tangent = include_residual
      ? node_adjoint_tangent.clone()
      : torch::zeros_like(target_adjoint);
  auto node_second_adjoint = torch::zeros_like(node_values);
  auto gate_second_adjoint = torch::zeros_like(edge_gates);
  const std::int64_t work =
      edge_gates.size(0) * node_values.size(1);
  if (work == 0) {
    return std::make_tuple(
        target_adjoint_tangent,
        node_second_adjoint,
        gate_second_adjoint);
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_gated_scatter_uses_real_gates_cuda(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      carrier_gated_scatter_real_gates_double_backward_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<float>>(),
              node_values.data_ptr<c10::complex<float>>(),
              edge_gates.data_ptr<float>(),
              node_adjoint_tangent.data_ptr<c10::complex<float>>(),
              gate_adjoint_tangent.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              edge_gates.size(0),
              node_values.size(0),
              target_adjoint.size(0),
              node_values.size(1),
              edge_gates.size(1),
              target_adjoint_tangent.data_ptr<c10::complex<float>>(),
              node_second_adjoint.data_ptr<c10::complex<float>>(),
              gate_second_adjoint.data_ptr<float>());
    } else {
      carrier_gated_scatter_real_gates_double_backward_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<double>>(),
              node_values.data_ptr<c10::complex<double>>(),
              edge_gates.data_ptr<double>(),
              node_adjoint_tangent.data_ptr<c10::complex<double>>(),
              gate_adjoint_tangent.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              edge_targets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              edge_gates.size(0),
              node_values.size(0),
              target_adjoint.size(0),
              node_values.size(1),
              edge_gates.size(1),
              target_adjoint_tangent.data_ptr<c10::complex<double>>(),
              node_second_adjoint.data_ptr<c10::complex<double>>(),
              gate_second_adjoint.data_ptr<double>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(
        target_adjoint_tangent,
        node_second_adjoint,
        gate_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_gated_scatter_double_backward_cuda",
      [&] {
        carrier_gated_scatter_double_backward_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                target_adjoint.data_ptr<scalar_t>(),
                node_values.data_ptr<scalar_t>(),
                edge_gates.data_ptr<scalar_t>(),
                node_adjoint_tangent.data_ptr<scalar_t>(),
                gate_adjoint_tangent.data_ptr<scalar_t>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                edge_gates.size(0),
                node_values.size(0),
                target_adjoint.size(0),
                node_values.size(1),
                edge_gates.size(1),
                target_adjoint_tangent.data_ptr<scalar_t>(),
                node_second_adjoint.data_ptr<scalar_t>(),
                gate_second_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_gated_scatter_double_backward_cuda(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& node_adjoint_tangent,
    const torch::Tensor& gate_adjoint_tangent,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  return carrier_gated_scatter_double_backward_cuda_impl(
      target_adjoint,
      node_values,
      edge_gates,
      node_adjoint_tangent,
      gate_adjoint_tangent,
      edge_sources,
      edge_targets,
      feature_channels,
      false);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_residual_gated_scatter_double_backward_cuda(
    const torch::Tensor& target_adjoint,
    const torch::Tensor& node_values,
    const torch::Tensor& edge_gates,
    const torch::Tensor& node_adjoint_tangent,
    const torch::Tensor& gate_adjoint_tangent,
    const torch::Tensor& edge_sources,
    const torch::Tensor& edge_targets,
    const torch::Tensor& feature_channels) {
  return carrier_gated_scatter_double_backward_cuda_impl(
      target_adjoint,
      node_values,
      edge_gates,
      node_adjoint_tangent,
      gate_adjoint_tangent,
      edge_sources,
      edge_targets,
      feature_channels,
      true);
}

void check_carrier_segmented_scatter_cuda_inputs(
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
  check_carrier_gated_scatter_cuda_inputs(
      node_values,
      edge_gates,
      edge_sources,
      edge_targets,
      feature_channels,
      node_values.size(0));
  check_cuda_int64_vector(target_offsets, "target_offsets");
  check_cuda_int64_vector(source_offsets, "source_offsets");
  check_cuda_int64_vector(source_edges, "source_edges");
  check_cuda_int64_vector(channel_offsets, "channel_offsets");
  check_cuda_int64_vector(channel_features, "channel_features");
  TORCH_CHECK(
      target_offsets.device() == node_values.device() &&
          source_offsets.device() == node_values.device() &&
          source_edges.device() == node_values.device() &&
          channel_offsets.device() == node_values.device() &&
          channel_features.device() == node_values.device(),
      "carrier scatter segments must share node_values device");
  TORCH_CHECK(
      target_offsets.numel() == node_values.size(0) + 1 &&
          source_offsets.numel() == node_values.size(0) + 1,
      "target_offsets and source_offsets must contain node_count + 1 entries");
  TORCH_CHECK(
      source_edges.numel() == edge_gates.size(0),
      "source_edges must contain one entry per edge");
  TORCH_CHECK(
      channel_offsets.numel() == edge_gates.size(1) + 1 &&
          channel_features.numel() == node_values.size(1),
      "channel segments must cover every feature exactly once");
}

torch::Tensor carrier_segmented_residual_gated_scatter_cuda(
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
  check_carrier_segmented_scatter_cuda_inputs(
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
  c10::cuda::CUDAGuard device_guard(node_values.device());
  auto target_values = torch::empty_like(node_values);
  const std::int64_t work = node_values.numel();
  if (work == 0) {
    return target_values;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_gated_scatter_uses_real_gates_cuda(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      carrier_segmented_residual_scatter_forward_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(work), threads, 0, stream>>>(
              node_values.data_ptr<c10::complex<float>>(),
              edge_gates.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              node_values.size(1),
              edge_gates.size(1),
              target_values.data_ptr<c10::complex<float>>());
    } else {
      carrier_segmented_residual_scatter_forward_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(work), threads, 0, stream>>>(
              node_values.data_ptr<c10::complex<double>>(),
              edge_gates.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              node_values.size(1),
              edge_gates.size(1),
              target_values.data_ptr<c10::complex<double>>());
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return target_values;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_segmented_residual_gated_scatter_cuda",
      [&] {
        carrier_segmented_residual_scatter_forward_kernel<
            scalar_t, scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                node_values.data_ptr<scalar_t>(),
                edge_gates.data_ptr<scalar_t>(),
                edge_sources.data_ptr<std::int64_t>(),
                target_offsets.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                node_values.size(0),
                node_values.size(1),
                edge_gates.size(1),
                target_values.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return target_values;
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_segmented_residual_gated_scatter_adjoint_cuda(
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
  check_carrier_segmented_scatter_cuda_inputs(
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
  TORCH_CHECK(
      target_adjoint.is_cuda() &&
          target_adjoint.is_contiguous() &&
          target_adjoint.device() == node_values.device() &&
          target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.sizes() == node_values.sizes(),
      "target_adjoint must match node_values");
  c10::cuda::CUDAGuard device_guard(node_values.device());
  auto node_adjoint = torch::empty_like(node_values);
  auto gate_adjoint = torch::empty_like(edge_gates);
  const std::int64_t node_work = node_values.numel();
  const std::int64_t gate_work = edge_gates.numel();
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_gated_scatter_uses_real_gates_cuda(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      carrier_segmented_residual_scatter_node_adjoint_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(node_work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<float>>(),
              edge_gates.data_ptr<float>(),
              edge_targets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              node_adjoint.data_ptr<c10::complex<float>>());
      if (gate_work > 0) {
        carrier_segmented_residual_scatter_gate_adjoint_kernel<
            c10::complex<float>, float>
            <<<launch_blocks(gate_work), threads, 0, stream>>>(
                target_adjoint.data_ptr<c10::complex<float>>(),
                node_values.data_ptr<c10::complex<float>>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                channel_features.data_ptr<std::int64_t>(),
                node_values.size(0),
                edge_gates.size(0),
                node_values.size(1),
                edge_gates.size(1),
                gate_adjoint.data_ptr<float>());
      }
    } else {
      carrier_segmented_residual_scatter_node_adjoint_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(node_work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<double>>(),
              edge_gates.data_ptr<double>(),
              edge_targets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              node_adjoint.data_ptr<c10::complex<double>>());
      if (gate_work > 0) {
        carrier_segmented_residual_scatter_gate_adjoint_kernel<
            c10::complex<double>, double>
            <<<launch_blocks(gate_work), threads, 0, stream>>>(
                target_adjoint.data_ptr<c10::complex<double>>(),
                node_values.data_ptr<c10::complex<double>>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                channel_features.data_ptr<std::int64_t>(),
                node_values.size(0),
                edge_gates.size(0),
                node_values.size(1),
                edge_gates.size(1),
                gate_adjoint.data_ptr<double>());
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(node_adjoint, gate_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_segmented_residual_gated_scatter_adjoint_cuda",
      [&] {
        carrier_segmented_residual_scatter_node_adjoint_kernel<
            scalar_t, scalar_t>
            <<<launch_blocks(node_work), threads, 0, stream>>>(
                target_adjoint.data_ptr<scalar_t>(),
                edge_gates.data_ptr<scalar_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                source_offsets.data_ptr<std::int64_t>(),
                source_edges.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                node_values.size(0),
                edge_gates.size(0),
                node_values.size(1),
                edge_gates.size(1),
                node_adjoint.data_ptr<scalar_t>());
        if (gate_work > 0) {
          carrier_segmented_residual_scatter_gate_adjoint_kernel<
              scalar_t, scalar_t>
              <<<launch_blocks(gate_work), threads, 0, stream>>>(
                  target_adjoint.data_ptr<scalar_t>(),
                  node_values.data_ptr<scalar_t>(),
                  edge_sources.data_ptr<std::int64_t>(),
                  edge_targets.data_ptr<std::int64_t>(),
                  channel_offsets.data_ptr<std::int64_t>(),
                  channel_features.data_ptr<std::int64_t>(),
                  node_values.size(0),
                  edge_gates.size(0),
                  node_values.size(1),
                  edge_gates.size(1),
                  gate_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(node_adjoint, gate_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_segmented_residual_gated_scatter_double_backward_cuda(
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
  check_carrier_segmented_scatter_cuda_inputs(
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
  TORCH_CHECK(
      target_adjoint.is_cuda() && target_adjoint.is_contiguous() &&
          target_adjoint.device() == node_values.device() &&
          target_adjoint.scalar_type() == node_values.scalar_type() &&
          target_adjoint.sizes() == node_values.sizes() &&
          node_adjoint_tangent.is_cuda() &&
          node_adjoint_tangent.is_contiguous() &&
          node_adjoint_tangent.device() == node_values.device() &&
          node_adjoint_tangent.scalar_type() == node_values.scalar_type() &&
          node_adjoint_tangent.sizes() == node_values.sizes() &&
          gate_adjoint_tangent.is_cuda() &&
          gate_adjoint_tangent.is_contiguous() &&
          gate_adjoint_tangent.device() == edge_gates.device() &&
          gate_adjoint_tangent.scalar_type() == edge_gates.scalar_type() &&
          gate_adjoint_tangent.sizes() == edge_gates.sizes(),
      "segmented scatter adjoint tangents must match primals");
  c10::cuda::CUDAGuard device_guard(node_values.device());
  auto target_adjoint_tangent = torch::empty_like(target_adjoint);
  auto node_second_adjoint = torch::empty_like(node_values);
  auto gate_second_adjoint = torch::empty_like(edge_gates);
  const std::int64_t node_work = node_values.numel();
  const std::int64_t gate_work = edge_gates.numel();
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_gated_scatter_uses_real_gates_cuda(
          node_values, edge_gates)) {
    if (node_values.scalar_type() == torch::kComplexFloat) {
      carrier_segmented_residual_scatter_target_tangent_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(node_work), threads, 0, stream>>>(
              node_values.data_ptr<c10::complex<float>>(),
              edge_gates.data_ptr<float>(),
              node_adjoint_tangent.data_ptr<c10::complex<float>>(),
              gate_adjoint_tangent.data_ptr<float>(),
              edge_sources.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              node_values.size(1),
              edge_gates.size(1),
              target_adjoint_tangent.data_ptr<c10::complex<float>>());
      carrier_segmented_residual_scatter_node_second_kernel<
          c10::complex<float>, float>
          <<<launch_blocks(node_work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<float>>(),
              gate_adjoint_tangent.data_ptr<float>(),
              edge_targets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              node_second_adjoint.data_ptr<c10::complex<float>>());
      if (gate_work > 0) {
        carrier_segmented_residual_scatter_gate_second_kernel<
            c10::complex<float>, float>
            <<<launch_blocks(gate_work), threads, 0, stream>>>(
                target_adjoint.data_ptr<c10::complex<float>>(),
                node_adjoint_tangent.data_ptr<c10::complex<float>>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                channel_features.data_ptr<std::int64_t>(),
                node_values.size(0),
                edge_gates.size(0),
                node_values.size(1),
                edge_gates.size(1),
                gate_second_adjoint.data_ptr<float>());
      }
    } else {
      carrier_segmented_residual_scatter_target_tangent_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(node_work), threads, 0, stream>>>(
              node_values.data_ptr<c10::complex<double>>(),
              edge_gates.data_ptr<double>(),
              node_adjoint_tangent.data_ptr<c10::complex<double>>(),
              gate_adjoint_tangent.data_ptr<double>(),
              edge_sources.data_ptr<std::int64_t>(),
              target_offsets.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              node_values.size(1),
              edge_gates.size(1),
              target_adjoint_tangent.data_ptr<c10::complex<double>>());
      carrier_segmented_residual_scatter_node_second_kernel<
          c10::complex<double>, double>
          <<<launch_blocks(node_work), threads, 0, stream>>>(
              target_adjoint.data_ptr<c10::complex<double>>(),
              gate_adjoint_tangent.data_ptr<double>(),
              edge_targets.data_ptr<std::int64_t>(),
              source_offsets.data_ptr<std::int64_t>(),
              source_edges.data_ptr<std::int64_t>(),
              feature_channels.data_ptr<std::int64_t>(),
              node_values.size(0),
              edge_gates.size(0),
              node_values.size(1),
              edge_gates.size(1),
              node_second_adjoint.data_ptr<c10::complex<double>>());
      if (gate_work > 0) {
        carrier_segmented_residual_scatter_gate_second_kernel<
            c10::complex<double>, double>
            <<<launch_blocks(gate_work), threads, 0, stream>>>(
                target_adjoint.data_ptr<c10::complex<double>>(),
                node_adjoint_tangent.data_ptr<c10::complex<double>>(),
                edge_sources.data_ptr<std::int64_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                channel_features.data_ptr<std::int64_t>(),
                node_values.size(0),
                edge_gates.size(0),
                node_values.size(1),
                edge_gates.size(1),
                gate_second_adjoint.data_ptr<double>());
      }
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(
        target_adjoint_tangent,
        node_second_adjoint,
        gate_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      node_values.scalar_type(),
      "ye3t_carrier_segmented_residual_gated_scatter_double_backward_cuda",
      [&] {
        carrier_segmented_residual_scatter_target_tangent_kernel<
            scalar_t, scalar_t>
            <<<launch_blocks(node_work), threads, 0, stream>>>(
                node_values.data_ptr<scalar_t>(),
                edge_gates.data_ptr<scalar_t>(),
                node_adjoint_tangent.data_ptr<scalar_t>(),
                gate_adjoint_tangent.data_ptr<scalar_t>(),
                edge_sources.data_ptr<std::int64_t>(),
                target_offsets.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                node_values.size(0),
                node_values.size(1),
                edge_gates.size(1),
                target_adjoint_tangent.data_ptr<scalar_t>());
        carrier_segmented_residual_scatter_node_second_kernel<
            scalar_t, scalar_t>
            <<<launch_blocks(node_work), threads, 0, stream>>>(
                target_adjoint.data_ptr<scalar_t>(),
                gate_adjoint_tangent.data_ptr<scalar_t>(),
                edge_targets.data_ptr<std::int64_t>(),
                source_offsets.data_ptr<std::int64_t>(),
                source_edges.data_ptr<std::int64_t>(),
                feature_channels.data_ptr<std::int64_t>(),
                node_values.size(0),
                edge_gates.size(0),
                node_values.size(1),
                edge_gates.size(1),
                node_second_adjoint.data_ptr<scalar_t>());
        if (gate_work > 0) {
          carrier_segmented_residual_scatter_gate_second_kernel<
              scalar_t, scalar_t>
              <<<launch_blocks(gate_work), threads, 0, stream>>>(
                  target_adjoint.data_ptr<scalar_t>(),
                  node_adjoint_tangent.data_ptr<scalar_t>(),
                  edge_sources.data_ptr<std::int64_t>(),
                  edge_targets.data_ptr<std::int64_t>(),
                  channel_offsets.data_ptr<std::int64_t>(),
                  channel_features.data_ptr<std::int64_t>(),
                  node_values.size(0),
                  edge_gates.size(0),
                  node_values.size(1),
                  edge_gates.size(1),
                  gate_second_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      target_adjoint_tangent,
      node_second_adjoint,
      gate_second_adjoint);
}

void check_source_arena_schedule_cuda(
    const torch::Tensor& data,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    std::int64_t producer_width,
    std::int64_t output_width) {
  check_cuda_data_tensor(data, "source arena data");
  check_cuda_int64_vector(gather_indices, "gather_indices");
  check_cuda_int64_vector(reverse_offsets, "reverse_offsets");
  check_cuda_int64_vector(reverse_output_indices, "reverse_output_indices");
  check_cuda_int64_vector(center_types, "center_types");
  check_cuda_int64_vector(atom_types, "atom_types");
  TORCH_CHECK(producer_width > 0, "source arena producer width must be positive");
  TORCH_CHECK(output_width > 0, "source arena output width must be positive");
  TORCH_CHECK(
      gather_indices.device() == data.device() &&
          reverse_offsets.device() == data.device() &&
          reverse_output_indices.device() == data.device() &&
          center_types.device() == data.device() &&
          atom_types.device() == data.device(),
      "source arena schedule tensors must share the data device");
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
      atom_types.numel() == data.size(0),
      "atom_types must contain one value per source arena row");
}

torch::Tensor source_arena_gather_cuda(
    const torch::Tensor& producer,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types) {
  const std::int64_t output_width = gather_indices.numel();
  check_source_arena_schedule_cuda(
      producer,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      producer.size(1),
      output_width);
  c10::cuda::CUDAGuard device_guard(producer.device());
  auto output = torch::empty(
      {producer.size(0), output_width}, producer.options());
  if (output.numel() == 0) {
    return output;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      producer.scalar_type(),
      "ye3t_source_arena_gather_cuda",
      [&] {
        source_arena_gather_kernel<scalar_t>
            <<<launch_blocks(output.numel()), threads, 0, stream>>>(
                producer.data_ptr<scalar_t>(),
                gather_indices.data_ptr<std::int64_t>(),
                center_types.data_ptr<std::int64_t>(),
                atom_types.data_ptr<std::int64_t>(),
                producer.size(0),
                producer.size(1),
                output_width,
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

torch::Tensor source_arena_gather_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types) {
  const std::int64_t producer_width = reverse_offsets.numel() - 1;
  check_source_arena_schedule_cuda(
      output_adjoint,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      producer_width,
      output_adjoint.size(1));
  c10::cuda::CUDAGuard device_guard(output_adjoint.device());
  auto producer_adjoint = torch::empty(
      {output_adjoint.size(0), producer_width},
      output_adjoint.options());
  if (producer_adjoint.numel() == 0) {
    return producer_adjoint;
  }
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      output_adjoint.scalar_type(),
      "ye3t_source_arena_gather_adjoint_cuda",
      [&] {
        source_arena_gather_adjoint_kernel<scalar_t>
            <<<launch_blocks(producer_adjoint.numel()),
               threads,
               0,
               stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                reverse_offsets.data_ptr<std::int64_t>(),
                reverse_output_indices.data_ptr<std::int64_t>(),
                center_types.data_ptr<std::int64_t>(),
                atom_types.data_ptr<std::int64_t>(),
                output_adjoint.size(0),
                producer_width,
                output_adjoint.size(1),
                producer_adjoint.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return producer_adjoint;
}

torch::Tensor source_arena_gather_double_backward_cuda(
    const torch::Tensor& producer_adjoint_tangent,
    const torch::Tensor& gather_indices,
    const torch::Tensor& reverse_offsets,
    const torch::Tensor& reverse_output_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types) {
  return source_arena_gather_cuda(
      producer_adjoint_tangent,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types);
}

void check_source_arena_channel_transform_cuda_inputs(
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
  check_source_arena_schedule_cuda(
      producer,
      gather_indices,
      reverse_offsets,
      reverse_output_indices,
      center_types,
      atom_types,
      producer.size(1),
      gather_indices.numel());
  TORCH_CHECK(
      channel_maps.is_cuda() && channel_maps.is_contiguous() &&
          channel_maps.dim() == 1 &&
          channel_maps.device() == producer.device(),
      "channel_maps must be a contiguous CUDA vector on producer device");
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
  check_cuda_int64_vector(input_feature_offsets, "input_feature_offsets");
  check_cuda_int64_vector(output_feature_offsets, "output_feature_offsets");
  check_cuda_int64_vector(input_channel_offsets, "input_channel_offsets");
  check_cuda_int64_vector(output_channel_offsets, "output_channel_offsets");
  check_cuda_int64_vector(map_offsets, "map_offsets");
  const std::int64_t offset_count = input_feature_offsets.numel();
  TORCH_CHECK(
      offset_count >= 2 &&
          output_feature_offsets.numel() == offset_count &&
          input_channel_offsets.numel() == offset_count &&
          output_channel_offsets.numel() == offset_count &&
          map_offsets.numel() == offset_count,
      "source-arena channel-transform offsets must share one block_count + "
      "1 length");
  TORCH_CHECK(
      input_feature_offsets.device() == producer.device() &&
          output_feature_offsets.device() == producer.device() &&
          input_channel_offsets.device() == producer.device() &&
          output_channel_offsets.device() == producer.device() &&
          map_offsets.device() == producer.device(),
      "source-arena channel-transform offsets must share producer device");
}

template <typename Scalar, typename Control>
void launch_source_arena_channel_transform_cuda(
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& gather_indices,
    const torch::Tensor& center_types,
    const torch::Tensor& atom_types,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& output,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  source_arena_channel_transform_kernel<Scalar, Control>
      <<<launch_blocks(output.numel()), threads, 0, stream>>>(
          producer.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
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
          output.size(1),
          input_feature_offsets.numel() - 1,
          output.data_ptr<Scalar>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor source_arena_channel_transform_cuda(
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
    std::int64_t output_width) {
  check_source_arena_channel_transform_cuda_inputs(
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
  TORCH_CHECK(
      output_width > 0,
      "source-arena channel-transform output width must be positive");
  c10::cuda::CUDAGuard device_guard(producer.device());
  auto output = torch::empty(
      {producer.size(0), output_width}, producer.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (producer.scalar_type() == torch::kComplexFloat &&
      channel_maps.scalar_type() == torch::kFloat) {
    launch_source_arena_channel_transform_cuda<c10::complex<float>, float>(
        producer,
        channel_maps,
        gather_indices,
        center_types,
        atom_types,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
        output,
        stream);
  } else if (
      producer.scalar_type() == torch::kComplexDouble &&
      channel_maps.scalar_type() == torch::kDouble) {
    launch_source_arena_channel_transform_cuda<c10::complex<double>, double>(
        producer,
        channel_maps,
        gather_indices,
        center_types,
        atom_types,
        input_feature_offsets,
        output_feature_offsets,
        input_channel_offsets,
        output_channel_offsets,
        map_offsets,
        output,
        stream);
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        producer.scalar_type(),
        "ye3t_source_arena_channel_transform_cuda",
        [&] {
          launch_source_arena_channel_transform_cuda<scalar_t, scalar_t>(
              producer,
              channel_maps,
              gather_indices,
              center_types,
              atom_types,
              input_feature_offsets,
              output_feature_offsets,
              input_channel_offsets,
              output_channel_offsets,
              map_offsets,
              output,
              stream);
        });
  }
  return output;
}

template <typename Scalar, typename Control>
void launch_source_arena_channel_transform_adjoint_cuda(
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
    const torch::Tensor& map_offsets,
    torch::Tensor& producer_adjoint,
    torch::Tensor& maps_adjoint,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  source_arena_channel_transform_producer_adjoint_kernel<Scalar, Control>
      <<<launch_blocks(producer_adjoint.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          reverse_offsets.data_ptr<std::int64_t>(),
          reverse_output_indices.data_ptr<std::int64_t>(),
          center_types.data_ptr<std::int64_t>(),
          atom_types.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          producer.size(0),
          producer.size(1),
          output_adjoint.size(1),
          input_feature_offsets.numel() - 1,
          producer_adjoint.data_ptr<Scalar>());
  if (maps_adjoint.numel() > 0) {
    source_arena_channel_transform_maps_adjoint_kernel<Scalar, Control>
        <<<launch_blocks(maps_adjoint.numel()), threads, 0, stream>>>(
            output_adjoint.data_ptr<Scalar>(),
            producer.data_ptr<Scalar>(),
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
            output_adjoint.size(1),
            maps_adjoint.numel(),
            input_feature_offsets.numel() - 1,
            maps_adjoint.data_ptr<Control>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor>
source_arena_channel_transform_adjoint_cuda(
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
  check_source_arena_channel_transform_cuda_inputs(
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
  TORCH_CHECK(
      output_adjoint.is_cuda() && output_adjoint.is_contiguous() &&
          output_adjoint.dim() == 2 &&
          output_adjoint.device() == producer.device() &&
          output_adjoint.scalar_type() == producer.scalar_type() &&
          output_adjoint.size(0) == producer.size(0),
      "output_adjoint must match the fused source-arena output");
  c10::cuda::CUDAGuard device_guard(producer.device());
  auto producer_adjoint = torch::empty_like(producer);
  auto maps_adjoint = torch::empty_like(channel_maps);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_ADJOINT(Scalar, Control) \
  launch_source_arena_channel_transform_adjoint_cuda<Scalar, Control>( \
      output_adjoint, producer, channel_maps, gather_indices, \
      reverse_offsets, reverse_output_indices, center_types, atom_types, \
      input_feature_offsets, output_feature_offsets, input_channel_offsets, \
      output_channel_offsets, map_offsets, producer_adjoint, maps_adjoint, \
      stream)
  if (producer.scalar_type() == torch::kComplexFloat &&
      channel_maps.scalar_type() == torch::kFloat) {
    YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_ADJOINT(
        c10::complex<float>, float);
  } else if (
      producer.scalar_type() == torch::kComplexDouble &&
      channel_maps.scalar_type() == torch::kDouble) {
    YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_ADJOINT(
        c10::complex<double>, double);
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        producer.scalar_type(),
        "ye3t_source_arena_channel_transform_adjoint_cuda",
        [&] {
          YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_ADJOINT(scalar_t, scalar_t);
        });
  }
#undef YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_ADJOINT
  return std::make_tuple(producer_adjoint, maps_adjoint);
}

template <typename Scalar, typename Control>
void launch_source_arena_channel_transform_double_backward_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& producer_adjoint_tangent,
    const torch::Tensor& maps_adjoint_tangent,
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
    torch::Tensor& output_tangent,
    torch::Tensor& producer_second,
    torch::Tensor& maps_second,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  source_arena_channel_transform_output_tangent_kernel<Scalar, Control>
      <<<launch_blocks(output_tangent.numel()), threads, 0, stream>>>(
          producer.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          producer_adjoint_tangent.data_ptr<Scalar>(),
          maps_adjoint_tangent.data_ptr<Control>(),
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
          output_adjoint.size(1),
          input_feature_offsets.numel() - 1,
          output_tangent.data_ptr<Scalar>());
  source_arena_channel_transform_producer_second_kernel<Scalar, Control>
      <<<launch_blocks(producer_second.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          maps_adjoint_tangent.data_ptr<Control>(),
          reverse_offsets.data_ptr<std::int64_t>(),
          reverse_output_indices.data_ptr<std::int64_t>(),
          center_types.data_ptr<std::int64_t>(),
          atom_types.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          producer.size(0),
          producer.size(1),
          output_adjoint.size(1),
          input_feature_offsets.numel() - 1,
          producer_second.data_ptr<Scalar>());
  if (maps_second.numel() > 0) {
    source_arena_channel_transform_maps_second_kernel<Scalar, Control>
        <<<launch_blocks(maps_second.numel()), threads, 0, stream>>>(
            output_adjoint.data_ptr<Scalar>(),
            producer_adjoint_tangent.data_ptr<Scalar>(),
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
            output_adjoint.size(1),
            maps_second.numel(),
            input_feature_offsets.numel() - 1,
            maps_second.data_ptr<Control>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
source_arena_channel_transform_double_backward_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& producer,
    const torch::Tensor& channel_maps,
    const torch::Tensor& producer_adjoint_tangent,
    const torch::Tensor& maps_adjoint_tangent,
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
  check_source_arena_channel_transform_cuda_inputs(
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
  TORCH_CHECK(
      output_adjoint.is_cuda() && output_adjoint.is_contiguous() &&
          output_adjoint.dim() == 2 &&
          output_adjoint.device() == producer.device() &&
          output_adjoint.scalar_type() == producer.scalar_type() &&
          output_adjoint.size(0) == producer.size(0) &&
          producer_adjoint_tangent.sizes() == producer.sizes() &&
          producer_adjoint_tangent.scalar_type() == producer.scalar_type() &&
          maps_adjoint_tangent.sizes() == channel_maps.sizes() &&
          maps_adjoint_tangent.scalar_type() == channel_maps.scalar_type(),
      "source-arena channel-transform double-adjoint tangents have "
      "incompatible shapes or dtypes");
  c10::cuda::CUDAGuard device_guard(producer.device());
  auto output_tangent = torch::empty_like(output_adjoint);
  auto producer_second = torch::empty_like(producer);
  auto maps_second = torch::empty_like(channel_maps);
  const auto stream = at::cuda::getCurrentCUDAStream();
#define YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_DOUBLE(Scalar, Control) \
  launch_source_arena_channel_transform_double_backward_cuda< \
      Scalar, Control>( \
      output_adjoint, producer, channel_maps, producer_adjoint_tangent, \
      maps_adjoint_tangent, gather_indices, reverse_offsets, \
      reverse_output_indices, center_types, atom_types, \
      input_feature_offsets, output_feature_offsets, input_channel_offsets, \
      output_channel_offsets, map_offsets, output_tangent, \
      producer_second, maps_second, stream)
  if (producer.scalar_type() == torch::kComplexFloat &&
      channel_maps.scalar_type() == torch::kFloat) {
    YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_DOUBLE(
        c10::complex<float>, float);
  } else if (
      producer.scalar_type() == torch::kComplexDouble &&
      channel_maps.scalar_type() == torch::kDouble) {
    YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_DOUBLE(
        c10::complex<double>, double);
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        producer.scalar_type(),
        "ye3t_source_arena_channel_transform_double_backward_cuda",
        [&] {
          YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_DOUBLE(scalar_t, scalar_t);
        });
  }
#undef YE3T_LAUNCH_SOURCE_ARENA_CHANNEL_DOUBLE
  return std::make_tuple(output_tangent, producer_second, maps_second);
}

void check_carrier_channel_transform_cuda_inputs(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_cuda_data_tensor(values, "values");
  TORCH_CHECK(
      channel_maps.device() == values.device() &&
          channel_maps.is_contiguous() && channel_maps.dim() == 1,
      "channel_maps must be a contiguous CUDA vector on values device");
  check_cuda_int64_vector(input_feature_offsets, "input_feature_offsets");
  check_cuda_int64_vector(output_feature_offsets, "output_feature_offsets");
  check_cuda_int64_vector(input_channel_offsets, "input_channel_offsets");
  check_cuda_int64_vector(output_channel_offsets, "output_channel_offsets");
  check_cuda_int64_vector(map_offsets, "map_offsets");
  const bool real_maps_for_complex_values =
      (values.scalar_type() == torch::kComplexFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (values.scalar_type() == torch::kComplexDouble &&
       channel_maps.scalar_type() == torch::kDouble);
  TORCH_CHECK(
      channel_maps.scalar_type() == values.scalar_type() ||
          real_maps_for_complex_values,
      "channel_maps must match values dtype or use its real component dtype");
  TORCH_CHECK(
      input_feature_offsets.device() == values.device() &&
          output_feature_offsets.device() == values.device() &&
          input_channel_offsets.device() == values.device() &&
          output_channel_offsets.device() == values.device() &&
          map_offsets.device() == values.device(),
      "channel-transform offsets must share values device");
  const auto offset_count = input_feature_offsets.numel();
  TORCH_CHECK(
      offset_count >= 2 &&
          output_feature_offsets.numel() == offset_count &&
          input_channel_offsets.numel() == offset_count &&
          output_channel_offsets.numel() == offset_count &&
          map_offsets.numel() == offset_count,
      "channel-transform offsets must share one block_count + 1 length");
  TORCH_CHECK(
      values.size(1) > 0 && channel_maps.numel() > 0,
      "channel-transform packed dimensions must be positive");
}

bool carrier_channel_transform_uses_real_maps_cuda(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps) {
  return
      (values.scalar_type() == torch::kComplexFloat &&
       channel_maps.scalar_type() == torch::kFloat) ||
      (values.scalar_type() == torch::kComplexDouble &&
       channel_maps.scalar_type() == torch::kDouble);
}

void check_carrier_role_channel_map_adjoint_cuda_inputs(
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
  check_carrier_channel_transform_cuda_inputs(
      edge_values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_cuda_data_tensor(role_weights, "role_weights");
  check_cuda_int64_vector(atom_centers, "atom_centers");
  TORCH_CHECK(
      atomic_output_adjoint.is_cuda() &&
          atomic_output_adjoint.is_contiguous() &&
          atomic_output_adjoint.dim() == 3 &&
          atomic_output_adjoint.device() == edge_values.device() &&
          atomic_output_adjoint.scalar_type() == edge_values.scalar_type(),
      "atomic_output_adjoint must be a contiguous three-dimensional CUDA "
      "tensor matching edge_values");
  const auto expected_real_type =
      edge_values.scalar_type() == torch::kComplexFloat
      ? torch::kFloat
      : edge_values.scalar_type() == torch::kComplexDouble
      ? torch::kDouble
      : edge_values.scalar_type();
  TORCH_CHECK(
      role_weights.device() == edge_values.device() &&
          role_weights.scalar_type() == expected_real_type,
      "role_weights must share the edge device and use its real component "
      "dtype");
  TORCH_CHECK(
      atom_centers.device() == edge_values.device() &&
          role_weights.size(0) == edge_values.size(0) &&
          atom_centers.numel() == edge_values.size(0),
      "role_weights and atom_centers must match the edge count");
  TORCH_CHECK(
      atomic_output_adjoint.size(0) > 0 &&
          role_weights.size(1) > 0 &&
          atomic_output_adjoint.size(1) == role_weights.size(1) &&
          atomic_output_adjoint.size(2) > 0,
      "atomic_output_adjoint and role_weights must share positive role and "
      "carrier axes");
}

template <typename Scalar, typename Control>
void launch_carrier_channel_transform_cuda(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& output,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_channel_transform_kernel<Scalar, Control>
      <<<launch_blocks(output.numel()), threads, 0, stream>>>(
          values.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          output.size(1),
          input_feature_offsets.numel() - 1,
          output.data_ptr<Scalar>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar, typename Control>
void launch_carrier_channel_transform_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& values_adjoint,
    torch::Tensor& maps_adjoint,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_channel_transform_values_adjoint_kernel<Scalar, Control>
      <<<launch_blocks(values_adjoint.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          output_adjoint.size(1),
          input_feature_offsets.numel() - 1,
          values_adjoint.data_ptr<Scalar>());
  carrier_channel_transform_maps_adjoint_kernel<Scalar, Control>
      <<<launch_blocks(maps_adjoint.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          values.data_ptr<Scalar>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          output_adjoint.size(1),
          maps_adjoint.numel(),
          input_feature_offsets.numel() - 1,
          maps_adjoint.data_ptr<Control>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar, typename Control>
void launch_carrier_channel_transform_double_backward_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& values_adjoint_tangent,
    const torch::Tensor& maps_adjoint_tangent,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& output_tangent,
    torch::Tensor& values_second,
    torch::Tensor& maps_second,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_channel_transform_output_tangent_kernel<Scalar, Control>
      <<<launch_blocks(output_tangent.numel()), threads, 0, stream>>>(
          values.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          values_adjoint_tangent.data_ptr<Scalar>(),
          maps_adjoint_tangent.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          output_adjoint.size(1),
          input_feature_offsets.numel() - 1,
          output_tangent.data_ptr<Scalar>());
  carrier_channel_transform_values_second_kernel<Scalar, Control>
      <<<launch_blocks(values_second.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          maps_adjoint_tangent.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          output_adjoint.size(1),
          input_feature_offsets.numel() - 1,
          values_second.data_ptr<Scalar>());
  carrier_channel_transform_maps_second_kernel<Scalar, Control>
      <<<launch_blocks(maps_second.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          values_adjoint_tangent.data_ptr<Scalar>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          output_adjoint.size(1),
          maps_second.numel(),
          input_feature_offsets.numel() - 1,
          maps_second.data_ptr<Control>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar, typename Control, typename Real>
void launch_carrier_role_channel_map_adjoint_cuda(
    const torch::Tensor& edge_values,
    const torch::Tensor& role_weights,
    const torch::Tensor& atomic_output_adjoint,
    const torch::Tensor& atom_centers,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& maps_adjoint,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  if (edge_values.size(0) * role_weights.size(1) >= 16384) {
    carrier_role_channel_maps_adjoint_parallel_kernel<Scalar, Control, Real>
        <<<maps_adjoint.numel(),
           threads,
           threads * sizeof(Scalar),
           stream>>>(
            edge_values.data_ptr<Scalar>(),
            role_weights.data_ptr<Real>(),
            atomic_output_adjoint.data_ptr<Scalar>(),
            atom_centers.data_ptr<std::int64_t>(),
            input_feature_offsets.data_ptr<std::int64_t>(),
            output_feature_offsets.data_ptr<std::int64_t>(),
            input_channel_offsets.data_ptr<std::int64_t>(),
            output_channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            edge_values.size(0),
            atomic_output_adjoint.size(0),
            role_weights.size(1),
            edge_values.size(1),
            atomic_output_adjoint.size(2),
            maps_adjoint.numel(),
            input_feature_offsets.numel() - 1,
            maps_adjoint.data_ptr<Control>());
  } else {
    carrier_role_channel_maps_adjoint_kernel<Scalar, Control, Real>
        <<<launch_blocks(maps_adjoint.numel()), threads, 0, stream>>>(
          edge_values.data_ptr<Scalar>(),
          role_weights.data_ptr<Real>(),
          atomic_output_adjoint.data_ptr<Scalar>(),
          atom_centers.data_ptr<std::int64_t>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          edge_values.size(1),
          atomic_output_adjoint.size(2),
          maps_adjoint.numel(),
          input_feature_offsets.numel() - 1,
          maps_adjoint.data_ptr<Control>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar, typename Control, typename Real>
void launch_carrier_role_channel_map_adjoint_double_backward_cuda(
    const torch::Tensor& edge_values,
    const torch::Tensor& role_weights,
    const torch::Tensor& atomic_output_adjoint,
    const torch::Tensor& atom_centers,
    const torch::Tensor& maps_adjoint_tangent,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& edge_values_second,
    torch::Tensor& role_weights_second,
    torch::Tensor& atomic_output_tangent,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_role_channel_edge_second_kernel<Scalar, Control, Real>
      <<<launch_blocks(edge_values_second.numel()), threads, 0, stream>>>(
          role_weights.data_ptr<Real>(),
          atomic_output_adjoint.data_ptr<Scalar>(),
          atom_centers.data_ptr<std::int64_t>(),
          maps_adjoint_tangent.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          edge_values.size(1),
          atomic_output_adjoint.size(2),
          input_feature_offsets.numel() - 1,
          edge_values_second.data_ptr<Scalar>());
  carrier_role_channel_role_second_kernel<Scalar, Control, Real>
      <<<launch_blocks(role_weights_second.numel()), threads, 0, stream>>>(
          edge_values.data_ptr<Scalar>(),
          atomic_output_adjoint.data_ptr<Scalar>(),
          atom_centers.data_ptr<std::int64_t>(),
          maps_adjoint_tangent.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          edge_values.size(1),
          atomic_output_adjoint.size(2),
          input_feature_offsets.numel() - 1,
          role_weights_second.data_ptr<Real>());
  carrier_role_channel_output_tangent_kernel<Scalar, Control, Real>
      <<<launch_blocks(
             edge_values.size(0) * role_weights.size(1) *
             atomic_output_adjoint.size(2)),
         threads,
         0,
         stream>>>(
          edge_values.data_ptr<Scalar>(),
          role_weights.data_ptr<Real>(),
          atom_centers.data_ptr<std::int64_t>(),
          maps_adjoint_tangent.data_ptr<Control>(),
          input_feature_offsets.data_ptr<std::int64_t>(),
          output_feature_offsets.data_ptr<std::int64_t>(),
          input_channel_offsets.data_ptr<std::int64_t>(),
          output_channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          edge_values.size(0),
          atomic_output_adjoint.size(0),
          role_weights.size(1),
          edge_values.size(1),
          atomic_output_adjoint.size(2),
          input_feature_offsets.numel() - 1,
          atomic_output_tangent.data_ptr<Scalar>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void check_carrier_channel_update_cuda_inputs(
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_cuda_data_tensor(values, "values");
  check_cuda_data_tensor(gates, "gates");
  TORCH_CHECK(
      channel_maps.device() == values.device() &&
          channel_maps.is_contiguous() &&
          channel_maps.dim() == 1,
      "channel_maps must be a contiguous CUDA vector on values device");
  check_cuda_int64_vector(feature_offsets, "feature_offsets");
  check_cuda_int64_vector(channel_offsets, "channel_offsets");
  check_cuda_int64_vector(map_offsets, "map_offsets");
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
      gates.device() == values.device() &&
          channel_maps.device() == values.device() &&
          ((gates.scalar_type() == values.scalar_type() &&
            channel_maps.scalar_type() == values.scalar_type()) ||
           real_controls_for_complex_values),
      "gates and channel_maps must share values device and either match "
      "its dtype or both use its real component dtype");
  TORCH_CHECK(
      feature_offsets.device() == values.device() &&
          channel_offsets.device() == values.device() &&
          map_offsets.device() == values.device(),
      "carrier update offsets must share values device");
  TORCH_CHECK(
      feature_offsets.numel() >= 2 &&
          feature_offsets.numel() == channel_offsets.numel() &&
          feature_offsets.numel() == map_offsets.numel(),
      "carrier update offsets must have one common block_count + 1 length");
  TORCH_CHECK(
      values.size(1) > 0 &&
          gates.size(1) > 0 &&
          channel_maps.numel() > 0,
      "carrier update packed dimensions must be positive");
}

bool carrier_channel_update_uses_real_controls_cuda(
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

template <typename Scalar, typename Control>
void launch_carrier_channel_update_real_controls_cuda(
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& output,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_channel_update_kernel<Scalar, Control>
      <<<launch_blocks(values.numel()), threads, 0, stream>>>(
          values.data_ptr<Scalar>(),
          gates.data_ptr<Control>(),
          channel_maps.data_ptr<Control>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          gates.size(1),
          feature_offsets.numel() - 1,
          output.data_ptr<Scalar>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar, typename Control>
void launch_carrier_channel_update_real_controls_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& values_adjoint,
    torch::Tensor& gates_adjoint,
    torch::Tensor& channel_maps_adjoint,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_channel_values_adjoint_kernel<Scalar, Control>
      <<<launch_blocks(values.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          gates.data_ptr<Control>(),
          channel_maps.data_ptr<Control>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          gates.size(1),
          feature_offsets.numel() - 1,
          values_adjoint.data_ptr<Scalar>());
  carrier_channel_gates_adjoint_kernel<Scalar, Control>
      <<<launch_blocks(gates.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          values.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          gates.size(1),
          feature_offsets.numel() - 1,
          gates_adjoint.data_ptr<Control>());
  if (values.size(0) >= 32) {
    carrier_channel_maps_adjoint_parallel_kernel<Scalar, Control>
        <<<channel_maps.numel(),
           threads,
           threads * sizeof(Scalar),
           stream>>>(
            output_adjoint.data_ptr<Scalar>(),
            values.data_ptr<Scalar>(),
            gates.data_ptr<Control>(),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            values.size(1),
            gates.size(1),
            channel_maps.numel(),
            feature_offsets.numel() - 1,
            channel_maps_adjoint.data_ptr<Control>());
  } else {
    carrier_channel_maps_adjoint_kernel<Scalar, Control>
        <<<launch_blocks(channel_maps.numel()),
           threads,
           0,
           stream>>>(
            output_adjoint.data_ptr<Scalar>(),
            values.data_ptr<Scalar>(),
            gates.data_ptr<Control>(),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            values.size(1),
            gates.size(1),
            channel_maps.numel(),
            feature_offsets.numel() - 1,
            channel_maps_adjoint.data_ptr<Control>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename Scalar, typename Control>
void launch_carrier_channel_update_real_controls_double_backward_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& values_adjoint_tangent,
    const torch::Tensor& gates_adjoint_tangent,
    const torch::Tensor& channel_maps_adjoint_tangent,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets,
    torch::Tensor& output_adjoint_tangent,
    torch::Tensor& values_second_adjoint,
    torch::Tensor& gates_second_adjoint,
    torch::Tensor& channel_maps_second_adjoint,
    const at::cuda::CUDAStream& stream) {
  constexpr int threads = 256;
  carrier_channel_output_tangent_kernel<Scalar, Control>
      <<<launch_blocks(values.numel()), threads, 0, stream>>>(
          values.data_ptr<Scalar>(),
          gates.data_ptr<Control>(),
          channel_maps.data_ptr<Control>(),
          values_adjoint_tangent.data_ptr<Scalar>(),
          gates_adjoint_tangent.data_ptr<Control>(),
          channel_maps_adjoint_tangent.data_ptr<Control>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          gates.size(1),
          feature_offsets.numel() - 1,
          output_adjoint_tangent.data_ptr<Scalar>());
  carrier_channel_values_second_kernel<Scalar, Control>
      <<<launch_blocks(values.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          values.data_ptr<Scalar>(),
          gates.data_ptr<Control>(),
          channel_maps.data_ptr<Control>(),
          gates_adjoint_tangent.data_ptr<Control>(),
          channel_maps_adjoint_tangent.data_ptr<Control>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          gates.size(1),
          feature_offsets.numel() - 1,
          values_second_adjoint.data_ptr<Scalar>());
  carrier_channel_gates_second_kernel<Scalar, Control>
      <<<launch_blocks(gates.numel()), threads, 0, stream>>>(
          output_adjoint.data_ptr<Scalar>(),
          values.data_ptr<Scalar>(),
          channel_maps.data_ptr<Control>(),
          values_adjoint_tangent.data_ptr<Scalar>(),
          channel_maps_adjoint_tangent.data_ptr<Control>(),
          feature_offsets.data_ptr<std::int64_t>(),
          channel_offsets.data_ptr<std::int64_t>(),
          map_offsets.data_ptr<std::int64_t>(),
          values.size(0),
          values.size(1),
          gates.size(1),
          feature_offsets.numel() - 1,
          gates_second_adjoint.data_ptr<Control>());
  if (values.size(0) >= 32) {
    carrier_channel_maps_second_parallel_kernel<Scalar, Control>
        <<<channel_maps.numel(),
           threads,
           threads * sizeof(Scalar),
           stream>>>(
            output_adjoint.data_ptr<Scalar>(),
            values.data_ptr<Scalar>(),
            gates.data_ptr<Control>(),
            values_adjoint_tangent.data_ptr<Scalar>(),
            gates_adjoint_tangent.data_ptr<Control>(),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            values.size(1),
            gates.size(1),
            channel_maps.numel(),
            feature_offsets.numel() - 1,
            channel_maps_second_adjoint.data_ptr<Control>());
  } else {
    carrier_channel_maps_second_kernel<Scalar, Control>
        <<<launch_blocks(channel_maps.numel()),
           threads,
           0,
           stream>>>(
            output_adjoint.data_ptr<Scalar>(),
            values.data_ptr<Scalar>(),
            gates.data_ptr<Control>(),
            values_adjoint_tangent.data_ptr<Scalar>(),
            gates_adjoint_tangent.data_ptr<Control>(),
            feature_offsets.data_ptr<std::int64_t>(),
            channel_offsets.data_ptr<std::int64_t>(),
            map_offsets.data_ptr<std::int64_t>(),
            values.size(0),
            values.size(1),
            gates.size(1),
            channel_maps.numel(),
            feature_offsets.numel() - 1,
            channel_maps_second_adjoint.data_ptr<Control>());
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor carrier_channel_update_cuda(
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_update_cuda_inputs(
      values,
      gates,
      channel_maps,
      feature_offsets,
      channel_offsets,
      map_offsets);
  c10::cuda::CUDAGuard device_guard(values.device());
  auto output = torch::empty_like(values);
  const std::int64_t work = values.numel();
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_channel_update_uses_real_controls_cuda(
          values, gates, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      launch_carrier_channel_update_real_controls_cuda<
          c10::complex<float>, float>(
          values,
          gates,
          channel_maps,
          feature_offsets,
          channel_offsets,
          map_offsets,
          output,
          stream);
    } else {
      launch_carrier_channel_update_real_controls_cuda<
          c10::complex<double>, double>(
          values,
          gates,
          channel_maps,
          feature_offsets,
          channel_offsets,
          map_offsets,
          output,
          stream);
    }
    return output;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_update_cuda",
      [&] {
        carrier_channel_update_kernel<scalar_t>
            <<<launch_blocks(work), threads, 0, stream>>>(
                values.data_ptr<scalar_t>(),
                gates.data_ptr<scalar_t>(),
                channel_maps.data_ptr<scalar_t>(),
                feature_offsets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                map_offsets.data_ptr<std::int64_t>(),
                values.size(0),
                values.size(1),
                gates.size(1),
                feature_offsets.numel() - 1,
                output.data_ptr<scalar_t>());
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_channel_update_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& gates,
    const torch::Tensor& channel_maps,
    const torch::Tensor& feature_offsets,
    const torch::Tensor& channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_update_cuda_inputs(
      values,
      gates,
      channel_maps,
      feature_offsets,
      channel_offsets,
      map_offsets);
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      output_adjoint.device() == values.device() &&
          output_adjoint.scalar_type() == values.scalar_type() &&
          output_adjoint.sizes() == values.sizes(),
      "output_adjoint must match values");
  c10::cuda::CUDAGuard device_guard(values.device());
  auto values_adjoint = torch::empty_like(values);
  auto gates_adjoint = torch::empty_like(gates);
  auto channel_maps_adjoint = torch::empty_like(channel_maps);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_channel_update_uses_real_controls_cuda(
          values, gates, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      launch_carrier_channel_update_real_controls_adjoint_cuda<
          c10::complex<float>, float>(
          output_adjoint,
          values,
          gates,
          channel_maps,
          feature_offsets,
          channel_offsets,
          map_offsets,
          values_adjoint,
          gates_adjoint,
          channel_maps_adjoint,
          stream);
    } else {
      launch_carrier_channel_update_real_controls_adjoint_cuda<
          c10::complex<double>, double>(
          output_adjoint,
          values,
          gates,
          channel_maps,
          feature_offsets,
          channel_offsets,
          map_offsets,
          values_adjoint,
          gates_adjoint,
          channel_maps_adjoint,
          stream);
    }
    return std::make_tuple(
        values_adjoint,
        gates_adjoint,
        channel_maps_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_update_adjoint_cuda",
      [&] {
        carrier_channel_values_adjoint_kernel<scalar_t>
            <<<launch_blocks(values.numel()), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                gates.data_ptr<scalar_t>(),
                channel_maps.data_ptr<scalar_t>(),
                feature_offsets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                map_offsets.data_ptr<std::int64_t>(),
                values.size(0),
                values.size(1),
                gates.size(1),
                feature_offsets.numel() - 1,
                values_adjoint.data_ptr<scalar_t>());
        carrier_channel_gates_adjoint_kernel<scalar_t>
            <<<launch_blocks(gates.numel()), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                channel_maps.data_ptr<scalar_t>(),
                feature_offsets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                map_offsets.data_ptr<std::int64_t>(),
                values.size(0),
                values.size(1),
                gates.size(1),
                feature_offsets.numel() - 1,
                gates_adjoint.data_ptr<scalar_t>());
        if (values.size(0) >= 32) {
          carrier_channel_maps_adjoint_parallel_kernel<scalar_t>
              <<<channel_maps.numel(),
                 threads,
                 threads * sizeof(scalar_t),
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  values.data_ptr<scalar_t>(),
                  gates.data_ptr<scalar_t>(),
                  feature_offsets.data_ptr<std::int64_t>(),
                  channel_offsets.data_ptr<std::int64_t>(),
                  map_offsets.data_ptr<std::int64_t>(),
                  values.size(0),
                  values.size(1),
                  gates.size(1),
                  channel_maps.numel(),
                  feature_offsets.numel() - 1,
                  channel_maps_adjoint.data_ptr<scalar_t>());
        } else {
          carrier_channel_maps_adjoint_kernel<scalar_t>
              <<<launch_blocks(channel_maps.numel()),
                 threads,
                 0,
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  values.data_ptr<scalar_t>(),
                  gates.data_ptr<scalar_t>(),
                  feature_offsets.data_ptr<std::int64_t>(),
                  channel_offsets.data_ptr<std::int64_t>(),
                  map_offsets.data_ptr<std::int64_t>(),
                  values.size(0),
                  values.size(1),
                  gates.size(1),
                  channel_maps.numel(),
                  feature_offsets.numel() - 1,
                  channel_maps_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
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
carrier_channel_update_double_backward_cuda(
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
  check_carrier_channel_update_cuda_inputs(
      values,
      gates,
      channel_maps,
      feature_offsets,
      channel_offsets,
      map_offsets);
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(values_adjoint_tangent, "values_adjoint_tangent");
  check_cuda_data_tensor(gates_adjoint_tangent, "gates_adjoint_tangent");
  check_cuda_value_vector(
      channel_maps_adjoint_tangent,
      channel_maps,
      "channel_maps_adjoint_tangent");
  TORCH_CHECK(
      output_adjoint.device() == values.device() &&
          output_adjoint.scalar_type() == values.scalar_type() &&
          output_adjoint.sizes() == values.sizes() &&
          values_adjoint_tangent.device() == values.device() &&
          values_adjoint_tangent.scalar_type() == values.scalar_type() &&
          values_adjoint_tangent.sizes() == values.sizes(),
      "output_adjoint and values_adjoint_tangent must match values");
  TORCH_CHECK(
      gates_adjoint_tangent.device() == gates.device() &&
          gates_adjoint_tangent.scalar_type() == gates.scalar_type() &&
          gates_adjoint_tangent.sizes() == gates.sizes(),
      "gates_adjoint_tangent must match gates");
  TORCH_CHECK(
      channel_maps_adjoint_tangent.sizes() == channel_maps.sizes(),
      "channel_maps_adjoint_tangent must match channel_maps");
  c10::cuda::CUDAGuard device_guard(values.device());
  auto output_adjoint_tangent = torch::empty_like(output_adjoint);
  auto values_second_adjoint = torch::empty_like(values);
  auto gates_second_adjoint = torch::empty_like(gates);
  auto channel_maps_second_adjoint = torch::empty_like(channel_maps);
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_channel_update_uses_real_controls_cuda(
          values, gates, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      launch_carrier_channel_update_real_controls_double_backward_cuda<
          c10::complex<float>, float>(
          output_adjoint,
          values,
          gates,
          channel_maps,
          values_adjoint_tangent,
          gates_adjoint_tangent,
          channel_maps_adjoint_tangent,
          feature_offsets,
          channel_offsets,
          map_offsets,
          output_adjoint_tangent,
          values_second_adjoint,
          gates_second_adjoint,
          channel_maps_second_adjoint,
          stream);
    } else {
      launch_carrier_channel_update_real_controls_double_backward_cuda<
          c10::complex<double>, double>(
          output_adjoint,
          values,
          gates,
          channel_maps,
          values_adjoint_tangent,
          gates_adjoint_tangent,
          channel_maps_adjoint_tangent,
          feature_offsets,
          channel_offsets,
          map_offsets,
          output_adjoint_tangent,
          values_second_adjoint,
          gates_second_adjoint,
          channel_maps_second_adjoint,
          stream);
    }
    return std::make_tuple(
        output_adjoint_tangent,
        values_second_adjoint,
        gates_second_adjoint,
        channel_maps_second_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_update_double_backward_cuda",
      [&] {
        carrier_channel_output_tangent_kernel<scalar_t>
            <<<launch_blocks(values.numel()), threads, 0, stream>>>(
                values.data_ptr<scalar_t>(),
                gates.data_ptr<scalar_t>(),
                channel_maps.data_ptr<scalar_t>(),
                values_adjoint_tangent.data_ptr<scalar_t>(),
                gates_adjoint_tangent.data_ptr<scalar_t>(),
                channel_maps_adjoint_tangent.data_ptr<scalar_t>(),
                feature_offsets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                map_offsets.data_ptr<std::int64_t>(),
                values.size(0),
                values.size(1),
                gates.size(1),
                feature_offsets.numel() - 1,
                output_adjoint_tangent.data_ptr<scalar_t>());
        carrier_channel_values_second_kernel<scalar_t>
            <<<launch_blocks(values.numel()), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                gates.data_ptr<scalar_t>(),
                channel_maps.data_ptr<scalar_t>(),
                gates_adjoint_tangent.data_ptr<scalar_t>(),
                channel_maps_adjoint_tangent.data_ptr<scalar_t>(),
                feature_offsets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                map_offsets.data_ptr<std::int64_t>(),
                values.size(0),
                values.size(1),
                gates.size(1),
                feature_offsets.numel() - 1,
                values_second_adjoint.data_ptr<scalar_t>());
        carrier_channel_gates_second_kernel<scalar_t>
            <<<launch_blocks(gates.numel()), threads, 0, stream>>>(
                output_adjoint.data_ptr<scalar_t>(),
                values.data_ptr<scalar_t>(),
                channel_maps.data_ptr<scalar_t>(),
                values_adjoint_tangent.data_ptr<scalar_t>(),
                channel_maps_adjoint_tangent.data_ptr<scalar_t>(),
                feature_offsets.data_ptr<std::int64_t>(),
                channel_offsets.data_ptr<std::int64_t>(),
                map_offsets.data_ptr<std::int64_t>(),
                values.size(0),
                values.size(1),
                gates.size(1),
                feature_offsets.numel() - 1,
                gates_second_adjoint.data_ptr<scalar_t>());
        if (values.size(0) >= 32) {
          carrier_channel_maps_second_parallel_kernel<scalar_t>
              <<<channel_maps.numel(),
                 threads,
                 threads * sizeof(scalar_t),
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  values.data_ptr<scalar_t>(),
                  gates.data_ptr<scalar_t>(),
                  values_adjoint_tangent.data_ptr<scalar_t>(),
                  gates_adjoint_tangent.data_ptr<scalar_t>(),
                  feature_offsets.data_ptr<std::int64_t>(),
                  channel_offsets.data_ptr<std::int64_t>(),
                  map_offsets.data_ptr<std::int64_t>(),
                  values.size(0),
                  values.size(1),
                  gates.size(1),
                  channel_maps.numel(),
                  feature_offsets.numel() - 1,
                  channel_maps_second_adjoint.data_ptr<scalar_t>());
        } else {
          carrier_channel_maps_second_kernel<scalar_t>
              <<<launch_blocks(channel_maps.numel()),
                 threads,
                 0,
                 stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  values.data_ptr<scalar_t>(),
                  gates.data_ptr<scalar_t>(),
                  values_adjoint_tangent.data_ptr<scalar_t>(),
                  gates_adjoint_tangent.data_ptr<scalar_t>(),
                  feature_offsets.data_ptr<std::int64_t>(),
                  channel_offsets.data_ptr<std::int64_t>(),
                  map_offsets.data_ptr<std::int64_t>(),
                  values.size(0),
                  values.size(1),
                  gates.size(1),
                  channel_maps.numel(),
                  feature_offsets.numel() - 1,
                  channel_maps_second_adjoint.data_ptr<scalar_t>());
        }
        C10_CUDA_KERNEL_LAUNCH_CHECK();
      });
  return std::make_tuple(
      output_adjoint_tangent,
      values_second_adjoint,
      gates_second_adjoint,
      channel_maps_second_adjoint);
}

torch::Tensor source_analysis_cuda(
    const torch::Tensor& source,
    const torch::Tensor& assembly_rows,
    const torch::Tensor& assembly_columns,
    const torch::Tensor& assembly_values,
    const torch::Tensor& synthesis_rows,
    const torch::Tensor& synthesis_columns,
    const torch::Tensor& synthesis_values,
    std::int64_t induced_dimension,
    std::int64_t output_dimension) {
  check_cuda_data_tensor(source, "source");
  check_cuda_indices(
      assembly_rows,
      assembly_columns,
      assembly_values,
      source,
      "assembly");
  check_cuda_indices(
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      source,
      "synthesis");
  TORCH_CHECK(induced_dimension > 0, "induced_dimension must be positive");
  TORCH_CHECK(output_dimension > 0, "output_dimension must be positive");
  c10::cuda::CUDAGuard device_guard(source.device());
  auto ambient = torch::zeros(
      {source.size(0), induced_dimension},
      source.options());
  auto output = torch::zeros(
      {source.size(0), output_dimension},
      source.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      source.scalar_type(),
      "ye3t_source_analysis_cuda",
      [&] {
        const std::int64_t assembly_work =
            source.size(0) * assembly_rows.numel();
        if (assembly_work > 0) {
          source_assembly_kernel<scalar_t>
              <<<launch_blocks(assembly_work), threads, 0, stream>>>(
                  source.data_ptr<scalar_t>(),
                  source.size(0),
                  source.size(1),
                  assembly_rows.data_ptr<std::int64_t>(),
                  assembly_columns.data_ptr<std::int64_t>(),
                  assembly_values.data_ptr<scalar_t>(),
                  assembly_rows.numel(),
                  induced_dimension,
                  ambient.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t synthesis_work =
            source.size(0) * synthesis_rows.numel();
        if (synthesis_work > 0) {
          synthesis_analysis_kernel<scalar_t>
              <<<launch_blocks(synthesis_work), threads, 0, stream>>>(
                  ambient.data_ptr<scalar_t>(),
                  source.size(0),
                  induced_dimension,
                  synthesis_rows.data_ptr<std::int64_t>(),
                  synthesis_columns.data_ptr<std::int64_t>(),
                  synthesis_values.data_ptr<scalar_t>(),
                  synthesis_rows.numel(),
                  output_dimension,
                  output.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return output;
}

torch::Tensor source_analysis_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& assembly_rows,
    const torch::Tensor& assembly_columns,
    const torch::Tensor& assembly_values,
    const torch::Tensor& synthesis_rows,
    const torch::Tensor& synthesis_columns,
    const torch::Tensor& synthesis_values,
    std::int64_t source_dimension,
    std::int64_t induced_dimension) {
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_indices(
      assembly_rows,
      assembly_columns,
      assembly_values,
      output_adjoint,
      "assembly");
  check_cuda_indices(
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      output_adjoint,
      "synthesis");
  TORCH_CHECK(source_dimension > 0, "source_dimension must be positive");
  TORCH_CHECK(induced_dimension > 0, "induced_dimension must be positive");
  c10::cuda::CUDAGuard device_guard(output_adjoint.device());
  auto ambient_adjoint = torch::zeros(
      {output_adjoint.size(0), induced_dimension},
      output_adjoint.options());
  auto source_adjoint = torch::zeros(
      {output_adjoint.size(0), source_dimension},
      output_adjoint.options());
  constexpr int threads = 256;
  const auto stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      output_adjoint.scalar_type(),
      "ye3t_source_analysis_adjoint_cuda",
      [&] {
        const std::int64_t synthesis_work =
            output_adjoint.size(0) * synthesis_rows.numel();
        if (synthesis_work > 0) {
          synthesis_adjoint_kernel<scalar_t>
              <<<launch_blocks(synthesis_work), threads, 0, stream>>>(
                  output_adjoint.data_ptr<scalar_t>(),
                  output_adjoint.size(0),
                  output_adjoint.size(1),
                  synthesis_rows.data_ptr<std::int64_t>(),
                  synthesis_columns.data_ptr<std::int64_t>(),
                  synthesis_values.data_ptr<scalar_t>(),
                  synthesis_rows.numel(),
                  induced_dimension,
                  ambient_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
        const std::int64_t assembly_work =
            output_adjoint.size(0) * assembly_rows.numel();
        if (assembly_work > 0) {
          source_assembly_adjoint_kernel<scalar_t>
              <<<launch_blocks(assembly_work), threads, 0, stream>>>(
                  ambient_adjoint.data_ptr<scalar_t>(),
                  output_adjoint.size(0),
                  induced_dimension,
                  assembly_rows.data_ptr<std::int64_t>(),
                  assembly_columns.data_ptr<std::int64_t>(),
                  assembly_values.data_ptr<scalar_t>(),
                  assembly_rows.numel(),
                  source_dimension,
                  source_adjoint.data_ptr<scalar_t>());
          C10_CUDA_KERNEL_LAUNCH_CHECK();
        }
      });
  return source_adjoint;
}

torch::Tensor source_analysis_linear_cuda(
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
  TORCH_CHECK(
      weight.device() == source.device() &&
          weight.is_contiguous() &&
          weight.dim() == 1 &&
          weight.scalar_type() == source.scalar_type(),
      "weight must be a contiguous CUDA vector with source dtype");
  TORCH_CHECK(
      bias.device() == source.device() &&
          bias.is_contiguous() &&
          bias.dim() == 0 &&
          bias.scalar_type() == source.scalar_type(),
      "bias must be a contiguous CUDA scalar with source dtype");
  auto features = source_analysis_cuda(
      source,
      assembly_rows,
      assembly_columns,
      assembly_values,
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      induced_dimension,
      weight.numel());
  return torch::matmul(features, weight) + bias;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
source_analysis_linear_adjoint_cuda(
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
      output_adjoint.is_cuda() &&
          output_adjoint.device() == source.device() &&
          output_adjoint.is_contiguous() &&
          output_adjoint.dim() == 1 &&
          output_adjoint.scalar_type() == source.scalar_type() &&
          output_adjoint.numel() == source.size(0),
      "output_adjoint must be a contiguous CUDA batch vector");
  TORCH_CHECK(
      weight.device() == source.device() &&
          weight.is_contiguous() &&
          weight.dim() == 1 &&
          weight.scalar_type() == source.scalar_type(),
      "weight must be a contiguous CUDA vector with source dtype");
  auto features = source_analysis_cuda(
      source,
      assembly_rows,
      assembly_columns,
      assembly_values,
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      induced_dimension,
      weight.numel());
  auto feature_adjoint =
      output_adjoint.unsqueeze(1) * weight.conj().unsqueeze(0);
  auto source_adjoint = source_analysis_adjoint_cuda(
      feature_adjoint.contiguous(),
      assembly_rows,
      assembly_columns,
      assembly_values,
      synthesis_rows,
      synthesis_columns,
      synthesis_values,
      source.size(1),
      induced_dimension);
  auto weight_adjoint = torch::matmul(
      features.conj().transpose(0, 1),
      output_adjoint);
  auto bias_adjoint = output_adjoint.sum();
  return std::make_tuple(
      source_adjoint,
      weight_adjoint,
      bias_adjoint);
}

}  // namespace

torch::Tensor carrier_channel_transform_cuda(
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets,
    std::int64_t output_width) {
  check_carrier_channel_transform_cuda_inputs(
      values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  TORCH_CHECK(output_width > 0, "channel-transform output width must be positive");
  c10::cuda::CUDAGuard device_guard(values.device());
  auto output = torch::empty({values.size(0), output_width}, values.options());
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_channel_transform_uses_real_maps_cuda(values, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      launch_carrier_channel_transform_cuda<c10::complex<float>, float>(
          values,
          channel_maps,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          output,
          stream);
    } else {
      launch_carrier_channel_transform_cuda<c10::complex<double>, double>(
          values,
          channel_maps,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          output,
          stream);
    }
    return output;
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_transform_cuda",
      [&] {
        launch_carrier_channel_transform_cuda<scalar_t, scalar_t>(
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            output,
            stream);
      });
  return output;
}

std::tuple<torch::Tensor, torch::Tensor>
carrier_channel_transform_adjoint_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_transform_cuda_inputs(
      values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  TORCH_CHECK(
      output_adjoint.device() == values.device() &&
          output_adjoint.scalar_type() == values.scalar_type() &&
          output_adjoint.size(0) == values.size(0),
      "output_adjoint must match the channel-transform batch and dtype");
  c10::cuda::CUDAGuard device_guard(values.device());
  auto values_adjoint = torch::empty_like(values);
  auto maps_adjoint = torch::empty_like(channel_maps);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_channel_transform_uses_real_maps_cuda(values, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      launch_carrier_channel_transform_adjoint_cuda<
          c10::complex<float>, float>(
          output_adjoint,
          values,
          channel_maps,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          values_adjoint,
          maps_adjoint,
          stream);
    } else {
      launch_carrier_channel_transform_adjoint_cuda<
          c10::complex<double>, double>(
          output_adjoint,
          values,
          channel_maps,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          values_adjoint,
          maps_adjoint,
          stream);
    }
    return std::make_tuple(values_adjoint, maps_adjoint);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_transform_adjoint_cuda",
      [&] {
        launch_carrier_channel_transform_adjoint_cuda<scalar_t, scalar_t>(
            output_adjoint,
            values,
            channel_maps,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            values_adjoint,
            maps_adjoint,
            stream);
      });
  return std::make_tuple(values_adjoint, maps_adjoint);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_channel_transform_double_backward_cuda(
    const torch::Tensor& output_adjoint,
    const torch::Tensor& values,
    const torch::Tensor& channel_maps,
    const torch::Tensor& values_adjoint_tangent,
    const torch::Tensor& maps_adjoint_tangent,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_channel_transform_cuda_inputs(
      values,
      channel_maps,
      input_feature_offsets,
      output_feature_offsets,
      input_channel_offsets,
      output_channel_offsets,
      map_offsets);
  check_cuda_data_tensor(output_adjoint, "output_adjoint");
  check_cuda_data_tensor(values_adjoint_tangent, "values_adjoint_tangent");
  check_cuda_value_vector(
      maps_adjoint_tangent,
      channel_maps,
      "maps_adjoint_tangent");
  TORCH_CHECK(
      output_adjoint.device() == values.device() &&
          output_adjoint.scalar_type() == values.scalar_type() &&
          output_adjoint.size(0) == values.size(0),
      "output_adjoint must match the channel-transform batch and dtype");
  TORCH_CHECK(
      values_adjoint_tangent.device() == values.device() &&
          values_adjoint_tangent.scalar_type() == values.scalar_type() &&
          values_adjoint_tangent.sizes() == values.sizes(),
      "values_adjoint_tangent must match values");
  c10::cuda::CUDAGuard device_guard(values.device());
  auto output_tangent = torch::empty_like(output_adjoint);
  auto values_second = torch::empty_like(values);
  auto maps_second = torch::empty_like(channel_maps);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (carrier_channel_transform_uses_real_maps_cuda(values, channel_maps)) {
    if (values.scalar_type() == torch::kComplexFloat) {
      launch_carrier_channel_transform_double_backward_cuda<
          c10::complex<float>, float>(
          output_adjoint,
          values,
          channel_maps,
          values_adjoint_tangent,
          maps_adjoint_tangent,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          output_tangent,
          values_second,
          maps_second,
          stream);
    } else {
      launch_carrier_channel_transform_double_backward_cuda<
          c10::complex<double>, double>(
          output_adjoint,
          values,
          channel_maps,
          values_adjoint_tangent,
          maps_adjoint_tangent,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          output_tangent,
          values_second,
          maps_second,
          stream);
    }
    return std::make_tuple(output_tangent, values_second, maps_second);
  }
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      values.scalar_type(),
      "ye3t_carrier_channel_transform_double_backward_cuda",
      [&] {
        launch_carrier_channel_transform_double_backward_cuda<
            scalar_t, scalar_t>(
            output_adjoint,
            values,
            channel_maps,
            values_adjoint_tangent,
            maps_adjoint_tangent,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            output_tangent,
            values_second,
            maps_second,
            stream);
      });
  return std::make_tuple(output_tangent, values_second, maps_second);
}

torch::Tensor carrier_role_channel_map_adjoint_cuda(
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
  check_carrier_role_channel_map_adjoint_cuda_inputs(
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
  c10::cuda::CUDAGuard device_guard(edge_values.device());
  auto maps_adjoint = torch::empty_like(channel_maps);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (edge_values.scalar_type() == torch::kComplexFloat) {
    if (channel_maps.scalar_type() == torch::kFloat) {
      launch_carrier_role_channel_map_adjoint_cuda<
          c10::complex<float>, float, float>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          maps_adjoint,
          stream);
    } else {
      launch_carrier_role_channel_map_adjoint_cuda<
          c10::complex<float>, c10::complex<float>, float>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          maps_adjoint,
          stream);
    }
    return maps_adjoint;
  }
  if (edge_values.scalar_type() == torch::kComplexDouble) {
    if (channel_maps.scalar_type() == torch::kDouble) {
      launch_carrier_role_channel_map_adjoint_cuda<
          c10::complex<double>, double, double>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          maps_adjoint,
          stream);
    } else {
      launch_carrier_role_channel_map_adjoint_cuda<
          c10::complex<double>, c10::complex<double>, double>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          maps_adjoint,
          stream);
    }
    return maps_adjoint;
  }
  AT_DISPATCH_FLOATING_TYPES(
      edge_values.scalar_type(),
      "ye3t_carrier_role_channel_map_adjoint_cuda",
      [&] {
        launch_carrier_role_channel_map_adjoint_cuda<
            scalar_t, scalar_t, scalar_t>(
            edge_values,
            role_weights,
            atomic_output_adjoint,
            atom_centers,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            maps_adjoint,
            stream);
      });
  return maps_adjoint;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
carrier_role_channel_map_adjoint_double_backward_cuda(
    const torch::Tensor& edge_values,
    const torch::Tensor& role_weights,
    const torch::Tensor& atomic_output_adjoint,
    const torch::Tensor& atom_centers,
    const torch::Tensor& channel_maps,
    const torch::Tensor& maps_adjoint_tangent,
    const torch::Tensor& input_feature_offsets,
    const torch::Tensor& output_feature_offsets,
    const torch::Tensor& input_channel_offsets,
    const torch::Tensor& output_channel_offsets,
    const torch::Tensor& map_offsets) {
  check_carrier_role_channel_map_adjoint_cuda_inputs(
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
  check_cuda_value_vector(
      maps_adjoint_tangent,
      channel_maps,
      "maps_adjoint_tangent");
  c10::cuda::CUDAGuard device_guard(edge_values.device());
  auto edge_values_second = torch::empty_like(edge_values);
  auto role_weights_second = torch::empty_like(role_weights);
  auto atomic_output_tangent = torch::zeros_like(atomic_output_adjoint);
  const auto stream = at::cuda::getCurrentCUDAStream();
  if (edge_values.scalar_type() == torch::kComplexFloat) {
    if (channel_maps.scalar_type() == torch::kFloat) {
      launch_carrier_role_channel_map_adjoint_double_backward_cuda<
          c10::complex<float>, float, float>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          maps_adjoint_tangent,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          edge_values_second,
          role_weights_second,
          atomic_output_tangent,
          stream);
    } else {
      launch_carrier_role_channel_map_adjoint_double_backward_cuda<
          c10::complex<float>, c10::complex<float>, float>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          maps_adjoint_tangent,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          edge_values_second,
          role_weights_second,
          atomic_output_tangent,
          stream);
    }
    return std::make_tuple(
        edge_values_second, role_weights_second, atomic_output_tangent);
  }
  if (edge_values.scalar_type() == torch::kComplexDouble) {
    if (channel_maps.scalar_type() == torch::kDouble) {
      launch_carrier_role_channel_map_adjoint_double_backward_cuda<
          c10::complex<double>, double, double>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          maps_adjoint_tangent,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          edge_values_second,
          role_weights_second,
          atomic_output_tangent,
          stream);
    } else {
      launch_carrier_role_channel_map_adjoint_double_backward_cuda<
          c10::complex<double>, c10::complex<double>, double>(
          edge_values,
          role_weights,
          atomic_output_adjoint,
          atom_centers,
          maps_adjoint_tangent,
          input_feature_offsets,
          output_feature_offsets,
          input_channel_offsets,
          output_channel_offsets,
          map_offsets,
          edge_values_second,
          role_weights_second,
          atomic_output_tangent,
          stream);
    }
    return std::make_tuple(
        edge_values_second, role_weights_second, atomic_output_tangent);
  }
  AT_DISPATCH_FLOATING_TYPES(
      edge_values.scalar_type(),
      "ye3t_carrier_role_channel_map_adjoint_double_backward_cuda",
      [&] {
        launch_carrier_role_channel_map_adjoint_double_backward_cuda<
            scalar_t, scalar_t, scalar_t>(
            edge_values,
            role_weights,
            atomic_output_adjoint,
            atom_centers,
            maps_adjoint_tangent,
            input_feature_offsets,
            output_feature_offsets,
            input_channel_offsets,
            output_channel_offsets,
            map_offsets,
            edge_values_second,
            role_weights_second,
            atomic_output_tangent,
            stream);
      });
  return std::make_tuple(
      edge_values_second, role_weights_second, atomic_output_tangent);
}

TORCH_LIBRARY_IMPL(ye3t_runtime, CUDA, library) {
  library.impl(
      "cheb_exp_cos_radial_with_derivative",
      &cheb_exp_cos_radial_with_derivative_cuda);
  library.impl(
      "cheb_exp_cos_radial_table_with_derivative",
      &cheb_exp_cos_radial_table_with_derivative_cuda);
  library.impl(
      "cheb_exp_cos_radial_table_double_backward",
      &cheb_exp_cos_radial_table_double_backward_cuda);
  library.impl(
      "spherical_harmonics_with_derivative",
      &spherical_harmonics_with_derivative_cuda);
  library.impl(
      "spherical_harmonics_table_with_derivative",
      &spherical_harmonics_table_with_derivative_cuda);
  library.impl(
      "spherical_harmonics_table_double_backward",
      &spherical_harmonics_table_double_backward_cuda);
  library.impl(
      "plain_site_basis_product_with_derivative",
      &plain_site_basis_product_with_derivative_cuda);
  library.impl(
      "scheduled_radial_angular_channels_with_derivative",
      &scheduled_radial_angular_channels_with_derivative_cuda);
  library.impl(
      "plain_site_basis_product_adjoint",
      &plain_site_basis_product_adjoint_cuda);
  library.impl("compact_pair_product", &compact_pair_product_cuda);
  library.impl(
      "compact_pair_product_adjoint",
      &compact_pair_product_adjoint_cuda);
  library.impl(
      "compact_exterior_power",
      &compact_exterior_power_cuda);
  library.impl(
      "compact_exterior_power_adjoint",
      &compact_exterior_power_adjoint_cuda);
  library.impl(
      "symmetric_power_monomial",
      &symmetric_power_monomial_cuda);
  library.impl(
      "symmetric_power_monomial_adjoint",
      &symmetric_power_monomial_adjoint_cuda);
  library.impl(
      "symmetric_power_monomial_double_backward",
      &symmetric_power_monomial_double_backward_cuda);
  library.impl(
      "symmetric_power_shared_monomial",
      &symmetric_power_shared_monomial_cuda);
  library.impl(
      "symmetric_power_shared_monomial_adjoint",
      &symmetric_power_shared_monomial_adjoint_cuda);
  library.impl(
      "symmetric_power_shared_monomial_double_backward",
      &symmetric_power_shared_monomial_double_backward_cuda);
  library.impl(
      "symmetric_power_shared_monomial_factored_double_backward",
      &symmetric_power_shared_monomial_factored_double_backward_cuda);
  library.impl(
      "symmetric_power_shared_sparse_monomial",
      &symmetric_power_shared_sparse_monomial_cuda);
  library.impl(
      "symmetric_power_shared_sparse_monomial_adjoint",
      &symmetric_power_shared_sparse_monomial_adjoint_cuda);
  library.impl(
      "symmetric_power_shared_sparse_monomial_double_backward",
      &symmetric_power_shared_sparse_monomial_double_backward_cuda);
  library.impl(
      "symmetric_power_shared_monomial_batched_adjoint",
      &symmetric_power_shared_monomial_batched_adjoint_cuda);
  library.impl("factorized_angular", &factorized_angular_cuda);
  library.impl(
      "factorized_angular_adjoint",
      &factorized_angular_adjoint_cuda);
  library.impl(
      "factorized_angular_double_backward",
      &factorized_angular_double_backward_cuda);
  library.impl(
      "factorized_angular_heterogeneous",
      &factorized_angular_heterogeneous_cuda);
  library.impl(
      "factorized_angular_heterogeneous_adjoint",
      &factorized_angular_heterogeneous_adjoint_cuda);
  library.impl(
      "factorized_angular_heterogeneous_adjoint_with_workspace",
      &factorized_angular_heterogeneous_adjoint_with_workspace_cuda);
  library.impl(
      "factorized_angular_heterogeneous_double_backward",
      &factorized_angular_heterogeneous_double_backward_cuda);
  library.impl(
      "factorized_angular_heterogeneous_double_backward_from_workspace",
      &factorized_angular_heterogeneous_double_backward_from_workspace_cuda);
  library.impl(
      "factorized_angular_segmented",
      &factorized_angular_segmented_cuda);
  library.impl(
      "factorized_angular_segmented_adjoint",
      &factorized_angular_segmented_adjoint_cuda);
  library.impl(
      "factorized_angular_segmented_double_backward",
      &factorized_angular_segmented_double_backward_cuda);
  library.impl(
      "factorized_angular_linear",
      &factorized_angular_linear_cuda);
  library.impl(
      "factorized_angular_linear_adjoint",
      &factorized_angular_linear_adjoint_cuda);
  library.impl("density_accumulate", &density_accumulate_cuda);
  library.impl(
      "density_accumulate_adjoint",
      &density_accumulate_adjoint_cuda);
  library.impl(
      "edge_outer_accumulate",
      &edge_outer_accumulate_cuda);
  library.impl(
      "edge_outer_accumulate_adjoint",
      &edge_outer_accumulate_adjoint_cuda);
  library.impl(
      "edge_outer_accumulate_double_backward",
      &edge_outer_accumulate_double_backward_cuda);
  library.impl(
      "softmax_gaussian_role_density",
      &softmax_gaussian_role_density_cuda);
  library.impl(
      "softmax_gaussian_role_density_adjoint",
      &softmax_gaussian_role_density_adjoint_cuda);
  library.impl(
      "softmax_gaussian_role_density_double_backward",
      &softmax_gaussian_role_density_double_backward_cuda);
  library.impl(
      "scheduled_softmax_gaussian_role_density",
      &scheduled_role_density_cuda);
  library.impl(
      "scheduled_softmax_gaussian_role_density_adjoint",
      &scheduled_role_density_adjoint_cuda);
  library.impl(
      "scheduled_softmax_gaussian_role_density_double_backward",
      &scheduled_role_density_double_backward_cuda);
  library.impl(
      "carrier_gated_scatter",
      &carrier_gated_scatter_cuda);
  library.impl(
      "carrier_gated_scatter_adjoint",
      &carrier_gated_scatter_adjoint_cuda);
  library.impl(
      "carrier_gated_scatter_double_backward",
      &carrier_gated_scatter_double_backward_cuda);
  library.impl(
      "carrier_residual_gated_scatter",
      &carrier_residual_gated_scatter_cuda);
  library.impl(
      "carrier_residual_gated_scatter_adjoint",
      &carrier_residual_gated_scatter_adjoint_cuda);
  library.impl(
      "carrier_residual_gated_scatter_double_backward",
      &carrier_residual_gated_scatter_double_backward_cuda);
  library.impl(
      "carrier_segmented_residual_gated_scatter",
      &carrier_segmented_residual_gated_scatter_cuda);
  library.impl(
      "carrier_segmented_residual_gated_scatter_adjoint",
      &carrier_segmented_residual_gated_scatter_adjoint_cuda);
  library.impl(
      "carrier_segmented_residual_gated_scatter_double_backward",
      &carrier_segmented_residual_gated_scatter_double_backward_cuda);
  library.impl(
      "source_arena_gather",
      &source_arena_gather_cuda);
  library.impl(
      "source_arena_gather_adjoint",
      &source_arena_gather_adjoint_cuda);
  library.impl(
      "source_arena_gather_double_backward",
      &source_arena_gather_double_backward_cuda);
  library.impl(
      "source_arena_channel_transform",
      &source_arena_channel_transform_cuda);
  library.impl(
      "source_arena_channel_transform_adjoint",
      &source_arena_channel_transform_adjoint_cuda);
  library.impl(
      "source_arena_channel_transform_double_backward",
      &source_arena_channel_transform_double_backward_cuda);
  library.impl(
      "carrier_channel_update",
      &carrier_channel_update_cuda);
  library.impl(
      "carrier_channel_update_adjoint",
      &carrier_channel_update_adjoint_cuda);
  library.impl(
      "carrier_channel_update_double_backward",
      &carrier_channel_update_double_backward_cuda);
  library.impl(
      "carrier_channel_transform",
      &carrier_channel_transform_cuda);
  library.impl(
      "carrier_channel_transform_adjoint",
      &carrier_channel_transform_adjoint_cuda);
  library.impl(
      "carrier_channel_transform_double_backward",
      &carrier_channel_transform_double_backward_cuda);
  library.impl(
      "carrier_role_channel_map_adjoint",
      &carrier_role_channel_map_adjoint_cuda);
  library.impl(
      "carrier_role_channel_map_adjoint_double_backward",
      &carrier_role_channel_map_adjoint_double_backward_cuda);
  library.impl("source_analysis", &source_analysis_cuda);
  library.impl("source_analysis_adjoint", &source_analysis_adjoint_cuda);
  library.impl("source_analysis_linear", &source_analysis_linear_cuda);
  library.impl(
      "source_analysis_linear_adjoint",
      &source_analysis_linear_adjoint_cuda);
}
