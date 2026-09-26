#pragma once

#include <complex>
#include <cstdint>

namespace ye3t {
namespace runtime {

template <typename Real>
void cheb_exp_cos_radial_with_derivative(
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t radial_index,
    Real* values,
    Real* derivatives);

template <typename Real>
void cheb_exp_cos_radial_table_with_derivative(
    const Real* radii,
    const Real* cutoffs,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t maximum_radial_index,
    Real* values,
    Real* derivatives);

// Direct PACE-compatible ChebExpCos base table. Output column k corresponds
// to the one-based `.yace` radial label n=k+1 and includes the outer dcut
// switch. Spline interpolation and inner cutoffs are separate operations.
template <typename Real>
void pace_cheb_exp_cos_radial_table_with_derivative(
    const Real* radii,
    const Real* cutoffs,
    const Real* cutoff_widths,
    const Real* lambdas,
    std::int64_t edge_count,
    std::int64_t radial_count,
    Real* values,
    Real* derivatives);

// PACE's lookup grid contains floor(cutoff / requested_spacing) uniform
// intervals. Its effective spacing is therefore cutoff / interval_count.
template <typename Real>
std::int64_t pace_uniform_spline_interval_count(
    Real requested_spacing,
    Real cutoff);

// Build PACE-compatible cubic-Hermite coefficients. Samples use the layout
// [node][function] for nodes 1..interval_count at r=node*cutoff/interval_count.
// Coefficients use [interval][function][c0,c1,c2,c3]; interval zero is unused.
template <typename Real>
void pace_uniform_cubic_spline_build(
    const Real* node_values,
    const Real* node_derivatives,
    std::int64_t interval_count,
    std::int64_t function_count,
    Real cutoff,
    Real* coefficients);

template <typename Real>
void pace_uniform_cubic_spline_evaluate_with_derivative(
    const Real* radii,
    const Real* coefficients,
    std::int64_t point_count,
    std::int64_t function_count,
    std::int64_t interval_count,
    Real cutoff,
    Real* values,
    Real* derivatives);

// Apply a row-major [output][input] radial coefficient table to batched base
// values and their radial derivatives. The output layout is [point][output].
template <typename Real>
void pace_radial_channel_contraction_with_derivative(
    const Real* input_values,
    const Real* input_derivatives,
    const Real* coefficients,
    std::int64_t point_count,
    std::int64_t input_count,
    std::int64_t output_count,
    Real* output_values,
    Real* output_derivatives);

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
    Real* radii_gradient);

template <typename Real>
void complex_spherical_harmonics_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t angular_momentum,
    Real epsilon,
    std::complex<Real>* values,
    std::complex<Real>* derivatives);

template <typename Real>
void complex_spherical_harmonics_table_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    std::complex<Real>* values,
    std::complex<Real>* derivatives,
    Real output_scale = Real(1));

std::int64_t complex_spherical_harmonics_table_plan_size(
    std::int64_t maximum_angular_momentum);

std::int64_t complex_spherical_harmonics_nonnegative_table_width(
    std::int64_t maximum_angular_momentum);

template <typename Real>
void build_complex_spherical_harmonics_table_plan(
    std::int64_t maximum_angular_momentum,
    Real* plan,
    std::int64_t plan_size,
    Real normalization_scale = Real(1));

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
    Real output_scale = Real(1));

// Store only m >= 0 in rows packed as l*(l+1)/2 + m. Negative-m values and
// derivatives are recovered exactly as (-1)^m conjugate(Y_lm).
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
    Real output_scale = Real(1));

// Equivalent nonnegative-m table for callers that already cache unit bond
// directions and positive radii. This avoids repeating a square root and
// normalization in the angular stage.
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
    std::complex<Real>* derivatives);

// Streaming degree/order recurrence for the same normalized nonnegative-m
// harmonics, packed as l*(l+1)/2 + m like the table above. The plan holds
// plan[0] = scaled Y_0^0 and, for index = l*(l+1)/2 + m, the pair
// (plan[1 + 2*index], plan[2 + 2*index]). For l == m the first entry is the
// azimuthal step Y_mm = a_mm (u_x + i u_y) Y_{m-1,m-1}; for l > m the pair is
// the degree step Y_lm = a_lm u_z Y_{l-1,m} - b_lm Y_{l-2,m}. Coefficients are
// built once; evaluation performs no trigonometry, division by sin(theta), or
// per-edge coefficient construction, and the work is proportional to the
// number of stored harmonics. The Condon-Shortley phase and normalization
// match the polynomial-table plan for the same normalization scale.
std::int64_t complex_spherical_harmonics_recurrence_plan_size(
    std::int64_t maximum_angular_momentum);

template <typename Real>
void build_complex_spherical_harmonics_recurrence_plan(
    std::int64_t maximum_angular_momentum,
    Real* plan,
    std::int64_t plan_size,
    Real normalization_scale = Real(1));

// Unit directions and positive radii are prevalidated by the caller. The
// Cartesian derivatives are with respect to the raw displacement, so they
// include the 1/radius chain rule of the unit-vector projection.
template <typename Real>
void complex_spherical_harmonics_nonnegative_unit_recurrence_with_derivative_prevalidated(
    const Real* unit_vectors,
    const Real* radii,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    const Real* plan,
    std::int64_t plan_size,
    std::complex<Real>* values,
    std::complex<Real>* derivatives);

template <typename Real>
void real_spherical_harmonics_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t angular_momentum,
    Real epsilon,
    Real* values,
    Real* derivatives);

template <typename Real>
void real_spherical_harmonics_table_with_derivative(
    const Real* edge_vectors,
    std::int64_t edge_count,
    std::int64_t maximum_angular_momentum,
    Real epsilon,
    Real* values,
    Real* derivatives);

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
    Scalar* edge_charge_derivatives_neighbor);

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
    Real* edge_derivatives);

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
    Scalar* edge_charge_adjoint_neighbor);

template <typename Scalar>
void density_accumulate_forward(
    const Scalar* edge_values,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t channel_count,
    std::int64_t atom_count,
    Scalar* atomic_values);

template <typename Scalar>
void density_accumulate_adjoint(
    const Scalar* atomic_adjoint,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t channel_count,
    Scalar* edge_adjoint);

template <typename Scalar>
void edge_outer_accumulate_forward(
    const Scalar* left,
    const Scalar* right,
    const std::int64_t* centers,
    std::int64_t edge_count,
    std::int64_t left_dimension,
    std::int64_t right_dimension,
    std::int64_t atom_count,
    Scalar* atomic_values);

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
    Scalar* right_adjoint);

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
    Scalar* right_second_adjoint);

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
    Scalar* atomic_values);

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
    Scalar* edge_adjoint);

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
    Scalar* edge_second_adjoint);

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
    Scalar* atomic_values);

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
    Scalar* soft_weight_adjoint);

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
    Scalar* soft_weight_second_adjoint);

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
    Scalar* target_values);

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
    Scalar* gate_adjoint);

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
    Scalar* gate_second_adjoint);

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
    Scalar* target_values);

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
    Scalar* gate_adjoint);

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
    Scalar* gate_second_adjoint);

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
    Scalar* target_values);

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
    Scalar* gate_adjoint);

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
    Scalar* gate_second_adjoint);

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
    Scalar* target_values);

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
    Real* gate_adjoint);

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
    Real* gate_second_adjoint);

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
    Scalar* target_values);

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
    Real* gate_adjoint);

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
    Real* gate_second_adjoint);

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
    Scalar* target_values);

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
    Real* gate_adjoint);

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
    Real* gate_second_adjoint);

template <typename Scalar>
void source_arena_gather_forward(
    const Scalar* producer,
    const std::int64_t* gather_indices,
    const std::int64_t* center_types,
    const std::int64_t* atom_types,
    std::int64_t batch_size,
    std::int64_t producer_width,
    std::int64_t output_width,
    Scalar* output);

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
    Scalar* producer_adjoint);

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
    Scalar* output);

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
    Control* channel_maps_adjoint);

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
    Control* channel_maps_second_adjoint);

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
    Scalar* output);

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
    Scalar* output);

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
    Scalar* channel_maps_adjoint);

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
    Scalar* channel_maps_second_adjoint);

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
    Scalar* output);

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
    Real* channel_maps_adjoint);

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
    Real* channel_maps_second_adjoint);

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
    Scalar* channel_maps_adjoint);

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
    Real* channel_maps_adjoint);

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
    Scalar* atomic_output_adjoint_tangent);

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
    Scalar* atomic_output_adjoint_tangent);

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
    Scalar* channel_maps_adjoint);

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
    Scalar* channel_maps_second_adjoint);

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
    Scalar* output);

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
    Real* channel_maps_adjoint);

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
    Real* channel_maps_second_adjoint);

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
    Scalar* output);

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
    Scalar* source_adjoint);

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
    Scalar* output);

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
    Scalar* bias_adjoint);

template <typename Scalar>
void compact_pair_product_forward(
    const Scalar* left,
    const Scalar* right,
    std::int64_t batch_size,
    std::int64_t dimension,
    bool antisymmetric,
    Scalar* output);

template <typename Scalar>
void compact_pair_product_adjoint(
    const Scalar* output_adjoint,
    const Scalar* left,
    const Scalar* right,
    std::int64_t batch_size,
    std::int64_t dimension,
    bool antisymmetric,
    Scalar* left_adjoint,
    Scalar* right_adjoint);

template <typename Scalar>
void compact_exterior_power_forward(
    const Scalar* factors,
    std::int64_t batch_size,
    std::int64_t order,
    std::int64_t dimension,
    Scalar* output);

template <typename Scalar>
void compact_exterior_power_adjoint(
    const Scalar* output_adjoint,
    const Scalar* factors,
    std::int64_t batch_size,
    std::int64_t order,
    std::int64_t dimension,
    Scalar* factors_adjoint);

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
    Scalar* output);

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
    Scalar* input_adjoint);

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
    Scalar* output);

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
    Scalar* input_adjoint);

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
    Scalar* input_adjoint);

// Evaluate a scalar linear readout of shared symmetric-power monomials and
// its unit-seed Hermitian adjoint in one pass. Each monomial stores only its
// nonzero factors in factor_offsets/factor_indices/factor_exponents. This is
// the validated convenience form used after a compiler or exact explicit
// polynomial loader has lowered and deduplicated the monomial table.
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
    Scalar* input_adjoint);

// Execute a loader-validated sparse plan without rescanning its table or
// allocating scratch. The caller supplies at least
// 4*maximum_term_factors+2 Scalar entries and certifies every factor index,
// exponent, offset, and maximum_term_factors bound before entering this path.
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
    Scalar* input_adjoint);

// Execute the same scalar polynomial through a loader-validated prefix DAG.
// Power nodes cache repeated x[channel]^exponent factors, while product nodes
// share common monomial prefixes. Node zero is the multiplicative root and the
// next root_node_count nodes are its direct children. All remaining nodes are
// topologically ordered after that contiguous root partition. The caller
// supplies 2*power_count+2*node_count Scalar workspace entries.
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
    Scalar* input_adjoint);

// Execute a loader-validated binary product DAG. Operand indices address one
// packed value/adjoint array: cached powers first, then topologically ordered
// product nodes. A negative monomial operand denotes the constant one.
template <typename Scalar, typename CoefficientScalar>
void symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_prevalidated(
    const Scalar *input, std::int64_t batch_size, std::int64_t input_dimension,
    const std::int64_t *power_channels, const std::int64_t *power_exponents,
    std::int64_t power_count, const std::int64_t *node_left,
    const std::int64_t *node_right, std::int64_t node_count,
    const std::int64_t *monomial_operands,
    const CoefficientScalar *monomial_coefficients, std::int64_t monomial_count,
    Scalar *workspace, std::int64_t workspace_size, Scalar *output,
    Scalar *input_adjoint);

// Execute the complex-input, real-coefficient binary DAG in small atom tiles.
// Real and imaginary lanes are split in caller-owned workspace so the product
// and reverse-product loops expose contiguous real arithmetic to the CPU.
// The caller supplies
// (4*(power_count+node_count)+2)*tile_size Real entries.
template <typename Real>
void symmetric_power_sparse_monomial_binary_dag_linear_forward_adjoint_split_real_tiled_prevalidated(
    const std::complex<Real> *input, std::int64_t batch_size,
    std::int64_t input_dimension, const std::int64_t *power_channels,
    const std::int64_t *power_exponents, std::int64_t power_count,
    const std::int64_t *node_left, const std::int64_t *node_right,
    std::int64_t node_count, const std::int64_t *monomial_operands,
    const Real *monomial_coefficients, std::int64_t monomial_count,
    std::int64_t tile_size, Real *workspace, std::int64_t workspace_size,
    std::complex<Real> *output, std::complex<Real> *input_adjoint);

// Execute a loader-validated linear readout of sparse bilinear coupling DAGs.
// Component indices in the coefficient and readout arrays address the packed
// node workspace directly. Leaf components gather arbitrary entries from each
// input row. Output values and input adjoints are accumulated, not overwritten.
// The caller supplies (4*total_node_components+2)*tile_size Real entries.
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
    std::complex<Real> *input_adjoint);

template <typename Scalar>
void factorized_angular_forward(
    const Scalar *packed_slots, std::int64_t batch_size,
    std::int64_t input_dimension, std::int64_t source_dimension,
    const std::int64_t *node_offsets, const std::int64_t *node_dimensions,
    const std::int64_t *node_leaf_offsets, const std::int64_t *node_left,
    const std::int64_t *node_right,
    const std::int64_t *node_coefficient_offsets, std::int64_t node_count,
    const std::int64_t *coefficient_rows,
    const std::int64_t *coefficient_columns, const Scalar *coefficient_values,
    const std::int64_t *root_nodes, std::int64_t root_count,
    const std::int64_t *root_projection_starts,
    const std::int64_t *root_projection_dimensions,
    const std::int64_t *root_output_offsets, const Scalar *projection_values,
    std::int64_t total_projection_dimension, std::int64_t output_dimension,
    Scalar *output);

template <typename Scalar>
void factorized_angular_adjoint(
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
    Scalar* packed_slots_adjoint);

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
    Scalar* packed_slots_tangent);

template <typename Scalar>
struct FactorizedAngularSegmentedPlanView {
  const std::int64_t* segment_source_offsets;
  const std::int64_t* segment_input_offsets;
  const std::int64_t* segment_input_dimensions;
  const std::int64_t* segment_workspace_offsets;
  const std::int64_t* segment_workspace_dimensions;
  const std::int64_t* segment_node_offsets;
  const std::int64_t* segment_root_offsets;
  const std::int64_t* segment_projection_offsets;
  const std::int64_t* segment_projection_dimensions;
  const std::int64_t* segment_output_offsets;
  const std::int64_t* node_offsets;
  const std::int64_t* node_dimensions;
  const std::int64_t* node_leaf_offsets;
  const std::int64_t* node_left;
  const std::int64_t* node_right;
  const std::int64_t* node_coefficient_offsets;
  const std::int64_t* coefficient_rows;
  const std::int64_t* coefficient_columns;
  const Scalar* coefficient_values;
  const std::int64_t* root_nodes;
  const std::int64_t* root_projection_starts;
  const std::int64_t* root_projection_dimensions;
  const std::int64_t* root_output_offsets;
  const Scalar* projection_values;
  std::int64_t segment_count;
  std::int64_t packed_input_dimension;
  std::int64_t workspace_dimension;
  std::int64_t output_dimension;
};

template <typename Scalar>
void factorized_angular_segmented_forward(
    const Scalar* packed_slots,
    std::int64_t batch_size,
    const FactorizedAngularSegmentedPlanView<Scalar>& plan,
    Scalar* output);

template <typename Scalar>
void factorized_angular_segmented_adjoint(
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    const FactorizedAngularSegmentedPlanView<Scalar>& plan,
    Scalar* packed_slots_adjoint);

template <typename Scalar>
void factorized_angular_segmented_double_backward(
    const Scalar* packed_adjoint_tangent,
    const Scalar* output_adjoint,
    const Scalar* packed_slots,
    std::int64_t batch_size,
    const FactorizedAngularSegmentedPlanView<Scalar>& plan,
    Scalar* output_tangent,
    Scalar* packed_slots_tangent);

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
    Scalar* output);

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
    Scalar* bias_adjoint);

}  // namespace runtime
}  // namespace ye3t
