"""Orbit-metric construction agrees with every ordered reference coefficient."""
import pytest

from ye3t.couplings import lifted_cauchy_scalar as core


@pytest.mark.parametrize('key', [
    (1,2,(2,),1,0), (2,3,(2,1),1,1), (3,3,(2,1),1,1),
    (3,4,(2,2),1,0), (2,6,(4,2),1,0), (3,3,(2,1),2,2),
])
def test_joint_orbit_metric_matches_ordered_metric_dual(key):
    reference = core._build_block_template(key)
    payload = reference['payload']
    role = payload['role_carrier']
    angular = payload['angular_carrier']
    role_states = tuple(map(tuple,role['state_order']))
    magnetic_states = tuple(map(tuple,angular['state_order']))
    role_basis = core._basis_matrix_from_carrier_payload(role,len(role_states),len(role['coordinate_order']))
    angular_basis = core._basis_matrix_from_carrier_payload(angular,len(magnetic_states),len(angular['coordinate_order']))
    role_vectors = {tuple(coordinate):role_basis[:,index] for index,coordinate in enumerate(role['coordinate_order'])}
    angular_vectors = {tuple(coordinate):angular_basis[:,index] for index,coordinate in enumerate(angular['coordinate_order'])}
    pairings = {(record['role_copy_index'],record['angular_copy_index']):
                core._exact_matrix_from_payload(record['matrix']) for record in payload['pairings']}
    labels = tuple((row['role_copy_index'],row['angular_copy_index'],row['M']) for row in payload['analysis_rows'])
    gram, rows = core._joint_symmetric_power_metric_dual(role_states,magnetic_states,
        labels,role_vectors,angular_vectors,pairings,payload['tableau_count'])
    assert gram == core._exact_matrix_from_payload(payload['synthesis_gram'])
    for label,row in zip(labels,rows,strict=True):
        actual = {tuple((role,magnetic+key[3]) for role,magnetic in word):value
                  for word,value in row.items()}
        assert actual == reference['canonical_rows'][label]
