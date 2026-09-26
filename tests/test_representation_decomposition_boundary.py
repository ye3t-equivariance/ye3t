import inspect

import pytest


pytestmark = pytest.mark.fast


def test_young_subgroup_inventory_is_owned_by_core_package():
    from ye3t import representation_decomposition as decomposition

    source = inspect.getsource(decomposition)
    assert "ye3t_ace" not in source

    inventory = decomposition.build_young_subgroup_irrep_inventory(
        subgroup_partitions=((1,), (1,)),
        l_in=(0, 0),
        target_partitions=((2,),),
        target_L_R_values=(0,),
        validate_coherence=False,
    )

    assert isinstance(inventory, decomposition.YE3IrrepInventory)
    assert inventory.validation.passed
    assert inventory.metadata["scope"] == "symmetric_group_young_subgroup_specht_inventory"
    assert inventory.metadata["coefficient_convention"] == "ye3t_young_orthogonal_subduction"
    assert inventory.records
    assert inventory.records[0].coefficient_backend


def test_arbitrary_decomposition_boundary_object_remains_explicitly_planned():
    from ye3t import representation_decomposition as decomposition

    backend = decomposition.RepresentationDecompositionBackend()
    report = backend.status_report()

    assert report["status"] == "planned"
    assert report["implemented_backend"] == "young_subgroup_specht_inventory_for_symmetric_groups"
    assert "matrix units" in report["not_implemented_scope"]
