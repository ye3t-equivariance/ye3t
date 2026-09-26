import math


def test_graph_template_automorphisms_and_decorated_intersection():
    from ye3t.representations import decorated_automorphism_group, graph_template_automorphisms, slot_orbits

    path = graph_template_automorphisms(3, ((0, 1), (1, 2)))
    assert set(path) == {(0, 1, 2), (2, 1, 0)}
    assert slot_orbits(3, path) == ((0, 2), (1,))
    assert decorated_automorphism_group(3, ((0, 1), (1, 2)), ("x", "y", "x")) == path
    assert decorated_automorphism_group(3, ((0, 1), (1, 2)), ("x", "y", "z")) == ((0, 1, 2),)


def test_rooted_typed_automorphisms_and_repeated_factor_lift():
    from ye3t.representations import induced_factor_role_permutations
    from ye3t.representations import rooted_typed_graph_automorphisms

    edges = ((0, 1), (0, 2))
    automorphisms = rooted_typed_graph_automorphisms(
        3,
        edges,
        root_vertex=0,
        vertex_types=("K", "O", "O"),
    )
    assert automorphisms == ((0, 1, 2), (0, 2, 1))
    lifts = induced_factor_role_permutations(
        edges,
        (0, 0, 1, 1),
        automorphisms,
        factor_labels=("a", "b", "a", "b"),
    )
    assert lifts == ((0, 1, 2, 3), (2, 3, 0, 1))


def test_rooted_typed_automorphisms_respect_chemical_labels():
    from ye3t.representations import rooted_typed_graph_automorphisms

    assert rooted_typed_graph_automorphisms(
        3,
        ((0, 1), (0, 2)),
        root_vertex=0,
        vertex_types=("K", "O", "H"),
    ) == ((0, 1, 2),)


def test_rooted_star_generators_have_exact_s8_order():
    from ye3t.representations import (
        rooted_typed_star_automorphism_generators,
    )

    result = rooted_typed_star_automorphism_generators(
        9,
        tuple((0, index) for index in range(1, 9)),
        root_vertex=0,
        vertex_types=("O",) + ("H",) * 8,
        edge_types=("O-H",) * 8,
        directed=True,
    )
    assert result["order"] == math.factorial(8)
    assert len(result["generators"]) == 8
    assert result["symmetric_vertex_blocks"] == (
        tuple(range(1, 9)),
    )
