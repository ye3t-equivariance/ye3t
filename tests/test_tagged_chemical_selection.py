"""Chemical selection is compiler-owned and precedes coefficient construction."""
import pytest
from ye3t.couplings import count, tagged_cauchy_image_request


def test_low_rank_complete_and_high_rank_selected_chemistry(monkeypatch):
    import ye3t.couplings as couplings
    def forbidden(*args, **kwargs):
        raise AssertionError('Counts must not construct coefficients')
    monkeypatch.setattr(couplings, 'compile', forbidden)
    request = tagged_cauchy_image_request(species=['H', 'C', 'O', 'S', 'K'], catalogue={
        'nmax_per_rank': {2: 1, 4: 1, 6: 1}, 'lmax_per_rank': {2: 0, 4: 0, 6: 0},
        'source_block_partitions_by_rank': {2: [[2], [1, 1]], 4: [[2, 2]], 6: [[4, 2]]},
        'tag_counts_by_rank': {2: [0], 4: [0], 6: [0]},
        'chemical_patterns_by_rank': {4: [['O', 'H', 'O', 'H']], 6: [['O']*4+['H']*2]},
        'max_records_per_rank': 1000,
    })
    report = count(request)
    components = report.resource_report['components']
    by_rank = {rank: [row for row in components if row['tensor_order'] == rank] for rank in (2, 4, 6)}
    assert len(by_rank[2]) == 15
    assert len(by_rank[4]) == 1
    assert len(by_rank[6]) == 1
    assert request['catalogue']['chemical_patterns_by_rank']['4'] == (('H', 'H', 'O', 'O'),)


def test_chemical_pattern_unknown_species_fails_before_materialization():
    with pytest.raises(ValueError, match='chemical_patterns_by_rank'):
        tagged_cauchy_image_request(species=['H', 'O'], catalogue={
            'nmax_per_rank': {4: 1}, 'lmax_per_rank': {4: 0},
            'source_block_partitions_by_rank': {4: [[4]]},
            'chemical_patterns_by_rank': {4: [['Na']*4]},
        })
