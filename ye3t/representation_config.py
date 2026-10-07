"""Rank-resolved public representation requests for compiler-backed catalogues."""

from collections.abc import Mapping
from numbers import Integral

from ye3t._record import recordclass
from ye3t.spec import _target_partition


def _keys(payload, allowed, name):
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    unknown = set(payload) - set(allowed)
    if unknown:
        raise ValueError(f"{name} has unsupported fields: {sorted(unknown)!r}.")


def _integer(value, name):
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not a boolean.")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise TypeError(f"{name} must be an integer.")


def _integer_key_map(value, name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    result = {}
    for raw_key, item in value.items():
        key = _integer(raw_key, f"{name} key")
        if key in result:
            raise ValueError(f"{name} contains duplicate normalized key {key}.")
        result[key] = item
    return result


def _rank_map(value, ranks, name, minimum):
    entries = {rank: _integer(number, name) for rank, number in _integer_key_map(value, name).items()}
    if set(entries) != set(ranks):
        raise ValueError(f"{name} must have exactly ranks {ranks!r}.")
    if any(number < minimum for number in entries.values()):
        raise ValueError(f"{name} entries must be at least {minimum}.")
    return tuple((rank, entries[rank]) for rank in ranks)


def _partition(value, rank):
    if isinstance(value, (tuple, list)):
        value = "young:" + ",".join(str(_integer(part, "Young partition part")) for part in value)
    else:
        value = str(value).strip()
        if value.startswith("(") and value.endswith(")"):
            value = "young:" + value
    return _target_partition(value, rank)


@recordclass(
    (
        "group", "ranks", "parent_partitions", "L", "parity", "factorization",
        "subspace", "eta_count_per_rank", "l_max_per_rank", "young_kappa",
        "block_rotation",
    ),
    frozen=True,
)
class YE3TRepresentation:
    """Purpose: Hold an immutable rank-resolved abstract representation request.

    Mathematical contract: Each rank has its own S_N parent partition and one
    O(3) or SO(3) output type. Fixed-content counts delegate to ye3t.couplings.
    Inputs: The representation section of the public config schema.
    Outputs: Parsed policies, rank partitions, and compiler count reports.
    Does not: Enumerate atomistic channels or materialize coefficients.
    """

    @classmethod
    def from_config(cls, config):
        """Validate the public representation section into immutable fields."""
        _keys(config, {
            "group", "ranks", "parent", "factorization", "subspace",
            "uncoupled_factor_inputs", "intermediates",
        }, "representation")
        group = str(config["group"])
        if group not in {"O3", "SO3"}:
            raise ValueError("representation.group must be 'O3' or 'SO3'.")
        raw_ranks = config["ranks"]
        if not isinstance(raw_ranks, (tuple, list)):
            raise TypeError("representation.ranks must be a sequence of positive integers.")
        ranks = tuple(_integer(rank, "representation rank") for rank in raw_ranks)
        if not ranks or any(rank < 1 for rank in ranks) or tuple(sorted(set(ranks))) != ranks:
            raise ValueError("representation.ranks must be nonempty, positive, unique, and ascending.")

        parent = config["parent"]
        _keys(parent, {"young_lambda", "L", "parity"}, "representation.parent")
        L = _integer(parent["L"], "representation.parent.L")
        if L < 0:
            raise ValueError("representation.parent.L must be nonnegative.")
        parity = parent.get("parity")
        if group == "O3":
            if parity not in {"even", "odd"}:
                raise ValueError("O3 representation.parent.parity must be 'even' or 'odd'.")
        elif parity not in {None, "none"}:
            raise ValueError("SO3 omits representation.parent.parity.")
        if isinstance(parent["young_lambda"], Mapping):
            partition_inputs = _integer_key_map(parent["young_lambda"], "parent.young_lambda")
            if set(partition_inputs) != set(ranks):
                raise ValueError("parent.young_lambda rank map must cover exactly representation.ranks.")
            partitions = tuple((rank, _partition(partition_inputs[rank], rank)) for rank in ranks)
        else:
            partitions = tuple((rank, _partition(parent["young_lambda"], rank)) for rank in ranks)

        factorization = str(config["factorization"])
        if factorization not in {"cauchy", "standard_split"}:
            raise ValueError("representation.factorization must be 'cauchy' or 'standard_split'.")
        subspace = str(config["subspace"])
        if subspace not in {"full", "primitive", "decomposable"}:
            raise ValueError("representation.subspace must be full, primitive, or decomposable.")

        inputs = config["uncoupled_factor_inputs"]
        _keys(inputs, {"eta_count_per_rank", "l_max_per_rank"}, "uncoupled_factor_inputs")
        eta = _rank_map(inputs["eta_count_per_rank"], ranks, "eta_count_per_rank", 1)
        angular = _rank_map(inputs["l_max_per_rank"], ranks, "l_max_per_rank", 0)

        intermediate = config.get("intermediates", {})
        _keys(intermediate, {"young_kappa", "block_rotation"}, "intermediates")
        young = intermediate.get("young_kappa", "all_valid")
        if isinstance(young, str):
            if young not in {"all_valid", "symmetric_only"}:
                raise ValueError("young_kappa must be all_valid, symmetric_only, or explicit.")
            young = (young, ())
        else:
            _keys(young, {"policy", "by_block_size"}, "young_kappa")
            if young.get("policy") != "explicit":
                raise ValueError("young_kappa mapping requires policy='explicit'.")
            blocks = young.get("by_block_size", {})
            if not isinstance(blocks, Mapping) or not blocks:
                raise ValueError("explicit young_kappa requires by_block_size.")
            parsed_blocks = _integer_key_map(blocks, "young_kappa.by_block_size")
            young = ("explicit", tuple(sorted(
                (size, tuple(_partition(part, size) for part in values))
                for size, values in parsed_blocks.items()
            )))
            if any(size < 1 or not parts for size, parts in young[1]):
                raise ValueError("explicit young_kappa blocks must be nonempty and positive.")
            if any(len(set(parts)) != len(parts) for _size, parts in young[1]):
                raise ValueError("explicit young_kappa cannot repeat a block partition.")

        rotation = intermediate.get("block_rotation", {"policy": "all_valid"})
        _keys(rotation, {"policy", "Lambda_block_max_per_rank", "Lambda_values_by_block_size"}, "block_rotation")
        policy = rotation.get("policy")
        if policy == "all_valid":
            _keys(rotation, {"policy"}, "block_rotation all_valid")
            rotation = (policy, ())
        elif policy == "capped":
            _keys(rotation, {"policy", "Lambda_block_max_per_rank"}, "block_rotation capped")
            rotation = (policy, _rank_map(
                rotation["Lambda_block_max_per_rank"], ranks, "Lambda_block_max_per_rank", 0
            ))
        elif policy == "explicit":
            _keys(rotation, {"policy", "Lambda_values_by_block_size"}, "block_rotation explicit")
            values = rotation["Lambda_values_by_block_size"]
            if not isinstance(values, Mapping) or not values:
                raise ValueError("explicit block_rotation requires Lambda_values_by_block_size.")
            parsed_blocks = _integer_key_map(values, "Lambda_values_by_block_size")
            rows = []
            for size, lambdas in parsed_blocks.items():
                selected = tuple(_integer(Lambda, "block Lambda") for Lambda in lambdas)
                if len(set(selected)) != len(selected):
                    raise ValueError("explicit block rotations cannot repeat a Lambda value.")
                rows.append((size, tuple(sorted(selected))))
            rotation = (policy, tuple(sorted(rows)))
            if any(size < 1 or not lambdas or min(lambdas) < 0 for size, lambdas in rotation[1]):
                raise ValueError("explicit block rotations must be nonempty and nonnegative.")
        else:
            raise ValueError("block_rotation.policy must be all_valid, capped, or explicit.")

        return cls(group, ranks, partitions, L, parity, factorization, subspace,
                   eta, angular, young, rotation)

    def parent_partition(self, rank):
        """Return the exact Young parent partition for a selected rank."""
        rank = int(rank)
        for selected_rank, partition in self.parent_partitions:
            if selected_rank == rank:
                return partition
        raise ValueError(f"Rank {rank} is not selected by this representation.")

    def count_fixed_content(self, content, input_Ls, *, carrier="external_tensor"):
        """Count a full fixed-content sector through the exact compiler API.

        Restricted intermediate policies require a separate certified plan.
        This method does not materialize coupling coefficients.
        """
        if self.subspace != "full" or self.young_kappa[0] != "all_valid" or self.block_rotation[0] != "all_valid":
            raise ValueError("Fixed-content count with restricted subspace or intermediates needs a specialized compiler plan.")
        content = tuple(content)
        input_Ls = tuple(int(value) for value in input_Ls)
        rank = len(content)
        if len(input_Ls) != rank:
            raise ValueError("content and input_Ls must have the same rank.")
        partition = self.parent_partition(rank)
        l_max = dict(self.l_max_per_rank)[rank]
        if any(value < 0 or value > l_max for value in input_Ls):
            raise ValueError(f"input_Ls must lie in 0..{l_max} for rank {rank}.")
        from ye3t.couplings import count

        return count(
            {
                "content": content,
                "target_rotation": {
                    "L_R": self.L,
                    "group": self.group,
                    "parity": self.parity,
                },
                "target_permutation": "young:" + ",".join(str(part) for part in partition),
                "carrier": carrier,
                "metadata": {"input_Ls": input_Ls},
            },
            input_Ls=input_Ls,
        )

    def to_dict(self):
        """Return a detached canonical config mapping for serialization."""
        if self.young_kappa[0] == "explicit":
            young = {"policy": "explicit", "by_block_size": {
                size: [list(part) for part in parts] for size, parts in self.young_kappa[1]
            }}
        else:
            young = self.young_kappa[0]
        rotation = {"policy": self.block_rotation[0]}
        if self.block_rotation[0] == "capped":
            rotation["Lambda_block_max_per_rank"] = dict(self.block_rotation[1])
        elif self.block_rotation[0] == "explicit":
            rotation["Lambda_values_by_block_size"] = {
                size: list(values) for size, values in self.block_rotation[1]
            }
        return {
            "group": self.group,
            "ranks": list(self.ranks),
            "parent": {
                "young_lambda": {rank: list(part) for rank, part in self.parent_partitions},
                "L": self.L,
                "parity": self.parity,
            },
            "factorization": self.factorization,
            "subspace": self.subspace,
            "uncoupled_factor_inputs": {
                "eta_count_per_rank": dict(self.eta_count_per_rank),
                "l_max_per_rank": dict(self.l_max_per_rank),
            },
            "intermediates": {"young_kappa": young, "block_rotation": rotation},
        }
