"""Tagged Cauchy catalogue builder: enumerate, screen, pool, and select.

Mathematical/procedural contract:
    A "content" is a tuple of (channel_index, k_b) blocks with strictly
    increasing channel indices and sum k_b = N (1..rank_max), skipping
    contents with an odd global parity sum(k_b * l_b) (no even scalar
    exists). For each content, a compiler request is built with
    ``lifted_cauchy_fixed_content_scalar_request``; blocks larger than
    ``lambda_block_size_max`` are policy-restricted to their Lambda=0
    (ordinary density) sector, all other contents use the full Lambda
    enumeration. Contents are screened by their exact label count
    (``ye3t.couplings.count``, no coefficients); survivors are compiled,
    cached on disk by an exact hash of their request, support-filtered
    (``tagged_cauchy.tag_support_report``) and pooled
    (``tagged_cauchy.pooled_tagged_basis``). This is an exact tag-relabel-
    reduced raw frame, not a certified post-pooling image basis. Every feature is
    classified into a stratum (N, kappa_class, lambda_class) read off its
    sector's shared block kappas/Lambdas (checked, not assumed), and
    stratum quotas select the final catalogue deterministically (stable
    sort, then a prefix). Nothing here performs any numerical symmetry-
    basis construction: enumeration, hashing, and selection are exact and
    order-stable; the only floating-point content is the coefficients the
    compiler itself already produced.
"""

from collections import OrderedDict
import itertools
import json
import os
import time

from ye3t.cache.artifacts import ArtifactCacheMiss, YE3TArtifactStore
from ye3t.couplings import (
    compile as compile_coupling,
    count as count_coupling,
    plan as plan_coupling,
    lifted_cauchy_fixed_content_scalar_request,
)
from ye3t.couplings.lifted_cauchy_scalar import (
    CompiledLiftedCauchyScalar,
    LIFTED_CAUCHY_BLOCK_CACHE_SCHEMA,
    LIFTED_CAUCHY_SCALAR_CONVENTION,
    _block_cache_dependencies,
    _block_cache_request,
    _block_template_from_payload,
    _freeze_json,
    _hook_content_dimension,
    _stable_hash,
    _template_cache_certificate,
    _validate_cached_block_template,
)
from ye3t.couplings.tagged_cauchy import (
    _all_contents,
    block_template_role_copy_contents,
    kostka_number,
    normalize_role_bindings,
    pooled_tagged_basis,
    tag_support_report,
)


CATALOGUE_SCHEMA = "ye3t_tagged_catalogue_v1"


def _normalize_catalogue_spec(spec):
    spec = dict(spec)
    species = tuple(str(value) for value in spec["species"])
    if not species or len(set(species)) != len(species):
        raise ValueError("species must be a nonempty tuple of unique symbols.")
    raw_channels = dict(spec["channels"])
    if set(raw_channels) != set(species):
        raise ValueError(
            "channels must have exactly one entry per species in `species`."
        )
    channels_by_species = {}
    for species_symbol in species:
        pairs = tuple(
            (int(radial_index), int(l)) for radial_index, l in raw_channels[species_symbol]
        )
        if not pairs:
            raise ValueError(f"Species {species_symbol!r} has no (radial_index, l) pairs.")
        if len(set(pairs)) != len(pairs):
            raise ValueError(
                f"Duplicate (radial_index, l) pair for species {species_symbol!r}."
            )
        if any(radial_index < 0 or l < 0 for radial_index, l in pairs):
            raise ValueError("radial_index and l must be nonnegative.")
        channels_by_species[species_symbol] = pairs
    source_family_id = str(spec["source_family_id"])
    rank_max = int(spec["rank_max"])
    if rank_max <= 0:
        raise ValueError("rank_max must be positive.")
    lambda_block_size_max = int(spec["lambda_block_size_max"])
    if lambda_block_size_max <= 0:
        raise ValueError("lambda_block_size_max must be positive.")
    role_bindings = tuple(tuple(binding) for binding in spec["role_bindings"])
    role_dimension = len(role_bindings)
    normalize_role_bindings(role_bindings, role_dimension)
    kappa_policy = str(spec["kappa_policy"]).strip().lower()
    if kappa_policy not in {"all", "trivial"}:
        raise ValueError("kappa_policy must be 'all' or 'trivial'.")
    raw_quotas = spec["quotas"]
    if raw_quotas == "all":
        quotas = "all"
    else:
        quotas = {}
        for key, value in dict(raw_quotas).items():
            key = tuple(key)
            if len(key) != 3:
                raise ValueError(f"Quota stratum key must have length 3: {key!r}.")
            value = dict(value)
            minimum = int(value.get("min", 0))
            maximum = value.get("max", None)
            if maximum is not None:
                maximum = int(maximum)
            if minimum < 0 or (maximum is not None and maximum < minimum):
                raise ValueError(f"Invalid quota bounds for stratum {key!r}: {value!r}.")
            quotas[key] = {"min": minimum, "max": maximum}
    resource_limits = (
        None if spec.get("resource_limits") is None else dict(spec["resource_limits"])
    )
    cache_dir = spec.get("cache_dir", None)
    if cache_dir is not None:
        cache_dir = str(cache_dir)
    count_only = bool(spec.get("count_only", False))
    return {
        "species": species,
        "channels": channels_by_species,
        "source_family_id": source_family_id,
        "rank_max": rank_max,
        "lambda_block_size_max": lambda_block_size_max,
        "role_bindings": role_bindings,
        "role_dimension": int(role_dimension),
        "kappa_policy": kappa_policy,
        "quotas": quotas,
        "resource_limits": resource_limits,
        "cache_dir": cache_dir,
        "count_only": count_only,
    }


def _build_channel_list(normalized_spec):
    """Channel dicts in fixed (species order, then l, then radial index) order."""

    channels = []
    channel_index = 0
    for species_symbol in normalized_spec["species"]:
        pairs = normalized_spec["channels"][species_symbol]
        for radial_index, l in sorted(pairs, key=lambda pair: (pair[1], pair[0])):
            channels.append(
                {
                    "channel_index": int(channel_index),
                    "neighbor_species": species_symbol,
                    "radial_channel": int(radial_index),
                    "l": int(l),
                    "source_family_id": normalized_spec["source_family_id"],
                }
            )
            channel_index += 1
    return tuple(channels)


def _compositions(total, parts):
    """All ordered tuples of `parts` positive ints summing to `total`."""

    if parts == 1:
        yield (total,)
        return
    for first in range(1, total - (parts - 1) + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def _enumerate_contents(num_channels, N):
    """All (channel_index, k_b) block tuples, strictly increasing channels, sum=N."""

    for num_blocks in range(1, min(N, num_channels) + 1):
        for chosen in itertools.combinations(range(num_channels), num_blocks):
            for sizes in _compositions(N, num_blocks):
                yield tuple(zip(chosen, sizes))


def _kappa_class(block_kappas):
    max_rows = max(len(kappa) for kappa in block_kappas)
    if max_rows == 1:
        return "trivial"
    if max_rows == 2:
        return "two_row"
    return "three_row"


def _lambda_class(block_lambdas):
    return "zero" if all(int(value) == 0 for value in block_lambdas) else "positive"


def _feature_stratum(feature, N):
    """(N, kappa_class, lambda_class) for one pooled feature; asserts agreement."""

    labels = feature["constituent_labels"]
    if not labels:
        raise RuntimeError("Pooled feature has no constituent labels.")
    reference_kappas = tuple(
        tuple(int(part) for part in kappa) for kappa in labels[0]["block_kappas"]
    )
    reference_lambdas = tuple(int(value) for value in labels[0]["block_Lambdas"])
    for label in labels[1:]:
        kappas = tuple(tuple(int(part) for part in kappa) for kappa in label["block_kappas"])
        lambdas = tuple(int(value) for value in label["block_Lambdas"])
        if kappas != reference_kappas or lambdas != reference_lambdas:
            raise RuntimeError(
                "Pooled feature constituents disagree on block kappas/Lambdas "
                f"({kappas!r}/{lambdas!r} vs {reference_kappas!r}/{reference_lambdas!r}); "
                "a sector must share these fields by construction."
            )
    return (int(N), _kappa_class(reference_kappas), _lambda_class(reference_lambdas))


def _cache_path(cache_dir, request_hash):
    return os.path.join(cache_dir, f"{request_hash}.json")


def _load_cached_artifact(cache_dir, request_hash):
    if cache_dir is None:
        return None
    path = _cache_path(cache_dir, request_hash)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            body = json.load(handle)
        return CompiledLiftedCauchyScalar.from_dict(body)
    except Exception:
        # Corrupted, truncated, or schema-mismatched cache entry: treat as a
        # miss and let the caller recompile (and overwrite it).
        return None


def _store_cached_artifact(cache_dir, request_hash, compiled):
    if cache_dir is None:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, request_hash)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(compiled.to_dict(), handle)
    os.replace(tmp_path, path)


def _stratum_key_to_string(key):
    N, kappa_class, lambda_class = key
    return f"{int(N)}|{kappa_class}|{lambda_class}"


def _stratum_key_from_string(text):
    N, kappa_class, lambda_class = text.split("|")
    return (int(N), kappa_class, lambda_class)


def build_tagged_catalogue(spec):
    """Enumerate, screen, pool, and quota-select a tagged Cauchy catalogue.

    See the module docstring for the full contract. Raises ValueError if a
    quota stratum's minimum cannot be met, naming the stratum and the
    stage at which its candidates ran out (no labels, all unsupported, or
    all annihilated by tag pooling).
    """

    normalized_spec = _normalize_catalogue_spec(spec)
    channels = _build_channel_list(normalized_spec)
    role_dimension = normalized_spec["role_dimension"]
    role_bindings = normalized_spec["role_bindings"]

    if normalized_spec["count_only"]:
        return _build_count_only_inventory(normalized_spec, channels)

    content_records = []
    candidate_features = []  # dicts with sort keys + payload, pre-selection

    content_index = 0
    for N in range(1, normalized_spec["rank_max"] + 1):
        for blocks in _enumerate_contents(len(channels), N):
            parity = sum(k * channels[channel_index]["l"] for channel_index, k in blocks)
            if parity % 2 != 0:
                continue

            has_oversized = any(
                k > normalized_spec["lambda_block_size_max"] for _, k in blocks
            )
            block_lambda_policy = "zero" if has_oversized else "all"
            channels_subset = tuple(channels[channel_index] for channel_index, _ in blocks)
            block_sizes = tuple(k for _, k in blocks)

            start = time.time()
            outcome = "empty"
            error_text = None
            label_count = 0
            supported_count = 0
            pooled_count = 0
            request_payload = None
            request_hash = None
            artifact_self_hash = None
            cache_hit = False

            request = None
            try:
                request = lifted_cauchy_fixed_content_scalar_request(
                    channels_subset,
                    block_sizes,
                    role_dimension=role_dimension,
                    kappa_policy=normalized_spec["kappa_policy"],
                    block_lambda_policy=block_lambda_policy,
                    emit_factored=False,
                    resource_limits=normalized_spec["resource_limits"],
                )
            except ValueError:
                request = None

            if request is not None:
                request_payload = _freeze_json(request)
                request_hash = _stable_hash(request_payload)
                try:
                    report = count_coupling(request)
                except ValueError:
                    report = None
                if report is not None:
                    label_count = int(report.descriptor_count)
                    if label_count > 0:
                        compiled = _load_cached_artifact(
                            normalized_spec["cache_dir"], request_hash
                        )
                        if compiled is not None:
                            cache_hit = True
                        else:
                            compiled = compile_coupling(plan_coupling(report))
                            _store_cached_artifact(
                                normalized_spec["cache_dir"], request_hash, compiled
                            )
                        artifact_self_hash = str(compiled.self_hash)
                        support = tag_support_report(compiled, role_bindings)
                        supported_count = int(sum(support["supported"]))
                        if supported_count > 0:
                            basis = pooled_tagged_basis(compiled, role_bindings)
                            pooled_count = len(basis["features"])
                            for feature_index, feature in enumerate(basis["features"]):
                                stratum = _feature_stratum(feature, N)
                                combination = tuple(
                                    (
                                        int(descriptor_index),
                                        [
                                            float(coefficient_payload["binary64"][0]),
                                            float(coefficient_payload["binary64"][1]),
                                        ],
                                    )
                                    for descriptor_index, coefficient_payload in feature[
                                        "combination"
                                    ]
                                )
                                candidate_features.append(
                                    {
                                        "sort_key": (
                                            int(N),
                                            int(content_index),
                                            int(feature["sector_index"]),
                                            int(feature_index),
                                        ),
                                        "content_index": int(content_index),
                                        "feature_index": int(feature_index),
                                        "stratum": stratum,
                                        "constituent_labels": feature["constituent_labels"],
                                        "combination": combination,
                                        "combination_exact": tuple(
                                            (
                                                int(descriptor_index),
                                                coefficient_payload,
                                            )
                                            for descriptor_index, coefficient_payload in feature[
                                                "combination"
                                            ]
                                        ),
                                    }
                                )
                            outcome = "ok" if pooled_count > 0 else "annihilated"
                        else:
                            outcome = "unsupported"

            content_records.append(
                {
                    "content_index": int(content_index),
                    "N": int(N),
                    "blocks": tuple((int(ci), int(k)) for ci, k in blocks),
                    "request_payload": request_payload,
                    "request_hash": request_hash,
                    "artifact_self_hash": artifact_self_hash,
                    "label_count": int(label_count),
                    "supported_count": int(supported_count),
                    "pooled_count": int(pooled_count),
                    "compile_seconds": float(time.time() - start),
                    "cache_hit": bool(cache_hit),
                    "outcome": outcome,
                    "error": error_text,
                }
            )
            content_index += 1

    # ---- Stratum grouping, quota selection ---------------------------------
    by_stratum = OrderedDict()
    for candidate in candidate_features:
        by_stratum.setdefault(candidate["stratum"], []).append(candidate)

    quotas = normalized_spec["quotas"]
    for key in quotas if quotas != "all" else ():
        by_stratum.setdefault(key, [])

    strata_counts = {}
    quota_report = {}
    selected_by_stratum = {}
    for stratum_key in sorted(by_stratum):
        members = sorted(by_stratum[stratum_key], key=lambda item: item["sort_key"])
        available = len(members)
        if quotas == "all":
            minimum, maximum = 0, None
        else:
            bounds = quotas.get(stratum_key, {"min": 0, "max": None})
            minimum, maximum = bounds["min"], bounds["max"]
        if available < minimum:
            content_by_index = {c["content_index"]: c for c in content_records}
            stratum_content_indices = {
                c["content_index"]
                for c in content_records
                if c["N"] == stratum_key[0]
            }
            label_total = sum(
                content_by_index[i]["label_count"] for i in stratum_content_indices
            )
            supported_total = sum(
                content_by_index[i]["supported_count"] for i in stratum_content_indices
            )
            if label_total == 0:
                reason = "no labels exist"
            elif supported_total == 0:
                reason = "all were unsupported"
            elif available == 0:
                reason = "all were annihilated"
            else:
                reason = "insufficient pooled features after selection"
            raise ValueError(
                f"Stratum {stratum_key!r} has {available} available pooled feature(s), "
                f"fewer than the required minimum {minimum} ({reason})."
            )
        selected = members if maximum is None else members[:maximum]
        selected_by_stratum[stratum_key] = selected
        strata_counts[_stratum_key_to_string(stratum_key)] = {
            "available": int(available),
            "selected": int(len(selected)),
        }
        if quotas != "all":
            quota_report[_stratum_key_to_string(stratum_key)] = {
                "min": int(minimum),
                "max": None if maximum is None else int(maximum),
                "available": int(available),
                "selected": int(len(selected)),
                "met_minimum": bool(available >= minimum),
            }

    feature_records = []
    global_feature_index = 0
    for stratum_key in sorted(selected_by_stratum):
        for candidate in selected_by_stratum[stratum_key]:
            feature_records.append(
                {
                    "global_feature_index": int(global_feature_index),
                    "content_index": candidate["content_index"],
                    "feature_index": candidate["feature_index"],
                    "stratum": list(stratum_key),
                    "constituent_labels": candidate["constituent_labels"],
                    "combination": candidate["combination"],
                    "combination_exact": candidate["combination_exact"],
                }
            )
            global_feature_index += 1

    record = {
        "schema": CATALOGUE_SCHEMA,
        "spec": _freeze_json(
            {
                **{k: v for k, v in normalized_spec.items() if k != "quotas"},
                "quotas": (
                    "all"
                    if normalized_spec["quotas"] == "all"
                    else {
                        _stratum_key_to_string(key): value
                        for key, value in normalized_spec["quotas"].items()
                    }
                ),
            }
        ),
        "channels": _freeze_json(channels),
        "contents": _freeze_json(tuple(content_records)),
        "features": _freeze_json(tuple(feature_records)),
        "strata_counts": strata_counts,
        "quota_report": quota_report,
        "quota_policy": "all" if quotas == "all" else "explicit",
        "coordinate_status": "tag_relabel_reduced_raw_frame",
        "physical_image_basis": False,
    }
    record["catalogue_hash"] = _stable_hash(_hashable_record_view(record))
    return record


_VOLATILE_CONTENT_FIELDS = ("compile_seconds", "cache_hit")


def _hashable_record_view(record):
    """Record view used for ``catalogue_hash``: drops process-state fields.

    ``compile_seconds`` (wall-clock timing) and ``cache_hit`` (whether this
    particular process happened to find a warm on-disk cache entry) are
    informational only; neither reflects the mathematical content of the
    catalogue, and including either would make the "same spec gives the
    same catalogue_hash" contract depend on machine speed or cache state.
    """

    body = _catalogue_body_without_hash(record)
    body = dict(body)
    body["contents"] = tuple(
        {key: value for key, value in content.items() if key not in _VOLATILE_CONTENT_FIELDS}
        for content in body["contents"]
    )
    return body


def _catalogue_body_without_hash(record):
    return {key: value for key, value in record.items() if key != "catalogue_hash"}


def save_catalogue(record, path):
    """Write a catalogue record to ``path`` as JSON."""

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(record, handle)


def load_catalogue(path):
    """Read a catalogue record from ``path``, verifying ``catalogue_hash``."""

    with open(path, "r", encoding="utf-8") as handle:
        record = json.load(handle)
    expected = record.get("catalogue_hash")
    actual = _stable_hash(_hashable_record_view(record))
    if not expected or str(expected) != str(actual):
        raise ValueError(
            f"Catalogue hash mismatch at {path!r}: stored {expected!r}, recomputed {actual!r}."
        )
    return record


def write_catalogue_review(record, path):
    """Write a markdown review of a catalogue record to ``path``."""

    lines = []
    lines.append(f"# Tagged Cauchy catalogue review ({record['schema']})")
    lines.append("")
    lines.append(f"catalogue_hash: `{record['catalogue_hash']}`")
    lines.append("")
    lines.append("## Counts by stratum")
    lines.append("")
    lines.append("| stratum (N, kappa_class, lambda_class) | available | selected |")
    lines.append("|---|---|---|")
    for stratum_text in sorted(record["strata_counts"]):
        counts = record["strata_counts"][stratum_text]
        lines.append(
            f"| {stratum_text} | {counts['available']} | {counts['selected']} |"
        )
    lines.append("")

    lines.append("## Counts by N")
    lines.append("")
    by_n = {}
    for stratum_text, counts in record["strata_counts"].items():
        N = _stratum_key_from_string(stratum_text)[0]
        entry = by_n.setdefault(N, {"available": 0, "selected": 0})
        entry["available"] += counts["available"]
        entry["selected"] += counts["selected"]
    lines.append("| N | available | selected |")
    lines.append("|---|---|---|")
    for N in sorted(by_n):
        lines.append(f"| {N} | {by_n[N]['available']} | {by_n[N]['selected']} |")
    lines.append("")

    lines.append("## Quota outcome")
    lines.append("")
    if record["quota_policy"] == "all":
        lines.append("Quota policy: `all` (every available pooled feature kept).")
    else:
        lines.append("| stratum | min | max | available | selected | met_minimum |")
        lines.append("|---|---|---|---|---|---|")
        for stratum_text in sorted(record["quota_report"]):
            entry = record["quota_report"][stratum_text]
            lines.append(
                f"| {stratum_text} | {entry['min']} | {entry['max']} | "
                f"{entry['available']} | {entry['selected']} | {entry['met_minimum']} |"
            )
    lines.append("")

    lines.append("## Contents")
    lines.append("")
    lines.append(
        "| content_index | N | blocks | outcome | label_count | supported_count "
        "| pooled_count | compile_seconds | cache_hit |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for content in record["contents"]:
        lines.append(
            f"| {content['content_index']} | {content['N']} | {content['blocks']} | "
            f"{content['outcome']} | {content['label_count']} | "
            f"{content['supported_count']} | {content['pooled_count']} | "
            f"{content['compile_seconds']:.3f} | {content['cache_hit']} |"
        )
    lines.append("")

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def _read_only_block_template(key):
    """Read-only lookup of a compiled block-template payload, never building.

    Returns the template's ``payload`` dict (the same shape a compiled
    artifact's own ``block_templates`` entries have) if it is already
    present and verifies in the on-disk artifact store, or ``None`` if it
    is absent. Uses ``YE3TArtifactStore(mode="read_only")``, which never
    invokes a builder and never writes or locks anything on a miss (it
    raises ``ArtifactCacheMiss`` instead); the read-only store was verified
    to create no lock file and never call its own builder callback.
    """

    key = (
        int(key[0]),
        int(key[1]),
        tuple(int(value) for value in key[2]),
        int(key[3]),
        int(key[4]),
    )
    request = _block_cache_request(key)
    store = YE3TArtifactStore(mode="read_only")

    def _must_not_build():
        raise RuntimeError(
            "Read-only block-template lookup must never build; this is a bug "
            "if reached."
        )

    try:
        result = store.resolve(
            "lifted_cauchy_block_template",
            LIFTED_CAUCHY_BLOCK_CACHE_SCHEMA,
            request,
            builder=_must_not_build,
            validator=lambda payload: _validate_cached_block_template(
                payload, key, verify_construction=False
            ),
            certificate=_template_cache_certificate,
            required_certificate_checks=(
                "pairing_invariance_exact",
                "simultaneous_factor_invariance_exact",
                "metric_analysis_identity_exact",
                "symmetric_power_on_joint_role_angular_coordinate",
                "division_free_reverse",
            ),
            dependency_hashes=_block_cache_dependencies(request),
            producer={
                "compiler_convention": LIFTED_CAUCHY_SCALAR_CONVENTION,
                "implementation": "lifted_cauchy_joint_block_v1",
            },
        )
    except ArtifactCacheMiss:
        return None
    return _block_template_from_payload(result["payload"])["payload"]


def _label_support_possible(label, role_dimension, channel_l, edge_roles, copy_content_cache):
    """True/False/None (unknown) whether a label's role copies could cover
    every edge role, using only already-cached block templates (read-only;
    never compiles). ``copy_content_cache`` memoizes block-key lookups
    across labels within one inventory run.
    """

    if not edge_roles:
        return True
    content = [0] * role_dimension
    for channel_index, size, kappa, Lambda, role_copy in zip(
        label.block_channel_indices,
        label.block_sizes,
        label.block_kappas,
        label.block_Lambdas,
        label.role_copy_indices,
        strict=True,
    ):
        key = (
            role_dimension,
            int(size),
            tuple(int(part) for part in kappa),
            int(channel_l[int(channel_index)]),
            int(Lambda),
        )
        if key not in copy_content_cache:
            template_payload = _read_only_block_template(key)
            if template_payload is None:
                copy_content_cache[key] = None
            else:
                try:
                    contents, _certificate = block_template_role_copy_contents(
                        template_payload
                    )
                except (KeyError, ValueError, TypeError, RuntimeError):
                    # The cache is shared with concurrently-running writers
                    # that may hold a different revision of the block-
                    # template payload shape than this process. The store's
                    # own hash check only certifies the entry is internally
                    # self-consistent, not that its shape matches what this
                    # process expects. Treat an unreadable-but-present entry
                    # exactly like an absent one: unknown, not a crash.
                    contents = None
                copy_content_cache[key] = contents
        copy_contents = copy_content_cache[key]
        if copy_contents is None:
            return None
        block_content = copy_contents[int(role_copy)]
        for role_index in range(role_dimension):
            content[role_index] += block_content[role_index]
    return all(content[role] >= 1 for role in edge_roles)


def _block_key_string(key):
    role_dimension, size, kappa, l, Lambda = key
    kappa_text = "-".join(str(int(value)) for value in kappa)
    return f"rd{role_dimension}_sz{size}_k{kappa_text}_l{l}_L{Lambda}"


def _block_key_proxy(key):
    """Compile-cost proxy for one distinct block key: the exact role-content
    class sizes (Kostka numbers) and the raw magnetic product-state count
    (2l+1)**size -- both pure combinatorics, computed without compiling or
    touching the cache, so compile cost can be estimated from these proxies.
    """

    role_dimension, size, kappa, l, Lambda = key
    content_class_sizes = {}
    for content in _all_contents(size, role_dimension):
        count = kostka_number(kappa, content)
        if count:
            content_class_sizes["-".join(str(value) for value in content)] = int(count)
    return {
        "role_dimension": int(role_dimension),
        "size": int(size),
        "kappa": tuple(int(value) for value in kappa),
        "l": int(l),
        "Lambda": int(Lambda),
        "role_copy_count": int(_hook_content_dimension(kappa, role_dimension)),
        "content_class_sizes": content_class_sizes,
        "magnetic_states": int((2 * int(l) + 1) ** int(size)),
    }


def _hashable_inventory_view(record):
    """Record view used for ``inventory_hash``: drops the one process-state
    field (``elapsed_seconds`` per content, wall-clock timing) that would
    otherwise make identical specs hash differently run to run.  Wall-clock
    fields are excluded from the hash so the hash is deterministic.
    """

    body = {key: value for key, value in record.items() if key != "inventory_hash"}
    body = dict(body)
    body["contents"] = tuple(
        {key: value for key, value in content.items() if key != "elapsed_seconds"}
        for content in body["contents"]
    )
    return _freeze_json(body)


def _build_count_only_inventory(normalized_spec, channels):
    """The count_only=True code path: enumerate and classify labels via
    ``ye3t.couplings.count`` only -- no ``plan``/``compile`` call anywhere,
    hence no write to the shared build cache (compiling is the only thing
    that ever writes to it). See the module docstring's count_only note.
    """

    role_dimension = normalized_spec["role_dimension"]
    role_bindings = normalized_spec["role_bindings"]
    edge_roles = normalize_role_bindings(role_bindings, role_dimension)["edge_roles"]
    channel_l = {int(channel["channel_index"]): int(channel["l"]) for channel in channels}

    content_records = []
    stratum_counts = {}
    block_structure_counts = {}
    max_lambda_size_counts = {}
    support_possible_counts = {}
    block_key_signatures = set()
    copy_content_cache = {}

    content_index = 0
    for N in range(1, normalized_spec["rank_max"] + 1):
        for blocks in _enumerate_contents(len(channels), N):
            parity = sum(k * channels[channel_index]["l"] for channel_index, k in blocks)
            if parity % 2 != 0:
                continue

            has_oversized = any(
                k > normalized_spec["lambda_block_size_max"] for _, k in blocks
            )
            block_lambda_policy = "zero" if has_oversized else "all"
            channels_subset = tuple(channels[channel_index] for channel_index, _ in blocks)
            block_sizes = tuple(k for _, k in blocks)

            start = time.time()
            outcome = "empty"
            error_text = None
            label_count = 0
            request_payload = None
            request_hash = None
            content_block_keys = []

            try:
                request = lifted_cauchy_fixed_content_scalar_request(
                    channels_subset,
                    block_sizes,
                    role_dimension=role_dimension,
                    kappa_policy=normalized_spec["kappa_policy"],
                    block_lambda_policy=block_lambda_policy,
                    emit_factored=False,
                    resource_limits=normalized_spec["resource_limits"],
                )
                request_payload = _freeze_json(request)
                request_hash = _stable_hash(request_payload)
                report = count_coupling(request)
                label_count = int(report.descriptor_count)
                if label_count > 0:
                    outcome = "ok"
                    for label in report.labels:
                        block_kappas = tuple(
                            tuple(int(part) for part in kappa) for kappa in label.block_kappas
                        )
                        block_Lambdas = tuple(int(value) for value in label.block_Lambdas)
                        block_sizes_label = tuple(int(value) for value in label.block_sizes)
                        stratum = (
                            int(label.rank),
                            _kappa_class(block_kappas),
                            _lambda_class(block_Lambdas),
                        )
                        stratum_counts[stratum] = stratum_counts.get(stratum, 0) + 1
                        block_structure_counts[block_sizes_label] = (
                            block_structure_counts.get(block_sizes_label, 0) + 1
                        )
                        positive_sizes = [
                            size
                            for size, Lambda in zip(block_sizes_label, block_Lambdas)
                            if Lambda > 0
                        ]
                        max_lambda_size = max(positive_sizes) if positive_sizes else 0
                        max_lambda_size_counts[max_lambda_size] = (
                            max_lambda_size_counts.get(max_lambda_size, 0) + 1
                        )

                        support = _label_support_possible(
                            label, role_dimension, channel_l, edge_roles, copy_content_cache
                        )
                        bucket = support_possible_counts.setdefault(
                            stratum,
                            {"support_possible": 0, "support_impossible": 0, "unknown": 0},
                        )
                        if support is None:
                            bucket["unknown"] += 1
                        elif support:
                            bucket["support_possible"] += 1
                        else:
                            bucket["support_impossible"] += 1

                        for channel_index, size, kappa, Lambda in zip(
                            label.block_channel_indices,
                            label.block_sizes,
                            label.block_kappas,
                            label.block_Lambdas,
                            strict=True,
                        ):
                            key = (
                                role_dimension,
                                int(size),
                                tuple(int(part) for part in kappa),
                                channel_l[int(channel_index)],
                                int(Lambda),
                            )
                            content_block_keys.append(key)
                            block_key_signatures.add(key)
            except ValueError as exc:
                outcome = "refused"
                error_text = str(exc)
            except Exception as exc:  # defensive: report, never crash the sweep
                outcome = "refused"
                error_text = f"{type(exc).__name__}: {exc}"

            content_records.append(
                {
                    "content_index": int(content_index),
                    "N": int(N),
                    "blocks": tuple((int(ci), int(k)) for ci, k in blocks),
                    "request_payload": request_payload,
                    "request_hash": request_hash,
                    "label_count": int(label_count),
                    "outcome": outcome,
                    "error": error_text,
                    "block_keys": tuple(
                        _block_key_string(key) for key in sorted(set(content_block_keys))
                    ),
                    "elapsed_seconds": float(time.time() - start),
                }
            )
            content_index += 1

    block_key_proxies = {
        _block_key_string(key): _block_key_proxy(key) for key in sorted(block_key_signatures)
    }

    record = {
        "schema": CATALOGUE_SCHEMA,
        "mode": "count_only",
        "note": (
            "LABEL INVENTORY ONLY: these are compiler labels counted via "
            "ye3t.couplings.count before tag support and pooling; no "
            "descriptor was compiled and the shared build cache was never "
            "touched. 'support_possible_by_stratum' is a per-label estimate "
            "from already-cached block templates only (read-only lookup); "
            "'unknown' means the needed block template was not present in "
            "the cache and was deliberately not built."
        ),
        "spec": _freeze_json(
            {key: value for key, value in normalized_spec.items() if key != "quotas"}
        ),
        "channels": _freeze_json(channels),
        "contents": _freeze_json(tuple(content_records)),
        "label_strata_counts": {
            _stratum_key_to_string(key): value for key, value in stratum_counts.items()
        },
        "block_structure_counts": {
            "-".join(str(value) for value in key): value
            for key, value in block_structure_counts.items()
        },
        "max_lambda_block_size_counts": {
            str(key): value for key, value in max_lambda_size_counts.items()
        },
        "support_possible_by_stratum": {
            _stratum_key_to_string(key): value for key, value in support_possible_counts.items()
        },
        "block_key_proxies": block_key_proxies,
        "total_labels": int(sum(stratum_counts.values())),
    }
    record["inventory_hash"] = _stable_hash(_hashable_inventory_view(record))
    return record


def write_inventory_review(record, path):
    """Write a markdown review of a count_only inventory record to ``path``."""

    lines = []
    lines.append(f"# Tagged Cauchy label inventory ({record['schema']}, mode={record['mode']})")
    lines.append("")
    lines.append(f"**{record['note']}**")
    lines.append("")
    lines.append(f"inventory_hash: `{record['inventory_hash']}`")
    lines.append("")
    lines.append(f"total_labels: {record['total_labels']}")
    lines.append("")

    lines.append("## Labels by (N, kappa_class, lambda_class)")
    lines.append("")
    lines.append("| stratum | labels | support_possible | support_impossible | unknown |")
    lines.append("|---|---|---|---|---|")
    for stratum_text in sorted(record["label_strata_counts"]):
        count = record["label_strata_counts"][stratum_text]
        support = record["support_possible_by_stratum"].get(
            stratum_text, {"support_possible": 0, "support_impossible": 0, "unknown": 0}
        )
        lines.append(
            f"| {stratum_text} | {count} | {support['support_possible']} | "
            f"{support['support_impossible']} | {support['unknown']} |"
        )
    lines.append("")

    lines.append("## Labels by block-structure signature (tuple of block sizes)")
    lines.append("")
    lines.append("| block sizes | labels |")
    lines.append("|---|---|")
    for structure_text in sorted(
        record["block_structure_counts"], key=lambda text: [int(v) for v in text.split("-")]
    ):
        lines.append(f"| {structure_text} | {record['block_structure_counts'][structure_text]} |")
    lines.append("")

    lines.append("## Labels by maximum block size carrying Lambda > 0 (0 = none)")
    lines.append("")
    lines.append("| max Lambda>0 block size | labels |")
    lines.append("|---|---|")
    for key_text in sorted(record["max_lambda_block_size_counts"], key=int):
        lines.append(f"| {key_text} | {record['max_lambda_block_size_counts'][key_text]} |")
    lines.append("")

    lines.append("## Distinct block keys (compile-cost proxy)")
    lines.append("")
    lines.append(
        "| block key | role_copy_count | magnetic_states (2l+1)^size | "
        "content classes (content: Kostka count) |"
    )
    lines.append("|---|---|---|---|")
    for key_text in sorted(record["block_key_proxies"]):
        proxy = record["block_key_proxies"][key_text]
        classes = ", ".join(
            f"{content}:{count}"
            for content, count in sorted(proxy["content_class_sizes"].items())
        )
        lines.append(
            f"| {key_text} | {proxy['role_copy_count']} | {proxy['magnetic_states']} | "
            f"{classes} |"
        )
    lines.append("")

    lines.append("## Contents")
    lines.append("")
    lines.append(
        "| content_index | N | blocks | outcome | label_count | elapsed_seconds | error |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for content in record["contents"]:
        error_text = content["error"] or ""
        lines.append(
            f"| {content['content_index']} | {content['N']} | {content['blocks']} | "
            f"{content['outcome']} | {content['label_count']} | "
            f"{content['elapsed_seconds']:.3f} | {error_text} |"
        )
    lines.append("")

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


__all__ = [
    "CATALOGUE_SCHEMA",
    "build_tagged_catalogue",
    "load_catalogue",
    "save_catalogue",
    "write_catalogue_review",
    "write_inventory_review",
]
