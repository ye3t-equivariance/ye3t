"""Cache utilities for exact symbolic feature construction."""

from .artifacts import (
    ARTIFACT_STORE_SCHEMA,
    ArtifactCacheError,
    ArtifactCacheMiss,
    ArtifactCacheValidationError,
    YE3TArtifactStore,
    artifact_cache_events,
    artifact_hash,
    canonical_json_bytes,
    default_artifact_cache_directory,
)

_SYMBOLIC_EXPORTS = {
    "CGCacheKey", "CGCachedEntry", "ClebschGordanCache", "DualCacheSystem",
    "FeatureCache", "FeatureCachedEntry", "FeatureCacheKey", "PrimitiveCacheKey",
    "PrimitiveCacheSystem", "PrimitiveFeature", "ProductRecipe",
}


def __getattr__(name):
    if name in _SYMBOLIC_EXPORTS:
        from . import symbolic
        value = getattr(symbolic, name)
        globals()[name] = value
        return value
    raise AttributeError(name)

__all__ = [
    "ARTIFACT_STORE_SCHEMA",
    "ArtifactCacheError",
    "ArtifactCacheMiss",
    "ArtifactCacheValidationError",
    "CGCacheKey",
    "CGCachedEntry",
    "ClebschGordanCache",
    "DualCacheSystem",
    "FeatureCache",
    "FeatureCachedEntry",
    "FeatureCacheKey",
    "PrimitiveCacheKey",
    "PrimitiveCacheSystem",
    "PrimitiveFeature",
    "ProductRecipe",
    "YE3TArtifactStore",
    "artifact_cache_events",
    "artifact_hash",
    "canonical_json_bytes",
    "default_artifact_cache_directory",
]
