import copy
import dataclasses
import json
import logging
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Generic, Optional, TypeVar

import torch
from torch._dynamo.package import (
    _BackendId,
    _DynamoCacheEntry,
    DynamoCache,
    PrecompileCacheEntry,
)


"""
Classes and implementations related to precompile
"""

T = TypeVar("T")
logger = logging.getLogger(__name__)


@dataclass
class BackendCacheArtifact(Generic[T]):
    """
    Represents a single serializable backend artifact from a dynamo backend.
    Each BackendCacheArtifact has a key associated with it along with some
    serializable content.

    Example implementation:

    class MyPrecompileCacheArtifact(PrecompileCacheArtifact[MySerializableType]):
        my_field: int

        def after_deserialization(self) -> MySerializableType:
            result = pickle.loads(self.content)
            # Do some extra work post deserialization
            result.my_post_deserialization_function(self.my_field)
            return result
    """

    key: str
    content: Any

    @abstractmethod
    def after_deserialization(self) -> T:
        """
        Code to be run after reading raw byte contents from disk.
        Generally converts self.content from raw bytes back into its original form.
        """
        ...

    def edit_contents(self, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the contents of the artifact.
        """
        self.content = edit_fn(self.content)


class EagerCacheArtifact(BackendCacheArtifact[Any]):
    def after_deserialization(self) -> Any:
        return self.content


class BypassDynamoCacheEntry(Exception):
    pass


class PrecompileContext:
    """
    PrecompileContext is a special CacheArtifactManager for handling precompilation
    It uses the same interface as CacheArtifactManager, but handles deserialization differently: instead
    of placing each artifact into respective caches, it will stitch all the cache artifacts for a single key
    together and place it into a global Precompile Cache.

    PrecompileContext has two main portions: dynamo_cache_entries and backend_cache_artifacts.
    When saving, PrecompileContext.serialize() will serialize all dynamo cache entries along with any PrecompileCacheArtifacts that
    are needed to save those dynamo cache entries.

    The following artifact types are supported by PrecompileContext:
     - BundledAOTAutogradCacheArtifact

    """

    # Protected by the compile_lock
    # _backend_artifacts_by_key organizes results by the key of each artifact.
    # Each object here must be serializable
    _backend_artifacts_by_key: dict[str, BackendCacheArtifact] = {}

    # On call to `serialize()`, all cache artifacts in _dynamo_cache_entries are converted
    # into DynamoCacheArtifacts and added to _new_cache_artifacts for serialization
    _dynamo_cache_entries: dict[str, _DynamoCacheEntry] = {}

    @classmethod
    def clear(cls) -> None:
        cls._backend_artifacts_by_key.clear()
        cls._dynamo_cache_entries.clear()

    @classmethod
    def record_artifact(
        cls,
        artifact: BackendCacheArtifact,
    ) -> None:
        """
        Records a backend artifact to be used with dynamo cache entries
        """
        cls._backend_artifacts_by_key[artifact.key] = copy.deepcopy(artifact)

    @classmethod
    def record_dynamo_cache_entry(
        cls, cache_entry: _DynamoCacheEntry, key: str
    ) -> None:
        cls._dynamo_cache_entries[key] = cache_entry

    @classmethod
    def edit_artifact(cls, key: str, edit_fn: Callable[..., Any]) -> None:
        """
        Edit the content of an existing artifact
        """
        assert key in cls._backend_artifacts_by_key, f"Key {key} not found in artifacts"
        artifact = cls._backend_artifacts_by_key[key]
        artifact.edit_contents(edit_fn)

    @classmethod
    def serialize_artifact_by_key(cls, key: str) -> Optional[BackendCacheArtifact]:
        """
        Return the backend cache artifact with the associated key
        """
        return cls._backend_artifacts_by_key.get(key, None)

    @staticmethod
    def dump_debug_info(
        dynamo_entries: dict[str, _DynamoCacheEntry],
        backend_artifacts: dict[str, BackendCacheArtifact],
    ) -> dict[str, Any]:
        """
        Return a JSON serializable debug dump of all entries in the precompile context
        Called in serialize before serialization, and in populate_caches after deserialization
        """
        # Print debug information
        debug_info: defaultdict[list[Any]] = defaultdict(list)
        for key, cache_entry in dynamo_entries.items():
            info = cache_entry.debug_info()
            info["key"] = key
            debug_info["dynamo"].append(info)

        for artifact in backend_artifacts.values():
            debug_info["backends"].append(artifact.key)

        return debug_info

    @classmethod
    def save_to_dynamo_cache(cls) -> dict[str, Any]:
        precompile_cache_entries, debug_info = cls.create_cache_entries()
        for key, entry in precompile_cache_entries.items():
            DynamoCache.write(entry, key)
        return debug_info

    @classmethod
    def create_cache_entries(
        cls,
    ) -> tuple[dict[str, PrecompileCacheEntry], dict[str, Any]]:
        """
        Grabs all the cache entries in the precompile context and
        stitches them together into full PrecompileCacheEntries.
        """
        dynamo_entries = cls._dynamo_cache_entries
        backend_artifacts = cls._backend_artifacts_by_key

        num_artifacts = len(dynamo_entries)

        debug_info = PrecompileContext.dump_debug_info(
            dynamo_entries, backend_artifacts
        )
        debug_str = json.dumps(
            {
                "num_entries": num_artifacts,
                "artifacts": debug_info,
            },
        )
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "dynamo_cache_entries",
                "encoding": "json",
            },
            payload_fn=lambda: debug_str,
            expect_trace_id=False,
        )

        precompile_cache_entries = {}

        for key, cache_entry in dynamo_entries.items():
            try:
                backends = cache_entry.backend_ids
                backend_content: dict[_BackendId, BackendCacheArtifact[Any]] = {}
                for id_ in backends:
                    if id_ not in backend_artifacts:
                        debug_str = json.dumps(
                            {
                                "entry": cache_entry.debug_info,
                                "key": key,
                            }
                        )
                        logger.warning("Backend not found")
                        torch._logging.trace_structured(
                            "artifact",
                            metadata_fn=lambda: {
                                "name": "dynamo_cache_bypass",
                                "encoding": "json",
                            },
                            payload_fn=lambda: debug_str,
                            expect_trace_id=False,
                        )
                        continue
                    artifact = backend_artifacts[id_]
                    assert isinstance(artifact, BackendCacheArtifact)
                    backend_content[id_] = artifact
                precompile_cache_entries[key] = PrecompileCacheEntry(
                    dynamo=cache_entry, backends=backend_content
                )
            except Exception as e:
                logger.warning("Failed to create cache entry %s: %s", key, str(e))

                error = e
                data = json.dumps(
                    {
                        "key": key,
                        "error": str(error),
                    }
                )
                torch._logging.trace_structured(
                    "artifact",
                    metadata_fn=lambda: {
                        "name": "dynamo_cache_exception",
                        "encoding": "json",
                    },
                    payload_fn=lambda: data,
                )
                continue
        return precompile_cache_entries, debug_info


@dataclasses.dataclass(frozen=True)
class SystemInfo:
    """
    System information including Python, PyTorch, and GPU details.
    This information is used to ensure compiled artifacts can only be loaded
    with compatible system configurations.
    """

    python_version: str
    torch_version: str
    cuda_version: Optional[str]
    triton_version: Optional[tuple[int, int]]
    gpu_name: Optional[str]

    @classmethod
    def current(cls) -> "SystemInfo":
        """Create a SystemInfo instance with current system information."""
        # Get GPU name if CUDA is available
        gpu_name = None
        if torch.cuda.is_available():
            try:
                gpu_name = torch.cuda.get_device_name()
            except Exception:
                # If we can't get GPU info, leave as None
                pass

        return cls(
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            triton_version=get_triton_version((0, 0)),
            gpu_name=gpu_name,
        )

    def check_compatibility(self, other: "SystemInfo", use_cuda: bool = False) -> None:
        """
        Check if this SystemInfo is compatible with another SystemInfo.
        Raises RuntimeError if incompatible.
        """
        if self.python_version != other.python_version:
            raise RuntimeError(
                f"Compile package was created with a different Python version: {self.python_version}"
            )

        if self.torch_version != other.torch_version:
            raise RuntimeError(
                f"Compile package was created with a different PyTorch version: {self.torch_version}"
            )

        if use_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available")
            if self.cuda_version != other.cuda_version:
                raise RuntimeError(
                    f"Compile package was created with a different CUDA version: {self.cuda_version}"
                )

            if (
                other.triton_version != (0, 0)
                and self.triton_version != other.triton_version
            ):
                raise RuntimeError(
                    f"Compile package was created with a different Triton version: {self.triton_version}"
                )

            # Check GPU name if CUDA was used
            if other.gpu_name is not None and self.gpu_name != other.gpu_name:
                raise RuntimeError(
                    f"Compile package was created with different GPU: "
                    f"cached={self.gpu_name}, current={other.gpu_name}"
                )
