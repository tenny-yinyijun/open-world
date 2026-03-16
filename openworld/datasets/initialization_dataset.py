from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

from openworld.datasets.initialization import Initialization
from openworld.utils.io import load_yaml


class InitializationDataset:
    """An in-memory collection of :class:`Initialization` entries.

    Supports iteration, indexing, and length queries.  Designed so that a
    file-backed subclass can override the loading logic without changing the
    consumer API.
    """

    def __init__(self, initializations: Optional[Sequence[Initialization]] = None):
        self._items: List[Initialization] = list(initializations or [])

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_list(
        cls, entries: List[Dict[str, Any]]
    ) -> "InitializationDataset":
        """Build a dataset from a list of dicts.

        Each dict should contain keys matching :class:`Initialization` fields
        (``id``, ``initial_state``, ``initial_observation``, and optionally
        ``instruction`` / ``metadata``).
        """
        items = [Initialization(**entry) for entry in entries]
        return cls(items)

    @classmethod
    def from_yaml(cls, path: str) -> "InitializationDataset":
        """Build a dataset from a YAML file or a suite directory.

        A YAML file may contain either a top-level list of initialization
        entries or a mapping with an ``initializations`` list.

        A directory is treated as a suite root. Each direct child directory
        containing an ``initialization.yaml`` file becomes one dataset entry.

        Entries may either embed the full initialization payload inline or
        reference a per-case YAML file via ``initialization_path``.
        """
        dataset_path = Path(path).resolve()
        if dataset_path.is_dir():
            entries = cls._load_directory_entries(dataset_path)
            return cls.from_list(entries)

        payload = load_yaml(str(dataset_path))
        if isinstance(payload, list):
            entries = payload
        else:
            entries = payload.get("initializations", [])

        if not isinstance(entries, list):
            raise ValueError(
                f"Initialization dataset at {dataset_path} must be a list or contain an "
                "'initializations' list."
            )

        resolved_entries = [
            cls._load_entry(entry, dataset_path.parent) for entry in entries
        ]
        return cls.from_list(resolved_entries)

    @staticmethod
    def _load_directory_entries(dataset_dir: Path) -> List[Dict[str, Any]]:
        entries = []
        for case_dir in sorted(child for child in dataset_dir.iterdir() if child.is_dir()):
            init_path = case_dir / "initialization.yaml"
            if not init_path.exists():
                continue
            entries.append(
                InitializationDataset._load_entry(
                    {"id": case_dir.name, "initialization_path": str(init_path)},
                    dataset_dir,
                )
            )
        if not entries:
            raise ValueError(
                f"No initialization.yaml files found under dataset directory {dataset_dir}."
            )
        return entries

    @staticmethod
    def _load_entry(entry: Dict[str, Any], base_dir: Path) -> Dict[str, Any]:
        resolved = dict(entry)
        initialization_path = resolved.pop("initialization_path", None)
        if initialization_path is not None:
            init_path = Path(InitializationDataset._resolve_path(initialization_path, base_dir))
            init_payload = load_yaml(str(init_path))
            if not isinstance(init_payload, dict):
                raise ValueError(
                    f"Initialization file at {init_path} must contain a mapping."
                )
            resolved = {**init_payload, **resolved}
            resolved.setdefault("id", init_path.parent.name)
            base_dir = init_path.parent

        metadata = resolved.pop("metadata", None)
        metadata_path = resolved.pop("metadata_path", None)
        if metadata_path is not None:
            meta_path = Path(InitializationDataset._resolve_path(metadata_path, base_dir))
            loaded_metadata = load_yaml(str(meta_path))
            if not isinstance(loaded_metadata, dict):
                raise ValueError(f"Metadata file at {meta_path} must contain a mapping.")
            resolved["metadata"] = {**loaded_metadata, **(metadata or {})}
        elif metadata is not None:
            resolved["metadata"] = metadata

        observation = resolved.get("initial_observation")
        if observation is None:
            inferred_observation = InitializationDataset._infer_observation_from_case_dir(
                base_dir
            )
            if inferred_observation is not None:
                resolved["initial_observation"] = inferred_observation
        elif isinstance(observation, dict):
            resolved["initial_observation"] = InitializationDataset._resolve_observation_paths(
                observation,
                base_dir,
            )
        return resolved

    @staticmethod
    def _infer_observation_from_case_dir(base_dir: Path) -> Optional[Dict[str, Any]]:
        view_filenames = {
            "exterior_left": "exterior_left.png",
            "exterior_right": "exterior_right.png",
            "wrist": "wrist.png",
        }
        resolved_views = {}
        for view_name, filename in view_filenames.items():
            path = base_dir / filename
            if not path.exists():
                return None
            resolved_views[view_name] = str(path.resolve())
        return {"views": resolved_views}

    @staticmethod
    def _resolve_observation_paths(
        observation: Dict[str, Any],
        base_dir: Path,
    ) -> Dict[str, Any]:
        resolved = dict(observation)
        views = resolved.get("views")
        if isinstance(views, dict):
            resolved["views"] = {
                name: InitializationDataset._resolve_path(value, base_dir)
                for name, value in views.items()
            }
        return resolved

    @staticmethod
    def _resolve_path(value: Any, base_dir: Path) -> Any:
        if not isinstance(value, str):
            return value
        path = Path(value)
        if path.is_absolute():
            return str(path)
        return str((base_dir / path).resolve())

    # ------------------------------------------------------------------
    # Sequence API
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int) -> Initialization:
        return self._items[index]

    def __iter__(self) -> Iterator[Initialization]:
        return iter(self._items)

    def add(self, initialization: Initialization) -> None:
        """Append a single initialization entry."""
        self._items.append(initialization)
