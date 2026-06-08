"""DatasetAdapter protocol shared by all dataset loaders."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from visualcue.harness.types import GTSample


class DatasetAdapter(Protocol):
    """Iterable ground-truth dataset contract with a stable display name."""

    name: str

    def __iter__(self) -> Iterator[GTSample]:
        """Yield ground-truth samples without network access."""
        ...

    def __len__(self) -> int:
        """Return the number of available samples."""
        ...
