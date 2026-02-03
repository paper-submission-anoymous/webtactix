from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from webtactix.preprocess.snapshot_dedup import build_snapshot_with_dedup


@dataclass(frozen=True)
class EncodedObservation:
    """
    A compact, model-facing observation derived from raw aria_snapshot() data.
    """
    actree_yaml: str
    roles: List[str]              # role's type
    names: List[Optional[str]]    # role's name
    nums: List[int]               # num of all (role, name)
    role_nums: List[int]          # num of all (role)


@dataclass(frozen=True)
class ObservationEncoderConfig:
    """
    Controls observation preprocessing behavior.
    """
    table_max_rows: int = 10


class ObservationEncoder:
    def __init__(self, cfg: ObservationEncoderConfig | None = None) -> None:
        self.cfg = cfg or ObservationEncoderConfig()

    def encode(
        self,
        cur_snapshot: Any,
        prev_snapshot: Any | None = None,
        *,
        after_navigation: bool = False,
        skip_indices: Optional[List[int]] = None,
        skip_role_name_specs: Optional[List[Dict[str, Any]]] = None,
    ) -> EncodedObservation:
        """
        If after_navigation is False, cross-page dedup is disabled by providing an empty prev snapshot.
        This matches the intended behavior described in your method.
        """
        prev_snapshot = {}
        if not after_navigation:
            prev_snapshot = {}  # disable cross-page dedup on same-page actions

        yaml_text, roles, names, nums, role_nums = build_snapshot_with_dedup(
            prev_snapshot,
            cur_snapshot,
            skip_indices=[],
            skip_role_name_specs=skip_role_name_specs or [],
            table_max_rows=self.cfg.table_max_rows,
        )

        return EncodedObservation(
            actree_yaml=yaml_text,
            roles=roles,
            names=names,
            nums=nums,
            role_nums=role_nums,
        )
