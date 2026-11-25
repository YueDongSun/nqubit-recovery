"""Validate that an HDF5 shard stores all dataset bytes in its own file."""

from __future__ import annotations

from typing import Set

import h5py


def _object_address(value: h5py.Group | h5py.Dataset) -> int:
    return int(h5py.h5o.get_info(value.id).addr)


def validate_self_contained_hdf5(handle: h5py.File) -> None:
    """Reject links and dataset storage layouts outside the opened HDF5 file."""
    visited_groups: Set[int] = set()

    def visit_group(group: h5py.Group) -> None:
        group_address = _object_address(group)
        if group_address in visited_groups:
            return
        visited_groups.add(group_address)

        for name in group:
            link = group.get(name, getlink=True)
            if not isinstance(link, h5py.HardLink):
                raise ValueError(
                    f"V2.4 shard contains unsupported symbolic HDF5 link: {name}"
                )
            value = group[name]
            if isinstance(value, h5py.Group):
                visit_group(value)
            elif isinstance(value, h5py.Dataset):
                creation_property = value.id.get_create_plist()
                if creation_property.get_external_count() > 0:
                    raise ValueError(
                        f"V2.4 shard dataset uses external raw storage: {name}"
                    )
                if value.is_virtual:
                    raise ValueError(f"V2.4 shard dataset is virtual: {name}")
            else:
                raise ValueError(
                    f"V2.4 shard contains unsupported HDF5 object: {name}"
                )

    visit_group(handle)


__all__ = ["validate_self_contained_hdf5"]
