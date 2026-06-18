from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .config import IMAGES_DIR


LEGACY_HOME_PREFIXES = (
    "/home/ngocnt/",
    "/home/",
)


def _dedupe_keep_order(paths: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered



def extract_image_filename(image_entry: object) -> str:
    if isinstance(image_entry, dict):
        for key in ("file", "file_name", "file_id", "path"):
            value = image_entry.get(key)
            if value:
                return Path(str(value)).name
        return ""
    if image_entry is None:
        return ""
    return Path(str(image_entry)).name



def canonical_case_image_path(case_id: str, image_ref: object, images_dir: Path = IMAGES_DIR) -> Path:
    filename = extract_image_filename(image_ref)
    if not case_id or not filename:
        return Path()
    return Path(images_dir) / str(case_id) / filename



def resolve_case_image_paths(
    case: Optional[Dict],
    images_dir: Path = IMAGES_DIR,
    max_images: Optional[int] = None,
) -> List[str]:
    if not case:
        return []

    case_id = str(case.get("case_id", "") or "").strip()
    if not case_id:
        return []

    resolved: List[str] = []
    for image_entry in case.get("images") or []:
        canonical = canonical_case_image_path(case_id, image_entry, images_dir=images_dir)
        if canonical and canonical.exists():
            resolved.append(str(canonical))
    resolved = _dedupe_keep_order(resolved)
    if max_images is not None:
        return resolved[:max_images]
    return resolved



def normalize_query_image_paths(
    case_id: str,
    query_images: Optional[Iterable[object]] = None,
    image_entries: Optional[Iterable[object]] = None,
    images_dir: Path = IMAGES_DIR,
    max_images: Optional[int] = None,
) -> List[str]:
    resolved: List[str] = []
    for image_entry in image_entries or []:
        canonical = canonical_case_image_path(case_id, image_entry, images_dir=images_dir)
        if canonical and canonical.exists():
            resolved.append(str(canonical))

    for image_ref in query_images or []:
        canonical = canonical_case_image_path(case_id, image_ref, images_dir=images_dir)
        if canonical and canonical.exists():
            resolved.append(str(canonical))
            continue

        path_obj = Path(str(image_ref))
        if path_obj.is_absolute() and path_obj.exists():
            resolved.append(str(path_obj))

    resolved = _dedupe_keep_order(resolved)
    if max_images is not None:
        return resolved[:max_images]
    return resolved



def has_legacy_host_paths(paths: Iterable[object]) -> bool:
    for path in paths:
        text = str(path or "")
        if any(text.startswith(prefix) for prefix in LEGACY_HOME_PREFIXES):
            return True
    return False
