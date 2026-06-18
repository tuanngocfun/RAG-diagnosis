"""
Pseudolabel Dataset Adapter

Builds pipeline-compatible train/test/query/qrels artifacts from multimodal
pseudolabel outputs and normalizes query images onto the current server's
`IMAGES_DIR/{case_id}/{filename}` layout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

from .config import (
    DATASET_VERSION,
    IMAGES_DIR,
    LEGACY_PROJECT_ROOT,
    PROJECT_ROOT,
    SILVER_LABEL_DISCLAIMER,
    SPLIT_DIR,
    dataset_reuses_shared_eval_artifacts,
    get_dataset_base_version,
    get_dataset_artifact_filenames,
)
from .image_resolver import has_legacy_host_paths, normalize_query_image_paths
from .query_templates import DIAGNOSIS_QUESTION_WITH_TYPE




def _resolve_existing_path(candidates: Iterable[Path]) -> Path:
    candidate_list = [Path(c) for c in candidates]
    for candidate in candidate_list:
        if candidate.exists():
            return candidate
    return candidate_list[0]



def _candidate_paths(*relative_paths: str) -> List[Path]:
    roots = [PROJECT_ROOT, LEGACY_PROJECT_ROOT]
    candidates: List[Path] = []
    for root in roots:
        for relative_path in relative_paths:
            candidates.append(root / relative_path)
    return candidates


LEGACY_TRAIN_PSEUDOLABEL_RESULTS = _resolve_existing_path(
    _candidate_paths(
        'testing/multimodal/outputs/train_pseudolabel_v2/results.jsonl',
        'rag/testing/multimodal/outputs/train_pseudolabel_v2/results.jsonl',
    )
)

LEGACY_TEST_PSEUDOLABEL_RESULTS = _resolve_existing_path(
    _candidate_paths(
        'testing/multimodal/outputs/test_pseudolabel_v2_strict/results.jsonl',
        'rag/testing/multimodal/outputs/test_pseudolabel_v2_strict/results.jsonl',
        'testing/multimodal/outputs/test_pseudolabel_v1/test-results.jsonl',
        'rag/testing/multimodal/outputs/test_pseudolabel_v1/test-results.jsonl',
    )
)

P14_V7_DATASET_VERSION = "p14_v7"
P14_V7_TRAIN_PSEUDOLABEL_RESULTS = _resolve_existing_path(
    _candidate_paths(
        'rag/testing/multimodal/v7/out/p14_train_structure_only_main_v7_batch_single_repair/results.jsonl',
    )
)
P14_V7_TEST_PSEUDOLABEL_RESULTS = _resolve_existing_path(
    _candidate_paths(
        'rag/testing/multimodal/v7/out/p14_test_held_out_structure_only_v7_local_reaudit/results.jsonl',
    )
)

_ACTIVE_ARTIFACTS = get_dataset_artifact_filenames(DATASET_VERSION)
TRAIN_NORMALIZED = SPLIT_DIR / _ACTIVE_ARTIFACTS['train']
TEST_NORMALIZED = SPLIT_DIR / _ACTIVE_ARTIFACTS['test']
QUERY_FILE = SPLIT_DIR / _ACTIVE_ARTIFACTS['query']
QUERY_FILE_MIXED56 = SPLIT_DIR / _ACTIVE_ARTIFACTS['query_mixed56']
QRELS_VERIFIED = SPLIT_DIR / _ACTIVE_ARTIFACTS['qrels_verified']
QRELS_PSEUDOLABEL = SPLIT_DIR / _ACTIVE_ARTIFACTS['qrels_pseudolabel']

ENV_TRAIN_RESULTS = 'STRUCTURED_CASES_TRAIN_PSEUDOLABEL_RESULTS'
ENV_TEST_RESULTS = 'STRUCTURED_CASES_TEST_PSEUDOLABEL_RESULTS'
ENV_PSEUDOLABEL_SUFFIX = 'STRUCTURED_CASES_PSEUDOLABEL_SUFFIX'


@dataclass
class BuildStats:
    train_rows: int
    test_rows: int
    query_rows: int
    query_mixed56_rows: int = 0
    dataset_version: str = ''
    train_source: str = ''
    test_source: str = ''
    suffix: str = ''
    output_dir: str = ''
    train_path: str = ''
    test_path: str = ''
    query_path: str = ''
    query_mixed56_path: str = ''
    qrels_verified_path: str = ''
    qrels_pseudolabel_path: str = ''


def _normalize_suffix(suffix: Optional[str]) -> str:
    raw = str(suffix or '').strip()
    if not raw:
        return ''
    normalized = re.sub(r'[^A-Za-z0-9._-]+', '_', raw)
    return normalized.strip('_')


def _with_suffix(path: Path, suffix: str) -> Path:
    cleaned = _normalize_suffix(suffix)
    if not cleaned:
        return path
    return path.with_name(f'{path.stem}_{cleaned}{path.suffix}')


def get_silver_label_contract(dataset_version: Optional[str] = None) -> Dict[str, str]:
    version = str(dataset_version or DATASET_VERSION).strip() or DATASET_VERSION
    return {
        "dataset_version": version,
        "ground_truth_status": "silver_reference_only",
        "verified_track": "reference_label propagated from the structured multimodal pipeline, not clinician gold ground truth",
        "pseudolabel_track": "Prompt-1 final_diagnosis extracted from the structured multimodal pipeline, not clinician gold ground truth",
        "disclaimer": SILVER_LABEL_DISCLAIMER,
    }


def _default_source_results(dataset_version: Optional[str] = None) -> Tuple[Path, Path]:
    version = get_dataset_base_version(dataset_version or DATASET_VERSION)
    if version == P14_V7_DATASET_VERSION:
        return P14_V7_TRAIN_PSEUDOLABEL_RESULTS, P14_V7_TEST_PSEUDOLABEL_RESULTS
    return LEGACY_TRAIN_PSEUDOLABEL_RESULTS, LEGACY_TEST_PSEUDOLABEL_RESULTS


def get_pseudolabel_artifact_paths(
    output_suffix: str = '',
    dataset_version: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Path]:
    cleaned = _normalize_suffix(output_suffix)
    artifact_names = get_dataset_artifact_filenames(dataset_version or DATASET_VERSION)
    resolved_output_dir = Path(output_dir).expanduser() if output_dir is not None else SPLIT_DIR
    return {
        'train': _with_suffix(resolved_output_dir / artifact_names['train'], cleaned),
        'test': _with_suffix(resolved_output_dir / artifact_names['test'], cleaned),
        'query': _with_suffix(resolved_output_dir / artifact_names['query'], cleaned),
        'query_mixed56': _with_suffix(resolved_output_dir / artifact_names['query_mixed56'], cleaned),
        'qrels_verified': _with_suffix(resolved_output_dir / artifact_names['qrels_verified'], cleaned),
        'qrels_pseudolabel': _with_suffix(resolved_output_dir / artifact_names['qrels_pseudolabel'], cleaned),
    }


def _resolve_source_results_path(override: Optional[Union[str, Path]], fallback: Path) -> Path:
    if override is None:
        return fallback

    raw = str(override).strip()
    if not raw:
        return fallback

    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate

    return _resolve_existing_path(
        [
            candidate,
            PROJECT_ROOT / candidate,
            LEGACY_PROJECT_ROOT / candidate,
        ]
    )



def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



def _write_jsonl(path: Path, rows: Iterable[Dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
            count += 1
    return count



def _parse_tagged_field(raw_text: str, tag_name: str) -> str:
    if not raw_text:
        return ''
    pattern = rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    match = re.search(pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ''



def _extract_case_prompt(record: Dict) -> str:
    prompt1 = record.get('prompt1_converted_case') or {}
    case_prompt = ''
    if isinstance(prompt1, dict):
        case_prompt = (prompt1.get('case_prompt') or '').strip()
        if not case_prompt:
            case_prompt = _parse_tagged_field(prompt1.get('raw', ''), 'case_prompt')
    return case_prompt



def _extract_final_diagnosis(record: Dict) -> str:
    prompt1 = record.get('prompt1_converted_case') or {}
    if isinstance(prompt1, dict):
        final_dx = (prompt1.get('final_diagnosis') or '').strip()
        if final_dx:
            return final_dx
        return _parse_tagged_field(prompt1.get('raw', ''), 'final_diagnosis')
    return ''



def _to_image_entries(case_id: str, image_paths_used: List[str]) -> Tuple[List[Dict], List[str]]:
    absolute_paths = normalize_query_image_paths(
        case_id=case_id,
        query_images=image_paths_used,
        images_dir=IMAGES_DIR,
    )
    images = [{'file': Path(path).name, 'caption': ''} for path in absolute_paths]
    return images, absolute_paths



def _diagnosis_type_from_text(text: str) -> str:
    lowered = (text or '').lower()
    if not lowered:
        return ''
    if 'non-leish' in lowered or 'not leish' in lowered:
        return 'Non-Leishmaniasis'
    if 'post-kala' in lowered or 'pkdl' in lowered:
        return 'PKDL'
    if 'mucocutaneous' in lowered:
        return 'MCL'
    if 'visceral' in lowered or 'kala-azar' in lowered or 'kala azar' in lowered:
        return 'VL'
    if 'cutaneous' in lowered:
        return 'CL'
    if 'ocular' in lowered:
        return 'Ocular'
    if 'veterinary' in lowered or 'canine' in lowered:
        return 'Veterinary'
    return 'Other'



def _resolve_source_jsonl(source_jsonl: str) -> Path:
    rel = Path(str(source_jsonl or '').strip())
    if not rel:
        return rel
    if rel.is_absolute() and rel.exists():
        return rel

    candidates: List[Path] = [
        PROJECT_ROOT / rel,
        LEGACY_PROJECT_ROOT / rel,
    ]

    if len(rel.parts) >= 2 and rel.parts[0] == 'data' and rel.parts[1] == 'leishmaniasis_verified_v2':
        candidates.extend(
            [
                SPLIT_DIR / rel.name,
                PROJECT_ROOT / 'leishmaniasis_verified_v2' / rel.name,
                LEGACY_PROJECT_ROOT / 'data' / 'leishmaniasis_verified_v2' / rel.name,
            ]
        )

    return _resolve_existing_path(candidates)



def _build_verified_map(source_rel_paths: Iterable[str]) -> Dict[str, Dict]:
    by_case: Dict[str, Dict] = {}
    for rel in sorted(set(source_rel_paths)):
        if not rel:
            continue
        src = _resolve_source_jsonl(rel)
        if not src.exists():
            continue
        with open(src, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                case_id = row.get('case_id')
                if not case_id:
                    continue
                by_case[case_id] = {
                    'diagnosis': row.get('diagnosis', ''),
                    'diagnosis_type': row.get('diagnosis_type', ''),
                    'species': row.get('species', ''),
                    'is_leishmaniasis': row.get('is_leishmaniasis', None),
                }
    return by_case



def _normalize_rows(rows: List[Dict], split_name: str, dataset_version: str) -> List[Dict]:
    verified_map = _build_verified_map(record.get('source_jsonl', '') for record in rows)
    normalized: List[Dict] = []
    label_contract = get_silver_label_contract(dataset_version)

    for record in rows:
        case_id = record.get('case_id')
        if not case_id:
            continue

        case_prompt = _extract_case_prompt(record)
        if not case_prompt:
            continue

        reference = record.get('reference_label') or {}
        verified = verified_map.get(case_id) or {
            'diagnosis': reference.get('diagnosis', ''),
            'diagnosis_type': reference.get('diagnosis_type', ''),
            'species': reference.get('species', ''),
            'is_leishmaniasis': reference.get('is_leishmaniasis', None),
        }

        pseudo_dx = _extract_final_diagnosis(record) or reference.get('diagnosis', '')
        pseudo_type = _diagnosis_type_from_text(pseudo_dx) or reference.get('diagnosis_type', '')
        pseudo_label = {
            'diagnosis': pseudo_dx,
            'diagnosis_type': pseudo_type,
            'species': reference.get('species', ''),
            'is_leishmaniasis': None if pseudo_type == '' else pseudo_type != 'Non-Leishmaniasis',
        }

        images, query_image_paths = _to_image_entries(case_id, record.get('image_paths_used') or [])
        normalized.append(
            {
                'case_id': case_id,
                'case_text': case_prompt,
                'images': images,
                'diagnosis': verified.get('diagnosis', ''),
                'diagnosis_type': verified.get('diagnosis_type', ''),
                'species': verified.get('species', ''),
                'is_leishmaniasis': verified.get('is_leishmaniasis', None),
                'labels': {
                    'verified': verified,
                    'pseudolabel': pseudo_label,
                },
                'query_image_paths': query_image_paths,
                'split': split_name,
                'source_jsonl': record.get('source_jsonl', ''),
                'reference_label': reference,
                'pipeline_metadata': record.get('pipeline_metadata') or {},
                'label_contract': label_contract,
            }
        )

    return normalized



def _build_queries(test_rows: List[Dict], dataset_version: str) -> List[Dict]:
    queries: List[Dict] = []
    label_contract = get_silver_label_contract(dataset_version)
    for row in test_rows:
        case_id = row['case_id']
        clinical_context = row.get('case_text', '')
        query_images = normalize_query_image_paths(
            case_id=case_id,
            query_images=row.get('query_image_paths') or [],
            image_entries=row.get('images') or [],
            images_dir=IMAGES_DIR,
        )
        first_image = query_images[0] if query_images else None

        gt_verified = (row.get('labels') or {}).get('verified') or {
            'diagnosis': row.get('diagnosis', ''),
            'diagnosis_type': row.get('diagnosis_type', ''),
            'species': row.get('species', ''),
        }
        gt_pseudo = (row.get('labels') or {}).get('pseudolabel') or gt_verified

        query_types = ['Q1_diagnosis', 'Q1_Q3_multimodal_diagnosis']
        if first_image:
            query_types.insert(1, 'Q3_image_diagnosis')

        for query_type in query_types:
            query_context = '' if query_type == 'Q3_image_diagnosis' else clinical_context
            formatted_query = (
                f"{DIAGNOSIS_QUESTION_WITH_TYPE}\n\nClinical Context: {query_context[:1200]}"
                if query_context
                else DIAGNOSIS_QUESTION_WITH_TYPE
            )
            queries.append(
                {
                    'case_id': case_id,
                    'query_type': query_type,
                    'question': DIAGNOSIS_QUESTION_WITH_TYPE,
                    'clinical_context': query_context,
                    'query_images': query_images,
                    'image_path': first_image,
                    'ground_truth': {
                        'diagnosis': gt_verified.get('diagnosis', ''),
                        'diagnosis_type': gt_verified.get('diagnosis_type', ''),
                        'species': gt_verified.get('species', ''),
                    },
                    'ground_truth_pseudolabel': {
                        'diagnosis': gt_pseudo.get('diagnosis', ''),
                        'diagnosis_type': gt_pseudo.get('diagnosis_type', ''),
                        'species': gt_pseudo.get('species', ''),
                    },
                    'label_contract': label_contract,
                    'formatted_query': formatted_query,
                }
            )

    return queries


def _build_mixed56_queries(query_rows: List[Dict]) -> List[Dict]:
    mixed56: List[Dict] = []
    seen_case_ids = set()
    for row in query_rows:
        if row.get('query_type') != 'Q1_Q3_multimodal_diagnosis':
            continue
        case_id = row.get('case_id')
        if not case_id or case_id in seen_case_ids:
            continue
        seen_case_ids.add(case_id)
        mixed56.append(dict(row))
    return mixed56



def _build_qrels(train_rows: List[Dict], test_rows: List[Dict], label_key: str) -> Dict[str, Dict[str, int]]:
    train_by_type: Dict[str, List[str]] = {}
    for row in train_rows:
        label = (row.get('labels') or {}).get(label_key) or {}
        diagnosis_type = (label.get('diagnosis_type') or '').strip()
        if not diagnosis_type:
            continue
        train_by_type.setdefault(diagnosis_type, []).append(row['case_id'])

    qrels: Dict[str, Dict[str, int]] = {}
    for row in test_rows:
        case_id = row['case_id']
        label = (row.get('labels') or {}).get(label_key) or {}
        diagnosis_type = (label.get('diagnosis_type') or '').strip()
        rel_docs: Dict[str, int] = {}
        for doc_id in train_by_type.get(diagnosis_type, []):
            rel_docs[doc_id] = 3
        qrels[case_id] = rel_docs
    return qrels



def _query_artifact_has_stale_paths(query_file: Path) -> bool:
    if not query_file.exists():
        return True

    with open(query_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get('case_id', '') or '').strip()
            query_images = row.get('query_images') or []
            if not query_images:
                continue
            normalized = normalize_query_image_paths(case_id, query_images=query_images, images_dir=IMAGES_DIR)
            if has_legacy_host_paths(query_images):
                return True
            if any(not Path(str(path)).exists() for path in query_images):
                return True
            if normalized != list(query_images):
                return True
    return False



def _needs_rebuild(outputs: List[Path], inputs: List[Path], query_file: Path) -> bool:
    if any(not path.exists() for path in outputs):
        return True
    if _query_artifact_has_stale_paths(query_file):
        return True
    newest_input = max(path.stat().st_mtime for path in inputs)
    oldest_output = min(path.stat().st_mtime for path in outputs)
    return newest_input > oldest_output



def build_pseudolabel_artifacts(
    force: bool = False,
    train_results_path: Optional[Union[str, Path]] = None,
    test_results_path: Optional[Union[str, Path]] = None,
    output_suffix: Optional[str] = None,
    dataset_version: Optional[str] = None,
    output_dir: Optional[Union[str, Path]] = None,
) -> BuildStats:
    resolved_dataset_version = str(dataset_version or DATASET_VERSION).strip() or DATASET_VERSION
    default_train_source, default_test_source = _default_source_results(resolved_dataset_version)
    resolved_train_source = _resolve_source_results_path(
        train_results_path if train_results_path is not None else os.getenv(ENV_TRAIN_RESULTS),
        default_train_source,
    )
    resolved_test_source = _resolve_source_results_path(
        test_results_path if test_results_path is not None else os.getenv(ENV_TEST_RESULTS),
        default_test_source,
    )
    resolved_suffix = _normalize_suffix(
        output_suffix if output_suffix is not None else os.getenv(ENV_PSEUDOLABEL_SUFFIX, ''),
    )
    resolved_output_dir = Path(output_dir).expanduser() if output_dir is not None else SPLIT_DIR
    artifact_paths = get_pseudolabel_artifact_paths(
        resolved_suffix,
        dataset_version=resolved_dataset_version,
        output_dir=resolved_output_dir,
    )

    if dataset_reuses_shared_eval_artifacts(resolved_dataset_version):
        missing_artifacts = [str(path) for path in artifact_paths.values() if not path.exists()]
        if missing_artifacts:
            raise FileNotFoundError(
                "Aliased dataset artifacts are missing and cannot be rebuilt automatically: "
                + ", ".join(missing_artifacts)
            )

        train_rows = _read_jsonl(artifact_paths['train'])
        test_rows = _read_jsonl(artifact_paths['test'])
        query_rows = _read_jsonl(artifact_paths['query'])
        mixed56_query_rows = _read_jsonl(artifact_paths['query_mixed56'])
        return BuildStats(
            train_rows=len(train_rows),
            test_rows=len(test_rows),
            query_rows=len(query_rows),
            query_mixed56_rows=len(mixed56_query_rows),
            dataset_version=resolved_dataset_version,
            train_source=str(resolved_train_source),
            test_source=str(resolved_test_source),
            suffix=resolved_suffix,
            output_dir=str(resolved_output_dir),
            train_path=str(artifact_paths['train']),
            test_path=str(artifact_paths['test']),
            query_path=str(artifact_paths['query']),
            query_mixed56_path=str(artifact_paths['query_mixed56']),
            qrels_verified_path=str(artifact_paths['qrels_verified']),
            qrels_pseudolabel_path=str(artifact_paths['qrels_pseudolabel']),
        )

    inputs = [resolved_train_source, resolved_test_source]
    outputs = [
        artifact_paths['train'],
        artifact_paths['test'],
        artifact_paths['query'],
        artifact_paths['query_mixed56'],
        artifact_paths['qrels_verified'],
        artifact_paths['qrels_pseudolabel'],
    ]

    missing_inputs = [str(path) for path in inputs if not path.exists()]
    if missing_inputs:
        raise FileNotFoundError('Pseudolabel source files not found: ' + ', '.join(missing_inputs))

    if not force and not _needs_rebuild(outputs, inputs, artifact_paths['query']):
        train_rows = _read_jsonl(artifact_paths['train'])
        test_rows = _read_jsonl(artifact_paths['test'])
        query_rows = _read_jsonl(artifact_paths['query'])
        mixed56_query_rows = _read_jsonl(artifact_paths['query_mixed56'])
        return BuildStats(
            train_rows=len(train_rows),
            test_rows=len(test_rows),
            query_rows=len(query_rows),
            query_mixed56_rows=len(mixed56_query_rows),
            dataset_version=resolved_dataset_version,
            train_source=str(resolved_train_source),
            test_source=str(resolved_test_source),
            suffix=resolved_suffix,
            output_dir=str(resolved_output_dir),
            train_path=str(artifact_paths['train']),
            test_path=str(artifact_paths['test']),
            query_path=str(artifact_paths['query']),
            query_mixed56_path=str(artifact_paths['query_mixed56']),
            qrels_verified_path=str(artifact_paths['qrels_verified']),
            qrels_pseudolabel_path=str(artifact_paths['qrels_pseudolabel']),
        )

    train_source = _read_jsonl(resolved_train_source)
    test_source = _read_jsonl(resolved_test_source)

    train_rows = _normalize_rows(train_source, split_name=f'train_{resolved_dataset_version}', dataset_version=resolved_dataset_version)
    test_rows = _normalize_rows(test_source, split_name=f'test_{resolved_dataset_version}', dataset_version=resolved_dataset_version)

    n_train = _write_jsonl(artifact_paths['train'], train_rows)
    n_test = _write_jsonl(artifact_paths['test'], test_rows)
    query_rows = _build_queries(test_rows, dataset_version=resolved_dataset_version)
    n_queries = _write_jsonl(artifact_paths['query'], query_rows)
    mixed56_queries = _build_mixed56_queries(query_rows)
    n_queries_mixed56 = _write_jsonl(artifact_paths['query_mixed56'], mixed56_queries)

    qrels_verified = _build_qrels(train_rows, test_rows, label_key='verified')
    qrels_pseudo = _build_qrels(train_rows, test_rows, label_key='pseudolabel')

    with open(artifact_paths['qrels_verified'], 'w', encoding='utf-8') as f:
        json.dump(qrels_verified, f, ensure_ascii=False, indent=2)
    with open(artifact_paths['qrels_pseudolabel'], 'w', encoding='utf-8') as f:
        json.dump(qrels_pseudo, f, ensure_ascii=False, indent=2)

    return BuildStats(
        train_rows=n_train,
        test_rows=n_test,
        query_rows=n_queries,
        query_mixed56_rows=n_queries_mixed56,
        dataset_version=resolved_dataset_version,
        train_source=str(resolved_train_source),
        test_source=str(resolved_test_source),
        suffix=resolved_suffix,
        output_dir=str(resolved_output_dir),
        train_path=str(artifact_paths['train']),
        test_path=str(artifact_paths['test']),
        query_path=str(artifact_paths['query']),
        query_mixed56_path=str(artifact_paths['query_mixed56']),
        qrels_verified_path=str(artifact_paths['qrels_verified']),
        qrels_pseudolabel_path=str(artifact_paths['qrels_pseudolabel']),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build pseudolabel train/test/query artifacts')
    parser.add_argument(
        '--train-results-path',
        default=None,
        help='Override train pseudolabel results.jsonl source path',
    )
    parser.add_argument(
        '--test-results-path',
        default=None,
        help='Override test pseudolabel results.jsonl source path',
    )
    parser.add_argument(
        '--pseudolabel-suffix',
        default='',
        help='Optional suffix appended to artifact filenames (for versioned outputs)',
    )
    parser.add_argument(
        '--dataset-version',
        default=None,
        help='Optional dataset version for canonical output names (for example: p14_v7)',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='Optional output directory for generated artifacts',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force artifact rebuild even when outputs are newer than inputs',
    )
    parser.add_argument(
        '--no-force',
        action='store_true',
        help='Skip forced rebuild and only rebuild when inputs are newer than outputs',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    force_rebuild = True
    if args.no_force:
        force_rebuild = False
    if args.force:
        force_rebuild = True
    stats = build_pseudolabel_artifacts(
        force=force_rebuild,
        train_results_path=args.train_results_path,
        test_results_path=args.test_results_path,
        output_suffix=args.pseudolabel_suffix,
        dataset_version=args.dataset_version,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                'train_rows': stats.train_rows,
                'test_rows': stats.test_rows,
                'query_rows': stats.query_rows,
                'query_mixed56_rows': stats.query_mixed56_rows,
                'dataset_version': stats.dataset_version,
                'train_source': stats.train_source,
                'test_source': stats.test_source,
                'suffix': stats.suffix,
                'images_dir': str(IMAGES_DIR),
                'output_dir': stats.output_dir,
                'train_path': stats.train_path,
                'test_path': stats.test_path,
                'query_path': stats.query_path,
                'query_mixed56_path': stats.query_mixed56_path,
                'qrels_verified': stats.qrels_verified_path,
                'qrels_pseudolabel': stats.qrels_pseudolabel_path,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
