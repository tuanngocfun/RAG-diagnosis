#!/usr/bin/env python3
"""Compare saved RAG and no-RAG ragas artifacts without re-judging answers."""
from __future__ import annotations

import json
import math
import hashlib
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

from .config import RUNS_DIR
from .confirmatory_signals import context_has_confirmatory_signal
from .diagnosis_output_parser import analyze_answer_format
from .failure_taxonomy import (
    build_failure_taxonomy,
    summarize_capability_caveats,
    summarize_failure_taxonomy,
)


METRICS = [
    'l3_top1_correct',
    'top3_hit',
    'diagnosis_accuracy',
    'diagnosis_type_accuracy',
    'diagnosis_family_accuracy',
    'reasoning_recall',
    'multimodal_faithfulness',
    'multimodal_relevance',
    'context_relevance',
]
BUCKETS = ['all', 'leish', 'nonleish']
NONLEISH_DIAGNOSIS_TYPES = {'NON-LEISHMANIASIS'}
ANSWER_FILE_GLOBS = ('answers*.jsonl',)


def _resolve_run_dir(run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return (RUNS_DIR / run_id_or_path).resolve()


def _load_ragas_rows(run_dir: Path) -> Dict[str, Dict]:
    ragas_path = run_dir / 'ragas.jsonl'
    if not ragas_path.exists():
        raise FileNotFoundError(f'Missing ragas.jsonl: {ragas_path}')
    rows: Dict[str, Dict] = {}
    with open(ragas_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get('qid')
            if qid:
                rows[qid] = row
    return rows


def _load_json(path: Path) -> Dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def _build_output_path(rag_dir: Path, norag_dir: Path) -> Path:
    filename = f'comparison_{rag_dir.name}_vs_{norag_dir.name}.json'
    if len(filename) <= 240:
        return RUNS_DIR / filename

    digest = hashlib.sha1(f'{rag_dir.name}::{norag_dir.name}'.encode('utf-8')).hexdigest()[:12]
    rag_stub = rag_dir.name[:60]
    norag_stub = norag_dir.name[:60]
    shortened = f'comparison_{rag_stub}_vs_{norag_stub}_{digest}.json'
    return RUNS_DIR / shortened


def _load_run_contract_metadata(run_dir: Path) -> Dict[str, object]:
    metadata: Dict[str, object] = {
        'run_dir': str(run_dir),
        'base_run': None,
        'is_rag': None,
        'control_type': None,
        'prompt_mode': None,
        'prompt_contract_version': None,
        'prompt_contract_notes': None,
        'prompt_contract_summary': None,
        'decision_support_level': None,
        'generator_model': None,
        'answer_generation_contract_source': None,
        'ordering_mode': None,
        'use_context_image_tensors': None,
        'support_image_tensor_budget': None,
        'multimodal_usage_summary': None,
    }

    answer_generation_contract = run_dir / 'answer_generation_contract.json'
    if answer_generation_contract.exists():
        payload = _load_json(answer_generation_contract)
        metadata['is_rag'] = payload.get('is_rag', metadata['is_rag'])
        metadata['prompt_mode'] = payload.get('prompt_mode_requested') or payload.get('prompt_mode')
        metadata['prompt_contract_version'] = payload.get('prompt_contract_version')
        metadata['prompt_contract_notes'] = payload.get('prompt_contract_notes')
        metadata['prompt_contract_summary'] = payload.get('prompt_contract_summary')
        metadata['generator_model'] = payload.get('generator_model')
        metadata['ordering_mode'] = payload.get('ordering_mode')
        metadata['use_context_image_tensors'] = payload.get('use_context_image_tensors')
        metadata['support_image_tensor_budget'] = payload.get('support_image_tensor_budget')
        metadata['multimodal_usage_summary'] = payload.get('multimodal_usage_summary')
        metadata['answer_generation_contract_source'] = str(answer_generation_contract)

    run_config = run_dir / 'run_config.json'
    if run_config.exists():
        payload = _load_json(run_config)
        metadata['is_rag'] = payload.get('is_rag', metadata['is_rag'])
        metadata['control_type'] = payload.get('control_type', metadata['control_type'])
        metadata['prompt_mode'] = metadata['prompt_mode'] or payload.get('prompt_mode')
        metadata['prompt_contract_version'] = metadata['prompt_contract_version'] or payload.get('prompt_contract_version')
        metadata['prompt_contract_notes'] = metadata['prompt_contract_notes'] or payload.get('prompt_contract_notes')
        metadata['generator_model'] = metadata['generator_model'] or payload.get('generator_model')
        nested_generation_contract = payload.get('answer_generation_contract') or {}
        if isinstance(nested_generation_contract, dict):
            metadata['prompt_mode'] = metadata['prompt_mode'] or nested_generation_contract.get('prompt_mode_requested') or nested_generation_contract.get('prompt_mode')
            metadata['prompt_contract_version'] = (
                metadata['prompt_contract_version']
                or nested_generation_contract.get('prompt_contract_version')
            )
            metadata['prompt_contract_notes'] = (
                metadata['prompt_contract_notes']
                or nested_generation_contract.get('prompt_contract_notes')
            )
            metadata['prompt_contract_summary'] = (
                metadata['prompt_contract_summary']
                or nested_generation_contract.get('prompt_contract_summary')
            )
            metadata['generator_model'] = metadata['generator_model'] or nested_generation_contract.get('generator_model')
            metadata['ordering_mode'] = metadata['ordering_mode'] or nested_generation_contract.get('ordering_mode')
            if metadata['use_context_image_tensors'] is None:
                metadata['use_context_image_tensors'] = nested_generation_contract.get('use_context_image_tensors')
            if metadata['support_image_tensor_budget'] is None:
                metadata['support_image_tensor_budget'] = nested_generation_contract.get('support_image_tensor_budget')
            metadata['multimodal_usage_summary'] = (
                metadata['multimodal_usage_summary']
                or nested_generation_contract.get('multimodal_usage_summary')
            )
            if metadata['answer_generation_contract_source'] is None and nested_generation_contract:
                metadata['answer_generation_contract_source'] = f'{run_config}::answer_generation_contract'

    seed_config = run_dir / 'seed_sweep_config.json'
    if seed_config.exists():
        payload = _load_json(seed_config)
        metadata['prompt_mode'] = metadata['prompt_mode'] or payload.get('prompt_mode')
        metadata['generator_model'] = metadata['generator_model'] or payload.get('model')
        metadata['base_run'] = payload.get('base_run') or payload.get('base_run_dir') or payload.get('base_run_id')
        metadata['ordering_mode'] = metadata['ordering_mode'] or payload.get('ordering_mode')
        if metadata['use_context_image_tensors'] is None:
            metadata['use_context_image_tensors'] = payload.get('use_context_image_tensors')
        if metadata['support_image_tensor_budget'] is None:
            metadata['support_image_tensor_budget'] = payload.get('support_image_tensor_budget')

    summary = run_dir / 'summary.json'
    if summary.exists():
        payload = _load_json(summary)
        nested_generation_contract = payload.get('answer_generation_contract') or {}
        if isinstance(nested_generation_contract, dict):
            metadata['prompt_mode'] = metadata['prompt_mode'] or nested_generation_contract.get('prompt_mode_requested') or nested_generation_contract.get('prompt_mode')
            metadata['prompt_contract_version'] = (
                metadata['prompt_contract_version']
                or nested_generation_contract.get('prompt_contract_version')
            )
            metadata['prompt_contract_notes'] = (
                metadata['prompt_contract_notes']
                or nested_generation_contract.get('prompt_contract_notes')
            )
            metadata['prompt_contract_summary'] = (
                metadata['prompt_contract_summary']
                or nested_generation_contract.get('prompt_contract_summary')
            )
            metadata['generator_model'] = metadata['generator_model'] or nested_generation_contract.get('generator_model')
            metadata['ordering_mode'] = metadata['ordering_mode'] or nested_generation_contract.get('ordering_mode')
            if metadata['use_context_image_tensors'] is None:
                metadata['use_context_image_tensors'] = nested_generation_contract.get('use_context_image_tensors')
            if metadata['support_image_tensor_budget'] is None:
                metadata['support_image_tensor_budget'] = nested_generation_contract.get('support_image_tensor_budget')
            metadata['multimodal_usage_summary'] = (
                metadata['multimodal_usage_summary']
                or nested_generation_contract.get('multimodal_usage_summary')
            )
            if metadata['answer_generation_contract_source'] is None and nested_generation_contract:
                metadata['answer_generation_contract_source'] = f'{summary}::answer_generation_contract'

    if metadata['is_rag'] is False and not metadata['prompt_contract_version']:
        metadata['control_type'] = metadata['control_type'] or 'provisional_legacy_norag_prompt'
        metadata['prompt_contract_version'] = 'legacy_custom_norag_prompt'
        metadata['decision_support_level'] = 'directional_only'
    elif metadata['is_rag'] is False and metadata['control_type'] in {'matched_norag', 'matched_norag_chunked_sync'}:
        metadata['decision_support_level'] = 'final_control_candidate'

    return metadata


def _normalize_text(value: object) -> str:
    return str(value or '').strip()


def _normalize_diagnosis_type(value: object) -> str:
    return _normalize_text(value).upper()


def _bucket_from_ground_truth(ground_truth: object) -> Optional[str]:
    if not isinstance(ground_truth, dict):
        return None
    diagnosis_type = _normalize_diagnosis_type(ground_truth.get('diagnosis_type'))
    diagnosis = _normalize_text(ground_truth.get('diagnosis')).lower()
    if diagnosis_type in NONLEISH_DIAGNOSIS_TYPES or 'non-leish' in diagnosis:
        return 'nonleish'
    if diagnosis_type or diagnosis:
        return 'leish'
    return None


def _find_answer_file(run_dir: Path) -> Optional[Path]:
    candidates: List[Path] = []
    for pattern in ANSWER_FILE_GLOBS:
        candidates.extend(sorted(run_dir.glob(pattern)))
    if not candidates:
        return None
    preferred = sorted(
        candidates,
        key=lambda path: (
            'norag' not in path.name.lower(),
            'rag' not in path.name.lower(),
            path.name,
        ),
    )
    return preferred[0]


def _infer_generation_mode(answer_row: Dict, answer_file: Optional[Path]) -> Optional[str]:
    generation_mode = _normalize_text(answer_row.get('generation_mode'))
    if generation_mode:
        return generation_mode
    if _normalize_text(answer_row.get('retrieval_support_status')):
        contexts = answer_row.get('contexts') or []
        return 'rag_prompt' if contexts else 'no_rag_fallback'
    contexts = answer_row.get('contexts') or []
    if answer_file and 'norag' in answer_file.name.lower():
        return 'no_rag'
    if contexts:
        return 'rag_prompt'
    if answer_file and 'rag' in answer_file.name.lower():
        return 'no_rag_fallback'
    return None


def _infer_retrieval_support_status(answer_row: Dict, generation_mode: Optional[str]) -> Optional[str]:
    status = _normalize_text(answer_row.get('retrieval_support_status'))
    if status:
        return status
    contexts = answer_row.get('contexts') or []
    if generation_mode == 'no_rag':
        return 'not_applicable_norag'
    if generation_mode == 'no_rag_fallback':
        return 'empty_context_fallback'
    if contexts:
        return 'context_available'
    return None


def _load_answer_metadata(run_dir: Path) -> Dict[str, Dict]:
    answer_file = _find_answer_file(run_dir)
    if answer_file is None or not answer_file.exists():
        return {}

    metadata_by_qid: Dict[str, Dict] = {}
    with open(answer_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get('qid')
            if not qid:
                continue
            generation_mode = _infer_generation_mode(row, answer_file)
            ground_truth = row.get('ground_truth')
            ground_truth_pseudolabel = row.get('ground_truth_pseudolabel')
            bucket = _bucket_from_ground_truth(ground_truth) or _bucket_from_ground_truth(ground_truth_pseudolabel)
            format_analysis = analyze_answer_format(row.get('answer', ''))
            prompt_context_doc_ids = row.get('prompt_context_doc_ids') or [
                ctx.get('doc_id') for ctx in (row.get('contexts') or []) if ctx.get('doc_id')
            ]
            metadata_by_qid[qid] = {
                'ground_truth': ground_truth,
                'ground_truth_pseudolabel': ground_truth_pseudolabel,
                'ground_truth_bucket': bucket,
                'query_type': row.get('query_type'),
                'generation_mode': generation_mode,
                'retrieval_support_status': _infer_retrieval_support_status(row, generation_mode),
                'diagnosis_family': row.get('diagnosis_family') or format_analysis.diagnosis_family,
                'diagnosis_family_source': row.get('diagnosis_family_source') or format_analysis.diagnosis_family_source,
                'answer_format_valid': row.get('answer_format_valid', format_analysis.answer_format_valid),
                'answer_format_error': row.get('answer_format_error') or format_analysis.answer_format_error,
                'predicted_rank1_diagnosis_text': format_analysis.rank1_diagnosis_text,
                'prompt_context_doc_ids': prompt_context_doc_ids,
                'prompt_context_count': row.get('prompt_context_count', len(prompt_context_doc_ids)),
                'query_image_count': len(row.get('query_images') or []),
                'context_image_count': len(row.get('context_images') or []),
                'query_image_tensor_attempt_count': row.get('query_image_tensor_attempt_count'),
                'support_image_tensor_attempt_count': row.get('support_image_tensor_attempt_count'),
                'query_image_tensor_count': row.get('query_image_tensor_count'),
                'support_image_tensor_count': row.get('support_image_tensor_count'),
                'image_tensor_fallback_used': row.get('image_tensor_fallback_used'),
                'image_tensor_fallback_reason': row.get('image_tensor_fallback_reason'),
                'ordering_mode': row.get('ordering_mode'),
                'use_context_image_tensors': row.get('use_context_image_tensors'),
                'context_k': row.get('context_k'),
                'answer_text': row.get('answer', ''),
            }
    return metadata_by_qid


def _enrich_rows_with_answer_metadata(run_dir: Path, rows: Dict[str, Dict]) -> Dict[str, Dict]:
    answer_metadata = _load_answer_metadata(run_dir)
    if not answer_metadata:
        return rows
    enriched: Dict[str, Dict] = {}
    for qid, row in rows.items():
        merged = dict(row)
        metadata = answer_metadata.get(qid) or {}
        for key, value in metadata.items():
            if merged.get(key) in (None, '', 'unknown') and value not in (None, ''):
                merged[key] = value
        enriched[qid] = merged
    return enriched


def _resolve_base_run_dir(run_dir: Path) -> Optional[Path]:
    seed_config = run_dir / 'seed_sweep_config.json'
    if not seed_config.exists():
        return None
    payload = _load_json(seed_config)
    for key in ('base_run', 'base_run_dir', 'base_run_id'):
        candidate = payload.get(key)
        if not candidate:
            continue
        resolved = _resolve_run_dir(str(candidate))
        if resolved.exists():
            return resolved
    return None


def _load_train_jsonl_from_run(run_dir: Path) -> Optional[Path]:
    candidate_paths = [run_dir / 'run_config.json', run_dir / 'summary.json']
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        payload = _load_json(candidate_path)
        runtime_metadata = payload.get('runtime_metadata') or {}
        candidate = runtime_metadata.get('train_jsonl') or payload.get('train_jsonl')
        if not candidate:
            continue
        train_jsonl = Path(str(candidate)).expanduser()
        if train_jsonl.exists():
            return train_jsonl
    return None


def _bucket_from_train_case(case_row: Dict[str, object]) -> Optional[str]:
    is_leish = case_row.get('is_leishmaniasis')
    if isinstance(is_leish, bool):
        return 'leish' if is_leish else 'nonleish'
    diagnosis_type = _normalize_diagnosis_type(case_row.get('diagnosis_type'))
    diagnosis = _normalize_text(case_row.get('diagnosis')).lower()
    if diagnosis_type in NONLEISH_DIAGNOSIS_TYPES or 'non-leish' in diagnosis:
        return 'nonleish'
    if diagnosis_type or diagnosis:
        return 'leish'
    return None


def _load_train_bucket_index(train_jsonl: Path) -> Dict[str, str]:
    bucket_index: Dict[str, str] = {}
    with open(train_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = _normalize_text(row.get('case_id'))
            if not case_id:
                continue
            bucket = _bucket_from_train_case(row)
            if bucket:
                bucket_index[case_id] = bucket
    return bucket_index


def _load_base_retrieval_diagnostics(run_dir: Path) -> Dict[str, Dict[str, object]]:
    base_run_dir = _resolve_base_run_dir(run_dir)
    if base_run_dir is None:
        return {}

    retrieval_path = base_run_dir / 'retrieval.jsonl'
    if not retrieval_path.exists():
        return {}

    train_jsonl = _load_train_jsonl_from_run(base_run_dir)
    train_bucket_index = _load_train_bucket_index(train_jsonl) if train_jsonl is not None else {}
    diagnostics: Dict[str, Dict[str, object]] = {}

    with open(retrieval_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = _normalize_text(row.get('qid'))
            if not qid:
                continue
            contexts = row.get('contexts') or []
            leish_count = 0
            nonleish_count = 0
            confirmatory_doc_ids: Set[str] = set()
            for ctx in contexts:
                doc_id = _normalize_text(ctx.get('doc_id'))
                if not doc_id:
                    continue
                bucket = train_bucket_index.get(doc_id)
                if bucket == 'leish':
                    leish_count += 1
                elif bucket == 'nonleish':
                    nonleish_count += 1
                if context_has_confirmatory_signal(str(ctx.get('text') or '')):
                    confirmatory_doc_ids.add(doc_id)
            diagnostics[qid] = {
                'retrieved_leish_context_count': leish_count,
                'retrieved_nonleish_context_count': nonleish_count,
                '_confirmatory_doc_ids': sorted(confirmatory_doc_ids),
            }

    return diagnostics


def _enrich_rows_with_base_retrieval_diagnostics(run_dir: Path, rows: Dict[str, Dict]) -> Dict[str, Dict]:
    diagnostics_by_qid = _load_base_retrieval_diagnostics(run_dir)
    if not diagnostics_by_qid:
        return rows

    enriched: Dict[str, Dict] = {}
    for qid, row in rows.items():
        merged = dict(row)
        diagnostics = diagnostics_by_qid.get(qid) or {}
        confirmatory_doc_ids = {str(doc_id) for doc_id in diagnostics.get('_confirmatory_doc_ids') or [] if doc_id}
        prompt_doc_ids = {str(doc_id) for doc_id in (merged.get('prompt_context_doc_ids') or []) if doc_id}
        if diagnostics:
            merged['retrieved_leish_context_count'] = diagnostics.get('retrieved_leish_context_count')
            merged['retrieved_nonleish_context_count'] = diagnostics.get('retrieved_nonleish_context_count')
            merged['confirmatory_in_topk'] = bool(confirmatory_doc_ids)
            merged['confirmatory_in_prompt_context'] = bool(confirmatory_doc_ids & prompt_doc_ids)
            merged['confirmatory_dropped_by_pruning'] = bool(confirmatory_doc_ids and not (confirmatory_doc_ids & prompt_doc_ids))
        enriched[qid] = merged

    return enriched


def _bucket_for_row(row: Dict) -> str:
    if row.get('ground_truth_bucket'):
        return str(row['ground_truth_bucket'])
    traces = row.get('traces') or {}
    bucket = traces.get('ground_truth_bucket')
    if bucket:
        return str(bucket)
    inferred = _bucket_from_ground_truth(row.get('ground_truth')) or _bucket_from_ground_truth(row.get('ground_truth_pseudolabel'))
    return inferred or 'unknown'


def _numeric(value: object) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value):
            return None
        return value
    return None


def _mean_metric(rows: Iterable[Dict], metric: str) -> Optional[float]:
    values = [_numeric(row.get(metric)) for row in rows]
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return None
    return sum(numeric_values) / len(numeric_values)


def _rate(rows: Iterable[Dict], key: str) -> Optional[float]:
    values = [row.get(key) for row in rows if isinstance(row.get(key), bool)]
    if not values:
        return None
    return sum(1.0 for value in values if value) / len(values)


def _family_metric_coverage(rows: Iterable[Dict]) -> Optional[float]:
    rows = list(rows)
    if not rows:
        return None
    covered = sum(1 for row in rows if _numeric(row.get('diagnosis_family_accuracy')) is not None)
    return covered / len(rows)


def _nonleish_rank1_on_leish_gt(rows: Iterable[Dict]) -> int:
    count = 0
    for row in rows:
        if _bucket_for_row(row) != 'leish':
            continue
        family = _normalize_text(row.get('diagnosis_family'))
        rank1_text = _normalize_text(row.get('predicted_rank1_diagnosis_text')).lower()
        if family == 'nonleish_family' or 'non-leish' in rank1_text:
            count += 1
    return count


def _counter(rows: Iterable[Dict], key: str) -> Dict[str, int]:
    counter = Counter()
    for row in rows:
        value = row.get(key)
        counter[str(value or 'MISSING')] += 1
    return dict(counter)


def _bucket_rows(rows: List[Dict], bucket: str) -> List[Dict]:
    if bucket == 'all':
        return rows
    return [row for row in rows if _bucket_for_row(row) == bucket]


def _summarize_rows(rows: List[Dict]) -> Dict[str, object]:
    reasoning_recall_available = sum(1 for row in rows if _numeric(row.get('reasoning_recall')) is not None)
    reasoning_recall_missing_groundtruth = sum(
        1 for row in rows if row.get('reasoning_recall_method') == 'skipped_missing_groundtruth_reasoning'
    )
    reasoning_recall_missing_trace = sum(
        1 for row in rows if row.get('reasoning_recall_method') == 'skipped_unparseable_predicted_reasoning'
    )
    return {
        'n_rows': len(rows),
        'generation_mode_counts': _counter(rows, 'generation_mode'),
        'retrieval_support_status_counts': _counter(rows, 'retrieval_support_status'),
        'rank_source_counts': _counter(rows, 'rank_source'),
        'judge_parser_disagreement_rate': _rate(rows, 'judge_parser_disagreement'),
        'diagnosis_family_source_counts': _counter(rows, 'diagnosis_family_source'),
        'answer_format_error_counts': _counter(rows, 'answer_format_error'),
        'answer_format_valid_rate': _rate(rows, 'answer_format_valid'),
        'family_metric_coverage_rate': _family_metric_coverage(rows),
        'nonleish_rank1_on_leish_gt': _nonleish_rank1_on_leish_gt(rows),
        'reasoning_recall_coverage_rate': (reasoning_recall_available / len(rows)) if rows else None,
        'reasoning_recall_available_count': reasoning_recall_available,
        'reasoning_recall_missing_groundtruth_count': reasoning_recall_missing_groundtruth,
        'reasoning_recall_missing_trace_count': reasoning_recall_missing_trace,
        'reasoning_recall_method_counts': _counter(rows, 'reasoning_recall_method'),
        'reasoning_recall_judge_model_counts': _counter(rows, 'reasoning_recall_judge_model'),
        'reasoning_recall_source_id_counts': _counter(rows, 'reasoning_recall_source_id'),
        'metrics': {
            metric: _mean_metric(rows, metric)
            for metric in METRICS
        },
    }


def _summarize_by_bucket(rows: List[Dict]) -> Dict[str, Dict[str, object]]:
    return {
        bucket: _summarize_rows(_bucket_rows(rows, bucket))
        for bucket in BUCKETS
    }


def _metric_delta(rag_value: Optional[float], norag_value: Optional[float]) -> Optional[float]:
    if rag_value is None or norag_value is None:
        return None
    return rag_value - norag_value


def _build_metric_deltas(rag_summary: Dict[str, Dict[str, object]], norag_summary: Dict[str, Dict[str, object]]) -> Dict[str, Dict[str, Optional[float]]]:
    result: Dict[str, Dict[str, Optional[float]]] = {}
    for bucket in BUCKETS:
        result[bucket] = {}
        rag_metrics = rag_summary[bucket]['metrics']
        norag_metrics = norag_summary[bucket]['metrics']
        for metric in METRICS:
            result[bucket][metric] = _metric_delta(rag_metrics.get(metric), norag_metrics.get(metric))
    return result


def _safe_rate(count: int, total: int) -> Optional[float]:
    if total <= 0:
        return None
    return count / total


def _mean_row_value(rows: Iterable[Dict], key: str) -> Optional[float]:
    numeric_values = [_numeric(row.get(key)) for row in rows]
    filtered = [value for value in numeric_values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def _build_label_index(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {
        str(record.get('qid')): record
        for record in records
        if record.get('qid')
    }


def _summarize_contamination(rows: List[Dict], taxonomy_by_qid: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    by_bucket: Dict[str, Dict[str, object]] = {}
    for bucket in BUCKETS:
        bucket_rows = _bucket_rows(rows, bucket)
        diagnostic_rows = [
            row
            for row in bucket_rows
            if _numeric(row.get('retrieved_leish_context_count')) is not None
            and _numeric(row.get('retrieved_nonleish_context_count')) is not None
        ]
        leish_dominated_count = sum(
            1
            for row in diagnostic_rows
            if int(row.get('retrieved_leish_context_count') or 0) > int(row.get('retrieved_nonleish_context_count') or 0)
        )
        nonleish_dominated_count = sum(
            1
            for row in diagnostic_rows
            if int(row.get('retrieved_nonleish_context_count') or 0) > int(row.get('retrieved_leish_context_count') or 0)
        )
        tie_count = sum(
            1
            for row in diagnostic_rows
            if int(row.get('retrieved_nonleish_context_count') or 0) == int(row.get('retrieved_leish_context_count') or 0)
        )
        confusion_count = sum(
            1
            for row in diagnostic_rows
            if 'nonleish_confusion_from_leish_context' in set((taxonomy_by_qid.get(str(row.get('qid'))) or {}).get('labels') or [])
        )
        by_bucket[bucket] = {
            'n_rows': len(bucket_rows),
            'rows_with_retrieval_diagnostics': len(diagnostic_rows),
            'mean_retrieved_leish_context_count': _mean_row_value(diagnostic_rows, 'retrieved_leish_context_count'),
            'mean_retrieved_nonleish_context_count': _mean_row_value(diagnostic_rows, 'retrieved_nonleish_context_count'),
            'leish_dominated_count': leish_dominated_count,
            'leish_dominated_rate': _safe_rate(leish_dominated_count, len(diagnostic_rows)),
            'nonleish_dominated_count': nonleish_dominated_count,
            'nonleish_dominated_rate': _safe_rate(nonleish_dominated_count, len(diagnostic_rows)),
            'tie_count': tie_count,
            'tie_rate': _safe_rate(tie_count, len(diagnostic_rows)),
            'nonleish_confusion_from_leish_context_count': confusion_count,
        }
    return {
        'analysis_type': 'retrieval_contamination_summary_v1',
        'by_bucket': by_bucket,
    }


def _summarize_modality_use(rows: List[Dict], taxonomy_by_qid: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    by_bucket: Dict[str, Dict[str, object]] = {}
    for bucket in BUCKETS:
        bucket_rows = _bucket_rows(rows, bucket)
        query_image_rows = [row for row in bucket_rows if int(row.get('query_image_count') or 0) > 0]
        ignored_query_image_count = sum(
            1
            for row in bucket_rows
            if 'ignored_query_image' in set((taxonomy_by_qid.get(str(row.get('qid'))) or {}).get('labels') or [])
        )
        support_image_tensor_used_count = sum(
            1 for row in bucket_rows if int(row.get('support_image_tensor_count') or 0) > 0
        )
        use_context_image_tensors_true_count = sum(
            1 for row in bucket_rows if bool(row.get('use_context_image_tensors'))
        )
        by_bucket[bucket] = {
            'n_rows': len(bucket_rows),
            'rows_with_query_images': len(query_image_rows),
            'rows_with_query_images_rate': _safe_rate(len(query_image_rows), len(bucket_rows)),
            'ignored_query_image_count': ignored_query_image_count,
            'ignored_query_image_rate': _safe_rate(ignored_query_image_count, len(query_image_rows)),
            'mean_query_image_count': _mean_row_value(bucket_rows, 'query_image_count'),
            'mean_query_image_tensor_count': _mean_row_value(bucket_rows, 'query_image_tensor_count'),
            'mean_support_image_tensor_count': _mean_row_value(bucket_rows, 'support_image_tensor_count'),
            'support_image_tensor_used_count': support_image_tensor_used_count,
            'support_image_tensor_used_rate': _safe_rate(support_image_tensor_used_count, len(bucket_rows)),
            'use_context_image_tensors_true_count': use_context_image_tensors_true_count,
        }
    return {
        'analysis_type': 'modality_use_summary_v1',
        'by_bucket': by_bucket,
    }


def _summarize_confirmatory_evidence(rows: List[Dict]) -> Dict[str, object]:
    by_bucket: Dict[str, Dict[str, object]] = {}
    for bucket in BUCKETS:
        bucket_rows = _bucket_rows(rows, bucket)
        diagnostic_rows = [row for row in bucket_rows if row.get('confirmatory_in_topk') is not None]
        confirmatory_in_topk_count = sum(1 for row in diagnostic_rows if bool(row.get('confirmatory_in_topk')))
        confirmatory_in_prompt_count = sum(1 for row in diagnostic_rows if bool(row.get('confirmatory_in_prompt_context')))
        confirmatory_dropped_count = sum(1 for row in diagnostic_rows if bool(row.get('confirmatory_dropped_by_pruning')))
        by_bucket[bucket] = {
            'n_rows': len(bucket_rows),
            'rows_with_confirmatory_diagnostics': len(diagnostic_rows),
            'confirmatory_in_topk_count': confirmatory_in_topk_count,
            'confirmatory_in_topk_rate': _safe_rate(confirmatory_in_topk_count, len(diagnostic_rows)),
            'confirmatory_in_prompt_context_count': confirmatory_in_prompt_count,
            'confirmatory_in_prompt_context_rate': _safe_rate(confirmatory_in_prompt_count, len(diagnostic_rows)),
            'confirmatory_dropped_by_pruning_count': confirmatory_dropped_count,
            'confirmatory_dropped_by_pruning_rate': _safe_rate(confirmatory_dropped_count, len(diagnostic_rows)),
        }
    return {
        'analysis_type': 'confirmatory_evidence_summary_v1',
        'by_bucket': by_bucket,
    }


def _build_run_modality_status(meta: Dict[str, object], rows: List[Dict]) -> Dict[str, object]:
    stored_summary = meta.get('multimodal_usage_summary')
    if isinstance(stored_summary, dict):
        rows_with_query_images = int(stored_summary.get('rows_with_query_images') or 0)
        rows_with_context_images_available = int(stored_summary.get('rows_with_context_images_available') or 0)
        rows_with_query_image_tensors = int(stored_summary.get('rows_with_query_image_tensors') or 0)
        rows_with_support_image_tensors = int(stored_summary.get('rows_with_support_image_tensors') or 0)
        mean_query_image_tensor_count = stored_summary.get('mean_query_image_tensor_count')
        mean_support_image_tensor_count = stored_summary.get('mean_support_image_tensor_count')
        image_tensor_fallback_count = int(stored_summary.get('image_tensor_fallback_count') or 0)
        true_multimodal_support_active = bool(stored_summary.get('true_multimodal_support_active'))
    else:
        rows_with_query_images = sum(1 for row in rows if int(row.get('query_image_count') or 0) > 0)
        rows_with_context_images_available = sum(1 for row in rows if int(row.get('context_image_count') or 0) > 0)
        rows_with_query_image_tensors = sum(1 for row in rows if int(row.get('query_image_tensor_count') or 0) > 0)
        rows_with_support_image_tensors = sum(1 for row in rows if int(row.get('support_image_tensor_count') or 0) > 0)
        mean_query_image_tensor_count = _mean_row_value(rows, 'query_image_tensor_count')
        mean_support_image_tensor_count = _mean_row_value(rows, 'support_image_tensor_count')
        image_tensor_fallback_count = sum(1 for row in rows if row.get('image_tensor_fallback_used'))
        true_multimodal_support_active = rows_with_support_image_tensors > 0

    return {
        'n_rows': len(rows),
        'ordering_mode': meta.get('ordering_mode'),
        'use_context_image_tensors_requested': bool(meta.get('use_context_image_tensors')),
        'support_image_tensor_budget': meta.get('support_image_tensor_budget'),
        'rows_with_query_images': rows_with_query_images,
        'rows_with_context_images_available': rows_with_context_images_available,
        'rows_with_query_image_tensors': rows_with_query_image_tensors,
        'rows_with_support_image_tensors': rows_with_support_image_tensors,
        'mean_query_image_tensor_count': mean_query_image_tensor_count,
        'mean_support_image_tensor_count': mean_support_image_tensor_count,
        'image_tensor_fallback_count': image_tensor_fallback_count,
        'true_multimodal_support_active': true_multimodal_support_active,
    }


def _build_details(
    common_qids: List[str],
    rag_rows: Dict[str, Dict],
    norag_rows: Dict[str, Dict],
    taxonomy_by_qid: Dict[str, Dict[str, object]],
) -> List[Dict]:
    details: List[Dict] = []
    for qid in common_qids:
        rag_row = rag_rows[qid]
        norag_row = norag_rows[qid]
        taxonomy = taxonomy_by_qid.get(qid) or {}
        record = {
            'qid': qid,
            'bucket': _bucket_for_row(rag_row),
            'query_type': rag_row.get('query_type'),
            'rag_generation_mode': rag_row.get('generation_mode'),
            'norag_generation_mode': norag_row.get('generation_mode'),
            'rag_gt_rank': rag_row.get('gt_rank'),
            'norag_gt_rank': norag_row.get('gt_rank'),
            'rag_rank_source': rag_row.get('rank_source'),
            'norag_rank_source': norag_row.get('rank_source'),
            'rag_judge_parser_disagreement': rag_row.get('judge_parser_disagreement'),
            'norag_judge_parser_disagreement': norag_row.get('judge_parser_disagreement'),
            'rag_retrieval_support_status': rag_row.get('retrieval_support_status'),
            'norag_retrieval_support_status': norag_row.get('retrieval_support_status'),
            'rag_answer_format_valid': rag_row.get('answer_format_valid'),
            'norag_answer_format_valid': norag_row.get('answer_format_valid'),
            'rag_diagnosis_family_source': rag_row.get('diagnosis_family_source'),
            'norag_diagnosis_family_source': norag_row.get('diagnosis_family_source'),
            'rag_reasoning_recall_method': rag_row.get('reasoning_recall_method'),
            'norag_reasoning_recall_method': norag_row.get('reasoning_recall_method'),
            'rag_reasoning_recall_judge_model': rag_row.get('reasoning_recall_judge_model'),
            'norag_reasoning_recall_judge_model': norag_row.get('reasoning_recall_judge_model'),
            'rag_reasoning_recall_source_id': rag_row.get('reasoning_recall_source_id'),
            'norag_reasoning_recall_source_id': norag_row.get('reasoning_recall_source_id'),
            'rag_reasoning_recall_source_path': rag_row.get('reasoning_recall_source_path') or rag_row.get('reasoning_recall_source'),
            'norag_reasoning_recall_source_path': norag_row.get('reasoning_recall_source_path') or norag_row.get('reasoning_recall_source'),
            'rag_prompt_context_count': rag_row.get('prompt_context_count'),
            'rag_query_image_count': rag_row.get('query_image_count'),
            'rag_query_image_tensor_attempt_count': rag_row.get('query_image_tensor_attempt_count'),
            'rag_support_image_tensor_attempt_count': rag_row.get('support_image_tensor_attempt_count'),
            'rag_query_image_tensor_count': rag_row.get('query_image_tensor_count'),
            'rag_support_image_tensor_count': rag_row.get('support_image_tensor_count'),
            'rag_use_context_image_tensors': rag_row.get('use_context_image_tensors'),
            'rag_image_tensor_fallback_used': rag_row.get('image_tensor_fallback_used'),
            'rag_image_tensor_fallback_reason': rag_row.get('image_tensor_fallback_reason'),
            'retrieved_leish_context_count': rag_row.get('retrieved_leish_context_count'),
            'retrieved_nonleish_context_count': rag_row.get('retrieved_nonleish_context_count'),
            'confirmatory_in_topk': rag_row.get('confirmatory_in_topk'),
            'confirmatory_in_prompt_context': rag_row.get('confirmatory_in_prompt_context'),
            'confirmatory_dropped_by_pruning': rag_row.get('confirmatory_dropped_by_pruning'),
            'rag_failure_labels': list(taxonomy.get('labels') or []),
            'rag_failure_reasons': list(taxonomy.get('reasons') or []),
            'metrics': {},
        }
        for metric in METRICS:
            rag_value = _numeric(rag_row.get(metric))
            norag_value = _numeric(norag_row.get(metric))
            record['metrics'][metric] = {
                'rag': rag_value,
                'norag': norag_value,
                'delta': _metric_delta(rag_value, norag_value),
            }
        details.append(record)
    return details


def _fmt(value: Optional[float]) -> str:
    return 'NA' if value is None else f'{value:.4f}'


def _write_markdown_summary(
    output_path: Path,
    rag_meta: Dict[str, object],
    norag_meta: Dict[str, object],
    rag_modality_status: Dict[str, object],
    norag_modality_status: Dict[str, object],
    rag_summary: Dict[str, Dict[str, object]],
    norag_summary: Dict[str, Dict[str, object]],
    deltas: Dict[str, Dict[str, Optional[float]]],
    taxonomy_summary: Dict[str, Dict[str, object]],
    capability_caveats: Dict[str, object],
    contamination_summary: Dict[str, object],
    modality_use_summary: Dict[str, object],
    confirmatory_evidence_summary: Dict[str, object],
) -> Path:
    comparison_label = (
        'provisional comparison'
        if norag_meta.get('decision_support_level') == 'directional_only'
        else 'final matched comparison'
    )
    lines = [
        '# RAG vs No-RAG Comparison',
        '',
        f'- Comparison label: `{comparison_label}`',
        f"- RAG prompt metadata: mode=`{rag_meta.get('prompt_mode') or 'unknown'}`",
        (
            f"- No-RAG prompt metadata: control_type=`{norag_meta.get('control_type') or 'unknown'}`, "
            f"prompt_mode=`{norag_meta.get('prompt_mode') or 'unknown'}`, "
            f"prompt_contract_version=`{norag_meta.get('prompt_contract_version') or 'unknown'}`"
        ),
        '- Taxonomy note: failure labels and caveat counts are heuristic silver-label analysis, not clinician-verified truth.',
    ]
    if norag_meta.get('prompt_contract_notes'):
        lines.append(f"- No-RAG prompt notes: `{norag_meta.get('prompt_contract_notes')}`")
    if norag_meta.get('decision_support_level') == 'directional_only':
        lines.append(
            '- Decision rule: this provisional legacy no-RAG comparison is directional only and cannot support the final keep/reject conclusion.'
        )
        lines.append(
            '- Final keep/reject rule: only revised RAG vs matched no-RAG v2 can support the final decision.'
        )
    else:
        lines.append('- Decision rule: this matched no-RAG comparison is eligible to support the final keep/reject conclusion.')
    lines.append(
        "- RAG modality status: "
        f"true_multimodal_support_active=`{rag_modality_status.get('true_multimodal_support_active')}` "
        f"query_tensor_rows={rag_modality_status.get('rows_with_query_image_tensors')} "
        f"support_tensor_rows={rag_modality_status.get('rows_with_support_image_tensors')} "
        f"fallback_rows={rag_modality_status.get('image_tensor_fallback_count')}"
    )
    lines.append(
        "- No-RAG modality status: "
        f"true_multimodal_support_active=`{norag_modality_status.get('true_multimodal_support_active')}` "
        f"query_tensor_rows={norag_modality_status.get('rows_with_query_image_tensors')} "
        f"support_tensor_rows={norag_modality_status.get('rows_with_support_image_tensors')} "
        f"fallback_rows={norag_modality_status.get('image_tensor_fallback_count')}"
    )

    for bucket in BUCKETS:
        contamination = (contamination_summary.get('by_bucket') or {}).get(bucket, {})
        modality = (modality_use_summary.get('by_bucket') or {}).get(bucket, {})
        confirmatory = (confirmatory_evidence_summary.get('by_bucket') or {}).get(bucket, {})
        lines.extend(
            [
                '',
                f'## {bucket}',
                f"- n={rag_summary[bucket]['n_rows']}",
                (
                    f"- l3_top1_correct: rag={_fmt(rag_summary[bucket]['metrics']['l3_top1_correct'])} "
                    f"norag={_fmt(norag_summary[bucket]['metrics']['l3_top1_correct'])} "
                    f"delta={_fmt(deltas[bucket]['l3_top1_correct'])}"
                ),
                (
                    f"- top3_hit: rag={_fmt(rag_summary[bucket]['metrics']['top3_hit'])} "
                    f"norag={_fmt(norag_summary[bucket]['metrics']['top3_hit'])} "
                    f"delta={_fmt(deltas[bucket]['top3_hit'])}"
                ),
                (
                    f"- diagnosis_accuracy: rag={_fmt(rag_summary[bucket]['metrics']['diagnosis_accuracy'])} "
                    f"norag={_fmt(norag_summary[bucket]['metrics']['diagnosis_accuracy'])} "
                    f"delta={_fmt(deltas[bucket]['diagnosis_accuracy'])}"
                ),
                (
                    f"- diagnosis_type_accuracy: rag={_fmt(rag_summary[bucket]['metrics']['diagnosis_type_accuracy'])} "
                    f"norag={_fmt(norag_summary[bucket]['metrics']['diagnosis_type_accuracy'])} "
                    f"delta={_fmt(deltas[bucket]['diagnosis_type_accuracy'])}"
                ),
                (
                    f"- reasoning_recall: rag={_fmt(rag_summary[bucket]['metrics']['reasoning_recall'])} "
                    f"norag={_fmt(norag_summary[bucket]['metrics']['reasoning_recall'])} "
                    f"delta={_fmt(deltas[bucket]['reasoning_recall'])}"
                ),
                (
                    f"- reasoning_recall_coverage: rag={_fmt(rag_summary[bucket]['reasoning_recall_coverage_rate'])} "
                    f"norag={_fmt(norag_summary[bucket]['reasoning_recall_coverage_rate'])}"
                ),
                (
                    f"- reasoning_recall_method_counts: rag={rag_summary[bucket]['reasoning_recall_method_counts']} "
                    f"norag={norag_summary[bucket]['reasoning_recall_method_counts']}"
                ),
                (
                    f"- reasoning_recall_source_id_counts: rag={rag_summary[bucket]['reasoning_recall_source_id_counts']} "
                    f"norag={norag_summary[bucket]['reasoning_recall_source_id_counts']}"
                ),
                (
                    f"- answer_format_valid_rate: rag={_fmt(rag_summary[bucket]['answer_format_valid_rate'])} "
                    f"norag={_fmt(norag_summary[bucket]['answer_format_valid_rate'])}"
                ),
                f"- failure_taxonomy_counts: {taxonomy_summary.get(bucket, {}).get('label_counts', {})}",
                f"- capability_caveats: {capability_caveats.get('by_bucket', {}).get(bucket, {}).get('label_counts', {})}",
                (
                    f"- contamination_summary: "
                    f"leish_dominated={contamination.get('leish_dominated_count')} "
                    f"nonleish_dominated={contamination.get('nonleish_dominated_count')} "
                    f"mean_leish_contexts={_fmt(contamination.get('mean_retrieved_leish_context_count'))} "
                    f"mean_nonleish_contexts={_fmt(contamination.get('mean_retrieved_nonleish_context_count'))}"
                ),
                (
                    f"- modality_use_summary: "
                    f"query_image_rows={modality.get('rows_with_query_images')} "
                    f"ignored_query_image={modality.get('ignored_query_image_count')} "
                    f"support_tensors_used={modality.get('support_image_tensor_used_count')}"
                ),
                (
                    f"- confirmatory_evidence_summary: "
                    f"in_topk={confirmatory.get('confirmatory_in_topk_count')} "
                    f"in_prompt={confirmatory.get('confirmatory_in_prompt_context_count')} "
                    f"dropped_by_pruning={confirmatory.get('confirmatory_dropped_by_pruning_count')}"
                ),
            ]
        )

    markdown_path = output_path.with_suffix('.md')
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    return markdown_path


def compare_rag_norag(rag_run_id: str, norag_run_id: str, sample_size: Optional[int] = None) -> Path:
    rag_dir = _resolve_run_dir(rag_run_id)
    norag_dir = _resolve_run_dir(norag_run_id)
    rag_meta = _load_run_contract_metadata(rag_dir)
    norag_meta = _load_run_contract_metadata(norag_dir)

    rag_rows = _enrich_rows_with_answer_metadata(rag_dir, _load_ragas_rows(rag_dir))
    rag_rows = _enrich_rows_with_base_retrieval_diagnostics(rag_dir, rag_rows)
    norag_rows = _enrich_rows_with_answer_metadata(norag_dir, _load_ragas_rows(norag_dir))

    common_qids = sorted(set(rag_rows) & set(norag_rows))
    if sample_size is not None:
        common_qids = common_qids[:sample_size]

    if not common_qids:
        raise RuntimeError('No common qids found between the selected RAG and no-RAG runs.')

    rag_common = [rag_rows[qid] for qid in common_qids]
    norag_common = [norag_rows[qid] for qid in common_qids]

    rag_summary = _summarize_by_bucket(rag_common)
    norag_summary = _summarize_by_bucket(norag_common)
    deltas = _build_metric_deltas(rag_summary, norag_summary)
    taxonomy_records = build_failure_taxonomy(common_qids, rag_rows, norag_rows)
    taxonomy_by_qid = _build_label_index(taxonomy_records)
    taxonomy_summary = summarize_failure_taxonomy(taxonomy_records)
    capability_caveats = summarize_capability_caveats(rag_common)
    contamination_summary = _summarize_contamination(rag_common, taxonomy_by_qid)
    modality_use_summary = _summarize_modality_use(rag_common, taxonomy_by_qid)
    confirmatory_evidence_summary = _summarize_confirmatory_evidence(rag_common)
    rag_modality_status = _build_run_modality_status(rag_meta, rag_common)
    norag_modality_status = _build_run_modality_status(norag_meta, norag_common)

    print('=' * 72)
    print('RAG HELPED OR HURT AUDIT')
    print('=' * 72)
    print(f'RAG:   {rag_dir}')
    print(f'No-RAG:{norag_dir}')
    print(f'Common qids: {len(common_qids)}')
    print(f'RAG prompt metadata: mode={rag_meta.get("prompt_mode") or "unknown"}')
    print(f'RAG modality status: {rag_modality_status}')
    print(
        'No-RAG prompt metadata: '
        f'control_type={norag_meta.get("control_type") or "unknown"} '
        f'prompt_mode={norag_meta.get("prompt_mode") or "unknown"} '
        f'prompt_contract_version={norag_meta.get("prompt_contract_version") or "unknown"}'
    )
    print(f'No-RAG modality status: {norag_modality_status}')
    if norag_meta.get('decision_support_level') == 'directional_only':
        print('Decision support: provisional legacy no-RAG control only; do not use this comparison for final keep/reject.')
    elif norag_meta.get('decision_support_level') == 'final_control_candidate':
        print('Decision support: matched no-RAG control; eligible for final keep/reject use.')

    for bucket in BUCKETS:
        print(f'\n[{bucket}] n={rag_summary[bucket]["n_rows"]}')
        print(f'  generation_mode_counts (RAG):   {rag_summary[bucket]["generation_mode_counts"]}')
        print(f'  generation_mode_counts (No-RAG):{norag_summary[bucket]["generation_mode_counts"]}')
        print(f'  rank_source_counts (RAG):   {rag_summary[bucket]["rank_source_counts"]}')
        print(f'  rank_source_counts (No-RAG):{norag_summary[bucket]["rank_source_counts"]}')
        print(f'  judge_parser_disagreement_rate: rag={_fmt(rag_summary[bucket]["judge_parser_disagreement_rate"])} norag={_fmt(norag_summary[bucket]["judge_parser_disagreement_rate"])}')
        print(f'  retrieval_support_status_counts (RAG):   {rag_summary[bucket]["retrieval_support_status_counts"]}')
        print(f'  retrieval_support_status_counts (No-RAG):{norag_summary[bucket]["retrieval_support_status_counts"]}')
        print(f'  diagnosis_family_source_counts (RAG):   {rag_summary[bucket]["diagnosis_family_source_counts"]}')
        print(f'  diagnosis_family_source_counts (No-RAG):{norag_summary[bucket]["diagnosis_family_source_counts"]}')
        print(f'  reasoning_recall_coverage_rate: rag={_fmt(rag_summary[bucket]["reasoning_recall_coverage_rate"])} norag={_fmt(norag_summary[bucket]["reasoning_recall_coverage_rate"])}')
        print(f'  reasoning_recall_method_counts (RAG):   {rag_summary[bucket]["reasoning_recall_method_counts"]}')
        print(f'  reasoning_recall_method_counts (No-RAG):{norag_summary[bucket]["reasoning_recall_method_counts"]}')
        print(f'  reasoning_recall_source_id_counts (RAG):   {rag_summary[bucket]["reasoning_recall_source_id_counts"]}')
        print(f'  reasoning_recall_source_id_counts (No-RAG):{norag_summary[bucket]["reasoning_recall_source_id_counts"]}')
        print(f'  answer_format_valid_rate: rag={_fmt(rag_summary[bucket]["answer_format_valid_rate"])} norag={_fmt(norag_summary[bucket]["answer_format_valid_rate"])}')
        print(f'  family_metric_coverage_rate: rag={_fmt(rag_summary[bucket]["family_metric_coverage_rate"])} norag={_fmt(norag_summary[bucket]["family_metric_coverage_rate"])}')
        print(f'  nonleish_rank1_on_leish_gt: rag={rag_summary[bucket]["nonleish_rank1_on_leish_gt"]} norag={norag_summary[bucket]["nonleish_rank1_on_leish_gt"]}')
        print(f'  failure_taxonomy_counts: {taxonomy_summary.get(bucket, {}).get("label_counts", {})}')
        print(f'  capability_caveats: {capability_caveats.get("by_bucket", {}).get(bucket, {}).get("label_counts", {})}')
        print(f'  contamination_summary: {contamination_summary.get("by_bucket", {}).get(bucket, {})}')
        print(f'  modality_use_summary: {modality_use_summary.get("by_bucket", {}).get(bucket, {})}')
        print(f'  confirmatory_evidence_summary: {confirmatory_evidence_summary.get("by_bucket", {}).get(bucket, {})}')
        for metric in METRICS:
            rag_value = rag_summary[bucket]['metrics'][metric]
            norag_value = norag_summary[bucket]['metrics'][metric]
            delta = deltas[bucket][metric]
            print(f'  {metric}: rag={_fmt(rag_value)} norag={_fmt(norag_value)} delta={_fmt(delta)}')

    output_path = _build_output_path(rag_dir, norag_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        'rag_run': str(rag_dir),
        'norag_run': str(norag_dir),
        'rag_prompt_contract': rag_meta,
        'norag_prompt_contract': norag_meta,
        'common_qids': len(common_qids),
        'rag_only_qids': sorted(set(rag_rows) - set(norag_rows)),
        'norag_only_qids': sorted(set(norag_rows) - set(rag_rows)),
        'rag_summary': rag_summary,
        'norag_summary': norag_summary,
        'metric_deltas': deltas,
        'rag_modality_status': rag_modality_status,
        'norag_modality_status': norag_modality_status,
        'taxonomy_metadata': {
            'analysis_type': 'silver_label_heuristic',
            'failure_taxonomy': 'posthoc_failure_taxonomy_v1',
            'capability_caveats': 'posthoc_capability_caveat_summary_v1',
        },
        'failure_taxonomy_summary': taxonomy_summary,
        'capability_caveats': capability_caveats,
        'contamination_summary': contamination_summary,
        'modality_use_summary': modality_use_summary,
        'confirmatory_evidence_summary': confirmatory_evidence_summary,
        'details': _build_details(common_qids, rag_rows, norag_rows, taxonomy_by_qid),
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    taxonomy_path = output_path.with_suffix('.taxonomy.jsonl')
    with open(taxonomy_path, 'w', encoding='utf-8') as f:
        for record in taxonomy_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    markdown_path = _write_markdown_summary(
        output_path,
        rag_meta,
        norag_meta,
        rag_modality_status,
        norag_modality_status,
        rag_summary,
        norag_summary,
        deltas,
        taxonomy_summary,
        capability_caveats,
        contamination_summary,
        modality_use_summary,
        confirmatory_evidence_summary,
    )
    print(f'\nSaved comparison report to {output_path}')
    print(f'Saved failure taxonomy to {taxonomy_path}')
    print(f'Saved comparison summary to {markdown_path}')
    return output_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Compare saved RAG and no-RAG ragas artifacts')
    parser.add_argument('--rag', required=True, help='RAG run id or absolute run directory path')
    parser.add_argument('--norag', required=True, help='No-RAG run id or absolute run directory path')
    parser.add_argument('--sample', type=int, default=None, help='Optional cap on common qids for quick inspection')
    args = parser.parse_args()

    compare_rag_norag(
        rag_run_id=args.rag,
        norag_run_id=args.norag,
        sample_size=args.sample,
    )
