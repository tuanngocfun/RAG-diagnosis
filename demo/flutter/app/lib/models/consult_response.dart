class DiagnosisRank {
  const DiagnosisRank({
    required this.rank,
    required this.label,
    required this.confidenceBand,
    required this.rationale,
  });

  factory DiagnosisRank.fromJson(Map<String, dynamic> json) {
    return DiagnosisRank(
      rank: json['rank'] as int? ?? 0,
      label: json['label'] as String? ?? '',
      confidenceBand: json['confidence_band'] as String? ?? '',
      rationale: json['rationale'] as String? ?? '',
    );
  }

  final int rank;
  final String label;
  final String confidenceBand;
  final String rationale;
}

class EvidenceItem {
  const EvidenceItem({
    required this.chunkId,
    required this.sourceCaseId,
    required this.title,
    required this.diagnosisLabel,
    required this.excerpt,
    required this.score,
    required this.confirmatory,
  });

  factory EvidenceItem.fromJson(Map<String, dynamic> json) {
    return EvidenceItem(
      chunkId: json['chunk_id'] as String? ?? '',
      sourceCaseId: json['source_case_id'] as String? ?? '',
      title: json['title'] as String? ?? '',
      diagnosisLabel: json['diagnosis_label'] as String? ?? '',
      excerpt: json['excerpt'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble() ?? 0,
      confirmatory: json['confirmatory'] as bool? ?? false,
    );
  }

  final String chunkId;
  final String sourceCaseId;
  final String title;
  final String diagnosisLabel;
  final String excerpt;
  final double score;
  final bool confirmatory;
}

class UncertaintyGate {
  const UncertaintyGate({
    required this.stage,
    required this.triggerCodes,
    required this.retrievalSupportStatus,
    required this.modelConfidence,
    required this.imageUsable,
    required this.escalationRequired,
    required this.topScore,
    required this.evidenceConflictFlag,
    required this.providerMode,
  });

  factory UncertaintyGate.fromJson(Map<String, dynamic> json) {
    return UncertaintyGate(
      stage: json['stage'] as String? ?? '',
      triggerCodes: (json['trigger_codes'] as List<dynamic>? ?? <dynamic>[])
          .map((dynamic item) => item.toString())
          .toList(),
      retrievalSupportStatus: json['retrieval_support_status'] as String? ?? '',
      modelConfidence: json['model_confidence'] as String? ?? '',
      imageUsable: json['image_usable'] as bool? ?? false,
      escalationRequired: json['escalation_required'] as bool? ?? true,
      topScore: (json['top_score'] as num?)?.toDouble() ?? 0,
      evidenceConflictFlag: json['evidence_conflict_flag'] as bool? ?? false,
      providerMode: json['provider_mode'] as String? ?? '',
    );
  }

  final String stage;
  final List<String> triggerCodes;
  final String retrievalSupportStatus;
  final String modelConfidence;
  final bool imageUsable;
  final bool escalationRequired;
  final double topScore;
  final bool evidenceConflictFlag;
  final String providerMode;
}

class ConsultResponse {
  const ConsultResponse({
    required this.requestId,
    required this.modelName,
    required this.decisionState,
    required this.topDiagnoses,
    required this.answerMarkdown,
    required this.evidence,
    required this.disclaimer,
    required this.timingMs,
    required this.uncertaintyGate,
    required this.neededNextInputs,
    required this.safeToShowRankedDifferential,
    required this.runtimeMetadata,
  });

  factory ConsultResponse.fromJson(Map<String, dynamic> json) {
    return ConsultResponse(
      requestId: json['request_id'] as String? ?? '',
      modelName: json['model_name'] as String? ?? '',
      decisionState: json['decision_state'] as String? ?? '',
      topDiagnoses: (json['top_diagnoses'] as List<dynamic>? ?? <dynamic>[])
          .map((dynamic item) =>
              DiagnosisRank.fromJson(item as Map<String, dynamic>))
          .toList(),
      answerMarkdown: json['answer_markdown'] as String? ?? '',
      evidence: (json['evidence'] as List<dynamic>? ?? <dynamic>[])
          .map((dynamic item) =>
              EvidenceItem.fromJson(item as Map<String, dynamic>))
          .toList(),
      disclaimer: json['disclaimer'] as String? ?? '',
      timingMs: json['timing_ms'] as int? ?? 0,
      uncertaintyGate: UncertaintyGate.fromJson(
        json['uncertainty_gate'] as Map<String, dynamic>? ??
            <String, dynamic>{},
      ),
      neededNextInputs:
          (json['needed_next_inputs'] as List<dynamic>? ?? <dynamic>[])
              .map((dynamic item) => item.toString())
              .toList(),
      safeToShowRankedDifferential:
          json['safe_to_show_ranked_differential'] as bool? ?? false,
      runtimeMetadata: Map<String, dynamic>.from(
        json['runtime_metadata'] as Map<String, dynamic>? ??
            <String, dynamic>{},
      ),
    );
  }

  final String requestId;
  final String modelName;
  final String decisionState;
  final List<DiagnosisRank> topDiagnoses;
  final String answerMarkdown;
  final List<EvidenceItem> evidence;
  final String disclaimer;
  final int timingMs;
  final UncertaintyGate uncertaintyGate;
  final List<String> neededNextInputs;
  final bool safeToShowRankedDifferential;
  final Map<String, dynamic> runtimeMetadata;

  bool get isAbstained => decisionState == 'abstained';
  bool get isProvisional => decisionState == 'provisional_parametric';
  bool get isGrounded => decisionState == 'rag_supported';
  bool get isRealGpuProvider => providerMode == 'real_gpu_gemma4';

  String get providerMode =>
      runtimeMetadata['provider_mode']?.toString() ??
      uncertaintyGate.providerMode;

  String get gpuName => runtimeMetadata['gpu_name']?.toString() ?? '';

  String get quantizationMode =>
      runtimeMetadata['quantization_mode']?.toString() ?? '';

  double? get generationLatencySeconds =>
      (runtimeMetadata['generation_latency_seconds'] as num?)?.toDouble();

  double? get modelLoadSeconds =>
      (runtimeMetadata['model_load_seconds'] as num?)?.toDouble();

  double? get gpuPeakMemoryAllocatedMib =>
      (runtimeMetadata['gpu_peak_memory_allocated_mib'] as num?)?.toDouble();

  int? get queryImageTensorCount =>
      (runtimeMetadata['query_image_tensor_count'] as num?)?.toInt();
}
