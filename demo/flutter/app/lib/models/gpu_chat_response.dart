import 'consult_response.dart';

class BackendHealth {
  const BackendHealth({
    required this.status,
    required this.modelName,
    required this.providerMode,
    required this.chatAvailable,
    required this.cudaAvailable,
    required this.gpuName,
    required this.bitsandbytesAvailable,
    required this.gpuFreeMemoryReady,
    required this.modelLoaded,
    required this.gpuMinFreeMib,
    this.gpuMemoryTotalMib,
    this.gpuMemoryFreeMib,
  });

  factory BackendHealth.fromJson(Map<String, dynamic> json) {
    return BackendHealth(
      status: json['status'] as String? ?? '',
      modelName: json['model_name'] as String? ?? '',
      providerMode: json['provider_mode'] as String? ?? '',
      chatAvailable: json['chat_available'] as bool? ?? false,
      cudaAvailable: json['cuda_available'] as bool? ?? false,
      gpuName: json['gpu_name'] as String? ?? '',
      bitsandbytesAvailable: json['bitsandbytes_available'] as bool? ?? false,
      gpuFreeMemoryReady: json['gpu_free_memory_ready'] as bool? ?? false,
      modelLoaded: json['model_loaded'] as bool? ?? false,
      gpuMinFreeMib: (json['gpu_min_free_mib'] as num?)?.toInt() ?? 12000,
      gpuMemoryTotalMib: (json['gpu_memory_total_mib'] as num?)?.toInt(),
      gpuMemoryFreeMib: (json['gpu_memory_free_mib'] as num?)?.toInt(),
    );
  }

  final String status;
  final String modelName;
  final String providerMode;
  final bool chatAvailable;
  final bool cudaAvailable;
  final String gpuName;
  final bool bitsandbytesAvailable;
  final bool gpuFreeMemoryReady;
  final bool modelLoaded;
  final int gpuMinFreeMib;
  final int? gpuMemoryTotalMib;
  final int? gpuMemoryFreeMib;

  bool get isGpuChatReady => providerMode == 'real_gpu_gemma4' && chatAvailable;
}

class GpuChatMessage {
  const GpuChatMessage({
    required this.role,
    required this.content,
  });

  final String role;
  final String content;

  Map<String, dynamic> toJson() => <String, dynamic>{
        'role': role,
        'content': content,
      };
}

class GpuChatResponse {
  const GpuChatResponse({
    required this.requestId,
    required this.modelName,
    required this.providerMode,
    required this.assistantMarkdown,
    required this.evidence,
    required this.disclaimer,
    required this.timingMs,
    required this.safetyState,
    required this.neededNextInputs,
    required this.runtimeMetadata,
    required this.retrievalAudit,
    required this.responseSourceMode,
    required this.sourceLabel,
    required this.sourcePath,
    required this.freshGenerationExecuted,
  });

  factory GpuChatResponse.fromJson(Map<String, dynamic> json) {
    return GpuChatResponse(
      requestId: json['request_id'] as String? ?? '',
      modelName: json['model_name'] as String? ?? '',
      providerMode: json['provider_mode'] as String? ?? '',
      assistantMarkdown: json['assistant_markdown'] as String? ?? '',
      evidence: (json['evidence'] as List<dynamic>? ?? <dynamic>[])
          .map((dynamic item) =>
              EvidenceItem.fromJson(item as Map<String, dynamic>))
          .toList(),
      disclaimer: json['disclaimer'] as String? ?? '',
      timingMs: json['timing_ms'] as int? ?? 0,
      safetyState: json['safety_state'] as String? ?? '',
      neededNextInputs:
          (json['needed_next_inputs'] as List<dynamic>? ?? <dynamic>[])
              .map((dynamic item) => item.toString())
              .toList(),
      runtimeMetadata: Map<String, dynamic>.from(
        json['runtime_metadata'] as Map<String, dynamic>? ??
            <String, dynamic>{},
      ),
      retrievalAudit: RetrievalAudit.fromJson(
        json['retrieval_audit'] as Map<String, dynamic>? ?? <String, dynamic>{},
      ),
      responseSourceMode: json['response_source_mode'] as String? ?? 'live_gpu',
      sourceLabel: json['source_label'] as String? ?? '',
      sourcePath: json['source_path'] as String? ?? '',
      freshGenerationExecuted:
          json['fresh_generation_executed'] as bool? ?? true,
    );
  }

  final String requestId;
  final String modelName;
  final String providerMode;
  final String assistantMarkdown;
  final List<EvidenceItem> evidence;
  final String disclaimer;
  final int timingMs;
  final String safetyState;
  final List<String> neededNextInputs;
  final Map<String, dynamic> runtimeMetadata;
  final RetrievalAudit retrievalAudit;
  final String responseSourceMode;
  final String sourceLabel;
  final String sourcePath;
  final bool freshGenerationExecuted;

  bool get isGenerated => safetyState == 'generated_support';
  bool get isBlocked => safetyState != 'generated_support';
  bool get isOfficialReplay => responseSourceMode == 'official_v12d_replay';

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

class RetrievalAudit {
  const RetrievalAudit({
    required this.retrievalBackend,
    required this.kbPath,
    required this.topKRequested,
    required this.candidateCount,
    required this.returnedCount,
    required this.scoringMethod,
    required this.liveRerankExecuted,
    required this.liveRerankMethod,
    required this.rerankBoundary,
    required this.returnedContexts,
    required this.officialReference,
  });

  factory RetrievalAudit.fromJson(Map<String, dynamic> json) {
    return RetrievalAudit(
      retrievalBackend: json['retrieval_backend'] as String? ?? '',
      kbPath: json['kb_path'] as String? ?? '',
      topKRequested: (json['top_k_requested'] as num?)?.toInt() ?? 0,
      candidateCount: (json['candidate_count'] as num?)?.toInt() ?? 0,
      returnedCount: (json['returned_count'] as num?)?.toInt() ?? 0,
      scoringMethod: json['scoring_method'] as String? ?? '',
      liveRerankExecuted: json['live_rerank_executed'] as bool? ?? false,
      liveRerankMethod: json['live_rerank_method']?.toString(),
      rerankBoundary: json['rerank_boundary'] as String? ?? '',
      returnedContexts:
          (json['returned_contexts'] as List<dynamic>? ?? <dynamic>[])
              .map(
                (dynamic item) => LiveRetrievedContext.fromJson(
                  item as Map<String, dynamic>,
                ),
              )
              .toList(),
      officialReference: OfficialRerankReference.fromJson(
        json['official_rerank_reference'] as Map<String, dynamic>? ??
            <String, dynamic>{},
      ),
    );
  }

  final String retrievalBackend;
  final String kbPath;
  final int topKRequested;
  final int candidateCount;
  final int returnedCount;
  final String scoringMethod;
  final bool liveRerankExecuted;
  final String? liveRerankMethod;
  final String rerankBoundary;
  final List<LiveRetrievedContext> returnedContexts;
  final OfficialRerankReference officialReference;

  bool get hasData =>
      retrievalBackend.isNotEmpty ||
      returnedContexts.isNotEmpty ||
      officialReference.hasData;
}

class LiveRetrievedContext {
  const LiveRetrievedContext({
    required this.rank,
    required this.evidence,
  });

  factory LiveRetrievedContext.fromJson(Map<String, dynamic> json) {
    return LiveRetrievedContext(
      rank: (json['rank'] as num?)?.toInt() ?? 0,
      evidence: EvidenceItem.fromJson(json),
    );
  }

  final int rank;
  final EvidenceItem evidence;
}

class OfficialRerankReference {
  const OfficialRerankReference({
    required this.available,
    required this.reason,
    required this.sourceLabel,
    required this.sourcePath,
    required this.caseId,
    required this.qid,
    required this.retrieverMethod,
    required this.rerank,
    required this.retrievalTopK,
    required this.contextCount,
    required this.contexts,
    required this.boundary,
  });

  factory OfficialRerankReference.fromJson(Map<String, dynamic> json) {
    return OfficialRerankReference(
      available: json['available'] as bool? ?? false,
      reason: json['reason'] as String? ?? '',
      sourceLabel: json['source_label'] as String? ?? '',
      sourcePath: json['source_path'] as String? ?? '',
      caseId: json['case_id'] as String? ?? '',
      qid: json['qid'] as String? ?? '',
      retrieverMethod: json['retriever_method'] as String? ?? '',
      rerank: json['rerank'] as bool?,
      retrievalTopK: (json['retrieval_top_k'] as num?)?.toInt(),
      contextCount: (json['context_count'] as num?)?.toInt() ?? 0,
      contexts: (json['contexts'] as List<dynamic>? ?? <dynamic>[])
          .map(
            (dynamic item) => OfficialRerankContext.fromJson(
              item as Map<String, dynamic>,
            ),
          )
          .toList(),
      boundary: json['boundary'] as String? ?? '',
    );
  }

  final bool available;
  final String reason;
  final String sourceLabel;
  final String sourcePath;
  final String caseId;
  final String qid;
  final String retrieverMethod;
  final bool? rerank;
  final int? retrievalTopK;
  final int contextCount;
  final List<OfficialRerankContext> contexts;
  final String boundary;

  bool get hasData => available || reason.isNotEmpty || sourcePath.isNotEmpty;
}

class OfficialRerankContext {
  const OfficialRerankContext({
    required this.rank,
    required this.docId,
    required this.score,
    required this.diagnosisType,
    required this.labelSource,
    required this.textPrefix,
    required this.textCharCount,
  });

  factory OfficialRerankContext.fromJson(Map<String, dynamic> json) {
    return OfficialRerankContext(
      rank: (json['rank'] as num?)?.toInt() ?? 0,
      docId: json['doc_id'] as String? ?? '',
      score: (json['score'] as num?)?.toDouble(),
      diagnosisType: json['diagnosis_type'] as String? ?? '',
      labelSource: json['label_source'] as String? ?? '',
      textPrefix: json['text_prefix_260'] as String? ?? '',
      textCharCount: (json['text_char_count'] as num?)?.toInt() ?? 0,
    );
  }

  final int rank;
  final String docId;
  final double? score;
  final String diagnosisType;
  final String labelSource;
  final String textPrefix;
  final int textCharCount;
}
