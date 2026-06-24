import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:medical_demo_app/models/gpu_chat_response.dart';
import 'package:medical_demo_app/widgets/retrieval_audit_panel.dart';

void main() {
  test('parses retrieval audit from GPU chat response', () {
    final GpuChatResponse response = GpuChatResponse.fromJson(
      <String, dynamic>{
        'request_id': 'manual-test-PMC7516301_01',
        'model_name': 'google/gemma-4-E4B-it',
        'provider_mode': 'real_gpu_gemma4',
        'assistant_markdown': 'answer',
        'evidence': <dynamic>[],
        'disclaimer': 'Decision support only.',
        'timing_ms': 1200,
        'safety_state': 'generated_support',
        'needed_next_inputs': <dynamic>[],
        'runtime_metadata': <String, dynamic>{},
        'retrieval_audit': _auditJson(),
      },
    );

    expect(response.retrievalAudit.retrievalBackend, 'local_demo_lexical');
    expect(response.retrievalAudit.returnedContexts, hasLength(1));
    expect(response.retrievalAudit.returnedContexts.first.rank, 1);
    expect(response.retrievalAudit.returnedContexts.first.evidence.chunkId,
        'mcl-002');
    expect(response.retrievalAudit.liveRerankExecuted, isFalse);
    expect(response.retrievalAudit.officialReference.available, isTrue);
    expect(response.retrievalAudit.officialReference.retrieverMethod, 'hybrid');
    expect(response.retrievalAudit.officialReference.rerank, isTrue);
    expect(response.retrievalAudit.officialReference.contexts.first.docId,
        'PMC7528117_01');
  });

  test('missing retrieval audit remains backward-compatible', () {
    final GpuChatResponse response = GpuChatResponse.fromJson(
      <String, dynamic>{
        'request_id': 'older-response',
        'model_name': 'google/gemma-4-E4B-it',
        'provider_mode': 'real_gpu_gemma4',
        'assistant_markdown': 'answer',
        'evidence': <dynamic>[],
        'disclaimer': 'Decision support only.',
        'timing_ms': 1200,
        'safety_state': 'generated_support',
        'needed_next_inputs': <dynamic>[],
        'runtime_metadata': <String, dynamic>{},
      },
    );

    expect(response.retrievalAudit.hasData, isFalse);
  });

  testWidgets('renders retriever and reranker audit panel',
      (WidgetTester tester) async {
    final RetrievalAudit audit = RetrievalAudit.fromJson(_auditJson());

    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: RetrievalAuditPanel(audit: audit),
          ),
        ),
      ),
    );

    expect(find.text('Retriever and reranker audit'), findsOneWidget);
    expect(find.text('Live retriever'), findsOneWidget);
    expect(find.text('Live reranker: not executed by this demo backend'),
        findsOneWidget);
    expect(find.text('Official rerank reference'), findsOneWidget);
    expect(find.text('PMC7516301_01'), findsOneWidget);
    expect(find.text('PMC7528117_01'), findsOneWidget);
  });
}

Map<String, dynamic> _auditJson() => <String, dynamic>{
      'retrieval_backend': 'local_demo_lexical',
      'kb_path': '/tmp/leishmaniasis_demo_pack.json',
      'top_k_requested': 4,
      'candidate_count': 6,
      'returned_count': 1,
      'scoring_method':
          'lexical token-overlap score with tag/title/confirmatory bonuses',
      'live_rerank_executed': false,
      'live_rerank_method': null,
      'rerank_boundary':
          'Live backend exposes lexical retrieval scores but no separate re-ranker contract.',
      'returned_contexts': <dynamic>[
        <String, dynamic>{
          'rank': 1,
          'chunk_id': 'mcl-002',
          'source_case_id': 'demo-mcl',
          'title': 'Mucosal disease evidence',
          'diagnosis_label': 'Mucocutaneous leishmaniasis',
          'excerpt': 'Palatal and nasal mucosal disease can support MCL.',
          'score': 0.3617,
          'confirmatory': true,
        },
      ],
      'official_rerank_reference': <String, dynamic>{
        'available': true,
        'source_label': 'official Gemma 4 experiment-pipeline trace',
        'source_path': '/tmp/trace_summary.json',
        'case_id': 'PMC7516301_01',
        'qid': 'PMC7516301_01::Q1_Q3_multimodal_diagnosis',
        'retriever_method': 'hybrid',
        'rerank': true,
        'retrieval_top_k': 20,
        'context_count': 10,
        'boundary':
            'Reference-only: this is the official rerank-enabled final context list.',
        'contexts': <dynamic>[
          <String, dynamic>{
            'rank': 1,
            'doc_id': 'PMC7528117_01',
            'score': 0.03301807008521583,
            'diagnosis_type': 'MCL',
            'label_source': 'train_verified',
            'text_prefix_260': 'A 52-year-old woman had a palatal lesion.',
            'text_char_count': 1157,
          },
        ],
      },
    };
