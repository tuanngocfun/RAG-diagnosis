import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:image_picker/image_picker.dart';
import 'package:medical_demo_app/models/consult_response.dart';
import 'package:medical_demo_app/models/gpu_chat_response.dart';
import 'package:medical_demo_app/screens/gpu_assistant_screen.dart';
import 'package:medical_demo_app/services/backend_client.dart';

class _HealthErrorClient extends BackendClient {
  _HealthErrorClient() : super(baseUri: Uri.parse('http://127.0.0.1:8010'));

  @override
  Future<BackendHealth> getHealth() async {
    throw Exception('test health failure');
  }

  @override
  Future<ConsultResponse> submitConsultation({
    required String patientText,
    XFile? image,
  }) {
    throw UnimplementedError('Not used by this test.');
  }

  @override
  Future<GpuChatResponse> submitGpuChat({
    required List<GpuChatMessage> messages,
    XFile? image,
    String responseMode = 'live_gpu',
  }) {
    throw UnimplementedError('Not used by this test.');
  }
}

class _ReadyLiveClient extends BackendClient {
  _ReadyLiveClient() : super(baseUri: Uri.parse('http://127.0.0.1:8021'));

  String? submittedResponseMode;

  @override
  Future<BackendHealth> getHealth() async {
    return const BackendHealth(
      status: 'ok',
      modelName: 'google/gemma-4-E4B-it',
      providerMode: 'real_gpu_gemma4',
      chatAvailable: true,
      cudaAvailable: true,
      gpuName: 'NVIDIA TITAN RTX',
      bitsandbytesAvailable: true,
      gpuFreeMemoryReady: true,
      modelLoaded: true,
      gpuMinFreeMib: 12000,
      gpuMemoryTotalMib: 24019,
      gpuMemoryFreeMib: 12461,
    );
  }

  @override
  Future<ConsultResponse> submitConsultation({
    required String patientText,
    XFile? image,
  }) {
    throw UnimplementedError('Not used by this test.');
  }

  @override
  Future<GpuChatResponse> submitGpuChat({
    required List<GpuChatMessage> messages,
    XFile? image,
    String responseMode = 'live_gpu',
  }) async {
    submittedResponseMode = responseMode;
    return GpuChatResponse.fromJson(<String, dynamic>{
      'request_id': 'test-live',
      'model_name': 'google/gemma-4-E4B-it',
      'provider_mode': 'real_gpu_gemma4',
      'assistant_markdown':
          '**Rank 1 supportive consideration:** Mucocutaneous Leishmaniasis',
      'evidence': <dynamic>[],
      'disclaimer': 'Decision support only.',
      'timing_ms': 12,
      'safety_state': 'generated_support',
      'needed_next_inputs': <dynamic>['clinician review'],
      'runtime_metadata': <String, dynamic>{
        'response_source_mode': 'live_gpu',
        'fresh_generation_executed': true,
      },
      'retrieval_audit': <String, dynamic>{},
      'response_source_mode': 'live_gpu',
      'source_label': 'fresh local Gemma 4 GPU generation',
      'source_path': '',
      'fresh_generation_executed': true,
    });
  }
}

void main() {
  testWidgets('health failure shows backend URL and recovery guidance',
      (WidgetTester tester) async {
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: GpuAssistantScreen(client: _HealthErrorClient()),
        ),
      ),
    );

    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(find.text('Backend and GPU status'), findsOneWidget);
    expect(find.text('Backend URL: http://127.0.0.1:8010'), findsOneWidget);
    expect(find.textContaining('test health failure'), findsWidgets);
    expect(
      find.textContaining('Open http://127.0.0.1:8010/health'),
      findsOneWidget,
    );
    expect(find.text('Ask GPU assistant'), findsOneWidget);
  });

  testWidgets('live GPU mode is default and renders source banner',
      (WidgetTester tester) async {
    final _ReadyLiveClient client = _ReadyLiveClient();
    await tester.pumpWidget(
      MaterialApp(
        home: Scaffold(
          body: GpuAssistantScreen(client: client),
        ),
      ),
    );

    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(find.textContaining('Live Gemma 4 + RAG generation'), findsOneWidget);
    expect(find.textContaining('Last checked:'), findsOneWidget);

    await tester.enterText(
      find.byType(TextField),
      'PMC7516301_01 selected thesis case.',
    );
    await tester.pump();
    await tester.ensureVisible(find.text('Ask GPU assistant'));
    await tester.pump();
    await tester.tap(find.text('Ask GPU assistant'));
    await tester.pump();
    await tester.pump(const Duration(milliseconds: 100));

    expect(client.submittedResponseMode, 'live_gpu');
    expect(
      find.text('fresh local Gemma 4 GPU generation'),
      findsOneWidget,
    );
    expect(
      find.text(
        'Live local Gemma 4 + RAG generation used for current demo evidence; clinician confirmation is still required.',
      ),
      findsOneWidget,
    );
  });
}
