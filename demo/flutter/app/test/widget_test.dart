import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:medical_demo_app/models/consult_response.dart';
import 'package:medical_demo_app/widgets/consultation_result_view.dart';

void main() {
  testWidgets('renders blocked uncertainty state', (WidgetTester tester) async {
    const ConsultResponse response = ConsultResponse(
      requestId: 'blocked-1',
      modelName: 'google/gemma-4-E4B-it',
      decisionState: 'abstained',
      topDiagnoses: <DiagnosisRank>[],
      answerMarkdown: 'Insufficient information.',
      evidence: <EvidenceItem>[],
      disclaimer: 'Decision support only.',
      timingMs: 12,
      uncertaintyGate: UncertaintyGate(
        stage: 'evidence',
        triggerCodes: <String>['missing_required_inputs'],
        retrievalSupportStatus: 'empty_contexts',
        modelConfidence: 'low',
        imageUsable: false,
        escalationRequired: true,
        topScore: 0,
        evidenceConflictFlag: false,
        providerMode: 'deterministic_demo',
      ),
      neededNextInputs: <String>['clinical description'],
      safeToShowRankedDifferential: false,
      runtimeMetadata: <String, dynamic>{},
    );

    await tester.pumpWidget(
      const MaterialApp(
        home: Scaffold(
          body: SingleChildScrollView(
            child: ConsultationResultView(response: response),
          ),
        ),
      ),
    );

    expect(find.text('Blocked by uncertainty gate'), findsOneWidget);
    expect(find.text('Additional information needed'), findsOneWidget);
  });
}
