import 'package:flutter/material.dart';

import '../models/consult_response.dart';
import 'decision_banner.dart';
import 'evidence_card.dart';

class ConsultationResultView extends StatelessWidget {
  const ConsultationResultView({
    required this.response,
    super.key,
  });

  final ConsultResponse response;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        DecisionBanner(response: response),
        const SizedBox(height: 16),
        Text(
          'Decision state: ${response.decisionState}',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        Text(
          'Gate stage: ${response.uncertaintyGate.stage}',
          style: Theme.of(context).textTheme.bodyMedium,
        ),
        const SizedBox(height: 12),
        _GateAuditPanel(response: response),
        if (response.runtimeMetadata.isNotEmpty) ...<Widget>[
          const SizedBox(height: 12),
          _RuntimeAuditPanel(response: response),
        ],
        const SizedBox(height: 16),
        if (response.safeToShowRankedDifferential &&
            !response.isRealGpuProvider)
          ...response.topDiagnoses.map(
            (DiagnosisRank item) => ListTile(
              contentPadding: EdgeInsets.zero,
              title: Text('${item.rank}. ${item.label}'),
              subtitle: Text(item.rationale),
              trailing: Chip(label: Text(item.confidenceBand)),
            ),
          )
        else
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: <Widget>[
                  Text(
                    'Additional information needed',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 12),
                  ...response.neededNextInputs.map(
                    (String item) => Padding(
                      padding: const EdgeInsets.only(bottom: 8),
                      child: Text('- $item'),
                    ),
                  ),
                ],
              ),
            ),
          ),
        const SizedBox(height: 16),
        Text(
          response.isRealGpuProvider ? 'Generated model answer' : 'Answer',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        SelectableText(response.answerMarkdown),
        if (response.evidence.isNotEmpty) ...<Widget>[
          const SizedBox(height: 16),
          Text(
            response.isRealGpuProvider
                ? 'Retrieved evidence sent to model'
                : 'Evidence',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: 8),
          ...response.evidence
              .map((EvidenceItem item) => EvidenceCard(item: item)),
        ],
        const SizedBox(height: 16),
        Text(
          response.disclaimer,
          style: Theme.of(context).textTheme.bodySmall,
        ),
      ],
    );
  }
}

class _RuntimeAuditPanel extends StatelessWidget {
  const _RuntimeAuditPanel({required this.response});

  final ConsultResponse response;

  @override
  Widget build(BuildContext context) {
    final List<Widget> chips = <Widget>[
      _GateChip(label: 'provider ${response.providerMode}'),
      _GateChip(label: 'model ${response.modelName}'),
      _GateChip(
          label: 'total ${(response.timingMs / 1000).toStringAsFixed(1)}s'),
    ];
    final double? generationLatency = response.generationLatencySeconds;
    if (generationLatency != null) {
      chips.add(_GateChip(
          label: 'generation ${generationLatency.toStringAsFixed(1)}s'));
    }
    final double? modelLoad = response.modelLoadSeconds;
    if (modelLoad != null) {
      chips.add(_GateChip(label: 'load ${modelLoad.toStringAsFixed(1)}s'));
    }
    if (response.gpuName.isNotEmpty) {
      chips.add(_GateChip(label: response.gpuName));
    }
    if (response.quantizationMode.isNotEmpty) {
      chips.add(_GateChip(label: 'quant ${response.quantizationMode}'));
    }
    final int? imageTensorCount = response.queryImageTensorCount;
    if (imageTensorCount != null) {
      chips.add(_GateChip(label: 'image tensors $imageTensorCount'));
    }
    final double? peakMemory = response.gpuPeakMemoryAllocatedMib;
    if (peakMemory != null) {
      chips.add(_GateChip(label: 'peak ${peakMemory.toStringAsFixed(0)} MiB'));
    }

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Runtime audit',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: chips,
            ),
          ],
        ),
      ),
    );
  }
}

class _GateAuditPanel extends StatelessWidget {
  const _GateAuditPanel({required this.response});

  final ConsultResponse response;

  @override
  Widget build(BuildContext context) {
    final UncertaintyGate gate = response.uncertaintyGate;
    final List<String> triggers =
        gate.triggerCodes.isEmpty ? <String>['none'] : gate.triggerCodes;

    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Gate audit',
              style: Theme.of(context).textTheme.titleMedium,
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                _GateChip(label: 'support ${gate.retrievalSupportStatus}'),
                _GateChip(
                    label: 'top score ${gate.topScore.toStringAsFixed(3)}'),
                _GateChip(label: 'confidence ${gate.modelConfidence}'),
                _GateChip(
                  label: gate.evidenceConflictFlag
                      ? 'evidence conflict'
                      : 'no evidence conflict',
                ),
                _GateChip(
                    label:
                        gate.imageUsable ? 'image usable' : 'no usable image'),
                _GateChip(label: '${response.timingMs} ms'),
              ],
            ),
            const SizedBox(height: 12),
            Text(
              'Triggers: ${triggers.join(', ')}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }
}

class _GateChip extends StatelessWidget {
  const _GateChip({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label),
      visualDensity: VisualDensity.compact,
    );
  }
}
