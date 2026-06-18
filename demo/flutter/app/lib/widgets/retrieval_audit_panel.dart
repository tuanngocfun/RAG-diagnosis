import 'package:flutter/material.dart';

import '../models/gpu_chat_response.dart';

class RetrievalAuditPanel extends StatelessWidget {
  const RetrievalAuditPanel({
    required this.audit,
    super.key,
  });

  final RetrievalAudit audit;

  @override
  Widget build(BuildContext context) {
    if (!audit.hasData) {
      return const SizedBox.shrink();
    }
    return Card(
      child: ExpansionTile(
        initiallyExpanded: true,
        title: Text(
          'Retriever and reranker audit',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        subtitle: const Text(
          'Live retriever trace plus official rerank reference when available',
        ),
        childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
        children: <Widget>[
          _LiveRetrieverSection(audit: audit),
          const SizedBox(height: 16),
          _RerankerBoundary(audit: audit),
          const SizedBox(height: 16),
          _OfficialReferenceSection(reference: audit.officialReference),
        ],
      ),
    );
  }
}

class _LiveRetrieverSection extends StatelessWidget {
  const _LiveRetrieverSection({required this.audit});

  final RetrievalAudit audit;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text('Live retriever', style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: <Widget>[
            _AuditChip(label: audit.retrievalBackend),
            _AuditChip(label: 'top-k ${audit.topKRequested}'),
            _AuditChip(label: 'candidates ${audit.candidateCount}'),
            _AuditChip(label: 'returned ${audit.returnedCount}'),
          ],
        ),
        if (audit.kbPath.isNotEmpty) ...<Widget>[
          const SizedBox(height: 8),
          SelectableText('KB: ${audit.kbPath}'),
        ],
        if (audit.scoringMethod.isNotEmpty) ...<Widget>[
          const SizedBox(height: 8),
          Text('Scoring: ${audit.scoringMethod}'),
        ],
        const SizedBox(height: 10),
        ...audit.returnedContexts.map(
          (LiveRetrievedContext context) => _LiveContextCard(context: context),
        ),
      ],
    );
  }
}

class _LiveContextCard extends StatelessWidget {
  const _LiveContextCard({required this.context});

  final LiveRetrievedContext context;

  @override
  Widget build(BuildContext context) {
    final evidence = this.context.evidence;
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: <Widget>[
              _AuditChip(label: 'rank ${this.context.rank}'),
              _AuditChip(label: evidence.chunkId),
              _AuditChip(label: evidence.sourceCaseId),
              _AuditChip(label: evidence.diagnosisLabel),
              _AuditChip(label: 'score ${evidence.score.toStringAsFixed(4)}'),
              _AuditChip(
                label:
                    evidence.confirmatory ? 'confirmatory' : 'not confirmatory',
              ),
            ],
          ),
          if (evidence.title.isNotEmpty) ...<Widget>[
            const SizedBox(height: 8),
            Text(
              evidence.title,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                  ),
            ),
          ],
          if (evidence.excerpt.isNotEmpty) ...<Widget>[
            const SizedBox(height: 8),
            SelectableText(evidence.excerpt),
          ],
        ],
      ),
    );
  }
}

class _RerankerBoundary extends StatelessWidget {
  const _RerankerBoundary({required this.audit});

  final RetrievalAudit audit;

  @override
  Widget build(BuildContext context) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    final String method = audit.liveRerankMethod?.isNotEmpty == true
        ? audit.liveRerankMethod!
        : 'not executed by this demo backend';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: scheme.surfaceContainerHighest,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text('Live reranker: $method'),
          if (audit.rerankBoundary.isNotEmpty) ...<Widget>[
            const SizedBox(height: 6),
            Text(audit.rerankBoundary),
          ],
        ],
      ),
    );
  }
}

class _OfficialReferenceSection extends StatelessWidget {
  const _OfficialReferenceSection({required this.reference});

  final OfficialRerankReference reference;

  @override
  Widget build(BuildContext context) {
    if (!reference.hasData) {
      return const SizedBox.shrink();
    }
    if (!reference.available) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'Official rerank reference',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: 8),
          Text(
            reference.reason.isEmpty
                ? 'No official rerank reference matched this request.'
                : reference.reason,
          ),
        ],
      );
    }
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          'Official rerank reference',
          style: Theme.of(context).textTheme.titleSmall,
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          runSpacing: 8,
          children: <Widget>[
            _AuditChip(label: reference.sourceLabel),
            _AuditChip(label: reference.caseId),
            _AuditChip(label: 'retriever ${reference.retrieverMethod}'),
            _AuditChip(label: 'rerank ${reference.rerank}'),
            if (reference.retrievalTopK != null)
              _AuditChip(label: 'top-k ${reference.retrievalTopK}'),
            _AuditChip(label: 'contexts ${reference.contextCount}'),
          ],
        ),
        const SizedBox(height: 8),
        SelectableText('qid: ${reference.qid}'),
        if (reference.boundary.isNotEmpty) ...<Widget>[
          const SizedBox(height: 8),
          Text(reference.boundary),
        ],
        const SizedBox(height: 10),
        ...reference.contexts.map(
          (OfficialRerankContext context) =>
              _OfficialContextCard(context: context),
        ),
      ],
    );
  }
}

class _OfficialContextCard extends StatelessWidget {
  const _OfficialContextCard({required this.context});

  final OfficialRerankContext context;

  @override
  Widget build(BuildContext context) {
    final double? score = this.context.score;
    return Container(
      width: double.infinity,
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: <Widget>[
              _AuditChip(label: 'rank ${this.context.rank}'),
              _AuditChip(label: this.context.docId),
              if (score != null)
                _AuditChip(label: 'score ${score.toStringAsPrecision(5)}'),
              _AuditChip(label: this.context.diagnosisType),
              _AuditChip(label: this.context.labelSource),
            ],
          ),
          if (this.context.textPrefix.isNotEmpty) ...<Widget>[
            const SizedBox(height: 8),
            SelectableText(this.context.textPrefix),
          ],
        ],
      ),
    );
  }
}

class _AuditChip extends StatelessWidget {
  const _AuditChip({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    if (label.isEmpty) {
      return const SizedBox.shrink();
    }
    return Chip(
      label: Text(label),
      visualDensity: VisualDensity.compact,
    );
  }
}
