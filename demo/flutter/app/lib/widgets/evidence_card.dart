import 'package:flutter/material.dart';

import '../models/consult_response.dart';

class EvidenceCard extends StatelessWidget {
  const EvidenceCard({
    required this.item,
    super.key,
  });

  final EvidenceItem item;

  @override
  Widget build(BuildContext context) {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              item.title.isEmpty ? item.sourceCaseId : item.title,
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: <Widget>[
                if (item.diagnosisLabel.isNotEmpty)
                  Chip(
                    label: Text(item.diagnosisLabel),
                    visualDensity: VisualDensity.compact,
                  ),
                Chip(
                  label: Text('score ${item.score.toStringAsFixed(3)}'),
                  visualDensity: VisualDensity.compact,
                ),
                if (item.confirmatory)
                  const Chip(
                    label: Text('confirmatory'),
                    visualDensity: VisualDensity.compact,
                  ),
                Chip(
                  label: Text(item.sourceCaseId),
                  visualDensity: VisualDensity.compact,
                ),
              ],
            ),
            const SizedBox(height: 8),
            Text(item.excerpt),
          ],
        ),
      ),
    );
  }
}
