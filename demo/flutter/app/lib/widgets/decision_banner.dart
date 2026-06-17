import 'package:flutter/material.dart';

import '../models/consult_response.dart';

class DecisionBanner extends StatelessWidget {
  const DecisionBanner({
    required this.response,
    super.key,
  });

  final ConsultResponse response;

  @override
  Widget build(BuildContext context) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    late final Color background;
    late final Color foreground;
    late final String title;
    late final String subtitle;

    if (response.isAbstained) {
      background = scheme.errorContainer;
      foreground = scheme.onErrorContainer;
      title = 'Blocked by uncertainty gate';
      subtitle =
          'No ranked differential is being shown because the inputs are not safe enough.';
    } else if (response.isProvisional) {
      background = scheme.tertiaryContainer;
      foreground = scheme.onTertiaryContainer;
      title = 'Provisional fallback result';
      subtitle = 'This differential is model-only and not evidence-grounded.';
    } else {
      background = scheme.primaryContainer;
      foreground = scheme.onPrimaryContainer;
      title = 'Grounded result';
      subtitle =
          'This differential was produced with retrieved evidence support.';
    }

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: background,
        borderRadius: BorderRadius.circular(20),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            title,
            style: TextStyle(
              color: foreground,
              fontWeight: FontWeight.w700,
              fontSize: 18,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            subtitle,
            style: TextStyle(color: foreground),
          ),
        ],
      ),
    );
  }
}
