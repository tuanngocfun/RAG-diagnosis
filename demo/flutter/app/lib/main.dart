import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import 'screens/main_shell.dart';
import 'services/backend_client.dart';

String _getBackendBaseUrl() {
  const String envUrl = String.fromEnvironment('MEDICAL_DEMO_BACKEND_URL');
  if (envUrl.isNotEmpty) {
    return envUrl;
  }
  if (kIsWeb) {
    // The web server proxies /health and /v1/* to the real backend,
    // so we use the same origin the page was loaded from.
    final Uri base = Uri.base;
    if (base.host.isNotEmpty) {
      return Uri(
        scheme: base.scheme,
        host: base.host,
        port: base.port,
      ).toString();
    }
  }
  return 'http://127.0.0.1:8010';
}

void main() {
  runApp(const MedicalDemoApp());
}

class MedicalDemoApp extends StatelessWidget {
  const MedicalDemoApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Medical Demo',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFF005C53),
          brightness: Brightness.light,
        ),
        useMaterial3: true,
      ),
      home: MainShell(
        client: BackendClient(
          baseUri: Uri.parse(_getBackendBaseUrl()),
        ),
      ),
    );
  }
}
