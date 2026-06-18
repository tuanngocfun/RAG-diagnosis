import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import '../models/consult_response.dart';
import '../models/gpu_chat_response.dart';

class BackendClient {
  BackendClient({required this.baseUri});

  final Uri baseUri;
  static const Duration _healthTimeout = Duration(seconds: 15);

  Future<BackendHealth> getHealth() async {
    final Uri endpoint = baseUri.replace(path: '/health');
    final http.Response response = await http.get(endpoint).timeout(
      _healthTimeout,
      onTimeout: () {
        throw TimeoutException(
          'Timed out waiting for backend health at $endpoint. '
          'Open that URL in the same browser and confirm port 8010 is forwarded.',
          _healthTimeout,
        );
      },
    );
    if (response.statusCode != 200) {
      throw Exception(
        'Backend health failed: ${response.statusCode} ${response.body}',
      );
    }
    return BackendHealth.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<ConsultResponse> submitConsultation({
    required String patientText,
    XFile? image,
  }) async {
    final Uri endpoint = baseUri.replace(path: '/v1/consult');
    final Map<String, dynamic> payload = <String, dynamic>{
      'patient_text': patientText,
      'device_platform': kIsWeb ? 'web' : defaultTargetPlatform.name,
    };
    if (image != null) {
      final List<int> bytes = await image.readAsBytes();
      payload['image_base64'] = base64Encode(bytes);
      payload['image_filename'] = image.name;
    }

    final http.Response response = await http.post(
      endpoint,
      headers: const <String, String>{
        'Content-Type': 'application/json',
      },
      body: jsonEncode(payload),
    );

    if (response.statusCode != 200) {
      throw Exception(
        'Backend request failed: ${response.statusCode} ${response.body}',
      );
    }
    return ConsultResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }

  Future<GpuChatResponse> submitGpuChat({
    required List<GpuChatMessage> messages,
    XFile? image,
    String responseMode = 'live_gpu',
  }) async {
    final Uri endpoint = baseUri.replace(path: '/v1/chat');
    final Map<String, dynamic> payload = <String, dynamic>{
      'messages': messages.map((GpuChatMessage item) => item.toJson()).toList(),
      'device_platform': kIsWeb ? 'web' : defaultTargetPlatform.name,
      'response_mode': responseMode,
    };
    if (image != null) {
      final List<int> bytes = await image.readAsBytes();
      payload['image_base64'] = base64Encode(bytes);
      payload['image_filename'] = image.name;
    }

    final http.Response response = await http.post(
      endpoint,
      headers: const <String, String>{
        'Content-Type': 'application/json',
      },
      body: jsonEncode(payload),
    );

    if (response.statusCode != 200) {
      throw Exception(
        'Backend chat failed: ${response.statusCode} ${response.body}',
      );
    }
    return GpuChatResponse.fromJson(
      jsonDecode(response.body) as Map<String, dynamic>,
    );
  }
}
