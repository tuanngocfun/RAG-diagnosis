import 'dart:async';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/consult_response.dart';
import '../services/backend_client.dart';
import '../widgets/consultation_result_view.dart';

class _DemoCase {
  const _DemoCase({
    required this.id,
    required this.label,
    required this.expectedState,
    required this.text,
  });

  final String id;
  final String label;
  final String expectedState;
  final String text;
}

const List<_DemoCase> _demoCases = <_DemoCase>[
  _DemoCase(
    id: 'supported',
    label: 'Supported cutaneous case',
    expectedState: 'rag_supported',
    text:
        'Ulcerated plaque on the forearm after sandfly exposure with a smear showing amastigotes.',
  ),
  _DemoCase(
    id: 'insufficient',
    label: 'Insufficient input',
    expectedState: 'abstained',
    text: 'Rash.',
  ),
  _DemoCase(
    id: 'provisional',
    label: 'Low-support provisional case',
    expectedState: 'provisional_parametric',
    text:
        'Chronic skin lesion with ulcerated border after travel to an endemic region.',
  ),
  _DemoCase(
    id: 'out_of_scope',
    label: 'Out-of-scope safety case',
    expectedState: 'abstained',
    text: 'Pigmented melanoma-like lesion with rapid growth and bleeding.',
  ),
];

class ConsultScreen extends StatefulWidget {
  const ConsultScreen({
    required this.client,
    super.key,
  });

  final BackendClient client;

  @override
  State<ConsultScreen> createState() => _ConsultScreenState();
}

class _ConsultScreenState extends State<ConsultScreen> {
  final TextEditingController _patientTextController = TextEditingController();
  final ImagePicker _imagePicker = ImagePicker();

  XFile? _selectedImage;
  ConsultResponse? _response;
  String? _error;
  String? _selectedDemoCaseId;
  bool _submitting = false;
  int _elapsedSeconds = 0;
  Timer? _elapsedTimer;

  @override
  void dispose() {
    _elapsedTimer?.cancel();
    _patientTextController.dispose();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final XFile? image = await _imagePicker.pickImage(source: source);
    if (!mounted) {
      return;
    }
    setState(() {
      _selectedImage = image;
    });
  }

  Future<void> _submit() async {
    _elapsedTimer?.cancel();
    setState(() {
      _submitting = true;
      _error = null;
      _elapsedSeconds = 0;
    });
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (Timer timer) {
      if (!mounted) {
        timer.cancel();
        return;
      }
      setState(() {
        _elapsedSeconds += 1;
      });
    });
    try {
      final ConsultResponse response = await widget.client.submitConsultation(
        patientText: _patientTextController.text,
        image: _selectedImage,
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _response = response;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _error = error.toString();
      });
    } finally {
      _elapsedTimer?.cancel();
      _elapsedTimer = null;
      if (mounted) {
        setState(() {
          _submitting = false;
        });
      }
    }
  }

  void _selectDemoCase(String? id) {
    final _DemoCase? demoCase = _demoCaseById(id);
    setState(() {
      _selectedDemoCaseId = id;
      _response = null;
      _error = null;
      _selectedImage = null;
      if (demoCase != null) {
        _patientTextController.text = demoCase.text;
      }
    });
  }

  _DemoCase? _demoCaseById(String? id) {
    for (final _DemoCase demoCase in _demoCases) {
      if (demoCase.id == id) {
        return demoCase;
      }
    }
    return null;
  }

  @override
  Widget build(BuildContext context) {
    final _DemoCase? selectedDemoCase = _demoCaseById(_selectedDemoCaseId);

    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Text(
              'Supervisor demo mode: local evidence retrieval, uncertainty gating, and an explicitly reported backend provider. Not real clinical inference.',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 16),
            DropdownButtonFormField<String>(
              initialValue: _selectedDemoCaseId,
              isExpanded: true,
              decoration: InputDecoration(
                labelText: 'Demo case',
                border: const OutlineInputBorder(),
                helperText: selectedDemoCase == null
                    ? 'Optional: choose a prepared supervisor demo case.'
                    : 'Expected state: ${selectedDemoCase.expectedState}',
              ),
              items: _demoCases
                  .map(
                    (_DemoCase item) => DropdownMenuItem<String>(
                      value: item.id,
                      child: Text(item.label),
                    ),
                  )
                  .toList(),
              onChanged: _submitting ? null : _selectDemoCase,
            ),
            const SizedBox(height: 16),
            TextField(
              controller: _patientTextController,
              maxLines: 6,
              decoration: const InputDecoration(
                labelText: 'Patient description',
                border: OutlineInputBorder(),
                hintText:
                    'Example: ulcerated plaque on the forearm after sandfly exposure...',
              ),
            ),
            const SizedBox(height: 16),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              children: <Widget>[
                FilledButton.tonalIcon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Camera'),
                ),
                FilledButton.tonalIcon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Gallery'),
                ),
                if (_selectedImage != null)
                  Chip(
                    label: Text(_selectedImage!.name),
                    onDeleted: () {
                      setState(() {
                        _selectedImage = null;
                      });
                    },
                  ),
              ],
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _submitting ? null : _submit,
              icon: const Icon(Icons.medical_services_outlined),
              label: Text(
                _submitting ? 'Running ${_elapsedSeconds}s' : 'Run consult',
              ),
            ),
            if (_submitting) ...<Widget>[
              const SizedBox(height: 12),
              const LinearProgressIndicator(),
              const SizedBox(height: 8),
              Text(
                'Waiting for backend result: ${_elapsedSeconds}s',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            if (_error != null) ...<Widget>[
              const SizedBox(height: 16),
              Text(
                _error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            if (_response != null) ...<Widget>[
              const SizedBox(height: 24),
              ConsultationResultView(response: _response!),
            ],
          ],
        ),
      ),
    );
  }
}
