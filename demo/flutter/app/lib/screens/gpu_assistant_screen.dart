import 'dart:async';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';

import '../models/consult_response.dart';
import '../models/gpu_chat_response.dart';
import '../services/backend_client.dart';
import '../widgets/evidence_card.dart';
import '../widgets/retrieval_audit_panel.dart';

enum GpuResponseMode {
  liveGpu('live_gpu', 'Live Gemma 4 + RAG');

  const GpuResponseMode(this.apiValue, this.label);

  final String apiValue;
  final String label;
}

class GpuAssistantScreen extends StatefulWidget {
  const GpuAssistantScreen({
    required this.client,
    super.key,
  });

  final BackendClient client;

  @override
  State<GpuAssistantScreen> createState() => _GpuAssistantScreenState();
}

class _GpuAssistantScreenState extends State<GpuAssistantScreen> {
  final TextEditingController _messageController = TextEditingController();
  final ImagePicker _imagePicker = ImagePicker();
  final List<GpuChatMessage> _messages = <GpuChatMessage>[];

  BackendHealth? _health;
  GpuChatResponse? _response;
  XFile? _selectedImage;
  String? _error;
  final GpuResponseMode _responseMode = GpuResponseMode.liveGpu;
  DateTime? _lastHealthCheckedAt;
  bool _loadingHealth = true;
  bool _submitting = false;
  int _elapsedSeconds = 0;
  Timer? _elapsedTimer;
  Timer? _healthPollTimer;

  @override
  void initState() {
    super.initState();
    _messageController.addListener(_onTextChanged);
    _loadHealth();
  }

  @override
  void dispose() {
    _healthPollTimer?.cancel();
    _elapsedTimer?.cancel();
    _messageController.removeListener(_onTextChanged);
    _messageController.dispose();
    super.dispose();
  }

  void _onTextChanged() {
    if (mounted) {
      setState(() {});
    }
  }

  void _scheduleHealthRetry() {
    _healthPollTimer?.cancel();
    if (_health?.status == 'ok') {
      return; // Backend health is confirmed. Use manual refresh for later changes.
    }
    _healthPollTimer = Timer(const Duration(seconds: 5), () {
      if (mounted) {
        _loadHealth();
      }
    });
  }

  Future<void> _loadHealth() async {
    setState(() {
      _loadingHealth = true;
      _error = null;
    });
    try {
      final BackendHealth health = await widget.client.getHealth();
      if (!mounted) {
        return;
      }
      setState(() {
        _health = health;
        _error = null;
      });
    } catch (error) {
      if (!mounted) {
        return;
      }
      setState(() {
        _health = null;
        _error = error.toString();
      });
    } finally {
      if (mounted) {
        setState(() {
          _loadingHealth = false;
          _lastHealthCheckedAt = DateTime.now();
        });
        _scheduleHealthRetry();
      }
    }
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
    final String content = _messageController.text.trim();
    final bool liveGpuReady = _health?.isGpuChatReady == true;
    final bool ready = liveGpuReady;
    if (content.isEmpty || !ready) {
      return;
    }
    final GpuChatMessage userMessage = GpuChatMessage(
      role: 'user',
      content: content,
    );
    final List<GpuChatMessage> requestMessages = <GpuChatMessage>[
      ..._messages,
      userMessage,
    ];

    _elapsedTimer?.cancel();
    setState(() {
      _messages.add(userMessage);
      _messageController.clear();
      _submitting = true;
      _error = null;
      _response = null;
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
      final GpuChatResponse response = await widget.client.submitGpuChat(
        messages: requestMessages,
        image: _selectedImage,
        responseMode: _responseMode.apiValue,
      );
      if (!mounted) {
        return;
      }
      setState(() {
        _response = response;
        _selectedImage = null;
        _messages.add(
          GpuChatMessage(
            role: 'assistant',
            content: response.assistantMarkdown,
          ),
        );
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

  void _resetConversation() {
    setState(() {
      _messages.clear();
      _response = null;
      _error = null;
      _selectedImage = null;
      _messageController.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final bool liveGpuReady = _health?.isGpuChatReady == true;
    final bool ready = liveGpuReady;
    final bool hasText = _messageController.text.trim().isNotEmpty;
    final bool canSubmit = ready && hasText && !_submitting;

    return SafeArea(
      child: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            const _SafetyBanner(),
            const SizedBox(height: 16),
            _HealthPanel(
              backendUri: widget.client.baseUri,
              health: _health,
              loading: _loadingHealth,
              error: _error,
              lastCheckedAt: _lastHealthCheckedAt,
              onRefresh: _loadHealth,
            ),
            if (!ready) ...<Widget>[
              const SizedBox(height: 12),
              _ReadinessBlocker(
                backendUri: widget.client.baseUri,
                health: _health,
                error: _error,
              ),
            ],
            const SizedBox(height: 16),
            const _LiveModeNotice(),
            const SizedBox(height: 16),
            TextField(
              controller: _messageController,
              enabled: !_submitting,
              maxLines: 6,
              decoration: const InputDecoration(
                labelText: 'Clinical summary for GPU assistant',
                border: OutlineInputBorder(),
                hintText:
                    'Example: adult patient with chronic ulcerated plaque after endemic travel...',
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 12,
              runSpacing: 12,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: <Widget>[
                FilledButton.tonalIcon(
                  onPressed:
                      _submitting ? null : () => _pickImage(ImageSource.camera),
                  icon: const Icon(Icons.photo_camera_outlined),
                  label: const Text('Camera'),
                ),
                FilledButton.tonalIcon(
                  onPressed: _submitting
                      ? null
                      : () => _pickImage(ImageSource.gallery),
                  icon: const Icon(Icons.photo_library_outlined),
                  label: const Text('Gallery'),
                ),
                if (_selectedImage != null)
                  Chip(
                    label: Text(_selectedImage!.name),
                    onDeleted: _submitting
                        ? null
                        : () {
                            setState(() {
                              _selectedImage = null;
                            });
                          },
                  ),
                OutlinedButton.icon(
                  onPressed: _submitting ? null : _resetConversation,
                  icon: const Icon(Icons.restart_alt_outlined),
                  label: const Text('Reset'),
                ),
              ],
            ),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: canSubmit ? _submit : null,
              icon: const Icon(Icons.smart_toy_outlined),
              label: Text(
                _submitting
                    ? 'Generating ${_elapsedSeconds}s'
                    : 'Ask GPU assistant',
              ),
            ),
            if (_submitting) ...<Widget>[
              const SizedBox(height: 12),
              const LinearProgressIndicator(),
              const SizedBox(height: 8),
              Text(
                'Waiting for local Gemma 4 result: ${_elapsedSeconds}s',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
            if (_error != null && !_loadingHealth) ...<Widget>[
              const SizedBox(height: 16),
              Text(
                _error!,
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ],
            if (_messages.isNotEmpty) ...<Widget>[
              const SizedBox(height: 24),
              _MessageHistory(messages: _messages),
            ],
            if (_response != null) ...<Widget>[
              const SizedBox(height: 24),
              _AssistantAnswerCard(response: _response!),
              if (_response!.evidence.isNotEmpty) ...<Widget>[
                const SizedBox(height: 16),
                Text(
                  'Retrieved evidence sent to model',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
                const SizedBox(height: 8),
                ..._response!.evidence
                    .map((EvidenceItem item) => EvidenceCard(item: item)),
              ],
              if (_response!.retrievalAudit.hasData) ...<Widget>[
                const SizedBox(height: 16),
                RetrievalAuditPanel(audit: _response!.retrievalAudit),
              ],
              if (_response!.runtimeMetadata.isNotEmpty) ...<Widget>[
                const SizedBox(height: 16),
                _GpuRuntimeAuditPanel(response: _response!),
              ],
              const SizedBox(height: 16),
              Text(
                _response!.disclaimer,
                style: Theme.of(context).textTheme.bodySmall,
              ),
            ],
          ],
        ),
      ),
    );
  }
}

class _SafetyBanner extends StatelessWidget {
  const _SafetyBanner();

  @override
  Widget build(BuildContext context) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: scheme.errorContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(
        'Research decision support only. The assistant is not clinically validated, does not provide ground truth, and requires clinician confirmation before any action.',
        style: TextStyle(
          color: scheme.onErrorContainer,
          fontWeight: FontWeight.w600,
        ),
      ),
    );
  }
}

class _HealthPanel extends StatelessWidget {
  const _HealthPanel({
    required this.backendUri,
    required this.health,
    required this.loading,
    required this.error,
    required this.lastCheckedAt,
    required this.onRefresh,
  });

  final Uri backendUri;
  final BackendHealth? health;
  final bool loading;
  final String? error;
  final DateTime? lastCheckedAt;
  final VoidCallback onRefresh;

  @override
  Widget build(BuildContext context) {
    final BackendHealth? value = health;
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Row(
              children: <Widget>[
                Expanded(
                  child: Text(
                    'Backend and GPU status',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                ),
                IconButton(
                  tooltip: 'Refresh status',
                  onPressed: loading ? null : onRefresh,
                  icon: const Icon(Icons.refresh),
                ),
              ],
            ),
            const SizedBox(height: 8),
            SelectableText(
              'Backend URL: $backendUri',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 4),
            Text(
              lastCheckedAt == null
                  ? 'Status check has not completed yet.'
                  : 'Last checked: ${_formatTime(lastCheckedAt!)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            if (loading) ...<Widget>[
              const SizedBox(height: 12),
              const LinearProgressIndicator(),
            ],
            if (error != null) ...<Widget>[
              const SizedBox(height: 12),
              Container(
                width: double.infinity,
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Theme.of(context).colorScheme.errorContainer,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: SelectableText(
                  error!,
                  style: TextStyle(
                    color: Theme.of(context).colorScheme.onErrorContainer,
                    fontWeight: FontWeight.w600,
                  ),
                ),
              ),
            ],
            if (value != null) ...<Widget>[
              const SizedBox(height: 12),
              Wrap(
                spacing: 8,
                runSpacing: 8,
                children: <Widget>[
                  _StatusChip(label: 'provider ${value.providerMode}'),
                  _StatusChip(label: 'model ${value.modelName}'),
                  _StatusChip(
                    label:
                        value.chatAvailable ? 'chat available' : 'chat blocked',
                  ),
                  _StatusChip(
                    label: value.cudaAvailable ? 'CUDA ready' : 'CUDA hidden',
                  ),
                  _StatusChip(
                    label: value.modelLoaded
                        ? 'model loaded'
                        : 'min VRAM ${value.gpuMinFreeMib} MiB',
                  ),
                  _StatusChip(
                    label: value.gpuFreeMemoryReady ? 'VRAM ready' : 'VRAM low',
                  ),
                  if (value.gpuName.isNotEmpty)
                    _StatusChip(label: value.gpuName),
                  if (value.gpuMemoryFreeMib != null &&
                      value.gpuMemoryTotalMib != null)
                    _StatusChip(
                      label:
                          'VRAM ${value.gpuMemoryFreeMib}/${value.gpuMemoryTotalMib} MiB free',
                    ),
                  _StatusChip(
                    label: value.bitsandbytesAvailable
                        ? 'bitsandbytes ready'
                        : 'bitsandbytes missing',
                  ),
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }

  static String _formatTime(DateTime value) {
    String two(int number) => number.toString().padLeft(2, '0');
    return '${two(value.hour)}:${two(value.minute)}:${two(value.second)}';
  }
}

class _LiveModeNotice extends StatelessWidget {
  const _LiveModeNotice();

  @override
  Widget build(BuildContext context) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        border: Border.all(color: scheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            'Response source',
            style: Theme.of(context).textTheme.titleSmall,
          ),
          const SizedBox(height: 8),
          Text(
            'Live Gemma 4 + RAG generation. The V12d slides are aligned to the locked live-demo case recapture; reruns remain research evidence and require clinician confirmation.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
        ],
      ),
    );
  }
}

class _ReadinessBlocker extends StatelessWidget {
  const _ReadinessBlocker({
    required this.backendUri,
    required this.health,
    required this.error,
  });

  final Uri backendUri;
  final BackendHealth? health;
  final String? error;

  @override
  Widget build(BuildContext context) {
    final String reason = health == null
        ? error == null
            ? 'Backend health has not been confirmed yet. Waiting for $backendUri/health.'
            : 'Backend health request failed. Open $backendUri/health in this browser, confirm the 8021 proxy can serve /health, then press Refresh status.'
        : 'Live Gemma 4 + RAG needs real GPU mode, CUDA, bitsandbytes, and enough free VRAM before sending a chat request.';
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).colorScheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Text(reason),
    );
  }
}

class _MessageHistory extends StatelessWidget {
  const _MessageHistory({required this.messages});

  final List<GpuChatMessage> messages;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: <Widget>[
        Text(
          'Conversation',
          style: Theme.of(context).textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        ...messages.map(
          (GpuChatMessage message) => _ChatBubble(message: message),
        ),
      ],
    );
  }
}

class _ChatBubble extends StatelessWidget {
  const _ChatBubble({required this.message});

  final GpuChatMessage message;

  @override
  Widget build(BuildContext context) {
    final bool fromUser = message.role == 'user';
    final ColorScheme scheme = Theme.of(context).colorScheme;
    return Align(
      alignment: fromUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        constraints: const BoxConstraints(maxWidth: 760),
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: fromUser
              ? scheme.primaryContainer
              : scheme.surfaceContainerHighest,
          borderRadius: BorderRadius.circular(8),
        ),
        child: SelectableText(
          message.content,
          style: TextStyle(
            color: fromUser ? scheme.onPrimaryContainer : scheme.onSurface,
          ),
        ),
      ),
    );
  }
}

class _AssistantAnswerCard extends StatelessWidget {
  const _AssistantAnswerCard({required this.response});

  final GpuChatResponse response;

  @override
  Widget build(BuildContext context) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    final Color accent = response.isGenerated ? scheme.primary : scheme.error;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        border: Border.all(color: scheme.outlineVariant),
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Row(
            children: <Widget>[
              Expanded(
                child: Text(
                  'Assistant answer',
                  style: Theme.of(context).textTheme.titleMedium,
                ),
              ),
              Chip(
                label: Text(response.safetyState),
                side: BorderSide(color: accent),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _ResponseSourceBanner(response: response),
          const SizedBox(height: 12),
          SelectableText(response.assistantMarkdown),
          if (response.neededNextInputs.isNotEmpty) ...<Widget>[
            const SizedBox(height: 16),
            Text(
              'Needed next inputs',
              style: Theme.of(context).textTheme.titleSmall,
            ),
            const SizedBox(height: 8),
            ...response.neededNextInputs.map(
              (String item) => Padding(
                padding: const EdgeInsets.only(bottom: 6),
                child: Text('- $item'),
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _ResponseSourceBanner extends StatelessWidget {
  const _ResponseSourceBanner({required this.response});

  final GpuChatResponse response;

  @override
  Widget build(BuildContext context) {
    final ColorScheme scheme = Theme.of(context).colorScheme;
    final bool replay = response.isOfficialReplay;
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: replay ? scheme.primaryContainer : scheme.secondaryContainer,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Text(
            response.sourceLabel.isEmpty
                ? response.responseSourceMode
                : response.sourceLabel,
            style: TextStyle(
              color: replay
                  ? scheme.onPrimaryContainer
                  : scheme.onSecondaryContainer,
              fontWeight: FontWeight.w700,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            replay
                ? 'Official experiment-pipeline replay; fresh model generation not executed.'
                : 'Live local Gemma 4 + RAG generation used for current demo evidence; clinician confirmation is still required.',
            style: TextStyle(
              color: replay
                  ? scheme.onPrimaryContainer
                  : scheme.onSecondaryContainer,
            ),
          ),
          if (response.sourcePath.isNotEmpty) ...<Widget>[
            const SizedBox(height: 4),
            SelectableText(
              response.sourcePath,
              style: TextStyle(
                color: replay
                    ? scheme.onPrimaryContainer
                    : scheme.onSecondaryContainer,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _GpuRuntimeAuditPanel extends StatelessWidget {
  const _GpuRuntimeAuditPanel({required this.response});

  final GpuChatResponse response;

  @override
  Widget build(BuildContext context) {
    final List<Widget> chips = <Widget>[
      _StatusChip(label: 'source ${response.responseSourceMode}'),
      _StatusChip(
        label: response.freshGenerationExecuted
            ? 'fresh generation yes'
            : 'fresh generation no',
      ),
      _StatusChip(label: 'provider ${response.providerMode}'),
      _StatusChip(label: 'model ${response.modelName}'),
      _StatusChip(
          label: 'total ${(response.timingMs / 1000).toStringAsFixed(1)}s'),
    ];
    final Object? temperature = response.runtimeMetadata['temperature'];
    if (temperature != null) {
      chips.add(_StatusChip(label: 'temperature $temperature'));
    }
    final Object? doSample = response.runtimeMetadata['do_sample'];
    if (doSample != null) {
      chips.add(_StatusChip(label: 'do_sample $doSample'));
    }
    final Object? seed = response.runtimeMetadata['random_seed'];
    if (seed != null) {
      chips.add(_StatusChip(label: 'seed $seed'));
    }
    final double? generationLatency = response.generationLatencySeconds;
    if (generationLatency != null) {
      chips.add(
        _StatusChip(
            label: 'generation ${generationLatency.toStringAsFixed(1)}s'),
      );
    }
    final double? modelLoad = response.modelLoadSeconds;
    if (modelLoad != null) {
      chips.add(_StatusChip(label: 'load ${modelLoad.toStringAsFixed(1)}s'));
    }
    if (response.gpuName.isNotEmpty) {
      chips.add(_StatusChip(label: response.gpuName));
    }
    if (response.quantizationMode.isNotEmpty) {
      chips.add(_StatusChip(label: 'quant ${response.quantizationMode}'));
    }
    final int? imageTensorCount = response.queryImageTensorCount;
    if (imageTensorCount != null) {
      chips.add(_StatusChip(label: 'image tensors $imageTensorCount'));
    }
    final double? peakMemory = response.gpuPeakMemoryAllocatedMib;
    if (peakMemory != null) {
      chips
          .add(_StatusChip(label: 'peak ${peakMemory.toStringAsFixed(0)} MiB'));
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

class _StatusChip extends StatelessWidget {
  const _StatusChip({required this.label});

  final String label;

  @override
  Widget build(BuildContext context) {
    return Chip(
      label: Text(label),
      visualDensity: VisualDensity.compact,
    );
  }
}
