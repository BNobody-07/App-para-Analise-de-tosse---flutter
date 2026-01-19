import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:file_picker/file_picker.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Análise de Tosse',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const CoughAnalysisPage(),
    );
  }
}

class CoughAnalysisPage extends StatefulWidget {
  const CoughAnalysisPage({super.key});

  @override
  _CoughAnalysisPageState createState() => _CoughAnalysisPageState();
}

class _CoughAnalysisPageState extends State<CoughAnalysisPage> {
  FlutterSoundRecorder? _recorder;
  FlutterSoundPlayer? _player;
  bool _isRecording = false;
  final bool _isPlaying = false;
  String _result = '';
  String _filePath = '';

  Interpreter? _interpreter;

  @override
  void initState() {
    super.initState();
    _recorder = FlutterSoundRecorder();
    _player = FlutterSoundPlayer();
    _initRecorder();
    _loadModel();
  }

  Future<void> _initRecorder() async {
    await Permission.microphone.request();
    await _recorder!.openRecorder();
  }

  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset(
      'assets/cough_model_quant.tflite',
    );
    _interpreter!.allocateTensors();

    // Debug: Print input and output tensor details
    var inputTensors = _interpreter!.getInputTensors();
    var outputTensors = _interpreter!.getOutputTensors();

    print('Input tensors: $inputTensors');
    print('Output tensors: $outputTensors');
  }

  Future<void> _startRecording() async {
    Directory tempDir = await getTemporaryDirectory();
    _filePath = '${tempDir.path}/cough.wav';
    await _recorder!.startRecorder(toFile: _filePath);
    setState(() {
      _isRecording = true;
      _result = '';
    });
  }

  Future<void> _stopRecording() async {
    await _recorder!.stopRecorder();
    setState(() {
      _isRecording = false;
    });
    _analyzeCough();
  }

  Future<void> _pickAndAnalyzeFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.audio,
      allowMultiple: false,
    );

    if (result != null && result.files.single.path != null) {
      _filePath = result.files.single.path!;
      _analyzeCough();
    }
  }

  Future<void> _analyzeCough() async {
    try {
      // Preprocess audio
      List<double> audioData = await _loadAudioData(_filePath);
      if (audioData.isEmpty) {
        setState(() => _result = 'Erro: Áudio vazio');
        return;
      }

      if (_interpreter == null) {
        throw Exception('Modelo ainda não carregado');
      }

      List<double> normalized = _normalizeAudio(audioData);
      List<double> denoised = _removeNoise(normalized);
      List<List<double>> spectrogram = _generateSpectrogram(denoised);

      // Run model - input shape [1, 128, 128, 3] for quantized int8 model
      // Flatten into single Int8List [1*128*128*3]
      List<int> flatInput = [];
      for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
          int val = ((spectrogram[i][j] * 255) - 128).toInt().clamp(-128, 127);
          flatInput.add(val); // R
          flatInput.add(val); // G
          flatInput.add(val); // B
        }
      }

      final Int8List input = Int8List.fromList(flatInput);
      final reshapedInput = input.reshape([1, 128, 128, 3]);

      final Int8List outputBuffer = Int8List(3);
      final reshapedOutput = outputBuffer.reshape([1, 3]);

      _interpreter!.run(reshapedInput, reshapedOutput);

      // Get result - scale back to probabilities
      List<double> probabilities = outputBuffer
          .map((e) => ((e + 128) / 255.0).clamp(0.0, 1.0))
          .toList();
      int maxIndex = probabilities.indexOf(probabilities.reduce(max));
      List<String> labels = ['Normal', 'Pneumonia', 'Bronquite'];
      setState(() {
        _result = labels[maxIndex];
      });
    } catch (e) {
      setState(() => _result = 'Erro na análise: $e');
    }
  }

  Future<List<double>> _loadAudioData(String path) async {
    // Simplified WAV parsing for 16-bit PCM
    File file = File(path);
    List<int> bytes = await file.readAsBytes();
    // Skip WAV header (44 bytes) and convert to double
    List<double> samples = [];
    for (int i = 44; i < bytes.length; i += 2) {
      int sample = (bytes[i + 1] << 8) | bytes[i];
      if (sample > 32767) sample -= 65536;
      samples.add(sample / 32768.0);
    }
    return samples;
  }

  List<double> _normalizeAudio(List<double> audio) {
    double maxVal = audio.reduce(max);
    return audio.map((e) => e / maxVal).toList();
  }

  List<double> _removeNoise(List<double> audio) {
    // Simple high-pass filter as noise removal
    List<double> filtered = [];
    for (int i = 1; i < audio.length; i++) {
      filtered.add(audio[i] - 0.9 * audio[i - 1]);
    }
    return filtered;
  }

  List<List<double>> _generateSpectrogram(List<double> audio) {
    // Simplified spectrogram generation (placeholder - real implementation would use FFT)
    // Assume 128x128 spectrogram
    List<List<double>> spectrogram = List.generate(
      128,
      (_) => List.filled(128, 0.0),
    );
    // Fill with audio data in a simple way
    int minLen = min(audio.length, 128 * 128);
    for (int i = 0; i < minLen; i++) {
      int row = i ~/ 128;
      int col = i % 128;
      spectrogram[row][col] = audio[i].abs();
    }
    return spectrogram;
  }

  @override
  void dispose() {
    _recorder?.closeRecorder();
    _player?.closePlayer();
    _interpreter?.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Análise de Tosse')),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            ElevatedButton(
              onPressed: _isRecording ? _stopRecording : _startRecording,
              child: Text(_isRecording ? 'Parar Gravação' : 'Gravar Tosse'),
            ),
            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _pickAndAnalyzeFile,
              child: const Text('Carregar Arquivo de Áudio'),
            ),
            const SizedBox(height: 20),
            if (_result.isNotEmpty)
              Text('Resultado: $_result', style: const TextStyle(fontSize: 24)),
            const SizedBox(height: 20),
            const Text(
              'Aviso: Este aplicativo não substitui diagnóstico médico.',
              style: TextStyle(color: Colors.red),
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }
}
