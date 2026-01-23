import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:fftea/fftea.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sound/flutter_sound.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:file_picker/file_picker.dart';

void main() {
  runApp(const MyApp());
}

// APLICATIVO MÓVEL (Root da estrutura)
class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Análise de Tosse',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const CoughAnalysisPage(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class CoughAnalysisPage extends StatefulWidget {
  const CoughAnalysisPage({super.key});

  @override
  // ignore: library_private_types_in_public_api
  _CoughAnalysisPageState createState() => _CoughAnalysisPageState();
}

// ANALIZADOR DA TOSSE
class _CoughAnalysisPageState extends State<CoughAnalysisPage> {
  FlutterSoundRecorder? _recorder;
  FlutterSoundPlayer? _player;
  bool _isRecording = false;
  final bool _isPlaying = false;
  String _result = '';
  String _filePath = '';

  Map<String, double> _probabilities = {};
  String _finalLabel = '';
  double _confidence = 0.0;

  // Intepretador do TensorFlowLite
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

  // GRAVADOR DE ÁUDIO
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

  // UPLOAD/ANALIZAR O AUDIO
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

  // MODELO TFLITE NO DISPOSITIVO
  Future<void> _loadModel() async {
    _interpreter = await Interpreter.fromAsset(
      'assets/cough_model_float32.tflite',
    );
    _interpreter!.allocateTensors();
  }

  Future<void> _analyzeCough() async {
    try {
      // Processa o audio
      List<double> audioData = await _loadAudioData(_filePath);
      if (audioData.isEmpty) {
        setState(() => _result = 'Erro: Áudio vazio');
        return;
      }

      if (_interpreter == null) {
        throw Exception('Modelo ainda não carregado');
      }

      // Converter para Float32
      Float32List signal = Float32List.fromList(audioData);

      // PRÉ-PROCESSAMENTO
      signal = _normalizeAndTrim(signal);

      // INICIALIZA A FUNÇÃO DO ESPECTROGRAMA
      final List<List<double>> spectrogram = _generateModelInput(signal);

      // Preparação do Buffer para o Modelo
      final Float32List inputBuffer = Float32List(1 * 128 * 128 * 3);
      int index = 0;
      for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 128; j++) {
          double val = spectrogram[i][j];

          inputBuffer[index++] = val; // Canal R
          inputBuffer[index++] = val; // Canal G
          inputBuffer[index++] = val; // Canal B
        }
      }

      final reshapedInput = inputBuffer.reshape([1, 128, 128, 3]);
      final Float32List outputBuffer = Float32List(3);
      final reshapedOutput = outputBuffer.reshape([1, 3]);

      // PREDIÇÃO (Execução da inferência pelo TFLite)
      _interpreter?.run(reshapedInput, reshapedOutput);

      // Obter resultado - escalar de volta para probabilidades
      final labels = ['Normal', 'Bronquite', 'Pneumonia'];

      // Cálculo de Softmax e Probabilidades
      List<double> logits = List.generate(
        3,
        (i) => reshapedOutput[0][i].toDouble(),
      );

      double maxLogit = logits.reduce(max);
      List<double> exps = logits.map((e) => exp(e - maxLogit)).toList();
      double sumExps = exps.reduce((a, b) => a + b);
      List<double> probsList = exps.map((e) => e / sumExps).toList();

      final Map<String, double> probs = {
        for (int i = 0; i < labels.length; i++) labels[i]: probsList[i],
      };

      final sorted = probs.entries.toList()
        ..sort((a, b) => b.value.compareTo(a.value));

      final best = sorted.first;

      setState(() {
        _probabilities = probs;
        _finalLabel = best.key;
        _confidence = best.value;
        _result = ''; // Limpa todos os erros antigos
      });
    } catch (e) {
      setState(() => _result = 'Erro na análise: $e');
    }
  }

  Future<List<double>> _loadAudioData(String path) async {
    // Análise simplificada de WAV para PCM de 16 bits
    File file = File(path);
    List<int> bytes = await file.readAsBytes();

    // Pular cabeçalho WAV (44 bytes) e converter para double
    List<double> samples = [];
    for (int i = 44; i < bytes.length; i += 2) {
      int sample = (bytes[i + 1] << 8) | bytes[i];
      if (sample > 32767) sample -= 65536;
      samples.add(sample / 32768.0);
    }
    return samples;
  }

  // NORMALIZAÇÃO
  Float32List _normalizeAndTrim(Float32List input) {
    final maxVal = input.map((e) => e.abs()).reduce(max);
    final normalized = input.map((e) => e / maxVal).toList();

    // REMOÇÃO DE RUÍDO (Filtro de threshold para silêncio/ruído de fundo)
    return Float32List.fromList(
      normalized.where((e) => e.abs() > 0.02).toList(),
    );
  }

  // GERAÇÃO DO ESPECTROGRAMA (FFT)
  List<List<double>> _generateModelInput(Float32List audio) {
    // Definimos as dimensões da "imagem" que o modelo espera (128x128 pixels/pontos)
    const int targetHeight = 128;
    const int targetWidth = 128;

    // O tamanho da FFT (Fast Fourier Transform) define a resolução da análise
    // Usamos o dobro da altura para obter a resolução de frequência correta
    const int fftSize = targetHeight * 2;

    // Inicializamos o motor matemático da FFT
    final fft = FFT(fftSize);

    // Janela de Hanning: Suaviza as bordas de cada pedaço de áudio para evitar ruído matemático (spectral leakage)
    final window = Window.hanning(fftSize);

    List<List<double>> spectrogram = [];

    // Calculamos o 'pulo' (hop) necessário para cobrir o áudio e gerar exatamente 128 colunas
    int hopSize = (audio.length / targetWidth).floor();
    if (hopSize < 1) hopSize = 1;

    // Loop principal: percorre o áudio criando as colunas da nossa imagem
    for (int i = 0; i < targetWidth; i++) {
      int start = i * hopSize;
      int end = start + fftSize;

      // Extração de um frame (pedaço) do áudio
      List<double> chunk;
      if (end < audio.length) {
        chunk = audio.sublist(start, end).toList();
      } else {
        // Zero Padding: se o áudio for curto, preenchemos com silêncio para manter o tamanho fixo
        chunk = audio.sublist(start, audio.length).toList();
        chunk.addAll(List.filled(fftSize - chunk.length, 0.0));
      }

      // JANELAMENTO: Aplicamos a função Hanning multiplicando o áudio pela curva da janela
      final windowedChunk = List<double>.generate(chunk.length, (idx) {
        return chunk[idx] * window[idx];
      });

      // FFT REAL: Converte o áudio (ondas no tempo) para o domínio da frequência
      final freqDomain = fft.realFft(windowedChunk);

      List<double> magnitudes = [];
      for (int j = 0; j < targetHeight; j++) {
        // A FFT retorna números complexos (Parte Real e Imaginária)
        final complex = freqDomain[j];
        final double real = complex.x;
        final double imag = complex.y;

        // CÁLCULO DA MAGNITUDE: Representa a "força" daquela frequência específica (Volume)
        double mag = sqrt(real * real + imag * imag);
        double logMag = log(mag + 1e-6);
        magnitudes.add(logMag);
      }

      // Adicionamos a coluna de frequências à nossa matriz final
      spectrogram.add(magnitudes);
    }

    // NORMALIZAÇÃO MIN-MAX: Ajustamos todos os valores para ficarem entre 0.0 e 1.0
    double maxVal = -double.infinity;
    double minVal = double.infinity;

    // Encontra os limites (mínimo e máximo) de toda a matriz
    for (var row in spectrogram) {
      for (var val in row) {
        if (val > maxVal) maxVal = val;
        if (val < minVal) minVal = val;
      }
    }

    List<List<double>> normalizedSpectrogram = [];
    for (var row in spectrogram) {
      List<double> normRow = [];
      for (var val in row) {
        double norm = (val - minVal) / (maxVal - minVal);
        // Proteção contra divisões por zero se o áudio estiver totalmente em silêncio
        normRow.add(norm.isNaN ? 0.0 : norm);
      }
      normalizedSpectrogram.add(normRow);
    }

    // TERMINA E RETORNA O RESULTADO DO ESPECTROGRAMA
    return normalizedSpectrogram;
  }

  @override
  void dispose() {
    _recorder?.closeRecorder();
    _player?.closePlayer();
    _interpreter?.close();
    super.dispose();
  }

  // INTERFACE DO APP
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
              Text(_result, style: const TextStyle(color: Colors.red))
            // INTERFACE DE RESULTADO (Exibição dos riscos de bronquite/pneumonia)
            else if (_probabilities.isNotEmpty)
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ..._probabilities.entries.map(
                    (e) => Text(
                      '${e.key}: ${(e.value * 100).toStringAsFixed(0)}%',
                      style: const TextStyle(fontSize: 16),
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Resultado: $_finalLabel',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  Text(
                    'Confiança: ${(_confidence * 100).toStringAsFixed(0)}%',
                    style: const TextStyle(fontSize: 16),
                  ),
                ],
              ),
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
