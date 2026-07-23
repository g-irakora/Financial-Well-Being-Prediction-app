import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

// Public API from Task 2. Change only if you redeploy to a different URL.
const String kApiBaseUrl = 'https://financial-wellbeing-api.onrender.com';
const String kPredictPath = '/predict';

void main() => runApp(const WellBeingApp());

class WellBeingApp extends StatelessWidget {
  const WellBeingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Financial Well-Being Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFF2A7DE1)),
        useMaterial3: true,
      ),
      home: const PredictPage(),
    );
  }
}

/// A labelled option for the coded categorical dropdowns (value sent to API +
/// human label shown to the user).
class Option {
  final int value;
  final String label;
  const Option(this.value, this.label);
}

class PredictPage extends StatefulWidget {
  const PredictPage({super.key});

  @override
  State<PredictPage> createState() => _PredictPageState();
}

class _PredictPageState extends State<PredictPage> {
  // 3 numeric inputs (typed) ...
  final _fsScore = TextEditingController(text: '55');
  final _khScore = TextEditingController(text: '0.4');
  final _hhSize = TextEditingController(text: '3');

  // ... and 5 coded categorical inputs (dropdowns) = 8 model variables total.
  int _income = 7;
  int _savings = 5;
  int _age = 2;
  int _educ = 4;
  int _employ = 2;

  static const _incomeOpts = [
    Option(1, '1 · Less than \$20k'), Option(2, '2 · \$20k–30k'),
    Option(3, '3 · \$30k–40k'), Option(4, '4 · \$40k–50k'),
    Option(5, '5 · \$50k–60k'), Option(6, '6 · \$60k–75k'),
    Option(7, '7 · \$75k–100k'), Option(8, '8 · \$100k–150k'),
    Option(9, '9 · \$150k or more'),
  ];
  static const _savingsOpts = [
    Option(1, '1 · \$0'), Option(2, '2 · \$1–99'), Option(3, '3 · \$100–999'),
    Option(4, '4 · \$1k–5k'), Option(5, '5 · \$5k–20k'),
    Option(6, '6 · \$20k–75k'), Option(7, '7 · \$75k or more'),
  ];
  static const _ageOpts = [
    Option(1, '1 · 18–24'), Option(2, '2 · 25–34'), Option(3, '3 · 35–44'),
    Option(4, '4 · 45–54'), Option(5, '5 · 55–61'), Option(6, '6 · 62–69'),
    Option(7, '7 · 70–74'), Option(8, '8 · 75+'),
  ];
  static const _educOpts = [
    Option(1, '1 · Less than high school'), Option(2, '2 · High school'),
    Option(3, '3 · Some college / Associate'), Option(4, '4 · Bachelor\'s degree'),
    Option(5, '5 · Graduate / Professional'),
  ];
  static const _employOpts = [
    Option(1, '1 · Self-employed'), Option(2, '2 · Full-time employee'),
    Option(3, '3 · Part-time employee'), Option(4, '4 · Homemaker'),
    Option(5, '5 · Full-time student'), Option(6, '6 · Sick / disabled'),
    Option(7, '7 · Unemployed'), Option(8, '8 · Retired'),
  ];

  bool _loading = false;
  String? _resultText;
  double? _score;
  bool _isError = false;

  @override
  void dispose() {
    _fsScore.dispose();
    _khScore.dispose();
    _hhSize.dispose();
    super.dispose();
  }

  Future<void> _predict() async {
    FocusScope.of(context).unfocus();

    // client-side checks: presence + type + range (mirrors the API contract)
    final fs = double.tryParse(_fsScore.text.trim());
    final kh = double.tryParse(_khScore.text.trim());
    final hh = int.tryParse(_hhSize.text.trim());

    if (_fsScore.text.trim().isEmpty ||
        _khScore.text.trim().isEmpty ||
        _hhSize.text.trim().isEmpty) {
      _show('Please fill in all fields before predicting.', isError: true);
      return;
    }
    if (fs == null || kh == null || hh == null) {
      _show('Skill score, knowledge score and household size must be numbers.',
          isError: true);
      return;
    }
    if (fs < 0 || fs > 100) {
      _show('Financial skill score must be between 0 and 100.', isError: true);
      return;
    }
    if (kh < -4 || kh > 4) {
      _show('Financial knowledge score must be between -4 and 4.', isError: true);
      return;
    }
    if (hh < 1 || hh > 12) {
      _show('Household size must be between 1 and 12.', isError: true);
      return;
    }

    final body = {
      'FSscore': fs,
      'KHscore': kh,
      'PPHHSIZE': hh,
      'PPINCIMP': _income,
      'SAVINGSRANGES': _savings,
      'agecat': _age,
      'PPEDUC': _educ,
      'EMPLOY': _employ,
    };

    setState(() {
      _loading = true;
      _resultText = null;
      _score = null;
    });

    try {
      final resp = await http
          .post(
            Uri.parse('$kApiBaseUrl$kPredictPath'),
            headers: {'Content-Type': 'application/json'},
            body: jsonEncode(body),
          )
          .timeout(const Duration(seconds: 90));

      final data = jsonDecode(resp.body);
      if (resp.statusCode == 200) {
        setState(() {
          _score = (data['predicted_wellbeing_score'] as num).toDouble();
          _resultText = data['interpretation'] as String?;
          _isError = false;
        });
      } else if (resp.statusCode == 422) {
        _show('Some values are invalid or out of range. ${_firstError(data)}',
            isError: true);
      } else {
        _show('Server error (${resp.statusCode}). Please try again.', isError: true);
      }
    } catch (e) {
      _show('Could not reach the API. Check your connection and try again.\n'
          '(Free hosting may take ~30–60s to wake up on the first request.)',
          isError: true);
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _firstError(dynamic data) {
    try {
      final d = data['detail'];
      if (d is List && d.isNotEmpty) {
        final loc = (d[0]['loc'] as List).last;
        return '($loc: ${d[0]['msg']})';
      }
    } catch (_) {}
    return '';
  }

  void _show(String text, {required bool isError}) {
    setState(() {
      _resultText = text;
      _isError = isError;
      _score = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final scheme = Theme.of(context).colorScheme;
    return Scaffold(
      appBar: AppBar(
        backgroundColor: scheme.primary,
        foregroundColor: Colors.white,
        title: const Text('Financial Well-Being Predictor'),
        centerTitle: true,
      ),
      body: SafeArea(
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 560),
            child: ListView(
              padding: const EdgeInsets.all(20),
              children: [
                const SizedBox(height: 4),
                Text('Estimate a person\'s financial well-being score (0–100)',
                    style: Theme.of(context).textTheme.titleMedium),
                const SizedBox(height: 4),
                Text('Fill the 8 fields, then tap Predict.',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.black54)),
                const SizedBox(height: 16),

                _numberField(_fsScore, 'Financial skill score (0–100)',
                    'How strong their day-to-day money skills are', decimal: false),
                _numberField(_khScore, 'Financial knowledge score (-4 to 4)',
                    'Standardised financial-knowledge score', decimal: true),
                _numberField(_hhSize, 'Household size (1–12)',
                    'Number of people in the household', decimal: false),

                _dropdown('Household income', _income, _incomeOpts,
                    (v) => setState(() => _income = v)),
                _dropdown('Total savings', _savings, _savingsOpts,
                    (v) => setState(() => _savings = v)),
                _dropdown('Age group', _age, _ageOpts,
                    (v) => setState(() => _age = v)),
                _dropdown('Education', _educ, _educOpts,
                    (v) => setState(() => _educ = v)),
                _dropdown('Employment status', _employ, _employOpts,
                    (v) => setState(() => _employ = v)),

                const SizedBox(height: 8),
                SizedBox(
                  height: 50,
                  child: FilledButton.icon(
                    onPressed: _loading ? null : _predict,
                    icon: _loading
                        ? const SizedBox(
                            width: 18,
                            height: 18,
                            child: CircularProgressIndicator(
                                strokeWidth: 2, color: Colors.white))
                        : const Icon(Icons.insights),
                    label: Text(_loading ? 'Predicting…' : 'Predict'),
                  ),
                ),
                const SizedBox(height: 16),
                _resultCard(scheme),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _numberField(TextEditingController c, String label, String hint,
      {required bool decimal}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: TextField(
        controller: c,
        keyboardType:
            TextInputType.numberWithOptions(decimal: decimal, signed: decimal),
        inputFormatters: [
          FilteringTextInputFormatter.allow(
              RegExp(decimal ? r'[0-9.\-]' : r'[0-9]')),
        ],
        decoration: InputDecoration(
          labelText: label,
          helperText: hint,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
      ),
    );
  }

  Widget _dropdown(String label, int value, List<Option> opts,
      ValueChanged<int> onChanged) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: DropdownButtonFormField<int>(
        initialValue: value,
        isExpanded: true,
        decoration: InputDecoration(
          labelText: label,
          border: const OutlineInputBorder(),
          isDense: true,
        ),
        items: opts
            .map((o) => DropdownMenuItem(value: o.value, child: Text(o.label)))
            .toList(),
        onChanged: (v) {
          if (v != null) onChanged(v);
        },
      ),
    );
  }

  Widget _resultCard(ColorScheme scheme) {
    if (_resultText == null && _score == null) {
      return const SizedBox.shrink();
    }
    final bg = _isError ? const Color(0xFFFDECEA) : const Color(0xFFEAF3FF);
    final fg = _isError ? const Color(0xFFB3261E) : scheme.primary;
    return Card(
      color: bg,
      elevation: 0,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      child: Padding(
        padding: const EdgeInsets.all(18),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(_isError ? Icons.error_outline : Icons.check_circle_outline,
                    color: fg),
                const SizedBox(width: 8),
                Text(_isError ? 'Cannot predict' : 'Predicted well-being score',
                    style: TextStyle(
                        color: fg, fontWeight: FontWeight.w600, fontSize: 16)),
              ],
            ),
            const SizedBox(height: 10),
            if (_score != null)
              Text('${_score!.toStringAsFixed(1)} / 100',
                  style: TextStyle(
                      color: fg, fontSize: 34, fontWeight: FontWeight.bold)),
            if (_resultText != null) ...[
              const SizedBox(height: 6),
              Text(_resultText!,
                  style: TextStyle(color: fg.withValues(alpha: 0.9), fontSize: 14)),
            ],
          ],
        ),
      ),
    );
  }
}
