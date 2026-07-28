import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;

// Public API from Task 2. Change only if you redeploy to a different URL.
const String kApiBaseUrl = 'https://financial-wellbeing-api.onrender.com';
const String kPredictPath = '/predict';

// ---- palette ----
const _teal = Color(0xFF0F766E);
const _tealLight = Color(0xFF14B8A6);
const _bg = Color(0xFFEEF2F6);
const _ink = Color(0xFF0F172A);

void main() => runApp(const WellBeingApp());

class WellBeingApp extends StatelessWidget {
  const WellBeingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Financial Well-Being Predictor',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        scaffoldBackgroundColor: _bg,
        colorScheme: ColorScheme.fromSeed(
          seedColor: _teal,
          primary: _teal,
          brightness: Brightness.light,
        ),
        fontFamily: 'Roboto',
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          isDense: true,
          contentPadding:
              const EdgeInsets.symmetric(horizontal: 14, vertical: 14),
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: Color(0xFFD5DCE6)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: Color(0xFFD5DCE6)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(14),
            borderSide: const BorderSide(color: _teal, width: 1.6),
          ),
        ),
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
    return Scaffold(
      body: Column(
        children: [
          const _Header(),
          Expanded(
            child: Center(
              child: ConstrainedBox(
                constraints: const BoxConstraints(maxWidth: 580),
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(16, 18, 16, 28),
                  children: [
                    _sectionCard(
                      title: 'Financial profile',
                      icon: Icons.savings_outlined,
                      children: [
                        _numberField(_fsScore, 'Financial skill score',
                            '0–100 · everyday money skills',
                            icon: Icons.psychology_outlined, decimal: false),
                        _numberField(_khScore, 'Financial knowledge score',
                            '-4 to 4 · standardised knowledge',
                            icon: Icons.school_outlined, decimal: true),
                        _dropdown('Household income', _income, _incomeOpts,
                            Icons.payments_outlined,
                            (v) => setState(() => _income = v)),
                        _dropdown('Total savings', _savings, _savingsOpts,
                            Icons.account_balance_outlined,
                            (v) => setState(() => _savings = v)),
                      ],
                    ),
                    const SizedBox(height: 14),
                    _sectionCard(
                      title: 'About you',
                      icon: Icons.person_outline,
                      children: [
                        _dropdown('Age group', _age, _ageOpts,
                            Icons.cake_outlined, (v) => setState(() => _age = v)),
                        _dropdown('Education', _educ, _educOpts,
                            Icons.menu_book_outlined,
                            (v) => setState(() => _educ = v)),
                        _dropdown('Employment status', _employ, _employOpts,
                            Icons.work_outline,
                            (v) => setState(() => _employ = v)),
                        _numberField(_hhSize, 'Household size',
                            '1–12 · people in the household',
                            icon: Icons.groups_outlined, decimal: false),
                      ],
                    ),
                    const SizedBox(height: 20),
                    _predictButton(),
                    const SizedBox(height: 18),
                    _resultCard(),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  // ---- section container ----
  Widget _sectionCard({
    required String title,
    required IconData icon,
    required List<Widget> children,
  }) {
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(20),
        boxShadow: const [
          BoxShadow(
              color: Color(0x0F0F172A), blurRadius: 18, offset: Offset(0, 6)),
        ],
      ),
      padding: const EdgeInsets.fromLTRB(18, 16, 18, 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: _teal.withValues(alpha: 0.10),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: _teal, size: 20),
              ),
              const SizedBox(width: 10),
              Text(title,
                  style: const TextStyle(
                      fontSize: 16, fontWeight: FontWeight.w700, color: _ink)),
            ],
          ),
          const SizedBox(height: 14),
          ...children,
        ],
      ),
    );
  }

  Widget _numberField(TextEditingController c, String label, String hint,
      {required IconData icon, required bool decimal}) {
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
          prefixIcon: Icon(icon, size: 20, color: _teal),
        ),
      ),
    );
  }

  Widget _dropdown(String label, int value, List<Option> opts, IconData icon,
      ValueChanged<int> onChanged) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: DropdownButtonFormField<int>(
        initialValue: value,
        isExpanded: true,
        icon: const Icon(Icons.expand_more, color: _teal),
        borderRadius: BorderRadius.circular(14),
        decoration: InputDecoration(
          labelText: label,
          prefixIcon: Icon(icon, size: 20, color: _teal),
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

  Widget _predictButton() {
    return SizedBox(
      height: 54,
      child: DecoratedBox(
        decoration: BoxDecoration(
          borderRadius: BorderRadius.circular(16),
          gradient: const LinearGradient(colors: [_teal, _tealLight]),
          boxShadow: const [
            BoxShadow(
                color: Color(0x330F766E), blurRadius: 16, offset: Offset(0, 8)),
          ],
        ),
        child: Material(
          color: Colors.transparent,
          child: InkWell(
            borderRadius: BorderRadius.circular(16),
            onTap: _loading ? null : _predict,
            child: Center(
              child: _loading
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(
                          strokeWidth: 2.4, color: Colors.white))
                  : const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(Icons.insights, color: Colors.white),
                        SizedBox(width: 10),
                        Text('Predict',
                            style: TextStyle(
                                color: Colors.white,
                                fontSize: 17,
                                fontWeight: FontWeight.w700)),
                      ],
                    ),
            ),
          ),
        ),
      ),
    );
  }

  // color band for a score
  Color _bandColor(double s) {
    if (s < 40) return const Color(0xFFDC2626);
    if (s < 55) return const Color(0xFFF59E0B);
    if (s < 70) return _teal;
    return const Color(0xFF16A34A);
  }

  Widget _resultCard() {
    if (_resultText == null && _score == null) {
      return const SizedBox.shrink();
    }

    if (_isError) {
      return Container(
        width: double.infinity,
        decoration: BoxDecoration(
          color: const Color(0xFFFEF2F2),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: const Color(0xFFFECACA)),
        ),
        padding: const EdgeInsets.all(18),
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Icon(Icons.error_outline, color: Color(0xFFB91C1C)),
            const SizedBox(width: 12),
            Expanded(
              child: Text(_resultText ?? 'Something went wrong.',
                  style: const TextStyle(
                      color: Color(0xFFB91C1C),
                      fontSize: 14,
                      height: 1.35,
                      fontWeight: FontWeight.w500)),
            ),
          ],
        ),
      );
    }

    final s = _score ?? 0;
    final band = _bandColor(s);
    return Container(
      width: double.infinity,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [band.withValues(alpha: 0.14), band.withValues(alpha: 0.04)],
        ),
        border: Border.all(color: band.withValues(alpha: 0.35)),
      ),
      padding: const EdgeInsets.all(22),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.check_circle, color: band, size: 20),
              const SizedBox(width: 8),
              Text('Predicted well-being score',
                  style: TextStyle(
                      color: band, fontWeight: FontWeight.w600, fontSize: 14)),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(s.toStringAsFixed(1),
                  style: TextStyle(
                      color: band,
                      fontSize: 46,
                      height: 1,
                      fontWeight: FontWeight.w800)),
              const SizedBox(width: 4),
              const Padding(
                padding: EdgeInsets.only(bottom: 6),
                child: Text('/ 100',
                    style: TextStyle(
                        color: Color(0xFF64748B),
                        fontSize: 16,
                        fontWeight: FontWeight.w600)),
              ),
            ],
          ),
          const SizedBox(height: 14),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: (s / 100).clamp(0, 1),
              minHeight: 10,
              backgroundColor: Colors.white,
              valueColor: AlwaysStoppedAnimation(band),
            ),
          ),
          if (_resultText != null) ...[
            const SizedBox(height: 14),
            Text(_resultText!,
                style: const TextStyle(
                    color: Color(0xFF334155), fontSize: 14, height: 1.4)),
          ],
        ],
      ),
    );
  }
}

/// Gradient hero header at the top of the page.
class _Header extends StatelessWidget {
  const _Header();

  @override
  Widget build(BuildContext context) {
    final top = MediaQuery.of(context).padding.top;
    return Container(
      width: double.infinity,
      padding: EdgeInsets.fromLTRB(22, top + 22, 22, 26),
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [Color(0xFF0B5A54), _teal, _tealLight],
        ),
        borderRadius: BorderRadius.vertical(bottom: Radius.circular(28)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: Colors.white.withValues(alpha: 0.18),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Icon(Icons.trending_up_rounded,
                    color: Colors.white, size: 26),
              ),
              const SizedBox(width: 14),
              const Expanded(
                child: Text('Financial Well-Being Predictor',
                    style: TextStyle(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.w800,
                        height: 1.15)),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Estimate a person\'s financial well-being score (0–100). '
            'Fill the 8 fields, then tap Predict.',
            style: TextStyle(
                color: Colors.white.withValues(alpha: 0.92),
                fontSize: 13.5,
                height: 1.4),
          ),
        ],
      ),
    );
  }
}
