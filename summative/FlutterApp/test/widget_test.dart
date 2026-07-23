import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:financial_wellbeing/main.dart';

void main() {
  testWidgets('renders the predictor with a Predict button', (tester) async {
    // Tall surface so the whole lazy ListView (8 inputs + button) is built.
    tester.view.physicalSize = const Size(1200, 3000);
    tester.view.devicePixelRatio = 1.0;
    addTearDown(tester.view.resetPhysicalSize);
    addTearDown(tester.view.resetDevicePixelRatio);

    await tester.pumpWidget(const WellBeingApp());
    await tester.pumpAndSettle();

    expect(find.text('Financial Well-Being Predictor'), findsOneWidget);
    expect(find.text('Predict'), findsOneWidget);
    // 3 numeric text fields + 5 dropdowns = 8 inputs for the 8 model variables
    expect(find.byType(TextField), findsNWidgets(3));
    expect(find.byType(DropdownButtonFormField<int>), findsNWidgets(5));
  });
}
