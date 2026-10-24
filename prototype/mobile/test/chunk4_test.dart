import 'package:flutter_test/flutter_test.dart';
import 'package:messmate_mobile/core/sensors/exif_extractor.dart';
import 'package:messmate_mobile/core/notifications/delayed_notification_service.dart';
import 'package:messmate_mobile/core/utils/uuid_generator.dart';
import 'package:messmate_mobile/features/timeline/state/reverse_friction_controller.dart';
import 'package:messmate_mobile/features/timeline/widgets/refill_chip_row.dart';

void main() {
  group('Chunk 4 - EXIF Sensor & CIPA DC-008 Diagonal FOV', () {
    test('Calculates exact CIPA DC-008 focal length for 4:3 12MP sensor', () {
      final fx = ExifSensorExtractor.calculateFocalLengthPx(
        f35mm: 24.0,
        width: 4032,
        height: 3024,
      );
      expect((fx - 2795.69).abs() < 0.05, isTrue);
    });

    test('Falls back to 24.0mm flagship prior when EXIF is missing', () {
      final intrinsics = ExifSensorExtractor.extractIntrinsics(
        exifData: {},
        imageWidth: 4032,
        imageHeight: 3024,
      );
      expect(intrinsics.focalLength35mm, equals(24.0));
      expect(intrinsics.focalLengthMm, equals(24.0));
      expect(intrinsics.sensorWidthMm, equals(36.0));
      expect(intrinsics.sensorHeightMm, equals(24.0));
      expect(intrinsics.isFallbackPrior, isTrue);
      expect(intrinsics.cx, equals(2016.0));
      expect(intrinsics.cy, equals(1512.0));
      expect((intrinsics.fx - 2795.69).abs() < 0.05, isTrue);
    });

    test('Parses rational strings in EXIF tags correctly and populates physical intrinsics', () {
      final intrinsics = ExifSensorExtractor.extractIntrinsics(
        exifData: {
          'FocalLengthIn35mmFilm': '24/1',
          'FocalLength': '4.25',
          'SensorWidth': '36.0',
          'SensorHeight': '24.0',
          'ImageWidth': 4032,
          'ImageLength': 3024,
        },
      );
      expect(intrinsics.focalLength35mm, equals(24.0));
      expect(intrinsics.focalLengthMm, equals(4.25));
      expect(intrinsics.sensorWidthMm, equals(36.0));
      expect(intrinsics.sensorHeightMm, equals(24.0));
      expect(intrinsics.isFallbackPrior, isFalse);
    });
  });

  group('Chunk 4 - Reverse Friction Curve & VoI Allocator', () {
    test('Stages transition correctly through Reverse Friction Curve', () {
      expect(
        ReverseFrictionController.determineStage(mealCount: 1, priorVariance: 0.08),
        equals(FrictionStage.dayOneMirror),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 3, priorVariance: 0.08),
        equals(FrictionStage.dayOneMirror),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 4, priorVariance: 0.05),
        equals(FrictionStage.weekOneLearning),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 21, priorVariance: 0.025),
        equals(FrictionStage.weekOneLearning),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 22, priorVariance: 0.018),
        equals(FrictionStage.weekTwoRefining),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 10, priorVariance: 0.015),
        equals(FrictionStage.weekTwoRefining),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 30, priorVariance: 0.010),
        equals(FrictionStage.steadyState),
      );
      expect(
        ReverseFrictionController.determineStage(mealCount: 5, priorVariance: 0.010),
        equals(FrictionStage.steadyState),
      );
    });

    test('Enforces The Chutney Theorem (V_FLOOR = 25.0 kcal)', () {
      final controller = ReverseFrictionController(
        initialMealCount: 10,
        initialPriorVariance: 0.05,
      );
      expect(controller.stage, equals(FrictionStage.weekOneLearning));

      // Suppressed below 25.0 kcal
      expect(controller.shouldRenderQuestionCard(askValueKcal: 24.9), isFalse);

      // Rendered at or above 25.0 kcal
      expect(controller.shouldRenderQuestionCard(askValueKcal: 25.0), isTrue);
      expect(controller.shouldRenderQuestionCard(askValueKcal: 45.0), isTrue);

      // Consuming budget
      expect(controller.budgetRemaining, equals(3));
      controller.recordQuestionAsked();
      controller.recordQuestionAsked();
      controller.recordQuestionAsked();
      expect(controller.budgetRemaining, equals(0));

      // Suppressed when budget exhausted
      expect(controller.shouldRenderQuestionCard(askValueKcal: 100.0), isFalse);
    });

    test('Steady-state completely silences questions', () {
      final controller = ReverseFrictionController(
        initialMealCount: 45,
        initialPriorVariance: 0.008,
      );
      expect(controller.stage, equals(FrictionStage.steadyState));
      expect(controller.isSilentMode, isTrue);
      expect(controller.maxDailyBudget, equals(0));
      expect(controller.shouldRenderQuestionCard(askValueKcal: 200.0), isFalse);
    });
  });

  group('Chunk 4 - Delayed Notifications (Invariant I4)', () {
    test('Enforces maximum 4 tap-only action buttons', () {
      final service = DelayedNotificationService();
      final notif = service.scheduleClarification(
        mealId: 'meal_123',
        promptText: 'Was that 1 or 2 rotis?',
        actions: const [
          NotificationActionButton(actionId: '1', label: '1 Roti'),
          NotificationActionButton(actionId: '2', label: '2 Rotis'),
          NotificationActionButton(actionId: '3', label: '3 Rotis'),
          NotificationActionButton(actionId: '4', label: '4 Rotis'),
          NotificationActionButton(actionId: '5', label: '5 Rotis'),
        ],
      );
      expect(notif.actions.length, lessThanOrEqualTo(4));
      service.cancelMealNotification('meal_123');
    });

    test('Cancellation removes pending notification', () {
      final service = DelayedNotificationService();
      service.scheduleClarification(
        mealId: 'meal_cancel_test',
        promptText: 'Check meal',
        actions: const [
          NotificationActionButton(actionId: 'yes', label: 'Yes'),
          NotificationActionButton(actionId: 'no', label: 'No'),
        ],
      );
      expect(service.getNotificationForMeal('meal_cancel_test'), isNotNull);
      final cancelled = service.cancelMealNotification('meal_cancel_test');
      expect(cancelled, isTrue);
      expect(service.getNotificationForMeal('meal_cancel_test'), isNull);
    });
  });

  group('Chunk 4 - UUID v4 Generator', () {
    test('Generates RFC 4122 compliant UUID v4 strings', () {
      final uuidRegex = RegExp(r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$');
      for (int i = 0; i < 50; i++) {
        final id = UuidGenerator.v4();
        expect(uuidRegex.hasMatch(id), isTrue, reason: 'Failed for UUID: $id');
      }
    });
  });

  group('Chunk 4 - Physical Device Connectivity & Presets', () {
    test('Configures USB adb reverse and LAN base URLs', () {
      expect(ApiEndpoints.usbAdbReverseUrl, equals('http://127.0.0.1:8000'));
      expect(ApiEndpoints.physicalDeviceLanUrl, equals('http://10.5.138.91:8000'));
      expect(ApiEndpoints.androidEmulatorUrl, equals('http://10.0.2.2:8000'));
      expect(ApiEndpoints.defaultBaseUrl, equals(ApiEndpoints.usbAdbReverseUrl));
    });
  });
}
