"""Static removal gates runnable without Xcode; not a substitute for macOS tests."""
import hashlib
import pathlib
import plistlib
import re
import textwrap
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = ROOT / "Voxt"


class PermissionMinimizationTests(unittest.TestCase):
    def test_removed_capture_and_attachment_symbols_have_no_runtime_consumers(self):
        retired = (
            "TranscriptionAppContext", "RewriteAppContextGuidance",
            "DebugRewriteAppContextPayload", "__VOXT_DEBUG_REWRITE_APP_CONTEXT_CAPTURE__",
            "LLMInputAttachment", "LLMImageAttachment", "LLMDebugImagePreview",
            ".appContext", "/usr/sbin/screencapture", '"input_image"',
            "CGRequestScreenCaptureAccess", "CGPreflightScreenCaptureAccess",
            "CGRequestListenEventAccess", "CGPreflightListenEventAccess",
            "HotkeyRecorderHIDMonitor", "IOHIDManagerRegisterInputValueCallback",
            "TCCAccessPreflight", "TCCAccessRequest", "PrivateFrameworks/TCC",
        )
        for path in APP.rglob("*.swift"):
            source = path.read_text()
            for symbol in retired:
                with self.subTest(path=str(path.relative_to(ROOT)), symbol=symbol):
                    self.assertNotIn(symbol, source)

    def test_hotkey_backends_use_accessibility_taps(self):
        for name in ("HotkeyEventTapInstallation.swift", "HotkeyRecorderView.swift"):
            source = (APP / "Hotkey" / name).read_text()
            self.assertIn("options: .defaultTap", source)
            self.assertNotIn(".listenOnly", source)
        source = (APP / "Hotkey/HotkeyManager.swift").read_text()
        self.assertIn("AccessibilityPermissionManager.isTrusted()", source)
        self.assertNotIn("requestInputMonitoring", source)

    def test_device_mute_does_not_capture_audio(self):
        source = (APP / "Core/SystemAudioMuteController.swift").read_text()
        for api in ("AudioHardwareCreateProcessTap", "AggregateDevice", "CapturePermission", "TCC"):
            self.assertNotIn(api, source)
        self.assertIn("kAudioDevicePropertyMute", source)
        self.assertIn("AudioObjectIsPropertySettable", source)
        self.assertIn("deviceUID(device.id) == device.uid", source)
        meeting = (APP / "Meeting/Capture/MeetingSystemAudioCapture.swift").read_text()
        self.assertIn("AudioHardwareCreateProcessTap", meeting)
        self.assertIn(".unmuted", meeting)

    def test_usage_descriptions_keep_meeting_audio_but_not_screenshots(self):
        with (APP / "Voxt/Info.plist").open("rb") as stream:
            info = plistlib.load(stream)
        self.assertNotIn("NSScreenCaptureUsageDescription", info)
        self.assertIn("meeting", info["NSAudioCaptureUsageDescription"])
        self.assertNotIn("mute", info["NSAudioCaptureUsageDescription"])
        self.assertIn("NSMicrophoneUsageDescription", info)

    def test_retired_prompt_fixtures_match_migration_digests(self):
        fixtures = (ROOT / "VoxtTests/RetiredRewritePromptTests.swift").read_text()
        templates = re.findall(r'"""\n(.*?)\n        """', fixtures, re.DOTALL)
        self.assertEqual(len(templates), 3)
        defaults = (APP / "Core/AppPromptDefaults.swift").read_text()
        for template in templates:
            digest = hashlib.sha256(textwrap.dedent(template).strip().encode()).hexdigest()
            self.assertIn(digest, defaults)

    def test_new_permission_copy_is_localized_and_retired_rows_are_gone(self):
        keys = [
            "Required for global shortcuts and inserting text into other apps.",
            "System Audio for Meetings",
            "Requested by macOS when you start a meeting using system audio. Not needed for microphone-only recording, imported files, or output-device mute.",
            "If access was denied, enable System Audio Recording for Voxt in System Settings, then retry the meeting.",
            "Mute output device while recording",
            "Mutes all sounds on the current output device, including Voxt. Some devices do not support software mute. No system audio recording permission is needed.",
            "This output device could not be muted. Recording will continue.",
            "Write stable cleanup rules only. Do not paste raw transcription here. Voxt injects the transcription and glossary automatically.",
        ]
        for language in ("en", "zh-Hans", "ja"):
            source = (APP / f"{language}.lproj/Localizable.strings").read_text()
            for key in keys:
                self.assertIn(f'"{key}" = ', source)
            for key in ("Input Monitoring Permission", "Screen Recording Permission", "Context Enhancement", "Screenshot Context"):
                self.assertNotIn(f'"{key}" = ', source)


if __name__ == "__main__":
    unittest.main()
