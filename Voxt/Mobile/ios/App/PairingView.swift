import AVFoundation
import SwiftUI
import VoxtMobileKit

/// Connect your Mac: the in-app camera for Voxt Connector's QR, or the same Wi-Fi.
struct ConnectMacView: View {
    @Environment(AppModel.self) private var model
    @State private var showsConnectorHint = false
    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    var body: some View {
        let _ = interfaceLanguage
        OnboardingPage(title: "Connect your Mac",
                       subtitle: "On your Mac, open Voxt Connector and choose Add iPhone. Then point this iPhone at the QR.",
                       button: model.localPairingSearching ? "Looking for your Mac…" : "Find my Mac on this Wi-Fi",
                       buttonEnabled: !model.localPairingSearching,
                       buttonID: "findMacButton",
                       action: { Task { await model.findMacOnWiFi() } }) {
            scanner
            if let code = model.localPairingCode {
                VStack(spacing: 6) {
                    Text(VoxtL10n.shared.string("Confirm on your Mac: code"))
                        .font(VoxtTheme.Typography.caption).foregroundStyle(.secondary)
                    Text(String(code.prefix(3)) + "  " + String(code.suffix(3)))
                        .font(.system(size: 40, weight: .semibold, design: .rounded).monospacedDigit())
                        .accessibilityIdentifier("localPairingCode")
                }
                .frame(maxWidth: .infinity)
                .voxtCard()
            } else if showsConnectorHint {
                VStack(alignment: .leading, spacing: 8) {
                    Text(VoxtL10n.shared.string("Don't have Voxt Connector yet?")).font(.system(size: 19, weight: .semibold))
                    Text(VoxtL10n.shared.string("Voxt Connector runs on your Mac next to Voxt and shows this QR. Set up both from the link below."))
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                        .accessibilityIdentifier("connectorHint")
                    let source = SourceRevision.repository(info: Bundle.main.infoDictionary)
                    Link(destination: source) {
                        Text(verbatim: (source.host() ?? "") + source.path())
                    }
                    .foregroundStyle(VoxtTheme.Colors.accent)
                    .accessibilityIdentifier("connectorSetupLink")
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .voxtCard()
            } else {
                Button(VoxtL10n.shared.string("Don't have Voxt Connector yet?")) { withAnimation { showsConnectorHint = true } }
                    .foregroundStyle(VoxtTheme.Colors.accent)
                    .frame(maxWidth: .infinity)
                    .accessibilityIdentifier("noConnector")
            }
            if let error = model.localPairingError {
                Text(VoxtL10n.shared.string(error))
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("localPairingError")
            }
            if model.connection == .checking {
                ProgressView(VoxtL10n.shared.string("Connecting…"))
            } else if case let .failed(message) = model.connection, message != model.pairingScan.hint {
                Text(VoxtL10n.shared.string(message))
                    .foregroundStyle(.red)
                    .accessibilityIdentifier("pairingError")
            }
        }
        .task { await requestCamera() }
    }

    @ViewBuilder private var scanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            if model.pairingScan.state == .cameraDenied {
                Text(VoxtL10n.shared.string("Camera access is off. Allow it in Settings, or scan with the Camera app."))
                    .font(VoxtTheme.Typography.body)
                    .accessibilityIdentifier("cameraDenied")
                Button(VoxtL10n.shared.string("Open Settings")) {
                    if let url = URL(string: UIApplication.openSettingsURLString) { UIApplication.shared.open(url) }
                }
                .accessibilityIdentifier("cameraSettingsButton")
            } else {
                ZStack {
                    // Shows through until the camera draws, and where there is no camera (the simulator).
                    LinearGradient(colors: [Color(white: 0.16), Color(white: 0.07)], startPoint: .topLeading, endPoint: .bottomTrailing)
                    if model.pairingScan.state != .starting, !Self.fakeScan, model.localPairingCode == nil {
                        QRScannerView { model.scanned($0) }
                    }
                    if model.localPairingCode != nil {
                        Text(VoxtL10n.shared.string("Camera paused")).font(VoxtTheme.Typography.caption).foregroundStyle(.white.opacity(0.85))
                    }
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .strokeBorder(.white.opacity(0.85), lineWidth: 3)
                        .frame(width: 170, height: 170)
                    if model.pairingScan.state == .pairing {
                        ProgressView().tint(.white).controlSize(.large)
                    }
                }
                .frame(height: 260)
                .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
                .accessibilityElement(children: .ignore)
                .accessibilityLabel(VoxtL10n.shared.string("QR scanner"))
                .accessibilityIdentifier("qrScanner")
                if let hint = model.pairingScan.hint {
                    Text(VoxtL10n.shared.string(hint))
                        .font(VoxtTheme.Typography.caption)
                        .foregroundStyle(.red)
                        .accessibilityIdentifier("scanHint")
                }
            }
        }
    }

    /// Debug: VOXT_TEST_SCAN feeds one decoded string as if the camera read it, and
    /// VOXT_TEST_CAMERA=denied simulates a refused camera (the simulator has no camera).
    private static var fakeScan: Bool {
        #if DEBUG
        ProcessInfo.processInfo.environment["VOXT_TEST_SCAN"] != nil
        #else
        false
        #endif
    }

    private func requestCamera() async {
        #if DEBUG
        let environment = ProcessInfo.processInfo.environment
        if environment["VOXT_TEST_CAMERA"] == "denied" {
            model.cameraAccessChanged(.denied)
            return
        }
        if let scans = environment["VOXT_TEST_SCAN"] {
            model.cameraAccessChanged(.authorized)
            let interval = environment["VOXT_TEST_SCAN_INTERVAL"].flatMap(Double.init) ?? 1.5
            for text in scans.components(separatedBy: "\n") {
                try? await Task.sleep(for: .seconds(interval))
                model.scanned(text)
            }
            return
        }
        #endif
        // No camera (the simulator): nothing to ask for; the viewfinder stays dark.
        guard AVCaptureDevice.default(for: .video) != nil else {
            model.cameraAccessChanged(.authorized)
            return
        }
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: model.cameraAccessChanged(.authorized)
        case .notDetermined:
            model.cameraAccessChanged(.notDetermined)
            let granted = await AVCaptureDevice.requestAccess(for: .video)
            model.cameraAccessChanged(granted ? .authorized : .denied)
        default: model.cameraAccessChanged(.denied)
        }
    }
}

/// Back-camera preview that reports QR codes. Runs while it is on screen.
private struct QRScannerView: UIViewRepresentable {
    let onCode: (String) -> Void

    func makeUIView(context: Context) -> ScannerView {
        let view = ScannerView()
        view.onCode = onCode
        view.start()
        return view
    }

    func updateUIView(_ view: ScannerView, context: Context) { view.onCode = onCode }

    static func dismantleUIView(_ view: ScannerView, coordinator: ()) { view.stop() }

    final class ScannerView: UIView, AVCaptureMetadataOutputObjectsDelegate {
        private struct Session: @unchecked Sendable { let value = AVCaptureSession() }
        private let session = Session()
        private var lastCode: (text: String, at: Date)?
        var onCode: ((String) -> Void)?

        override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }

        func start() {
            let capture = session.value
            guard let camera = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back),
                  let input = try? AVCaptureDeviceInput(device: camera), capture.canAddInput(input) else { return }
            capture.addInput(input)
            let output = AVCaptureMetadataOutput()
            guard capture.canAddOutput(output) else { return }
            capture.addOutput(output)
            output.setMetadataObjectsDelegate(self, queue: .main)
            output.metadataObjectTypes = [.qr]
            if let preview = layer as? AVCaptureVideoPreviewLayer {
                preview.session = capture
                preview.videoGravity = .resizeAspectFill
            }
            let session = session
            DispatchQueue.global(qos: .userInitiated).async { session.value.startRunning() }
        }

        func stop() {
            let session = session
            DispatchQueue.global(qos: .userInitiated).async { session.value.stopRunning() }
        }

        nonisolated func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject],
                                        from connection: AVCaptureConnection) {
            let texts = metadataObjects.compactMap { ($0 as? AVMetadataMachineReadableCodeObject)?.stringValue }
            MainActor.assumeIsolated {
                for text in texts {
                    // The camera reports the same code many times a second.
                    if let lastCode, lastCode.text == text, Date().timeIntervalSince(lastCode.at) < 2 { continue }
                    lastCode = (text, Date())
                    onCode?(text)
                }
            }
        }
    }
}
