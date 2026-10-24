import Darwin
import Foundation
import SystemConfiguration

@MainActor public protocol ConnectorPairingRuntime: AnyObject {
    var onEvent: ((ConnectorPairingProcess.Event) -> Void)? { get set }
    func start(localPort: Int, token: String, configuration: ConnectorConfiguration) async throws
    func beginPairing() throws
    func cancelPairing() throws
    func stop()
}

@MainActor public final class ConnectorPairingProcess: ConnectorPairingRuntime {
    public enum Event: Equatable {
        case ready(address: String)
        case pairing(ConnectorPairingCode)
        case paired(address: String, client: String)
        case pairingClosed
        case failed
    }
    public enum Failure: Error { case configuration, unavailable, protocolError, timeout, stopped }
    public var onEvent: ((Event) -> Void)?
    private let executableURL: URL
    private let persist: (ConnectorConfiguration) throws -> Void
    private var child: Child?
    private var readSource: DispatchSourceRead?
    private var writeSource: DispatchSourceWrite?
    private var outgoing = Data()
    private var incoming = Data()
    private var ready = false
    private var generation = UUID()
    private var authorized: [String] = []
    /// Shown on the phone; the helper puts it on the link the phone keeps after pairing.
    private let macName: String

    public init(executableURL: URL, persist: @escaping (ConnectorConfiguration) throws -> Void,
                macName: String? = ConnectorPairingProcess.computerName()) {
        self.executableURL = executableURL; self.persist = persist
        self.macName = String((macName ?? "").trimmingCharacters(in: .whitespacesAndNewlines).prefix(64))
    }

    /// The Computer Name from Sharing settings. Host.current().localizedName can wait on the network.
    public nonisolated static func computerName() -> String? {
        SCDynamicStoreCopyComputerName(nil, nil) as String?
    }
    isolated deinit { stop() }

    public func start(localPort: Int, token: String, configuration: ConnectorConfiguration) async throws {
        stop()
        guard (1024...65534).contains(localPort), !token.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              Set(configuration.allowedClients).count == configuration.allowedClients.count,
              configuration.allowedClients.allSatisfy(TailcatConfiguration.isPublicKey) else { throw Failure.configuration }
        authorized = configuration.allowedClients
        let attempt = generation
        let running = Child(url: executableURL)
        child = running
        do {
            try send(["version": 2, "localPort": localPort, "token": token, "name": macName,
                      "allowedClients": configuration.allowedClients, "state": configuration.state.map { $0 as Any } ?? NSNull()])
            try running.launch()
            let source = DispatchSource.makeReadSource(fileDescriptor: running.output.fileDescriptor, queue: .main)
            source.setEventHandler { [weak self] in
                MainActor.assumeIsolated { if self?.generation == attempt { self?.pump() } }
            }
            readSource = source; source.resume()
            running.process.terminationHandler = { [weak self] _ in
                Task { @MainActor in
                    guard let self, self.generation == attempt else { return }
                    self.stop(); self.onEvent?(.failed)
                }
            }
            armWrite()
            let deadline = ContinuousClock.now.advanced(by: .seconds(20))
            while !ready {
                try Task.checkCancellation()
                guard generation == attempt, child != nil else { throw Failure.stopped }
                guard ContinuousClock.now < deadline else { throw Failure.timeout }
                try await Task.sleep(for: .milliseconds(10))
            }
            guard generation == attempt else { throw Failure.stopped }
        } catch {
            if generation == attempt { stop() }
            throw error
        }
    }

    public func beginPairing() throws { guard ready else { throw Failure.stopped }; try send(["command": "pair"]) }
    public func cancelPairing() throws { guard ready else { throw Failure.stopped }; try send(["command": "cancelPairing"]) }
    public func stop() {
        generation = UUID(); ready = false
        readSource?.cancel(); readSource = nil
        writeSource?.cancel(); writeSource = nil
        child?.close(); child = nil
        outgoing.removeAll(); incoming.removeAll(); authorized = []
    }
    private func send(_ object: [String: Any]) throws {
        var data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        data.append(10)
        guard outgoing.count + data.count <= 64 * 1024 else { throw Failure.protocolError }
        outgoing.append(data)
        if child?.process.isRunning == true { armWrite() }
    }
    private func armWrite() {
        guard writeSource == nil, let child, !outgoing.isEmpty else { return }
        let attempt = generation
        let source = DispatchSource.makeWriteSource(fileDescriptor: child.input.fileDescriptor, queue: .main)
        source.setEventHandler { [weak self] in
            MainActor.assumeIsolated { if self?.generation == attempt { self?.pump() } }
        }
        writeSource = source; source.resume()
    }
    private func pump() {
        guard let child else { return }
        do {
            if !outgoing.isEmpty {
                let count = outgoing.withUnsafeBytes { Darwin.write(child.input.fileDescriptor, $0.baseAddress!, $0.count) }
                if count > 0 { outgoing.removeFirst(count) }
                else if count < 0 && errno != EAGAIN && errno != EINTR { throw Failure.protocolError }
            }
            if outgoing.isEmpty { writeSource?.cancel(); writeSource = nil }
            var bytes = [UInt8](repeating: 0, count: 4096)
            let count = Darwin.read(child.output.fileDescriptor, &bytes, bytes.count)
            if count > 0 {
                incoming.append(contentsOf: bytes.prefix(count))
                guard incoming.count <= 64 * 1024 else { throw Failure.protocolError }
                while let end = incoming.firstIndex(of: 10) {
                    let line = Data(incoming.prefix(upTo: end))
                    incoming.removeSubrange(...end)
                    try consume(line)
                }
            } else if count == 0 || (count < 0 && errno != EAGAIN && errno != EINTR) { throw Failure.protocolError }
            if !child.process.isRunning { throw Failure.unavailable }
        } catch { stop(); onEvent?(.failed) }
    }
    private func consume(_ line: Data) throws {
        guard let object = try JSONSerialization.jsonObject(with: line) as? [String: Any], let event = object["event"] as? String else { throw Failure.protocolError }
        switch event {
        case "ready":
            guard let address = object["address"] as? String, address.isEmpty || address.hasPrefix("tc") else { throw Failure.protocolError }
            ready = true; onEvent?(.ready(address: address))
        case "persist":
            guard let id = object["id"] as? String, !id.isEmpty, let state = object["state"] as? String, !state.isEmpty,
                  let clients = object["allowedClients"] as? [String], !clients.isEmpty,
                  Set(clients).count == clients.count, clients.allSatisfy(TailcatConfiguration.isPublicKey) else { throw Failure.protocolError }
            do {
                try persist(.init(allowedClients: clients, state: state))
                authorized = clients
                try send(["command": "ack", "id": id, "ok": true])
            } catch {
                try send(["command": "ack", "id": id, "ok": false])
                onEvent?(.failed)
            }
        case "pairing":
            guard let text = object["url"] as? String, let url = URL(string: text) else { throw Failure.protocolError }
            let code = try ConnectorPairingCode(url: url)
            onEvent?(.pairing(code))
        case "paired":
            guard let address = object["address"] as? String, address.hasPrefix("tc"),
                  let client = object["client"] as? String, authorized.contains(client) else { throw Failure.protocolError }
            onEvent?(.paired(address: address, client: client))
        case "pairingClosed": onEvent?(.pairingClosed)
        case "error": onEvent?(.failed)
        default: throw Failure.protocolError
        }
    }

    @MainActor private final class Child {
        let process = Process()
        let inputPipe = Pipe(), outputPipe = Pipe()
        var input: FileHandle { inputPipe.fileHandleForWriting }
        var output: FileHandle { outputPipe.fileHandleForReading }
        private var closed = false
        init(url: URL) {
            process.executableURL = url; process.arguments = ["--connector"]
            process.environment = ["PATH": "/usr/bin:/bin", "TMPDIR": FileManager.default.temporaryDirectory.path]
            process.standardInput = inputPipe; process.standardOutput = outputPipe; process.standardError = FileHandle.nullDevice
        }
        func launch() throws {
            try TailcatProcess.configureInputPipe(input.fileDescriptor)
            guard fcntl(output.fileDescriptor, F_SETFL, O_NONBLOCK) == 0 else { throw Failure.unavailable }
            try process.run()
            try? inputPipe.fileHandleForReading.close(); try? outputPipe.fileHandleForWriting.close()
        }
        func close() {
            guard !closed else { return }; closed = true
            try? input.close(); try? output.close()
            try? inputPipe.fileHandleForReading.close(); try? outputPipe.fileHandleForWriting.close()
            if process.isRunning {
                process.terminate()
                let owned = process
                Task { @MainActor in
                    try? await Task.sleep(for: .seconds(1))
                    if owned.isRunning { kill(owned.processIdentifier, SIGKILL) }
                }
            }
        }
    }
}
