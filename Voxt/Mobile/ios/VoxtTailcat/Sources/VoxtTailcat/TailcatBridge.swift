import Foundation

public struct TailcatOptions: Equatable, Sendable {
    public let address: String
    public let clientKey: String
    public let remotePort: UInt16
    public init(address: String, clientKey: String, remotePort: UInt16) {
        self.address = address
        self.clientKey = clientKey
        self.remotePort = remotePort
    }
}

public enum TailcatError: Error, Equatable, Sendable { case invalidConfiguration, startFailed, invalidResponse }
struct NativeEndpoint: Sendable { let handle: Int64; let port: Int }
protocol NativeTailcatDriver: Sendable {
    func start(_ options: TailcatOptions) throws -> NativeEndpoint
    func stop(_ handle: Int64)
}

public actor TailcatBridge {
    private let driver: any NativeTailcatDriver
    private var invalidated = false
    private var current: (options: TailcatOptions, endpoint: NativeEndpoint)?
    init(driver: any NativeTailcatDriver) { self.driver = driver }
    deinit { if let current { driver.stop(current.endpoint.handle) } }

    public func endpoint(for options: TailcatOptions) throws -> URL {
        try Task.checkCancellation()
        guard !invalidated else { throw CancellationError() }
        guard options.remotePort != 0, !options.address.isEmpty, !options.clientKey.isEmpty,
              !options.address.utf8.contains(0), !options.clientKey.utf8.contains(0) else {
            throw TailcatError.invalidConfiguration
        }
        if let current, current.options == options {
            return Self.url(port: current.endpoint.port)
        }
        close()
        let endpoint = try driver.start(options)
        guard endpoint.handle > 0, (1...65535).contains(endpoint.port) else {
            if endpoint.handle > 0 { driver.stop(endpoint.handle) }
            throw TailcatError.invalidResponse
        }
        do { try Task.checkCancellation() } catch {
            driver.stop(endpoint.handle)
            throw error
        }
        current = (options, endpoint)
        return Self.url(port: endpoint.port)
    }

    /// Permanently revoke this pairing, including requests still holding this actor.
    public func invalidate() { invalidated = true; close() }

    /// Drop the current session; the next request may establish a new one.
    public func close() {
        guard let previous = current else { return }
        current = nil
        driver.stop(previous.endpoint.handle)
    }

    private static func url(port: Int) -> URL {
        URL(string: "http://127.0.0.1:\(port)")!
    }
}
