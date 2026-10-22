import Foundation
import CVoxtTailcat

struct CNativeDriver: NativeTailcatDriver {
    func start(_ options: TailcatOptions) throws -> NativeEndpoint {
        guard !options.address.utf8.contains(0), !options.clientKey.utf8.contains(0) else {
            throw TailcatError.invalidConfiguration
        }
        let data = try options.address.withCString { address in
            try options.clientKey.withCString { clientKey in
                try Self.copyAndFree(VoxtTailcatStart(UnsafeMutablePointer(mutating: address),
                                                    UnsafeMutablePointer(mutating: clientKey), Int32(options.remotePort)))
            }
        }
        return try NativeResponse.endpoint(from: data)
    }
    func stop(_ handle: Int64) { _ = VoxtTailcatStop(handle) }
    static func identity() throws -> TailcatIdentity {
        try NativeResponse.identity(from: copyAndFree(VoxtTailcatIdentity()))
    }
    private static func copyAndFree(_ pointer: UnsafeMutablePointer<CChar>?) throws -> Data {
        guard let pointer else { throw TailcatError.invalidResponse }
        defer { VoxtTailcatFree(pointer) }
        return Data(String(cString: pointer).utf8)
    }
}

extension TailcatBridge {
    public init() { self.init(driver: CNativeDriver()) }
}
extension TailcatIdentity {
    public static func generate() throws -> TailcatIdentity { try CNativeDriver.identity() }
}
