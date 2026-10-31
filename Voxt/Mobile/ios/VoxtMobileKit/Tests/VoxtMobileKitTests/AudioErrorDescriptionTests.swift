import Foundation
import Testing
@testable import VoxtMobileKit

struct AudioErrorDescriptionTests {
    @Test func audioFailuresHaveActionableDescriptionsInsteadOfOpaqueErrorCodes() {
        let unsupported: any Error = PCMDownsampler.Error.unsupportedFormat
        #expect((unsupported as? any LocalizedError)?.errorDescription == "This audio format is not supported.")
        let conversion: any Error = PCMDownsampler.Error.conversionFailed("Could not allocate output buffer")
        #expect((conversion as? any LocalizedError)?.errorDescription == "Could not allocate output buffer")
    }
}
