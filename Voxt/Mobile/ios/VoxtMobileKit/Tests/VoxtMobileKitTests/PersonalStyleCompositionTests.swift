import Foundation
import Testing
@testable import VoxtMobileKit

struct PersonalStyleCompositionTests {
    @Test func styleExamplesSurviveRequestRoundTrip() throws {
        var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(CompositionOptions())) as? [String: Any])
        let examples = [["before": "Thanks, I'll send the file.", "after": "Cheers — I'll send the file."]]
        object["styleExamples"] = examples
        let decoded = try JSONDecoder().decode(CompositionOptions.self, from: JSONSerialization.data(withJSONObject: object))
        try decoded.validate()
        let encoded = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        #expect(encoded["styleExamples"] as? [[String: String]] == examples)
    }

    @Test func invalidStyleExamplesAreRejectedByTheMobileClient() throws {
        let examples = [
            [["before": "Pay 50.", "after": "Pay 60."]],
            [["before": "", "after": "Hello"]],
            Array(repeating: ["before": "Thanks.", "after": "Cheers."], count: 9),
            Array(repeating: ["before": "Thanks.", "after": "Cheers."], count: 2)
        ]
        for invalid in examples {
            var object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(CompositionOptions())) as? [String: Any])
            object["styleExamples"] = invalid
            let decoded = try JSONDecoder().decode(CompositionOptions.self, from: JSONSerialization.data(withJSONObject: object))
            #expect(throws: CompositionOptions.ValidationError.self) { try decoded.validate() }
        }
    }
}
