import Foundation
#if canImport(DictationCore)
import DictationCore
#endif

extension ClientPreferences {
    var serviceConfiguration: ServiceConfiguration {
        get {
            guard let data = defaults.data(forKey: "serviceConfiguration"),
                  let value = try? JSONDecoder().decode(ServiceConfiguration.self, from: data) else {
                return ServiceConfiguration()
            }
            return value
        }
        set { defaults.set(try? JSONEncoder().encode(newValue), forKey: "serviceConfiguration") }
    }
}
