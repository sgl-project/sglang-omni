import Foundation

/// note (luojiaxuan): `--name value` options and `--flag` switches, with every accepted name declared up front.
struct Args {
    private var values: [String: String] = [:]
    private var switches: Set<String> = []

    init(_ arguments: [String], options: Set<String>, flags: Set<String>) throws {
        var iterator = arguments.makeIterator()
        while let arg = iterator.next() {
            if flags.contains(arg) {
                switches.insert(arg)
            } else if options.contains(arg) {
                guard let value = iterator.next() else { throw ReplayError("\(arg) needs a value") }
                values[arg] = value
            } else {
                throw ReplayError("unknown argument \(arg)")
            }
        }
    }

    func flag(_ name: String) -> Bool { switches.contains(name) }

    func string(_ name: String) -> String? { values[name] }

    func required(_ name: String) throws -> String {
        guard let value = values[name] else { throw ReplayError("\(name) is required") }
        return value
    }

    func double(_ name: String, default fallback: Double) throws -> Double {
        guard let text = values[name] else { return fallback }
        guard let value = Double(text) else { throw ReplayError("\(name): not a number: \(text)") }
        return value
    }

    func int(_ name: String, default fallback: Int) throws -> Int {
        guard let text = values[name] else { return fallback }
        guard let value = Int(text) else { throw ReplayError("\(name): not an integer: \(text)") }
        return value
    }
}
