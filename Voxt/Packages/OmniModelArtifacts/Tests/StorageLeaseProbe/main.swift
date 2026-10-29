import Foundation
import OmniModelArtifacts

let directory = URL(fileURLWithPath: CommandLine.arguments[1])
let mode: OmniModelDirectoryLease.Mode = CommandLine.arguments[2] == "shared" ? .sharedRead : .exclusiveMutation
let readyFile = URL(fileURLWithPath: CommandLine.arguments[3])
do {
    let lease = try OmniModelDirectoryLease.acquire(for: directory, mode: mode)
    defer { withExtendedLifetime(lease) {} }
    if CommandLine.arguments[2] == "pending" {
        let destination = URL(fileURLWithPath: CommandLine.arguments[4])
        var checks = 0
        _ = try OmniModelPublication.publish(stagingLease: lease, to: destination, checkCancellation: {
            checks += 1
            if checks == 2 {
                try Data("ready".utf8).write(to: readyFile)
                _ = readLine()
            }
        })
    } else {
        try Data("ready".utf8).write(to: readyFile)
        _ = readLine()
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(2)
}
