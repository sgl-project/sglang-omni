import Foundation

// voxtnew-helper: run by VoxtNew.app (internal/helper supervises it),
// helper protocol 1 on stdin/stdout (docs/helper-protocol.md), free-form
// diagnostics on stderr (~/Library/Logs/VoxtNew/helper.log). It exits
// when stdin closes or on helper.shutdown.
//
// The main thread runs the main run loop: NSWorkspace's notifications,
// and NSPasteboard live there. Requests are read on a
// thread of their own (Server.serve).

// capture.start's echoCancel: the helper's own child (VoiceProcessing.swift).
if CommandLine.arguments.dropFirst().first == "--vp-capture" { runVoiceProcessingChild() }

let out = Writer()
let server = Server(out: out)
server.serve()
RunLoop.main.run()
