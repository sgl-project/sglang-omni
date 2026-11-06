import Foundation
import VoxtNewLLMKit

// voxtnew-llm: run by VoxtNew.app (internal/llmrt supervises it), llm
// protocol 1 on stdin/stdout (docs/llm-protocol.md), free-form
// diagnostics on stderr (~/Library/Logs/VoxtNew/llm.log). It exits when
// stdin closes or on llm.shutdown.

// The protocol takes stdout over; fd 1 then points at stderr, so that
// stray prints of a library (llama.cpp, MLX) cannot corrupt the stream.
let stdoutFD = dup(STDOUT_FILENO)
dup2(STDERR_FILENO, STDOUT_FILENO)
setvbuf(stdout, nil, _IONBF, 0)
let stdoutHandle = FileHandle(fileDescriptor: stdoutFD, closeOnDealloc: true)

let writer = LineWriter { data in try? stdoutHandle.write(contentsOf: data) }
let server = Server(
    out: writer,
    engine: Runtime(),
    info: {
        HelloResult(protocol: protocolVersion, version: llmVersion(), pid: Int(getpid()), methods: Method.requests, device: device())
    },
    exit: { code in exit(code) }
)
server.serveStdin()
dispatchMain()
