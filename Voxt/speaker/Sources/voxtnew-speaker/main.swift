import CoreML
import FluidAudio
import Foundation

struct Request: Decodable {
    let op: String
    let audio: Data?
}

struct Reply: Encodable {
    var ready: Bool = false
    var error: String?
    var start_frame: Int = 0
    var frames: [[Float]] = []
}

func write(_ reply: Reply) throws {
    var data = try JSONEncoder().encode(reply)
    data.append(10)
    try FileHandle.standardOutput.write(contentsOf: data)
}

func reply(_ chunks: [Nemotron3ChunkResult], start: Int) -> Reply {
    let probabilities = chunks.flatMap { $0.probabilities }
    let frames = stride(from: 0, to: probabilities.count, by: 8).map {
        Array(probabilities[$0..<($0 + 8)])
    }
    return Reply(start_frame: start, frames: frames)
}

do {
    let args = CommandLine.arguments
    guard args.count == 3, args[1] == "--model" else {
        throw CocoaError(.fileReadInvalidFileName)
    }
    AppLogger.minimumLevel = .warning
    AppLogger.mirrorsToConsole = false
    let config = Nemotron3Config.fast
    let compute = MLModelConfiguration()
    compute.computeUnits = .cpuOnly
    let root = URL(fileURLWithPath: args[2])
    let modelURL = root.appendingPathComponent("monolithic/v2/" + config.modelFileName)
    let model = try MLModel(contentsOf: modelURL, configuration: compute)
    let silence = try Data(contentsOf: root.appendingPathComponent("learnable_sil_emb.bin"))
    let embedding = silence.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    let models = try Nemotron3Models(config: config, model: model, silenceEmbedding: embedding)
    let diarizer = Nemotron3Diarizer(config: config, models: models)
    try write(Reply(ready: true))
    while let line = readLine() {
        try autoreleasepool {
        // A protocol request contains at most one second of PCM. The Go
        // dispatcher bounds its own queue while Core ML loads or runs.
        guard line.utf8.count <= 48_000 else { throw CocoaError(.coderReadCorrupt) }
        let request = try JSONDecoder().decode(Request.self, from: Data(line.utf8))
        switch request.op {
        case "audio":
            guard let pcm = request.audio, pcm.count % 2 == 0, pcm.count <= 32_000 else {
                throw CocoaError(.coderReadCorrupt)
            }
            let samples: [Float] = stride(from: 0, to: pcm.count, by: 2).map {
                Float(Int16(bitPattern: UInt16(pcm[$0]) | UInt16(pcm[$0 + 1]) << 8)) / 32768
            }
            let start = diarizer.streamedFrameCount
            diarizer.appendAudio(samples)
            try write(reply(try diarizer.processBufferedAudio(), start: start))
        case "finish":
            let start = diarizer.streamedFrameCount
            try write(reply(try diarizer.finishStream(), start: start))
        default:
            throw CocoaError(.coderReadCorrupt)
        }
        }
    }
} catch {
    try? write(Reply(error: String(describing: error)))
    exit(1)
}
