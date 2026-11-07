import Foundation

/// A model's state after the shared part of its prompts (the system prompt
/// and any turns before the user's message), kept to skip evaluating it
/// again: the most recently used first, bounded by count and by bytes.
/// What a snapshot is (an MLX model's KV and linear-attention caches) is
/// the engine's; this decides which one a prompt reuses and which to keep.
///
/// A snapshot is only ever reused whole: Qwen3.5's linear-attention state
/// cannot be rewound, so an entry serves a prompt that starts with all of
/// its tokens and has at least one more.
///
/// The evaluation of a prompt's shared prefix stops at every multiple of
/// gridTokens and at the prefix's end, whatever the cache holds (and with
/// maximumEntries 0, which keeps nothing): a snapshot is only kept, and
/// only reused, at one of those points. A prompt's evaluation is then the
/// same chunks whether it starts cold or from a snapshot, and a greedy
/// answer does not depend on the cache.
///
/// Besides the prefix's end, a prompt that parts from a kept one inside
/// the system prompt (a glossary that lists the terms the text mentions
/// first, a scene) keeps a branch point: the last grid point before they
/// part, which the next prompt of the kind starts from.
public struct PromptCache<Snapshot> {
    public struct Entry {
        public let tokens: [Int]
        public let snapshot: Snapshot
        public let bytes: Int
    }

    /// What a chat does with the cache: start from entry (nil: from
    /// nothing), evaluate the prompt up to prefixEnd (nil: no shared
    /// prefix past the entry) in chunks ending at the grid's points,
    /// keeping a snapshot at each point of keepAt, then evaluate the rest.
    public struct Plan {
        public let entry: Entry?
        public let prefixEnd: Int?
        public let keepAt: [Int]

        /// Prompt tokens whose evaluation the entry saves.
        public var reusedTokens: Int { entry?.tokens.count ?? 0 }
    }

    public let maximumEntries: Int
    public let maximumBytes: Int
    /// A point under this many tokens is not kept (its snapshot would
    /// cost more memory than its evaluation costs time).
    public let minimumTokens: Int
    public let gridTokens: Int
    public private(set) var entries: [Entry] = []

    public init(maximumEntries: Int, maximumBytes: Int, minimumTokens: Int, gridTokens: Int) {
        self.maximumEntries = maximumEntries
        self.maximumBytes = maximumBytes
        self.minimumTokens = minimumTokens
        self.gridTokens = gridTokens
    }

    public var bytes: Int { entries.reduce(0) { $0 + $1.bytes } }

    /// The plan for a prompt whose first sharedPrefix tokens are the part
    /// other prompts share (0: none). The entry it reuses becomes the most
    /// recently used.
    public mutating func plan(tokens: [Int], sharedPrefix: Int) -> Plan {
        let grid = gridTokens
        let usable = { (e: Entry) in
            e.tokens.count < tokens.count && (e.tokens.count == sharedPrefix || e.tokens.count % grid == 0)
                && tokens.starts(with: e.tokens)
        }
        var best: Int?
        for (i, e) in entries.enumerated() where usable(e) {
            if best.map({ e.tokens.count > entries[$0].tokens.count }) ?? true {
                best = i
            }
        }
        let entry = best.map { entries.remove(at: $0) }
        if let entry {
            entries.insert(entry, at: 0)
        }
        let reused = entry?.tokens.count ?? 0
        guard sharedPrefix > reused, sharedPrefix < tokens.count else {
            return Plan(entry: entry, prefixEnd: nil, keepAt: [])
        }
        var keep: [Int] = []
        if maximumEntries > 0 {
            let parted = entries.map { sharedPrefixLength($0.tokens, tokens) }.filter { $0 < sharedPrefix }.max() ?? 0
            let branch = parted / gridTokens * gridTokens
            if branch > reused, branch >= minimumTokens {
                keep.append(branch)
            }
            if sharedPrefix >= minimumTokens {
                keep.append(sharedPrefix)
            }
        }
        return Plan(entry: entry, prefixEnd: sharedPrefix, keepAt: keep)
    }

    /// Where the evaluation from start to end stops: the grid's points
    /// between them, then end.
    public func stops(from start: Int, to end: Int) -> [Int] {
        var out: [Int] = []
        var point = (start / gridTokens + 1) * gridTokens
        while point < end {
            out.append(point)
            point += gridTokens
        }
        return out + [end]
    }

    /// Keeps a snapshot of the state after tokens, as the most recently
    /// used, then drops the least recently used beyond the bounds. A
    /// snapshot larger than maximumBytes is not kept.
    public mutating func insert(tokens: [Int], snapshot: Snapshot, bytes: Int) {
        entries.removeAll { $0.tokens == tokens }
        guard maximumEntries > 0, bytes <= maximumBytes else { return }
        entries.insert(Entry(tokens: tokens, snapshot: snapshot, bytes: bytes), at: 0)
        while entries.count > maximumEntries || self.bytes > maximumBytes {
            entries.removeLast()
        }
    }

    public mutating func removeAll() {
        entries.removeAll()
    }
}

/// How many leading tokens a and b share.
public func sharedPrefixLength(_ a: [Int], _ b: [Int]) -> Int {
    zip(a, b).prefix { $0 == $1 }.count
}
