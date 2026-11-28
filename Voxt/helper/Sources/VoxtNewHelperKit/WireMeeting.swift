import Foundation

public struct MeetingWindowParams: Codable, Sendable {
    public var pid: Int
    public var screenshot: Bool?
    public init(pid: Int, screenshot: Bool? = nil) { self.pid = pid; self.screenshot = screenshot }
}

public struct MeetingWindowResult: Codable, Sendable {
    public var nodes: [MeetingNode]
    public var tiles: [MeetingScreenTile]?
    public init(nodes: [MeetingNode] = [], tiles: [MeetingScreenTile]? = nil) { self.nodes = nodes; self.tiles = tiles }
}

public struct MeetingNode: Codable, Sendable {
    public var parent: Int
    public var role: String
    public var title: String?
    public var description: String?
    public var value: String?
    public var identifier: String?
    public var url: String?

    public init(parent: Int, role: String, title: String? = nil, description: String? = nil,
                value: String? = nil, identifier: String? = nil, url: String? = nil) {
        self.parent = parent
        self.role = role
        self.title = title
        self.description = description
        self.value = value
        self.identifier = identifier
        self.url = url
    }
}

public struct MeetingScreenTile: Codable, Sendable {
 public var name: String
 public var speaking: Bool
 public init(name: String, speaking: Bool) { self.name=name; self.speaking=speaking }
}
