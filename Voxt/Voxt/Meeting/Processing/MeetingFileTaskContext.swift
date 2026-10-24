import Foundation

/// Operation identity shared by checkpoint recovery and diagnostic correlation.
/// Detached work must explicitly inherit it; logging may be disabled or removed.
nonisolated enum MeetingFileTaskContext {
    @TaskLocal static var taskID: UUID?
}
