import Foundation
import Testing
@testable import TypingHost

@Suite struct JapaneseInputQueueTests {
    @Test func burstStaysQueuedUntilThePendingWriteIsAcknowledged() {
        var queue = CompositionWriteQueue<String>()
        let write = UUID()
        #expect({ queue.enqueue("か") }())
        #expect({ queue.enqueue("い") }())
        #expect({ queue.enqueue("き") }())
        queue.beginWrite(id: write)
        #expect(queue.pendingID == write)
        #expect({ queue.next() }() == nil)
        #expect({ queue.next() }() == nil)
        #expect({ queue.acknowledge(id: write) }())
        #expect(queue.pendingID == nil)
        #expect({ queue.next() }() == "か")
        #expect({ queue.next() }() == "い")
        #expect({ queue.next() }() == "き")
        #expect({ queue.next() }() == nil)
    }

    @Test func oldAcknowledgementLeavesTheNewPendingWrite() {
        var queue = CompositionWriteQueue<String>()
        let old = UUID()
        let current = UUID()
        #expect({ queue.enqueue("き") }())
        queue.beginWrite(id: old)
        queue.beginWrite(id: current)
        #expect(queue.pendingID == current)
        #expect(!{ queue.acknowledge(id: old) }())
        #expect(queue.pendingID == current)
        #expect({ queue.next() }() == nil)
        #expect(!{ queue.acknowledge(id: UUID()) }())
        #expect({ queue.acknowledge(id: current) }())
        #expect({ queue.next() }() == "き")
    }

    @Test func cancelClearsThePendingWriteAndQueuedActions() {
        var queue = CompositionWriteQueue<String>()
        let write = UUID()
        #expect({ queue.enqueue("か") }())
        #expect({ queue.enqueue("い") }())
        queue.beginWrite(id: write)
        queue.cancel()
        #expect(queue.pendingID == nil)
        #expect(!{ queue.acknowledge(id: write) }())
        #expect({ queue.next() }() == nil)
        #expect({ queue.enqueue("き") }())
        #expect({ queue.next() }() == "き")
    }

    @Test func overflowRejectsThe129thActionAndKeepsTheFirst128() {
        var queue = CompositionWriteQueue<Int>()
        for action in 0..<128 {
            #expect({ queue.enqueue(action) }())
        }
        #expect(!{ queue.enqueue(128) }())
        let write = UUID()
        queue.beginWrite(id: write)
        #expect({ queue.next() }() == nil)
        #expect({ queue.acknowledge(id: write) }())
        for action in 0..<128 {
            #expect({ queue.next() }() == action)
        }
        #expect({ queue.next() }() == nil)
        #expect({ queue.enqueue(128) }())
    }
}
