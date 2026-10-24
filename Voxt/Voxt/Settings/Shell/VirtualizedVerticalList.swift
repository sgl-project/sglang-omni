// VirtualizedVerticalList.swift
// Provides Virtualized Vertical List for settings shell.

import SwiftUI

private struct VirtualizedScrollOffsetPreferenceKey: PreferenceKey {
    static var defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = nextValue()
    }
}

struct VirtualizedVerticalList<Item: Identifiable, Row: View>: View {
    let items: [Item]
    let rowHeight: CGFloat
    let rowSpacing: CGFloat
    let overscan: Int
    @ViewBuilder let row: (Item) -> Row

    @State private var scrollOffset: CGFloat = 0
    @State private var coordinateSpaceName = UUID().uuidString

    init(
        items: [Item],
        rowHeight: CGFloat,
        rowSpacing: CGFloat = 8,
        overscan: Int = 6,
        @ViewBuilder row: @escaping (Item) -> Row
    ) {
        self.items = items
        self.rowHeight = rowHeight
        self.rowSpacing = rowSpacing
        self.overscan = overscan
        self.row = row
    }

    var body: some View {
        GeometryReader { proxy in
            ScrollView {
                ZStack(alignment: .topLeading) {
                    GeometryReader { marker in
                        Color.clear.preference(
                            key: VirtualizedScrollOffsetPreferenceKey.self,
                            value: max(0, -marker.frame(in: .named(coordinateSpaceName)).minY)
                        )
                    }
                    .frame(height: 0)

                    Color.clear
                        .frame(height: totalContentHeight)

                    ForEach(Array(visibleRange(viewportHeight: proxy.size.height)), id: \.self) { index in
                        row(items[index])
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .frame(height: rowHeight)
                            .offset(y: CGFloat(index) * rowStride)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .coordinateSpace(name: coordinateSpaceName)
            .onPreferenceChange(VirtualizedScrollOffsetPreferenceKey.self) { value in
                scrollOffset = value
            }
        }
    }

    private var rowStride: CGFloat {
        rowHeight + rowSpacing
    }

    private var totalContentHeight: CGFloat {
        guard !items.isEmpty else { return 0 }
        return CGFloat(items.count) * rowHeight + CGFloat(max(0, items.count - 1)) * rowSpacing
    }

    private func visibleRange(viewportHeight: CGFloat) -> Range<Int> {
        guard !items.isEmpty, rowStride > 0 else { return 0..<0 }

        let resolvedViewportHeight = max(viewportHeight, rowHeight)
        let lower = max(0, Int(floor(scrollOffset / rowStride)) - overscan)
        let upper = min(
            items.count,
            Int(ceil((scrollOffset + resolvedViewportHeight) / rowStride)) + overscan
        )
        return lower..<max(lower, upper)
    }
}
