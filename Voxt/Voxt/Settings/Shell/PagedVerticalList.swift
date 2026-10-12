// PagedVerticalList.swift
// Provides Paged Vertical List for settings shell.

import AppKit
import SwiftUI

private let pagedVerticalListColumnIdentifier = NSUserInterfaceItemIdentifier("PagedVerticalListColumn")
private let pagedVerticalListRowIdentifier = NSUserInterfaceItemIdentifier("PagedVerticalListRow")

private final class HostedTableCell: NSTableCellView {
    let hostingView: NSHostingView<AnyView>

    override init(frame frameRect: NSRect) {
        hostingView = NSHostingView(rootView: AnyView(EmptyView()))
        super.init(frame: frameRect)

        wantsLayer = true
        layer?.backgroundColor = NSColor.clear.cgColor
        hostingView.translatesAutoresizingMaskIntoConstraints = false
        hostingView.setContentHuggingPriority(.defaultLow, for: .horizontal)
        hostingView.setContentHuggingPriority(.defaultLow, for: .vertical)
        addSubview(hostingView)

        NSLayoutConstraint.activate([
            hostingView.leadingAnchor.constraint(equalTo: leadingAnchor),
            hostingView.trailingAnchor.constraint(equalTo: trailingAnchor),
            hostingView.topAnchor.constraint(equalTo: topAnchor),
            hostingView.bottomAnchor.constraint(equalTo: bottomAnchor)
        ])
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
}

struct PagedVerticalList<Item: Identifiable, Row: View>: NSViewRepresentable {
    let items: [Item]
    let totalCount: Int
    let rowHeight: CGFloat
    let rowSpacing: CGFloat
    let rowHeightForItem: ((Item) -> CGFloat)?
    let rowHeightForItemAtWidth: ((Item, CGFloat) -> CGFloat)?
    let isLoading: Bool
    let onLoadMore: () -> Void
    let row: (Item) -> Row

    init(
        items: [Item],
        totalCount: Int,
        rowHeight: CGFloat,
        rowSpacing: CGFloat = 8,
        rowHeightForItem: ((Item) -> CGFloat)? = nil,
        rowHeightForItemAtWidth: ((Item, CGFloat) -> CGFloat)? = nil,
        isLoading: Bool,
        onLoadMore: @escaping () -> Void,
        @ViewBuilder row: @escaping (Item) -> Row
    ) {
        self.items = items
        self.totalCount = totalCount
        self.rowHeight = rowHeight
        self.rowSpacing = rowSpacing
        self.rowHeightForItem = rowHeightForItem
        self.rowHeightForItemAtWidth = rowHeightForItemAtWidth
        self.isLoading = isLoading
        self.onLoadMore = onLoadMore
        self.row = row
    }

    private var state: PagedVerticalListState {
        let items = items
        let rowHeight = rowHeight
        let rowHeightForItem = rowHeightForItem
        let rowHeightForItemAtWidth = rowHeightForItemAtWidth
        return PagedVerticalListState(
            itemCount: items.count,
            totalCount: totalCount,
            rowHeight: rowHeight,
            rowSpacing: rowSpacing,
            rowHeightForIndex: { index, width in
                guard index >= 0, index < items.count else { return rowHeight }
                if let rowHeightForItemAtWidth {
                    return rowHeightForItemAtWidth(items[index], width)
                }
                return rowHeightForItem?(items[index]) ?? rowHeight
            },
            isLoading: isLoading,
            onLoadMore: onLoadMore,
            row: { index in AnyView(row(items[index])) }
        )
    }

    func makeCoordinator() -> PagedVerticalListCoordinator {
        PagedVerticalListCoordinator(state: state)
    }

    func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSScrollView()
        scrollView.drawsBackground = false
        scrollView.borderType = .noBorder
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true

        let tableView = NSTableView()
        tableView.headerView = nil
        tableView.backgroundColor = .clear
        tableView.selectionHighlightStyle = .none
        tableView.gridStyleMask = []
        tableView.columnAutoresizingStyle = .firstColumnOnlyAutoresizingStyle
        tableView.rowHeight = rowHeight
        tableView.intercellSpacing = NSSize(width: 0, height: rowSpacing)
        tableView.usesAutomaticRowHeights = false
        tableView.dataSource = context.coordinator
        tableView.delegate = context.coordinator
        tableView.postsFrameChangedNotifications = true
        if #available(macOS 11.0, *) {
            tableView.style = .plain
        }

        let column = NSTableColumn(identifier: pagedVerticalListColumnIdentifier)
        column.resizingMask = .autoresizingMask
        tableView.addTableColumn(column)

        scrollView.documentView = tableView
        context.coordinator.tableView = tableView
        context.coordinator.scrollView = scrollView
        context.coordinator.observeFrameChanges(for: tableView)
        return scrollView
    }

    func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.update(state: state)
        guard let tableView = context.coordinator.tableView else { return }
        tableView.rowHeight = rowHeight
        tableView.intercellSpacing = NSSize(width: 0, height: rowSpacing)
        tableView.tableColumns.first?.width = scrollView.contentSize.width
        tableView.reloadData()
    }

}

final class PagedVerticalListCoordinator: NSObject, NSTableViewDataSource, NSTableViewDelegate {
    var state: PagedVerticalListState
    weak var tableView: NSTableView?
    weak var scrollView: NSScrollView?
    private var lastLoadMoreItemCount = -1
    private var lastKnownTotalCount = -1
    private var lastKnownTableWidth: CGFloat = 0
    private var frameObserver: NSObjectProtocol?

    init(state: PagedVerticalListState) {
        self.state = state
        lastKnownTotalCount = state.totalCount
    }

    deinit {
        if let frameObserver {
            NotificationCenter.default.removeObserver(frameObserver)
        }
    }

    func observeFrameChanges(for tableView: NSTableView) {
        frameObserver = NotificationCenter.default.addObserver(
            forName: NSView.frameDidChangeNotification,
            object: tableView,
            queue: .main
        ) { [weak self, weak tableView] _ in
            guard let self, let tableView else { return }
            self.reloadIfWidthChanged(in: tableView)
        }
    }

    func update(state newState: PagedVerticalListState) {
        if newState.itemCount < state.itemCount || newState.totalCount != lastKnownTotalCount {
            lastLoadMoreItemCount = -1
        }
        state = newState
        lastKnownTotalCount = newState.totalCount
    }

    func numberOfRows(in tableView: NSTableView) -> Int {
        state.itemCount + (showsFooter ? 1 : 0)
    }

    func tableView(_ tableView: NSTableView, heightOfRow row: Int) -> CGFloat {
        isFooterRow(row) ? 40 : state.height(for: row, width: availableWidth(in: tableView))
    }

    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row rowIndex: Int) -> NSView? {
        if isFooterRow(rowIndex) {
            return hostedCell(
                in: tableView,
                rootView: AnyView(footerView)
            )
        }

        guard rowIndex >= 0, rowIndex < state.itemCount else { return nil }
        requestNextPageIfNeeded(displaying: rowIndex)
        return hostedCell(
            in: tableView,
            rootView: AnyView(
                state.row(rowIndex)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .frame(height: state.height(for: rowIndex, width: availableWidth(in: tableView)))
            )
        )
    }

    private func availableWidth(in tableView: NSTableView) -> CGFloat {
        let width = tableView.bounds.width
        guard width.isFinite, width > 0 else { return state.rowHeight }
        return width
    }

    private func reloadIfWidthChanged(in tableView: NSTableView) {
        let width = availableWidth(in: tableView)
        guard abs(width - lastKnownTableWidth) >= 1 else { return }
        lastKnownTableWidth = width
        let indexes = IndexSet(integersIn: 0..<tableView.numberOfRows)
        tableView.noteHeightOfRows(withIndexesChanged: indexes)
        tableView.reloadData(forRowIndexes: indexes, columnIndexes: IndexSet(integer: 0))
    }

    private var showsFooter: Bool {
        state.isLoading || state.itemCount < state.totalCount
    }

    private func isFooterRow(_ rowIndex: Int) -> Bool {
        showsFooter && rowIndex == state.itemCount
    }

    @ViewBuilder
    private var footerView: some View {
        if state.isLoading {
            ProgressView()
                .controlSize(.small)
                .frame(maxWidth: .infinity, minHeight: 36)
        } else {
            Button(AppLocalization.localizedString("Load More")) {
                self.state.onLoadMore()
            }
            .buttonStyle(SettingsPillButtonStyle())
            .frame(maxWidth: .infinity, minHeight: 36)
        }
    }

    private func requestNextPageIfNeeded(displaying rowIndex: Int) {
        guard !state.isLoading, state.itemCount < state.totalCount else { return }
        guard rowIndex >= max(0, state.itemCount - 12) else { return }
        guard lastLoadMoreItemCount != state.itemCount else { return }
        lastLoadMoreItemCount = state.itemCount
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.state.onLoadMore()
        }
    }

    private func hostedCell(in tableView: NSTableView, rootView: AnyView) -> HostedTableCell {
        let cell = tableView.makeView(
            withIdentifier: pagedVerticalListRowIdentifier,
            owner: self
        ) as? HostedTableCell ?? HostedTableCell()
        cell.identifier = pagedVerticalListRowIdentifier
        cell.hostingView.rootView = rootView
        return cell
    }
}

struct PagedVerticalListState {
    let itemCount: Int
    let totalCount: Int
    let rowHeight: CGFloat
    let rowSpacing: CGFloat
    let rowHeightForIndex: (Int, CGFloat) -> CGFloat
    let isLoading: Bool
    let onLoadMore: () -> Void
    let row: (Int) -> AnyView

    func height(for index: Int, width: CGFloat = 0) -> CGFloat {
        rowHeightForIndex(index, width)
    }
}
