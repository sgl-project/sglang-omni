// ReportSettingsView.swift
// Provides Report Settings View for settings screens.

import SwiftUI
import AppKit

private func reportLocalized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct ReportSettingsView: View {
    @Environment(\.locale) private var locale
    @ObservedObject var historyStore: TranscriptionHistoryStore
    @ObservedObject var dictionaryStore: DictionaryStore
    @ObservedObject var mainWindowState: MainWindowVisibilityState
    let isActive: Bool
    @State private var cachedSummary: ReportSummary?
    @State private var branchRange: ReportTimeRange = .today
    @State private var vocabularyRange: ReportTimeRange = .today
    @State private var summaryGeneration = 0
    @State private var dashboardAnimationToken = UUID()

    private let metricColumns = [
        GridItem(.flexible(), spacing: 10),
        GridItem(.flexible(), spacing: 10),
    ]

    var body: some View {
        let summary = cachedSummary ?? ReportSummary.empty(locale: locale)

        GeometryReader { proxy in
            let leftWidth = max(168, floor(proxy.size.width * 0.36))
            let topRowHeight: CGFloat = 190
            let metricCardHeight = (topRowHeight - 10) / 2
            let bottomRowHeight: CGFloat = 226

            VStack(alignment: .leading, spacing: 12) {
                HStack(alignment: .top, spacing: 12) {
                    LazyVGrid(columns: metricColumns, spacing: 10) {
                        ReportMetricCard(
                            iconName: "clock.badge.checkmark",
                            title: reportLocalized("Total Dictation Time"),
                            value: formattedDuration(summary.totalDictationSeconds),
                            unit: nil
                        )
                        .frame(height: metricCardHeight)
                        ReportMetricCard(
                            iconName: "character.textbox",
                            title: reportLocalized("Total Dictation Characters"),
                            value: localizedNumber(summary.totalCharacters),
                            unit: reportLocalized("chars")
                        )
                        .frame(height: metricCardHeight)
                        ReportMetricCard(
                            iconName: "globe",
                            title: reportLocalized("Total Translation Characters"),
                            value: localizedNumber(summary.totalTranslationCharacters),
                            unit: reportLocalized("chars")
                        )
                        .frame(height: metricCardHeight)
                        ReportMetricCard(
                            iconName: "speedometer",
                            title: reportLocalized("Average Dictation Speed"),
                            value: localizedNumber(Int(summary.averageCharactersPerMinute)),
                            unit: reportLocalized("char/min")
                        )
                        .frame(height: metricCardHeight)
                    }
                    .frame(maxWidth: .infinity)

                    BranchRankingCard(
                        items: summary.branchItems,
                        selectedRange: $branchRange,
                        localizedNumber: localizedNumber,
                        animationToken: dashboardAnimationToken
                    )
                    .frame(width: leftWidth, height: topRowHeight)
                }

                HStack(alignment: .top, spacing: 12) {
                    DailyCharactersTrendCard(data: summary.dailyCharacters, animationToken: dashboardAnimationToken)
                        .frame(maxWidth: .infinity)
                        .frame(height: bottomRowHeight)

                    VocabularyCard(
                        entries: vocabularyEntries,
                        selectedRange: $vocabularyRange
                    )
                    .frame(width: leftWidth, height: bottomRowHeight)
                }
            }
            .frame(width: proxy.size.width, height: proxy.size.height, alignment: .topLeading)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .onAppear {
            refreshSummary()
            triggerDashboardAnimation()
        }
        .onReceive(historyStore.$entries) { _ in
            refreshSummary()
        }
        .onChange(of: branchRange) { _, _ in
            refreshSummary()
        }
        .onChange(of: locale.identifier) { _, _ in
            refreshSummary()
        }
        .onChange(of: mainWindowState.isVisible) { _, isVisible in
            guard isVisible, isActive else { return }
            triggerDashboardAnimation()
        }
        .onChange(of: isActive) { _, active in
            guard active else { return }
            triggerDashboardAnimation()
        }
    }

    private var vocabularyEntries: [DictionaryEntry] {
        let startDate = vocabularyRange.startDate()
        return dictionaryStore.entries
            .filter { entry in
                entry.source == .auto && entry.createdAt >= startDate
            }
            .sorted { lhs, rhs in
                if lhs.createdAt == rhs.createdAt {
                    return lhs.term.localizedCaseInsensitiveCompare(rhs.term) == .orderedAscending
                }
                return lhs.createdAt > rhs.createdAt
            }
    }

    private func formattedDuration(_ seconds: TimeInterval) -> String {
        let totalSeconds = max(0, Int(seconds.rounded()))
        let hours = totalSeconds / 3600
        let minutes = (totalSeconds % 3600) / 60
        let remainSeconds = totalSeconds % 60
        if hours > 0 {
            return String(format: reportLocalized("%dh %dm"), locale: locale, hours, minutes)
        }
        if minutes > 0 {
            return String(format: reportLocalized("%dm %ds"), locale: locale, minutes, remainSeconds)
        }
        return String(format: reportLocalized("%d s"), locale: locale, remainSeconds)
    }

    private func localizedNumber(_ value: Int) -> String {
        let formatter = NumberFormatter()
        formatter.locale = locale
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: value)) ?? "\(value)"
    }

    private func refreshSummary() {
        summaryGeneration += 1
        let generation = summaryGeneration
        let locale = locale
        let dayStarts = ReportSummary.lastSevenDayStarts()
        let branchStartDate = branchRange.startDate()

        historyStore.reportMetrics(dayStarts: dayStarts, branchStartDate: branchStartDate) { metrics in
            guard generation == summaryGeneration else { return }
            cachedSummary = ReportSummary(
                metrics: metrics ?? .empty(dayStarts: dayStarts),
                locale: locale
            )
        }
    }

    private func triggerDashboardAnimation() {
        dashboardAnimationToken = UUID()
    }
}

private enum ReportTimeRange: String, CaseIterable, Identifiable {
    case today
    case sevenDays
    case thirtyDays

    var id: String { rawValue }

    var title: String {
        switch self {
        case .today: return reportLocalized("Today")
        case .sevenDays: return reportLocalized("7 Days")
        case .thirtyDays: return reportLocalized("30 Days")
        }
    }

    func startDate(calendar: Calendar = .current, now: Date = Date()) -> Date {
        let today = calendar.startOfDay(for: now)
        switch self {
        case .today:
            return today
        case .sevenDays:
            return calendar.date(byAdding: .day, value: -6, to: today) ?? today
        case .thirtyDays:
            return calendar.date(byAdding: .day, value: -29, to: today) ?? today
        }
    }
}

private struct DashboardCard<Content: View>: View {
    let content: Content

    init(@ViewBuilder content: () -> Content) {
        self.content = content()
    }

    var body: some View {
        content
            .padding(12)
            .background(
                RoundedRectangle(cornerRadius: 18, style: .continuous)
                    .fill(SettingsUIStyle.groupedFillColor)
            )
    }
}

private struct DashboardCardHeader: View {
    let title: String
    @Binding var selectedRange: ReportTimeRange

    var body: some View {
        HStack(spacing: 8) {
            Text(title)
                .font(.system(size: 14, weight: .bold))
                .lineLimit(1)

            Spacer(minLength: 0)

            SettingsMenuPicker(
                selection: $selectedRange,
                options: ReportTimeRange.allCases.map { range in
                    SettingsMenuOption(value: range, title: range.title)
                },
                selectedTitle: selectedRange.title,
                width: 72,
                allowsCompactWidth: true,
                isCompact: true
            )
        }
    }
}

private struct BranchRankingCard: View {
    let items: [HistoryBranchMetricItem]
    @Binding var selectedRange: ReportTimeRange
    let localizedNumber: (Int) -> String
    let animationToken: UUID

    var body: some View {
        DashboardCard {
            VStack(alignment: .leading, spacing: 10) {
                DashboardCardHeader(title: reportLocalized("Branch"), selectedRange: $selectedRange)

                if items.isEmpty {
                    DashboardEmptyState(text: reportLocalized("No usage records yet"))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView(.vertical) {
                        VStack(spacing: 0) {
                            ForEach(Array(items.enumerated()), id: \.element.id) { index, item in
                                BranchRankingRow(
                                    item: item,
                                    maxValue: max(items.map(\.characterCount).max() ?? 1, 1),
                                    localizedNumber: localizedNumber,
                                    animationToken: animationToken
                                )

                                if index < items.count - 1 {
                                    Divider()
                                        .opacity(0.45)
                                        .padding(.leading, 26)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .top)
                    }
                    .scrollIndicators(.hidden)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
    }
}

private struct BranchRankingRow: View {
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    let item: HistoryBranchMetricItem
    let maxValue: Int
    let localizedNumber: (Int) -> String
    let animationToken: UUID
    @State private var barProgress: CGFloat = 0

    var body: some View {
        HStack(spacing: 8) {
            BranchMetricIcon(item: item)
                .frame(width: 18, height: 18)

            VStack(alignment: .leading, spacing: 2) {
                Text(item.title)
                    .font(.system(size: 11, weight: .semibold))
                    .lineLimit(1)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            VStack(alignment: .trailing, spacing: 4) {
                GeometryReader { proxy in
                    let targetWidth = max(8, proxy.size.width * CGFloat(item.characterCount) / CGFloat(maxValue))

                    Capsule(style: .continuous)
                        .fill(Color.accentColor.opacity(0.16))
                        .overlay(alignment: .leading) {
                            Capsule(style: .continuous)
                                .fill(Color.accentColor.opacity(0.88))
                                .frame(width: targetWidth * barProgress)
                        }
                }
                .frame(width: 54, height: 5)

                Text("\(localizedNumber(item.characterCount)) \(reportLocalized("chars"))")
                    .font(.system(size: 10, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }
        }
        .padding(.vertical, 7)
        .onAppear(perform: animateBar)
        .onChange(of: animationSignature) { _, _ in
            animateBar()
        }
        .onChange(of: animationToken) { _, _ in
            animateBar()
        }
    }

    private var animationSignature: String {
        "\(item.id)-\(item.characterCount)-\(maxValue)"
    }

    private func animateBar() {
        if reduceMotion {
            barProgress = 1
            return
        }

        barProgress = 0
        withAnimation(.easeOut(duration: 0.42)) {
            barProgress = 1
        }
    }
}

private struct BranchMetricIcon: View {
    let item: HistoryBranchMetricItem
    @State private var image: NSImage?
    private var imageSize: CGFloat { item.kind == .url ? 14 : 18 }
    private var cornerRadius: CGFloat { item.kind == .url ? 3 : 4 }

    var body: some View {
        Group {
            if let image {
                Image(nsImage: image)
                    .resizable()
                    .aspectRatio(contentMode: .fit)
                    .frame(width: imageSize, height: imageSize)
                    .clipShape(RoundedRectangle(cornerRadius: cornerRadius, style: .continuous))
            } else {
                Image(systemName: item.kind == .url ? "globe" : "app.dashed")
                    .font(.system(size: item.kind == .url ? 10 : 12, weight: .semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .frame(width: 18, height: 18)
        .task(id: item.id) {
            await loadImage()
        }
    }

    @MainActor
    private func loadImage() async {
        switch item.kind {
        case .app:
            guard let bundleID = item.bundleID else { return }
            image = EnhancementOverlayIconResolver.appIcon(bundleID: bundleID)
        case .url:
            guard let origin = item.urlOrigin else { return }
            if let cached = EnhancementOverlayIconResolver.cachedFavicon(forOrigin: origin) {
                image = cached
                return
            }
            image = await EnhancementOverlayIconResolver.favicon(forOrigin: origin)
        }
    }
}

private struct ReportMetricCard: View {
    let iconName: String
    let title: String
    let value: String
    let unit: String?

    var body: some View {
        DashboardCard {
            VStack(alignment: .leading, spacing: 9) {
                HStack(spacing: 6) {
                    Image(systemName: iconName)
                        .font(.system(size: 13, weight: .semibold))
                    Text(title)
                        .font(.system(size: 12, weight: .semibold))
                }
                .foregroundStyle(.secondary)
                .lineLimit(1)

                HStack(alignment: .firstTextBaseline, spacing: 4) {
                    Text(value)
                        .font(.system(size: 19, weight: .bold))
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)

                    if let unit {
                        Text(unit)
                            .font(.system(size: 11, weight: .semibold))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
        }
    }
}

private struct VocabularyCard: View {
    let entries: [DictionaryEntry]
    @Binding var selectedRange: ReportTimeRange

    var body: some View {
        DashboardCard {
            VStack(alignment: .leading, spacing: 10) {
                DashboardCardHeader(title: reportLocalized("Vocabulary"), selectedRange: $selectedRange)

                if entries.isEmpty {
                    DashboardEmptyState(text: reportLocalized("No auto-learned vocabulary yet"))
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    ScrollView(.vertical) {
                        VStack(spacing: 0) {
                            ForEach(Array(entries.prefix(10).enumerated()), id: \.element.id) { index, entry in
                                HStack(spacing: 8) {
                                    Text(entry.term)
                                        .font(.system(size: 11, weight: .semibold))
                                        .lineLimit(1)
                                    Spacer(minLength: 0)
                                }
                                .padding(.vertical, 7)

                                if index < min(entries.count, 10) - 1 {
                                    Divider()
                                        .opacity(0.45)
                                }
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .top)
                    }
                    .scrollIndicators(.hidden)
                    .frame(maxHeight: .infinity, alignment: .top)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
    }
}

private struct DailyCharactersTrendCard: View {
    let data: [DailyCharactersTrendChart.DayValue]
    let animationToken: UUID

    var body: some View {
        DashboardCard {
            VStack(alignment: .leading, spacing: 10) {
                HStack(alignment: .firstTextBaseline) {
                    Text(reportLocalized("Daily Characters (Last 7 Days)"))
                        .font(.system(size: 14, weight: .bold))
                        .lineLimit(1)

                    Spacer(minLength: 0)

                    Text("\(data.map(\.value).reduce(0, +)) \(reportLocalized("chars"))")
                        .font(.system(size: 14, weight: .bold))
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .minimumScaleFactor(0.72)
                }

                DailyCharactersTrendChart(data: data, animationToken: animationToken)
                    .frame(height: 178)
            }
        }
    }
}

private struct DailyCharactersTrendChart: View {
    struct DayValue: Identifiable {
        let dayStart: Date
        let label: String
        let value: Int

        var id: Date { dayStart }
    }

    let data: [DayValue]
    let animationToken: UUID
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    @State private var revealProgress: CGFloat = 0

    var body: some View {
        let maxValue = max(data.map(\.value).max() ?? 0, 1)

        GeometryReader { proxy in
            let graphRect = CGRect(
                x: 0,
                y: 22,
                width: max(proxy.size.width, 1),
                height: max(proxy.size.height - 48, 1)
            )

            ZStack(alignment: .topLeading) {
                HStack(spacing: 0) {
                    ForEach(data) { item in
                        VStack(alignment: .leading, spacing: 0) {
                            Text(item.label)
                                .font(.system(size: 10, weight: .medium))
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                                .padding(.leading, 6)
                            Spacer(minLength: 0)
                            Text("\(item.value)")
                                .font(.system(size: 10, weight: .bold))
                                .foregroundStyle(.primary.opacity(0.82))
                                .lineLimit(1)
                                .padding(.leading, 6)
                                .padding(.bottom, 8)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .leading)
                        .overlay(alignment: .leading) {
                            Rectangle()
                                .fill(SettingsUIStyle.subtleBorderColor)
                                .frame(width: 1)
                        }
                    }
                }

                ZStack(alignment: .topLeading) {
                    TrendAreaShape(data: data.map(\.value), maxValue: maxValue, graphRect: graphRect)
                        .fill(
                            LinearGradient(
                                colors: [Color.accentColor.opacity(0.26), Color.accentColor.opacity(0.02)],
                                startPoint: .top,
                                endPoint: .bottom
                            )
                        )

                    TrendLineShape(data: data.map(\.value), maxValue: maxValue, graphRect: graphRect)
                        .stroke(Color.accentColor.opacity(0.78), style: StrokeStyle(lineWidth: 2.5, lineCap: .round, lineJoin: .round))

                    ForEach(Array(points(in: graphRect, maxValue: maxValue).enumerated()), id: \.offset) { _, point in
                        Circle()
                            .fill(SettingsUIStyle.panelFillColor)
                            .frame(width: 12, height: 12)
                            .overlay(
                                Circle()
                                    .stroke(Color.accentColor.opacity(0.78), lineWidth: 2)
                            )
                            .position(point)
                    }
                }
                .mask(alignment: .leading) {
                    Rectangle()
                        .frame(width: proxy.size.width * revealProgress, height: proxy.size.height)
                }

                if data.allSatisfy({ $0.value == 0 }) {
                    DashboardEmptyState(text: reportLocalized("No trend data yet"))
                        .frame(width: proxy.size.width, height: graphRect.height)
                        .offset(y: graphRect.minY)
                }
            }
        }
        .onAppear(perform: animateReveal)
        .onChange(of: chartAnimationSignature) { _, _ in
            animateReveal()
        }
        .onChange(of: animationToken) { _, _ in
            animateReveal()
        }
    }

    private var chartAnimationSignature: String {
        data.map { "\($0.dayStart.timeIntervalSince1970):\($0.value)" }.joined(separator: "|")
    }

    private func animateReveal() {
        if reduceMotion {
            revealProgress = 1
            return
        }

        revealProgress = 0
        withAnimation(.easeOut(duration: 0.62)) {
            revealProgress = 1
        }
    }

    private func points(in rect: CGRect, maxValue: Int) -> [CGPoint] {
        guard !data.isEmpty else { return [] }
        if data.count == 1 {
            return [CGPoint(x: rect.midX, y: yPosition(for: data[0].value, maxValue: maxValue, rect: rect))]
        }
        return data.enumerated().map { index, item in
            let x = chartXPosition(index: index, count: data.count, rect: rect)
            return CGPoint(x: x, y: yPosition(for: item.value, maxValue: maxValue, rect: rect))
        }
    }
}

private struct TrendLineShape: Shape {
    let data: [Int]
    let maxValue: Int
    let graphRect: CGRect

    func path(in rect: CGRect) -> Path {
        chartLinePath(data: data, maxValue: maxValue, graphRect: graphRect)
    }
}

private struct TrendAreaShape: Shape {
    let data: [Int]
    let maxValue: Int
    let graphRect: CGRect

    func path(in rect: CGRect) -> Path {
        var path = chartLinePath(data: data, maxValue: maxValue, graphRect: graphRect)
        guard !data.isEmpty else { return path }
        path.addLine(to: CGPoint(x: graphRect.maxX, y: graphRect.maxY))
        path.addLine(to: CGPoint(x: graphRect.minX, y: graphRect.maxY))
        path.closeSubpath()
        return path
    }
}

private func chartLinePath(data: [Int], maxValue: Int, graphRect: CGRect) -> Path {
    var path = Path()
    guard !data.isEmpty else { return path }
    if data.count == 1 {
        let y = yPosition(for: data[0], maxValue: maxValue, rect: graphRect)
        let point = CGPoint(
            x: graphRect.midX,
            y: y
        )
        path.move(to: CGPoint(x: graphRect.minX, y: y))
        path.addLine(to: point)
        path.addLine(to: CGPoint(x: graphRect.maxX, y: y))
        return path
    }

    let points = chartPoints(data: data, maxValue: maxValue, graphRect: graphRect)
    guard let firstPoint = points.first, let lastPoint = points.last else { return path }

    path.move(to: chartEdgePoint(
        x: graphRect.minX,
        from: firstPoint,
        toward: points[1],
        graphRect: graphRect
    ))
    for point in points {
        path.addLine(to: point)
    }
    path.addLine(to: chartEdgePoint(
        x: graphRect.maxX,
        from: lastPoint,
        toward: points[points.count - 2],
        graphRect: graphRect
    ))
    return path
}

private func chartPoints(data: [Int], maxValue: Int, graphRect: CGRect) -> [CGPoint] {
    data.enumerated().map { index, value in
        CGPoint(
            x: chartXPosition(index: index, count: data.count, rect: graphRect),
            y: yPosition(for: value, maxValue: maxValue, rect: graphRect)
        )
    }
}

private func chartEdgePoint(x: CGFloat, from point: CGPoint, toward referencePoint: CGPoint, graphRect: CGRect) -> CGPoint {
    let dx = point.x - referencePoint.x
    guard abs(dx) > .ulpOfOne else {
        return CGPoint(x: x, y: point.y)
    }

    let slope = (point.y - referencePoint.y) / dx
    let y = point.y + (x - point.x) * slope
    return CGPoint(x: x, y: min(max(y, graphRect.minY), graphRect.maxY))
}

private func chartXPosition(index: Int, count: Int, rect: CGRect) -> CGFloat {
    guard count > 1 else { return rect.midX }
    let columnWidth = rect.width / CGFloat(count)
    return rect.minX + columnWidth * (CGFloat(index) + 0.5)
}

private func yPosition(for value: Int, maxValue: Int, rect: CGRect) -> CGFloat {
    let normalized = CGFloat(value) / CGFloat(max(maxValue, 1))
    return rect.maxY - normalized * max(rect.height * 0.78, 1) - rect.height * 0.08
}

private struct DashboardEmptyState: View {
    let text: String

    var body: some View {
        Text(text)
            .font(.system(size: 12, weight: .medium))
            .foregroundStyle(.secondary)
            .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}

private struct ReportSummary {
    let totalDictationSeconds: TimeInterval
    let totalCharacters: Int
    let totalTranslationCharacters: Int
    let averageCharactersPerMinute: Double
    let branchItems: [HistoryBranchMetricItem]
    let dailyCharacters: [DailyCharactersTrendChart.DayValue]

    init(metrics: HistoryReportMetrics, locale: Locale) {
        totalDictationSeconds = metrics.totalDictationSeconds
        totalCharacters = metrics.totalCharacters
        totalTranslationCharacters = metrics.totalTranslationCharacters
        averageCharactersPerMinute = totalDictationSeconds > 0
            ? Double(totalCharacters) / (totalDictationSeconds / 60.0)
            : 0
        branchItems = metrics.branchItems
        dailyCharacters = Self.dailyCharacters(from: metrics.dailyCharacters, locale: locale)
    }

    static func empty(locale: Locale) -> ReportSummary {
        let dayStarts = lastSevenDayStarts()
        return ReportSummary(metrics: .empty(dayStarts: dayStarts), locale: locale)
    }

    static func lastSevenDayStarts() -> [Date] {
        let calendar = Calendar.current
        let today = calendar.startOfDay(for: Date())
        let startDay = calendar.date(byAdding: .day, value: -6, to: today) ?? today
        return (0..<7).compactMap { offset in
            calendar.date(byAdding: .day, value: offset, to: startDay)
        }
    }

    private static func dailyCharacters(
        from valuesByDay: [Date: Int],
        locale: Locale
    ) -> [DailyCharactersTrendChart.DayValue] {
        lastSevenDayStarts().map { day in
            DailyCharactersTrendChart.DayValue(
                dayStart: day,
                label: day.formatted(.dateTime.weekday(.abbreviated).locale(locale)),
                value: valuesByDay[day, default: 0]
            )
        }
    }
}

private extension HistoryReportMetrics {
    static func empty(dayStarts: [Date]) -> HistoryReportMetrics {
        HistoryReportMetrics(
            totalDictationSeconds: 0,
            totalCharacters: 0,
            totalTranslationCharacters: 0,
            dailyCharacters: Dictionary(uniqueKeysWithValues: dayStarts.map { ($0, 0) }),
            branchItems: []
        )
    }
}
