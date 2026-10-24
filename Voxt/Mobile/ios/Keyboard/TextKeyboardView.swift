import SwiftUI
import VoxtMobileKit

/// Four text layouts and a suggestion strip. Callbacks insert text; this view does not own the document.
/// Swipe is only on the layout toolbar, so letter keys stay free of flick gestures. Long-press opens accent alternates.
public struct TextKeyboardView: View {
    private let state: TextKeyboardState
    /// Letter rows from a layout package; nil uses the kit catalog's rows.
    private let letterRows: [[TextKeyboardKey]]?
    /// Conversion candidates (pinyin), without the reading itself.
    private let candidates: [String]
    /// A converting layout (pinyin): one row above the keys holds the layout toolbar,
    /// or the candidate bar while composing.
    private let reservesCandidateBar: Bool
    private let composing: Bool
    /// Shown small at the space bar's corner instead of "space" (pinyin: 拼).
    private let spaceCaption: String?
    private let onCandidate: (String) -> Void
    private let suggestions: [String]
    private let onPress: (TextKeyboardKey) -> Void
    private let onAlternate: (String) -> Void
    private let onSuggestion: (String) -> Void
    private let onSelectLayout: (TextKeyboardLayoutID) -> Void
    private let onCycleLayout: (Int) -> Void

    public init(
        state: TextKeyboardState,
        letterRows: [[TextKeyboardKey]]? = nil,
        candidates: [String] = [],
        reservesCandidateBar: Bool = false,
        composing: Bool = false,
        spaceCaption: String? = nil,
        onCandidate: @escaping (String) -> Void = { _ in },
        suggestions: [String],
        onPress: @escaping (TextKeyboardKey) -> Void,
        onAlternate: @escaping (String) -> Void,
        onSuggestion: @escaping (String) -> Void,
        onSelectLayout: @escaping (TextKeyboardLayoutID) -> Void,
        onCycleLayout: @escaping (Int) -> Void
    ) {
        self.state = state
        self.letterRows = letterRows
        self.candidates = candidates
        self.reservesCandidateBar = reservesCandidateBar
        self.composing = composing
        self.spaceCaption = spaceCaption
        self.onCandidate = onCandidate
        self.suggestions = suggestions
        self.onPress = onPress
        self.onAlternate = onAlternate
        self.onSuggestion = onSuggestion
        self.onSelectLayout = onSelectLayout
        self.onCycleLayout = onCycleLayout
    }

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""
    /// The ⌄ grid of every candidate, in place of the keys.
    @State private var expanded = false

    private typealias Geometry = TextKeyboardGeometry

    public var body: some View {
        let _ = interfaceLanguage
        VStack(spacing: Geometry.rowGap) {
            if reservesCandidateBar {
                Group {
                    if composing { candidateBar } else { layoutToolbar }
                }
                .frame(height: 40)
            } else {
                layoutToolbar
                if !suggestions.isEmpty { suggestionStrip }
            }
            if expanded && composing { candidateGrid } else { keyRows }
        }
        .padding(.vertical, 8)
        .background(VoxtTheme.Colors.panel)
        .onChange(of: composing) { _, now in if !now { expanded = false } }
    }

    private var rows: [[TextKeyboardKey]] {
        state.plane == .letters ? letterRows ?? state.layout.rows(.letters) : state.layout.rows(state.plane)
    }

    private var keyAreaHeight: CGFloat {
        let count = CGFloat(rows.count)
        return count * Geometry.keyHeight + max(0, count - 1) * Geometry.rowGap
    }

    /// Standard iOS proportions (TextKeyboardGeometry): every letter key the same width.
    private var keyRows: some View {
        let rows = self.rows
        return GeometryReader { geometry in
            let frames = Geometry.layout(rows.map { $0.map(\.kind) }, width: geometry.size.width)
            VStack(spacing: Geometry.rowGap) {
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    ZStack(alignment: .topLeading) {
                        ForEach(Array(row.enumerated()), id: \.element.id) { column, key in
                            let frame = frames[index][column]
                            keyButton(display(key))
                                .frame(width: frame.width, height: Geometry.keyHeight)
                                .position(x: frame.x + frame.width / 2, y: Geometry.keyHeight / 2)
                        }
                    }
                    .frame(width: geometry.size.width, height: Geometry.keyHeight, alignment: .topLeading)
                }
            }
        }
        .frame(height: keyAreaHeight)
    }

    /// Plain words; the first, which Space takes, sits on a grey pill. ⌄ opens them all.
    private var candidateBar: some View {
        HStack(spacing: 0) {
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 4) {
                    ForEach(Array(candidates.enumerated()), id: \.element) { index, candidate in
                        candidateButton(candidate, first: index == 0, id: "candidate-\(candidate)")
                    }
                }
                .padding(.horizontal, 8)
            }
            .modifier(NoScrollEdgeEffect())
            Rectangle().fill(VoxtTheme.Colors.keyGray).frame(width: 1, height: 24)
            Button { expanded.toggle() } label: {
                Image(systemName: expanded ? "chevron.up" : "chevron.down")
                    .font(.system(size: 17, weight: .medium))
                    .foregroundStyle(VoxtTheme.Colors.ink)
                    .frame(width: 52, height: 40)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .disabled(candidates.isEmpty)
            .accessibilityLabel(VoxtL10n.shared.string(expanded ? "Back to keys" : "More candidates"))
            .accessibilityIdentifier(expanded ? "candidateCollapse" : "candidateExpand")
            .keyFeedback(.action)
        }
    }

    private var candidateGrid: some View {
        ScrollView {
            // Each word as wide as its text (a fixed grid cut 我就是是 to 我…).
            WrappingLayout(spacing: 6) {
                ForEach(Array(candidates.enumerated()), id: \.element) { index, candidate in
                    candidateButton(candidate, first: index == 0, id: "candidateGrid-\(candidate)", minWidth: 56)
                }
            }
            .padding(.horizontal, 8)
        }
        .modifier(NoScrollEdgeEffect())
        .frame(height: keyAreaHeight)
        .accessibilityIdentifier("candidateGrid")
    }

    private func candidateButton(_ candidate: String, first: Bool, id: String, minWidth: CGFloat = 40) -> some View {
        Button {
            expanded = false
            onCandidate(candidate)
        } label: {
            Text(candidate)
                .font(.system(size: 19))
                .lineLimit(1)
                .foregroundStyle(VoxtTheme.Colors.ink)
                .padding(.horizontal, 11)
                .frame(minWidth: minWidth)
                .frame(height: 34)
                .background(first ? VoxtTheme.Colors.keyGray : .clear, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .keyFeedback(.action)
        .accessibilityIdentifier(id)
        .accessibilityAddTraits(first ? .isSelected : [])
    }

    private var suggestionStrip: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(suggestions, id: \.self) { term in
                    Button(term) { onSuggestion(term) }
                        .buttonStyle(.plain)
                        .keyFeedback(.action)
                        .font(.system(size: 15, weight: .semibold))
                        .foregroundStyle(VoxtTheme.Colors.ink)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(VoxtTheme.Colors.card, in: Capsule())
                        .accessibilityIdentifier("suggestion-\(term)")
                }
            }
            .padding(.horizontal, 10)
        }
    }

    private var layoutToolbar: some View {
        TextLayoutToolbar(state: state,
                          onVoice: { onPress(.init(id: "voice", kind: .voice, label: "Voice")) },
                          onSelectLayout: onSelectLayout, onCycleLayout: onCycleLayout)
            .padding(.horizontal, 8)
    }

    private func display(_ key: TextKeyboardKey) -> TextKeyboardKey {
        guard state.plane == .letters, state.shifted, key.kind == .character, let text = key.text else { return key }
        var copy = key
        let upper = text.uppercased()
        copy.text = upper.count == 1 ? upper : text
        copy.label = copy.text ?? copy.label
        return copy
    }

    private func keyButton(_ key: TextKeyboardKey) -> some View {
        // While composing, Return enters the letters as typed: a white key with ↑.
        let highlighted = key.kind == .return && composing
        return Button {
            onPress(key)
        } label: {
            Group {
                switch key.kind {
                case .shift: Image(systemName: state.shifted ? "shift.fill" : "shift")
                case .delete: Image(systemName: "delete.left")
                case .voice: Image(systemName: "mic")
                case .return: Image(systemName: highlighted ? "arrow.up" : "return")
                case .space where spaceCaption != nil: Color.clear
                default: Text(key.kind == .character ? key.label : VoxtL10n.shared.string(key.label)).lineLimit(1).minimumScaleFactor(0.7)
                }
            }
                .font(.system(size: key.kind == .space ? 14 : 16, weight: .medium))
                .foregroundStyle(highlighted ? VoxtTheme.Colors.onInk : VoxtTheme.Colors.ink)
                .frame(maxWidth: .infinity)
                .frame(height: Geometry.keyHeight)
                .background(highlighted ? VoxtTheme.Colors.ink : VoxtTheme.Colors.keyGray,
                            in: RoundedRectangle(cornerRadius: Geometry.cornerRadius, style: .continuous))
                .overlay(alignment: .bottomTrailing) {
                    if key.kind == .space, let spaceCaption {
                        Text(spaceCaption)
                            .font(.system(size: 12))
                            .foregroundStyle(VoxtTheme.Colors.caption)
                            .padding(.trailing, 10)
                            .padding(.bottom, 5)
                            .accessibilityHidden(true)
                    }
                }
        }
        .buttonStyle(.plain)
        .accessibilityLabel(key.kind == .character ? key.label : VoxtL10n.shared.string(key.label))
        .accessibilityIdentifier(key.kind == .character ? "textKey-\(key.text ?? key.label)" : "textKey-\(key.kind.rawValue)")
        .modifier(AccentMenu(alternates: key.alternates, onAlternate: onAlternate))
        .keyFeedback(.key(key.kind))
    }
}

/// Lays its views out left to right at their own width, starting a new line when one doesn't fit.
private struct WrappingLayout: Layout {
    var spacing: CGFloat

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        let height = place(subviews, width: width).last.map { $0.origin.y + $0.height } ?? 0
        return CGSize(width: proposal.width ?? 0, height: height)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        for (subview, frame) in zip(subviews, place(subviews, width: bounds.width)) {
            subview.place(at: CGPoint(x: bounds.minX + frame.minX, y: bounds.minY + frame.minY), proposal: ProposedViewSize(frame.size))
        }
    }

    private func place(_ subviews: Subviews, width: CGFloat) -> [CGRect] {
        var frames: [CGRect] = []
        var x: CGFloat = 0, y: CGFloat = 0, line: CGFloat = 0
        for subview in subviews {
            let size = subview.sizeThatFits(.unspecified)
            if x > 0, x + size.width > width { x = 0; y += line + spacing; line = 0 }
            frames.append(CGRect(origin: CGPoint(x: x, y: y), size: size))
            x += size.width + spacing
            line = max(line, size.height)
        }
        return frames
    }
}

/// iOS 26 blurs content under a scroll view's edges; candidates stay crisp.
private struct NoScrollEdgeEffect: ViewModifier {
    func body(content: Content) -> some View {
        if #available(iOS 26.0, *) {
            content.scrollEdgeEffectHidden(true, for: .all)
        } else {
            content
        }
    }
}

private struct AccentMenu: ViewModifier {
    let alternates: [String]
    let onAlternate: (String) -> Void

    func body(content: Content) -> some View {
        if alternates.isEmpty {
            content
        } else {
            content.contextMenu {
                ForEach(alternates, id: \.self) { alternate in
                    Button(alternate) { onAlternate(alternate) }
                }
            }
        }
    }
}

/// The same fixed language row is used by Latin, Japanese and Zhuyin layouts.
struct TextLayoutToolbar: View {
    let state: TextKeyboardState
    let onVoice: () -> Void
    let onSelectLayout: (TextKeyboardLayoutID) -> Void
    let onCycleLayout: (Int) -> Void
    var identifier = "textLayoutToolbar"

    var body: some View {
        HStack(spacing: 4) {
            Button(action: onVoice) {
                Image(systemName: "mic").frame(width: 32, height: 30)
                    .background(VoxtTheme.Colors.keyGray, in: Capsule())
            }
            .accessibilityLabel(VoxtL10n.shared.string("Voice"))
            .accessibilityIdentifier("textLayout-voice")
            .keyFeedback(.key(.voice))
            ForEach(state.enabled, id: \.self) { id in
                Button { onSelectLayout(id) } label: {
                    Text(id.shortLabel).lineLimit(1).minimumScaleFactor(0.65)
                        .font(.system(size: 13, weight: .semibold))
                        .padding(.horizontal, 8).frame(height: 30)
                        .foregroundStyle(id == state.active ? VoxtTheme.Colors.onInk : VoxtTheme.Colors.ink)
                        .background(id == state.active ? VoxtTheme.Colors.ink : VoxtTheme.Colors.keyGray, in: Capsule())
                }
                .accessibilityLabel(id.title)
                .accessibilityIdentifier("textLayout-\(id.rawValue)")
                .keyFeedback(.mode)
                .accessibilityAddTraits(id == state.active ? .isSelected : [])
            }
            Spacer(minLength: 0)
        }
        .buttonStyle(.plain)
        .foregroundStyle(VoxtTheme.Colors.ink)
        .contentShape(Rectangle())
        .simultaneousGesture(DragGesture(minimumDistance: 30).onEnded { value in
            guard abs(value.translation.width) > abs(value.translation.height) else { return }
            onCycleLayout(value.translation.width < 0 ? 1 : -1)
        })
        .accessibilityElement(children: .contain)
        .accessibilityIdentifier(identifier)
    }
}
