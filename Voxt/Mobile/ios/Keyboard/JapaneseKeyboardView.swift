import SwiftUI
#if canImport(JapaneseInput)
import JapaneseInput
import TypingInput
import VoxtMobileKit

struct JapaneseKeyboardView: View {
    let model: KeyboardModel
    private let rows: [[JapaneseFlickKey]] = [[.a, .ka, .sa], [.ta, .na, .ha], [.ma, .ya, .ra]]

    var body: some View {
        VStack(spacing: 6) {
            TextLayoutToolbar(state: model.textKeyboard, onVoice: model.compositionVoice,
                              onSelectLayout: model.selectTextLayout, onCycleLayout: model.cycleTextLayout)
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    if let error = model.conversionError {
                        Text(VoxtL10n.shared.string(error))
                            .font(.system(size: 12))
                            .accessibilityIdentifier("japaneseDictionaryError")
                    }
                    ForEach(model.composition.candidates, id: \.self) { candidate in
                        Button(candidate) { model.selectCandidate(candidate) }
                            .font(.system(size: 18))
                            .padding(.horizontal, 10)
                            .frame(height: 34)
                            .background(VoxtTheme.Colors.card, in: RoundedRectangle(cornerRadius: 6))
                            .accessibilityIdentifier("japaneseCandidate-\(candidate)")
                    }
                }
            }
            .frame(height: 34)
            ForEach(rows.indices, id: \.self) { row in
                HStack(spacing: 6) {
                    ForEach(rows[row].indices, id: \.self) { column in kana(rows[row][column]) }
                    if row == 0 { control("⌫", id: "japaneseDelete", action: { model.press(.delete) }) }
                    if row == 1 { control("空白", id: "japaneseSpace", action: { model.press(.space) }) }
                    if row == 2 { control("↵", id: "japaneseReturn", action: { model.press(.return) }) }
                }
            }
            HStack(spacing: 6) {
                control("小゛゜", id: "japaneseModifier", action: { model.press(.modifier) })
                kana(.wa)
                kana(.punctuation)
                control("123", id: "japaneseNumbers", action: model.compositionNumbers)
            }

        }
        .buttonStyle(.plain)
        .foregroundStyle(VoxtTheme.Colors.ink)
        .padding(8)
        .background(VoxtTheme.Colors.panel)
    }

    private func kana(_ key: JapaneseFlickKey) -> some View {
        let center = JapaneseFlick.character(for: key, direction: .center) ?? ""
        return ZStack {
            Text(center).font(.system(size: 20))
            VStack(spacing: 0) {
                Text(JapaneseFlick.character(for: key, direction: .up) ?? " ")
                Spacer(minLength: 0)
                Text(JapaneseFlick.character(for: key, direction: .down) ?? " ")
            }
            .font(.system(size: 8))
            .foregroundStyle(VoxtTheme.Colors.caption)
            .padding(.vertical, 2)
            HStack {
                Text(JapaneseFlick.character(for: key, direction: .left) ?? " ")
                Spacer()
                Text(JapaneseFlick.character(for: key, direction: .right) ?? " ")
            }
            .font(.system(size: 9))
            .foregroundStyle(VoxtTheme.Colors.caption)
            .padding(.horizontal, 8)
        }
        .frame(maxWidth: .infinity).frame(height: 48)
        .background(VoxtTheme.Colors.keyGray, in: RoundedRectangle(cornerRadius: 8))
        .contentShape(Rectangle())
        .gesture(DragGesture(minimumDistance: 0).onEnded { value in
            press(key, JapaneseFlick.direction(dx: value.translation.width, dy: value.translation.height))
        })
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(center)
        .accessibilityAddTraits(.isButton)
        .accessibilityIdentifier("kanaKey-\(key.rawValue)")
        .keyFeedback(.key(.character))
        .accessibilityAction { press(key, .center) }
    }

    private func press(_ key: JapaneseFlickKey, _ direction: JapaneseFlickDirection) {
        guard let kana = JapaneseFlick.character(for: key, direction: direction) else { return }
        model.press(.text(kana))
    }

    private func control(_ title: String, id: String, action: @escaping () -> Void) -> some View {
        Button(title, action: action)
            .font(.system(size: 17))
            .frame(maxWidth: .infinity).frame(height: 48)
            .background(VoxtTheme.Colors.card, in: RoundedRectangle(cornerRadius: 8))
            .accessibilityIdentifier(id)
            .keyFeedback(.key(.numbers))
    }
}
#else
/// Without ios/InputMethods.yml the Japanese layout is never offered, so this never shows.
struct JapaneseKeyboardView: View {
    let model: KeyboardModel
    var body: some View { EmptyView() }
}
#endif
