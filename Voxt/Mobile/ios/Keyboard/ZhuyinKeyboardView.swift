import SwiftUI
import TypingInput
import VoxtMobileKit

struct ZhuyinKeyboardView: View {
    let model: KeyboardModel
    private let rows = ["ㄅㄉˇˋㄓˊ˙ㄚㄞㄢㄦ", "ㄆㄊㄍㄐㄔㄗㄧㄛㄟㄣ", "ㄇㄋㄎㄑㄕㄘㄨㄜㄠㄤ", "ㄈㄌㄏㄒㄖㄙㄩㄝㄡㄥ"]

    var body: some View {
        VStack(spacing: 5) {
            TextLayoutToolbar(state: model.textKeyboard, onVoice: model.compositionVoice,
                              onSelectLayout: model.selectTextLayout, onCycleLayout: model.cycleTextLayout, identifier: "zhuyinLayoutToolbar")
            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 8) {
                    if let error = model.conversionError {
                        Text(VoxtL10n.shared.string(error)).font(.system(size: 12))
                    }
                    ForEach(model.composition.candidates, id: \.self) { candidate in
                        Button(candidate) { model.selectCandidate(candidate) }
                            .font(.system(size: 18)).padding(.horizontal, 8).frame(height: 32)
                            .background(VoxtTheme.Colors.card, in: RoundedRectangle(cornerRadius: 6))
                            .accessibilityIdentifier("zhuyinCandidate-\(candidate)")
                    }
                }
            }.frame(height: 32)
            ForEach(rows, id: \.self) { row in
                HStack(spacing: 4) {
                    ForEach(Array(row), id: \.self) { symbol in
                        Button(String(symbol)) { model.press(.text(String(symbol))) }
                            .font(.system(size: 20))
                            .frame(maxWidth: .infinity).frame(height: 40)
                            .background(VoxtTheme.Colors.keyGray, in: RoundedRectangle(cornerRadius: 6))
                            .accessibilityIdentifier("zhuyinKey-\(symbol)")
                            .keyFeedback(.key(.character))
                    }
                }
            }
            GeometryReader { geometry in
                let unit = (geometry.size.width - 15) / 5
                HStack(spacing: 5) {
                    control("123", id: "zhuyinNumbers", action: model.compositionNumbers).frame(width: unit)
                    control("空白", id: "zhuyinSpace", action: { model.press(.space) }).frame(width: unit * 2)
                    control("⌫", id: "zhuyinDelete", action: { model.press(.delete) }).frame(width: unit)
                    control("↵", id: "zhuyinReturn", action: { model.press(.return) }).frame(width: unit)
                }
            }.frame(height: 40)
        }
        .buttonStyle(.plain)
        .foregroundStyle(VoxtTheme.Colors.ink)
        .padding(8)
        .background(VoxtTheme.Colors.panel)
    }

    private func control(_ title: String, id: String, action: @escaping () -> Void) -> some View {
        Button(title, action: action)
            .font(.system(size: 17)).frame(maxWidth: .infinity).frame(height: 40)
            .background(VoxtTheme.Colors.card, in: RoundedRectangle(cornerRadius: 6))
            .accessibilityIdentifier(id)
            .keyFeedback(.key(.numbers))
    }
}
