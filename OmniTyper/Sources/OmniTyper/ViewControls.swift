// SPDX-License-Identifier: Apache-2.0
import SwiftUI

struct IconButtonStyle: ButtonStyle {
    static let hitSize: CGFloat = 32

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .frame(width: Self.hitSize, height: Self.hitSize)
            .contentShape(Rectangle())
            .background(configuration.isPressed ? Color.primary.opacity(0.08) : .clear,
                        in: RoundedRectangle(cornerRadius: 6))
    }
}

struct FullRowToggleStyle: ToggleStyle {
    func makeBody(configuration: Configuration) -> some View {
        HStack(spacing: 0) {
            Button { configuration.isOn.toggle() } label: {
                configuration.label
                    .frame(maxWidth: .infinity, minHeight: IconButtonStyle.hitSize, alignment: .leading)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityHidden(true)
            Toggle(isOn: configuration.$isOn) { configuration.label }
                .labelsHidden().toggleStyle(.switch)
        }
    }
}

struct FullRowDisclosureStyle: DisclosureGroupStyle {
    var language: String?

    func makeBody(configuration: Configuration) -> some View {
        VStack(alignment: .leading, spacing: 0) {
            Button { configuration.isExpanded.toggle() } label: {
                HStack(spacing: 8) {
                    Image(systemName: configuration.isExpanded ? "chevron.down" : "chevron.right")
                        .font(.caption.weight(.semibold)).foregroundStyle(.secondary).frame(width: 12)
                    configuration.label
                    Spacer(minLength: 0)
                }
                .frame(minHeight: IconButtonStyle.hitSize)
                .contentShape(Rectangle())
            }
            .buttonStyle(.plain)
            .accessibilityValue(L10n.string(configuration.isExpanded ? "control.expanded" : "control.collapsed", in: language))
            if configuration.isExpanded {
                configuration.content.padding(.leading, 20)
            }
        }
    }
}
