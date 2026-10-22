import SwiftUI
import VoxtMobileKit

/// Settings › Translation: up to 3 languages the keyboard fans out above the
/// mic when it is held. Add from a searchable list, drag to reorder, swipe to remove.
struct TranslationLanguagesView: View {
    @Environment(AppModel.self) private var model
    @State private var showsAdd = false

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    private var targets: [String] { model.keyboardSettings.translationTargets }

    var body: some View {
        let _ = interfaceLanguage
        List {
            Section {
                TranslationPillPreview(targets: targets)
                    .listRowInsets(EdgeInsets())
                    .listRowBackground(Color.clear)
            } footer: {
                Text(VoxtL10n.shared.string(targets.isEmpty
                     ? "Choose up to 3 languages. On the keyboard, hold the mic and slide up to one."
                     : "On the keyboard, hold the mic and slide up to a language. Your words come out in that language."))
            }
            Section {
                ForEach(Array(targets.enumerated()), id: \.element) { index, code in
                    HStack(spacing: 14) {
                        Text("\(index + 1)")
                            .font(.system(size: 15, weight: .semibold))
                            .frame(width: 26, height: 26)
                            .background(VoxtTheme.Colors.keyGray.opacity(0.6), in: Circle())
                            .accessibilityHidden(true)
                        VStack(alignment: .leading, spacing: 2) {
                            Text(TranslationLanguages.nativeName(for: code))
                                .accessibilityIdentifier("targetName-\(code)")
                            Text(englishName(code))
                                .font(VoxtTheme.Typography.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "line.3.horizontal")
                            .foregroundStyle(.secondary)
                            .accessibilityHidden(true)
                    }
                    .padding(.vertical, 4)
                    .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                        Button(VoxtL10n.shared.string("Remove"), role: .destructive) { model.removeTranslationTarget(code) }
                    }
                }
                .onMove { model.moveTranslationTargets(fromOffsets: $0, toOffset: $1) }
                Button { showsAdd = true } label: {
                    Label(VoxtL10n.shared.string(model.keyboardSettings.isFull ? "Up to 3 languages" : "Add a language"), systemImage: "plus")
                }
                .disabled(model.keyboardSettings.isFull)
                .foregroundStyle(model.keyboardSettings.isFull ? Color.secondary : VoxtTheme.Colors.accent)
                .accessibilityIdentifier("addLanguage")
            } footer: {
                if targets.count > 1 {
                    Text(VoxtL10n.shared.string("Drag to change the order. The first one sits straight above the mic."))
                }
            }
        }
        .scrollContentBackground(.hidden)
        .background(VoxtTheme.Colors.screenBackground)
        .navigationTitle(VoxtL10n.shared.string("Translation"))
        .navigationBarTitleDisplayMode(.inline)
        .sheet(isPresented: $showsAdd) { AddLanguageSheet().environment(model) }
    }
}

/// The Translation screen on its own, as the keyboard's link opens it.
struct TranslationSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            TranslationLanguagesView()
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button(VoxtL10n.shared.string("Done")) { dismiss() }.accessibilityIdentifier("translationDone")
                    }
                }
        }
        .environment(model)
        .tint(VoxtTheme.Colors.ink)
    }
}

/// What the keyboard shows while the mic is held: the 1st language straight above it,
/// the 2nd to the left, the 3rd to the right. With none, the keyboard's own prompt.
struct TranslationPillPreview: View {
    let targets: [String]

    var body: some View {
        VStack(spacing: 18) {
            ZStack(alignment: .top) {
                if targets.isEmpty {
                    pill(VoxtL10n.shared.string("Choose translation languages"), selected: false)
                } else {
                    HStack {
                        if targets.count > 1 { pill(TranslationLanguages.nativeName(for: targets[1]), selected: false, index: 1) } else { Spacer() }
                        Spacer(minLength: 8)
                        if targets.count > 2 { pill(TranslationLanguages.nativeName(for: targets[2]), selected: false, index: 2) } else { Spacer() }
                    }
                    .padding(.top, 28)
                    pill(TranslationLanguages.nativeName(for: targets[0]), selected: true, index: 0)
                }
            }
            Image(systemName: targets.isEmpty ? "mic.fill" : "waveform")
                .font(.system(size: 26, weight: .semibold))
                .foregroundStyle(VoxtTheme.Colors.onInk)
                .frame(width: 72, height: 72)
                .background(VoxtTheme.Colors.ink, in: Circle())
                .accessibilityHidden(true)
        }
        .padding(.vertical, 22)
        .padding(.horizontal, 20)
        .frame(maxWidth: .infinity)
        .background(VoxtTheme.Colors.panel, in: RoundedRectangle(cornerRadius: 22, style: .continuous))
        .accessibilityElement(children: .contain)
    }

    private func pill(_ title: String, selected: Bool, index: Int? = nil) -> some View {
        Text(title)
            .font(.system(size: 17, weight: .semibold))
            .lineLimit(1)
            .foregroundStyle(selected ? VoxtTheme.Colors.onInk : VoxtTheme.Colors.ink)
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(selected ? VoxtTheme.Colors.ink : VoxtTheme.Colors.card, in: Capsule())
            .shadow(color: .black.opacity(selected ? 0 : 0.08), radius: 6, y: 2)
            .accessibilityIdentifier(index.map { "pill-\($0)" } ?? "pillPrompt")
    }
}

private struct AddLanguageSheet: View {
    @Environment(AppModel.self) private var model
    @Environment(\.dismiss) private var dismiss
    @State private var query = ""

    @AppStorage(VoxtL10n.preferenceKey, store: VoxtPreferences.shared) private var interfaceLanguage = ""

    private var languages: [String] {
        let needle = query.trimmingCharacters(in: .whitespaces)
        guard !needle.isEmpty else { return TranslationLanguages.catalog }
        return TranslationLanguages.catalog.filter {
            TranslationLanguages.nativeName(for: $0).localizedCaseInsensitiveContains(needle)
                || englishName($0).localizedCaseInsensitiveContains(needle)
        }
    }

    var body: some View {
        let _ = interfaceLanguage
        let settings = model.keyboardSettings
        NavigationStack {
            List {
                Section {
                    ForEach(languages, id: \.self) { code in
                        let chosen = settings.translationTargets.contains(code)
                        Button {
                            if chosen { model.removeTranslationTarget(code) } else { model.addTranslationTarget(code) }
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(TranslationLanguages.nativeName(for: code)).foregroundStyle(VoxtTheme.Colors.ink)
                                    Text(englishName(code)).font(VoxtTheme.Typography.caption).foregroundStyle(.secondary)
                                }
                                Spacer()
                                if chosen { Image(systemName: "checkmark").foregroundStyle(VoxtTheme.Colors.accent) }
                            }
                            .contentShape(Rectangle())
                        }
                        .disabled(!chosen && settings.isFull)
                        .opacity(!chosen && settings.isFull ? 0.4 : 1)
                        .accessibilityIdentifier("languageOption-\(code)")
                        .accessibilityAddTraits(chosen ? .isSelected : [])
                    }
                } header: {
                    if settings.isFull {
                        Text(VoxtL10n.shared.string("3 of 3 chosen. Remove one to add another."))
                            .font(VoxtTheme.Typography.caption)
                            .fontWeight(.regular)
                            .foregroundStyle(.secondary)
                            .textCase(nil)
                    }
                }
            }
            .searchable(text: $query, placement: .navigationBarDrawer(displayMode: .always), prompt: VoxtL10n.shared.string("Search languages"))
            .navigationTitle(VoxtL10n.shared.string("Add a language"))
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button(VoxtL10n.shared.string("Done")) { dismiss() }.accessibilityIdentifier("addLanguageDone")
                }
            }
        }
        .tint(VoxtTheme.Colors.ink)
    }
}

/// The language's name in the app's interface language ("Spanish").
private func englishName(_ code: String) -> String {
    VoxtL10n.shared.locale.localizedString(forIdentifier: code) ?? code
}
