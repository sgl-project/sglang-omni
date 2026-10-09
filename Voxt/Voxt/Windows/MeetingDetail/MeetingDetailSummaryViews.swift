// MeetingDetailSummaryViews.swift
// Provides Meeting Detail Summary Views for meeting detail windows.

import SwiftUI

private struct MeetingSummaryBottomVisibilityPreferenceKey: PreferenceKey {
    static var defaultValue = true

    static func reduce(value: inout Bool, nextValue: () -> Bool) {
        value = nextValue()
    }
}

struct MeetingDetailSummarySidebar: View {
    @ObservedObject var viewModel: MeetingDetailViewModel
    @State private var isScrolledToSummaryBottom = true
    @State private var hasUnreadSummaryMessages = false

    private let summaryBottomAnchorID = "meeting-summary-bottom-anchor"

    var body: some View {
        VStack(spacing: 12) {
            summaryHeader
            summaryBodyPane
            summaryComposerPane
        }
    }

    private var summaryHeader: some View {
        HStack(alignment: .top, spacing: 10) {
            VStack(alignment: .leading, spacing: 6) {
                Text(AppLocalization.localizedString("Meeting Summary"))
                    .font(.system(size: 16, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(summaryStatusTitle)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(viewModel.isSummaryStale ? Color.orange : Color.secondary)
            }

            Spacer(minLength: 8)

            Button(AppLocalization.localizedString("Settings")) {
                viewModel.presentSummarySettings()
            }
            .buttonStyle(MeetingToolbarButtonStyle())

            if viewModel.isSummaryStale, viewModel.canRegenerateSummary {
                Button(AppLocalization.localizedString("Regenerate")) {
                    viewModel.regenerateSummary()
                }
                .buttonStyle(MeetingToolbarButtonStyle(isActive: true))
            }
        }
    }

    private var summaryStatusTitle: String {
        if viewModel.isSummaryStale {
            return AppLocalization.localizedString("Transcript changed · summary needs regeneration")
        }
        if viewModel.isFinalizing {
            return AppLocalization.localizedString("Preparing saved meeting…")
        }
        if viewModel.summaryState == .loading {
            return AppLocalization.localizedString("Generating meeting summary…")
        }
        if viewModel.summary != nil {
            return AppLocalization.localizedString("Saved summary")
        }
        return AppLocalization.localizedString("Generated after the detail view loads")
    }

    private var summaryBodyPane: some View {
        GeometryReader { outerProxy in
            ScrollViewReader { proxy in
                ZStack(alignment: .bottomTrailing) {
                    ScrollView {
                        VStack(alignment: .leading, spacing: 14) {
                            if let summary = viewModel.summary {
                                if viewModel.summaryState == .loading {
                                    HStack(spacing: 10) {
                                        ProgressView()
                                            .controlSize(.small)

                                        Text(AppLocalization.localizedString("Generating meeting summary…"))
                                            .font(.system(size: 12, weight: .semibold))
                                            .foregroundStyle(.primary)
                                    }
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 10)
                                    .background(
                                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                                            .fill(Color.accentColor.opacity(0.08))
                                    )
                                    .overlay(
                                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                                            .strokeBorder(Color.accentColor.opacity(0.16), lineWidth: 1)
                                    )
                                }

                                VStack(alignment: .leading, spacing: 10) {
                                    Text(summary.title)
                                        .font(.system(size: 16, weight: .semibold))
                                        .foregroundStyle(.primary)
                                        .textSelection(.enabled)

                                    Text(summary.generatedAt.formatted(date: .abbreviated, time: .shortened))
                                        .font(.system(size: 11, weight: .medium))
                                        .foregroundStyle(.secondary)
                                        .textSelection(.enabled)
                                }

                                if !summary.body.isEmpty {
                                    VStack(alignment: .leading, spacing: 10) {
                                        HStack(spacing: 8) {
                                            Text(AppLocalization.localizedString("Summary"))
                                                .font(.system(size: 12, weight: .semibold))
                                                .foregroundStyle(.secondary)

                                            Spacer(minLength: 8)

                                            MeetingDetailCopyButton(text: summary.body)
                                        }

                                        ForEach(MeetingDetailFormatting.summaryParagraphs(summary.body), id: \.self) { paragraph in
                                            Text(paragraph)
                                                .font(.system(size: 13, weight: .medium))
                                                .foregroundStyle(.primary.opacity(0.92))
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                                .textSelection(.enabled)
                                        }
                                    }
                                    .textSelection(.enabled)
                                }

                                if !summary.todoItems.isEmpty {
                                    VStack(alignment: .leading, spacing: 10) {
                                        Text(AppLocalization.localizedString("TODO"))
                                            .font(.system(size: 12, weight: .semibold))
                                            .foregroundStyle(.secondary)

                                        ForEach(Array(summary.todoItems.enumerated()), id: \.offset) { index, item in
                                            HStack(alignment: .top, spacing: 10) {
                                                Text(String(index + 1))
                                                    .font(.system(size: 11, weight: .bold))
                                                    .foregroundStyle(.secondary)
                                                    .frame(width: 18, height: 18)
                                                    .background(
                                                        Circle()
                                                            .fill(MeetingDetailUIStyle.mutedFillColor)
                                                    )

                                                Text(item)
                                                    .font(.system(size: 13, weight: .medium))
                                                    .foregroundStyle(.primary.opacity(0.92))
                                                    .frame(maxWidth: .infinity, alignment: .leading)
                                                    .textSelection(.enabled)
                                            }
                                        }
                                    }
                                    .textSelection(.enabled)
                                }

                                if !viewModel.summaryChatMessages.isEmpty || viewModel.isSummaryChatLoading || viewModel.summaryChatErrorMessage != nil {
                                    VStack(alignment: .leading, spacing: 12) {
                                        Text(AppLocalization.localizedString("Follow-up"))
                                            .font(.system(size: 12, weight: .semibold))
                                            .foregroundStyle(.secondary)

                                        ForEach(viewModel.summaryChatMessages) { message in
                                            SummaryChatMessageRow(message: message)
                                        }

                                        if viewModel.isSummaryChatLoading {
                                            HStack(spacing: 10) {
                                                ProgressView()
                                                    .controlSize(.small)

                                                Text(AppLocalization.localizedString("Thinking…"))
                                                    .font(.system(size: 12, weight: .medium))
                                                    .foregroundStyle(.secondary)
                                            }
                                        }

                                        if let errorMessage = viewModel.summaryChatErrorMessage,
                                           !errorMessage.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                            Text(errorMessage)
                                                .font(.system(size: 12, weight: .medium))
                                                .foregroundStyle(.red.opacity(0.9))
                                                .frame(maxWidth: .infinity, alignment: .leading)
                                        }
                                    }
                                }
                            } else {
                                summaryStateView
                            }

                            GeometryReader { geo in
                                Color.clear
                                    .preference(
                                        key: MeetingSummaryBottomVisibilityPreferenceKey.self,
                                        value: abs(geo.frame(in: .named("MeetingSummaryScroll")).maxY - outerProxy.size.height) < 36
                                    )
                            }
                            .frame(height: 1)
                            .id(summaryBottomAnchorID)
                        }
                        .padding(16)
                        .textSelection(.enabled)
                    }
                    .coordinateSpace(name: "MeetingSummaryScroll")
                    .onPreferenceChange(MeetingSummaryBottomVisibilityPreferenceKey.self) { isVisible in
                        isScrolledToSummaryBottom = isVisible
                        if isVisible {
                            hasUnreadSummaryMessages = false
                        }
                    }
                    .onChange(of: viewModel.summaryChatMessages.count) { oldValue, newValue in
                        guard newValue > oldValue else { return }
                        handleSummaryMessagesUpdate(using: proxy)
                    }

                    if hasUnreadSummaryMessages {
                        Button {
                            hasUnreadSummaryMessages = false
                            scrollSummaryToBottom(using: proxy)
                        } label: {
                            HStack(spacing: 6) {
                                Image(systemName: "arrow.down")
                                    .font(.system(size: 10, weight: .semibold))
                                Text(AppLocalization.localizedString("New Message"))
                                    .font(.system(size: 12, weight: .semibold))
                            }
                            .foregroundStyle(.white.opacity(0.92))
                            .padding(.horizontal, 10)
                            .padding(.vertical, 7)
                            .background(
                                Capsule(style: .continuous)
                                    .fill(.black.opacity(0.78))
                            )
                            .overlay(
                                Capsule(style: .continuous)
                                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
                            )
                        }
                        .buttonStyle(.plain)
                        .padding(.trailing, 8)
                        .padding(.bottom, 8)
                    }
                }
            }
        }
        .meetingDetailPanelSurface(cornerRadius: 14)
    }

    private func handleSummaryMessagesUpdate(using proxy: ScrollViewProxy) {
        if isScrolledToSummaryBottom {
            scrollSummaryToBottom(using: proxy)
        } else {
            hasUnreadSummaryMessages = true
        }
    }

    private func scrollSummaryToBottom(using proxy: ScrollViewProxy) {
        DispatchQueue.main.async {
            withAnimation(.easeOut(duration: 0.18)) {
                proxy.scrollTo(summaryBottomAnchorID, anchor: .bottom)
            }
        }
    }

    @ViewBuilder
    private var summaryStateView: some View {
        switch viewModel.summaryState {
        case .loading:
            VStack(alignment: .leading, spacing: 14) {
                ProgressView()
                    .controlSize(.small)

                Text(AppLocalization.localizedString("Generating meeting summary…"))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(viewModel.summaryProviderMessage)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, minHeight: 240, alignment: .topLeading)

        case .unavailable(let message):
            summaryEmptyState(
                icon: "sparkles",
                title: AppLocalization.localizedString("Summary Unavailable"),
                message: message
            )

        case .failed(let message):
            VStack(alignment: .leading, spacing: 12) {
                summaryEmptyState(
                    icon: "exclamationmark.triangle",
                    title: AppLocalization.localizedString("Summary Failed"),
                    message: message
                )

                if viewModel.canRegenerateSummary {
                    Button(AppLocalization.localizedString("Try Again")) {
                        viewModel.regenerateSummary()
                    }
                    .buttonStyle(MeetingPillButtonStyle())
                }
            }

        case .idle:
            if viewModel.isFinalizing {
                summaryEmptyState(
                    icon: "clock.arrow.circlepath",
                    title: AppLocalization.localizedString("Preparing Summary"),
                    message: AppLocalization.localizedString("Voxt will start summary generation after the final meeting record is saved.")
                )
            } else if viewModel.mode == .live {
                summaryEmptyState(
                    icon: "clock.arrow.circlepath",
                    title: AppLocalization.localizedString("Waiting For Saved Record"),
                    message: AppLocalization.localizedString("Summary generation starts after this meeting is saved to history.")
                )
            } else if !viewModel.summaryAutoGenerate {
                summaryEmptyState(
                    icon: "sparkles",
                    title: AppLocalization.localizedString("Auto Summary Disabled"),
                    message: AppLocalization.localizedString("Enable automatic summary generation or use the settings dialog to regenerate manually.")
                )
            } else {
                summaryEmptyState(
                    icon: "doc.text.magnifyingglass",
                    title: AppLocalization.localizedString("No Summary Yet"),
                    message: AppLocalization.localizedString("Open the settings dialog to trigger a manual summary generation.")
                )
            }
        }
    }

    private func summaryEmptyState(icon: String, title: String, message: String) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Image(systemName: icon)
                .font(.system(size: 18, weight: .semibold))
                .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 8) {
                Text(title)
                    .font(.system(size: 14, weight: .semibold))
                    .foregroundStyle(.primary)

                Text(message)
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity, minHeight: 240, alignment: .topLeading)
    }

    private var summaryComposerPane: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(AppLocalization.localizedString("Follow-up Input"))
                .font(.system(size: 12, weight: .semibold))
                .foregroundStyle(.secondary)

            HStack(spacing: 10) {
                TextField(AppLocalization.localizedString("Ask a follow-up about this meeting"), text: $viewModel.summaryChatDraft)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .background(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(MeetingDetailUIStyle.controlFillColor)
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .strokeBorder(MeetingDetailUIStyle.borderColor, lineWidth: 1)
                    )
                    .disabled(viewModel.mode != .history || viewModel.summary == nil || !viewModel.hasSummaryModelOptions || viewModel.segments.isEmpty || viewModel.isSummaryChatLoading)
                    .onSubmit {
                        viewModel.sendSummaryChat()
                    }

                MeetingDetailFollowUpSendButton(
                    action: { viewModel.sendSummaryChat() },
                    isDisabled: !viewModel.canSendSummaryChat
                )
            }

        }
        .padding(14)
        .meetingDetailPanelSurface(cornerRadius: 14)
    }
}

struct MeetingDetailSummarySettingsDialog: View {
    @ObservedObject var viewModel: MeetingDetailViewModel
    @State private var isSummaryModelSelectorPresented = false

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 12) {
                Text(AppLocalization.localizedString("Summary Settings"))
                    .font(.title3.weight(.semibold))

                Spacer(minLength: 8)
            }

            HStack(alignment: .center, spacing: 12) {
                Text(AppLocalization.localizedString("Auto-generate Summary"))
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.primary)

                Spacer(minLength: 8)

                Toggle(
                    "",
                    isOn: Binding(
                        get: { viewModel.summaryAutoGenerate },
                        set: { viewModel.setSummaryAutoGenerate($0) }
                    )
                )
                .labelsHidden()
                .toggleStyle(.switch)
            }

            if viewModel.hasSummaryModelOptions {
                FeatureSelectorRow(
                    title: AppLocalization.localizedString("Summary Model"),
                    value: summaryModelSelectionTitle,
                    action: { isSummaryModelSelectorPresented = true }
                )
                .sheet(isPresented: $isSummaryModelSelectorPresented) {
                    FeatureModelSelectorDialog(
                        title: AppLocalization.localizedString("Choose Meeting Summary Model"),
                        entries: summaryModelSelectorEntries,
                        selectedID: summaryModelSelectorSelectionID,
                        onSelect: { selectionID in
                            viewModel.setSummaryModelSelectionID(
                                MeetingSummaryModelSelectorAdapter.summaryModelID(for: selectionID)
                            )
                        }
                    )
                }
            } else {
                Text(AppLocalization.localizedString("No summary model is available right now."))
                    .font(.system(size: 12, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            VStack(alignment: .leading, spacing: 7) {
                HStack(alignment: .center, spacing: 10) {
                    Text(AppLocalization.localizedString("Summary Prompt"))
                        .font(.system(size: 12, weight: .semibold))
                        .foregroundStyle(.secondary)

                    Spacer(minLength: 8)

                    Button(AppLocalization.localizedString("Reset")) {
                        viewModel.resetSummaryPromptTemplate()
                    }
                    .buttonStyle(SettingsPillButtonStyle())
                }

                PromptEditorView(
                    text: Binding(
                        get: { viewModel.summaryPromptTemplate },
                        set: { viewModel.setSummaryPromptTemplate($0) }
                    ),
                    height: 112,
                    contentPadding: 10,
                    variables: summaryPromptVariables,
                    variablesLayout: .twoColumns,
                    variablesTitle: AppLocalization.localizedString("Available variables")
                )

                VStack(alignment: .leading, spacing: 5) {
                    Text(AppLocalization.localizedString("Expected result: return JSON only with transcript_summary.title, transcript_summary.content, and todo_list. Do not wrap it in markdown or add extra keys."))
                        .font(.system(size: 11, weight: .medium))
                        .foregroundStyle(.secondary)
                }
                .fixedSize(horizontal: false, vertical: true)

                Text(AppLocalization.localizedString("Add custom summary instructions, tone constraints, or output emphasis here."))
                    .font(.system(size: 11, weight: .medium))
                    .foregroundStyle(.secondary)
            }

            SettingsDialogActionRow {
                Button(AppLocalization.localizedString("Cancel")) {
                    viewModel.isSummarySettingsPresented = false
                }
                .buttonStyle(SettingsPillButtonStyle())
                .keyboardShortcut(.cancelAction)

                Button(AppLocalization.localizedString("Regenerate Summary")) {
                    viewModel.isSummarySettingsPresented = false
                    viewModel.regenerateSummary()
                }
                .buttonStyle(SettingsPrimaryButtonStyle())
                .disabled(!viewModel.canRegenerateSummary || !viewModel.hasSummaryModelOptions)
                .keyboardShortcut(.defaultAction)
            }
        }
        .settingsDialogChrome(
            width: 520,
            maxHeight: 540,
            onClose: { viewModel.isSummarySettingsPresented = false }
        )
    }

    private var summaryModelSelectionTitle: String {
        let selectedID = viewModel.resolvedSummaryModelSelectionID
        guard let option = viewModel.summaryModelOptions.first(where: { $0.id == selectedID }) else {
            return AppLocalization.localizedString("No summary model is available right now.")
        }
        return "\(option.title) · \(option.subtitle)"
    }

    private var summaryModelSelectorSelectionID: FeatureModelSelectionID {
        MeetingSummaryModelSelectorAdapter.selectionID(forSummaryModelID: viewModel.resolvedSummaryModelSelectionID)
    }

    private var summaryModelSelectorEntries: [FeatureModelSelectorEntry] {
        MeetingSummaryModelSelectorAdapter.entries(
            for: viewModel.summaryModelOptions,
            selectedSummaryModelID: viewModel.resolvedSummaryModelSelectionID
        )
    }

    private var summaryPromptVariables: [PromptTemplateVariableDescriptor] {
        TranscriptSummarySupport.promptTemplateVariables.map {
            PromptTemplateVariableDescriptor(token: $0, tipKey: "Template tip \($0)")
        }
    }
}

private enum MeetingSummaryModelSelectorAdapter {
    private static func localized(_ key: String) -> String {
        AppLocalization.localizedString(key)
    }

    static func selectionID(forSummaryModelID summaryModelID: String) -> FeatureModelSelectionID {
        let trimmedID = summaryModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedID.hasPrefix("custom-llm:") {
            return .localLLM(String(trimmedID.dropFirst("custom-llm:".count)))
        }
        return FeatureModelSelectionID(rawValue: trimmedID)
    }

    static func summaryModelID(for selectionID: FeatureModelSelectionID) -> String {
        switch selectionID.textSelection {
        case .appleIntelligence:
            return FeatureModelSelectionID.appleIntelligence.rawValue
        case .localLLM(let repo):
            return "custom-llm:\(repo)"
        case .remoteLLM:
            return selectionID.rawValue
        case .none:
            return selectionID.rawValue
        }
    }

    static func entries(
        for options: [MeetingSummaryModelOption],
        selectedSummaryModelID: String
    ) -> [FeatureModelSelectorEntry] {
        options.map { option in
            let selectionID = selectionID(forSummaryModelID: option.id)
            let isSelected = option.id == selectedSummaryModelID
            let classification = classification(for: selectionID)
            let inUseTags = isSelected ? [localized("In Use")] : []

            return FeatureModelSelectorEntry(
                selectionID: selectionID,
                title: option.title,
                engine: classification.engine,
                sizeText: classification.sizeText,
                ratingText: classification.ratingText,
                filterTags: classification.filterTags + inUseTags,
                displayTags: classification.displayTags + inUseTags,
                statusText: option.subtitle,
                usageLocations: [localized("Meeting")],
                badgeText: nil,
                isSelectable: true,
                disabledReason: nil
            )
        }
    }

    private static func classification(for selectionID: FeatureModelSelectionID) -> ModelClassification {
        switch selectionID.textSelection {
        case .appleIntelligence:
            return ModelClassification(
                engine: localized("Apple"),
                sizeText: localized("Built-in"),
                ratingText: "4.2",
                filterTags: [localized("Local"), localized("Multilingual"), localized("Installed")],
                displayTags: [localized("Local"), localized("Multilingual"), localized("Installed")]
            )
        case .localLLM:
            return ModelClassification(
                engine: localized("Local LLM"),
                sizeText: localized("Installed"),
                ratingText: "4.5",
                filterTags: [localized("Local"), localized("Installed")],
                displayTags: [localized("Local"), localized("Installed")]
            )
        case .remoteLLM:
            return ModelClassification(
                engine: localized("Remote LLM"),
                sizeText: localized("Cloud"),
                ratingText: "4.5",
                filterTags: [localized("Remote"), localized("Configured")],
                displayTags: [localized("Remote"), localized("Configured")]
            )
        case .none:
            return ModelClassification(
                engine: "",
                sizeText: "",
                ratingText: "",
                filterTags: [],
                displayTags: []
            )
        }
    }

    private struct ModelClassification {
        let engine: String
        let sizeText: String
        let ratingText: String
        let filterTags: [String]
        let displayTags: [String]
    }
}

struct SummaryChatMessageRow: View {
    let message: MeetingSummaryChatMessage

    @State private var isHovered = false

    private var isAssistantMessage: Bool {
        message.role == .assistant
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(message.role == .user ? AppLocalization.localizedString("You") : AppLocalization.localizedString("Assistant"))
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)

            Text(message.content)
                .font(.system(size: 13, weight: .medium))
                .foregroundStyle(.primary.opacity(0.94))
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
        }
        .padding(12)
        .background(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .fill(message.role == .user ? Color.accentColor.opacity(0.08) : MeetingDetailUIStyle.mutedFillColor)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14, style: .continuous)
                .strokeBorder(
                    message.role == .user ? Color.accentColor.opacity(0.18) : MeetingDetailUIStyle.softBorderColor,
                    lineWidth: 1
                )
        )
        .overlay(alignment: .topTrailing) {
            if isAssistantMessage && isHovered {
                MeetingDetailHoverCopyCapsule(text: message.content)
                    .padding(.top, 8)
                    .padding(.trailing, 8)
                    .transition(.opacity.combined(with: .scale(scale: 0.94, anchor: .topTrailing)))
            }
        }
        .onHover { hovering in
            guard isAssistantMessage else { return }
            withAnimation(.easeOut(duration: 0.12)) {
                isHovered = hovering
            }
        }
    }
}

private struct MeetingDetailCopyButton: View {
    let text: String

    @State private var didCopy = false
    @State private var copyFeedbackToken = UUID()

    var body: some View {
        Button(action: copyToPasteboard) {
            Group {
                if didCopy {
                    CopySuccessIconView(color: Color.accentColor)
                } else {
                    CopyIconView(color: .secondary)
                }
            }
            .frame(width: 13, height: 13)
            .frame(width: 22, height: 22)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .help(AppLocalization.localizedString("Copy"))
        .accessibilityLabel(AppLocalization.localizedString("Copy"))
    }

    private func copyToPasteboard() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        copyStringToPasteboard(trimmed)
        copyFeedbackToken = UUID()
        let token = copyFeedbackToken
        didCopy = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard token == copyFeedbackToken else { return }
            didCopy = false
        }
    }
}

private struct MeetingDetailHoverCopyCapsule: View {
    let text: String

    @State private var didCopy = false
    @State private var copyFeedbackToken = UUID()

    var body: some View {
        Button(action: copyToPasteboard) {
            HStack(spacing: 4) {
                Group {
                    if didCopy {
                        CopySuccessIconView()
                    } else {
                        CopyIconView()
                    }
                }
                .frame(width: 12, height: 12)

                Text(didCopy ? AppLocalization.localizedString("Copied") : AppLocalization.localizedString("Copy"))
                    .font(.system(size: 10, weight: .semibold))
            }
            .foregroundStyle(.white.opacity(0.96))
            .padding(.horizontal, 8)
            .frame(height: 22)
            .background(
                Capsule(style: .continuous)
                    .fill(.black.opacity(0.74))
            )
            .overlay(
                Capsule(style: .continuous)
                    .strokeBorder(.white.opacity(0.12), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .help(AppLocalization.localizedString("Copy"))
        .accessibilityLabel(AppLocalization.localizedString("Copy"))
    }

    private func copyToPasteboard() {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        copyStringToPasteboard(trimmed)
        copyFeedbackToken = UUID()
        let token = copyFeedbackToken
        didCopy = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 1.2) {
            guard token == copyFeedbackToken else { return }
            didCopy = false
        }
    }
}
