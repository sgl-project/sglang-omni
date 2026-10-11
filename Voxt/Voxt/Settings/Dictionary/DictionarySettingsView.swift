// DictionarySettingsView.swift
// Provides Dictionary Settings View for dictionary settings.

import SwiftUI
import Combine
import AppKit
import UniformTypeIdentifiers

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct DictionarySettingsView: View {
    @AppStorage(AppPreferenceKey.dictionaryAutoLearningEnabled) private var dictionaryAutoLearningEnabled = true
    @AppStorage(AppPreferenceKey.dictionaryAutoLearningPrompt) private var storedAutomaticLearningPrompt = ""
    @AppStorage(AppPreferenceKey.dictionaryHighConfidenceCorrectionEnabled) private var dictionaryHighConfidenceCorrectionEnabled = true
    @AppStorage(AppPreferenceKey.dictionarySuggestionIngestModelOptionID) private var preferredHistoryScanModelID = ""

    @ObservedObject var historyStore: TranscriptionHistoryStore
    @ObservedObject var dictionaryStore: DictionaryStore
    @ObservedObject var dictionarySuggestionStore: DictionarySuggestionStore
    let availableHistoryScanModels: () -> [DictionaryHistoryScanModelOption]
    let onIngestSuggestionsFromHistory: (DictionaryHistoryScanRequest, Bool) -> Void
    let onCancelIngestSuggestionsFromHistory: () -> Void
    let navigationRequest: SettingsNavigationRequest?

    @State private var selectedTab: DictionaryEntriesTab = .hotwords
    @State private var selectedHotwordCategoryID: UUID?
    @State private var dialog: DictionaryDialog?
    @State private var categoryDialog: DictionaryCategoryDialog?
    @State private var pendingDeleteCategory: DictionaryCategory?
    @State private var showDictionaryAdvancedSettings = false
    @State private var showDictionaryIngestDialog = false
    @State private var suggestionFilterDraft = DictionarySuggestionFilterSettings.defaultValue
    @State private var automaticLearningPromptDraft = AppPromptDefaults.text(for: .dictionaryAutoLearning)
    @State private var historyScanModelOptions: [DictionaryHistoryScanModelOption] = []
    @State private var selectedHistoryScanModelID = ""
    @State private var dictionaryToastMessage = ""
    @State private var dictionaryToastDismissTask: Task<Void, Never>?
    @State private var suggestionActionMessage: String?
    @State private var pendingHistoryScanCount = 0
    @State private var dictionarySearchText = ""
    @State private var showDictionarySearchDialog = false
    @State private var visibleReplacementEntries: [DictionaryEntry] = []
    @State private var totalReplacementEntryCount = 0
    @State private var isLoadingReplacementEntries = false
    @State private var replacementEntryPageGeneration = 0

    private let entryPageSize = 80

    private var localHistoryScanModelOptions: [DictionaryHistoryScanModelOption] {
        historyScanModelOptions.filter { $0.source == .local }
    }

    private var remoteHistoryScanModelOptions: [DictionaryHistoryScanModelOption] {
        historyScanModelOptions.filter { $0.source == .remote }
    }

    private var selectedHistoryScanModelOption: DictionaryHistoryScanModelOption? {
        historyScanModelOptions.first(where: { $0.id == selectedHistoryScanModelID })
    }

    private var appBranchGroups: [AppBranchGroup] {
        guard let data = UserDefaults.standard.data(forKey: AppPreferenceKey.appBranchGroups),
              let groups = try? JSONDecoder().decode([AppBranchGroup].self, from: data)
        else {
            return []
        }
        return groups.sorted { $0.name.localizedCaseInsensitiveCompare($1.name) == .orderedAscending }
    }

    private var historyScanProgress: DictionaryHistoryScanProgress {
        dictionarySuggestionStore.historyScanProgress
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            dictionaryListCard
                .settingsNavigationAnchor(.dictionaryEntries)
                .frame(maxHeight: .infinity, alignment: .top)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        .overlay(alignment: .top) {
            if !dictionaryToastMessage.isEmpty {
                ModelDebugToast(message: dictionaryToastMessage) {
                    dismissDictionaryToast()
                }
                .padding(.top, 12)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .animation(.easeInOut(duration: 0.16), value: dictionaryToastMessage)
        .sheet(item: $dialog) { currentDialog in
            dialogView(for: currentDialog)
        }
        .sheet(item: $categoryDialog) { currentDialog in
            categoryDialogView(for: currentDialog)
        }
        .sheet(isPresented: $showDictionaryAdvancedSettings) {
            DictionaryAdvancedSettingsDialog(
                dictionaryAutoLearningEnabled: $dictionaryAutoLearningEnabled,
                automaticLearningPromptDraft: $automaticLearningPromptDraft,
                dictionaryHighConfidenceCorrectionEnabled: $dictionaryHighConfidenceCorrectionEnabled,
                isPresented: $showDictionaryAdvancedSettings,
                onRestoreDefaultAutomaticLearningPrompt: restoreAutomaticLearningPromptToDefault,
                onSave: saveDictionaryAdvancedSettings
            )
        }
        .sheet(isPresented: $showDictionaryIngestDialog) {
            DictionaryOneClickIngestDialog(
                isPresented: $showDictionaryIngestDialog,
                pendingHistoryScanCount: pendingHistoryScanCount,
                localModelOptions: localHistoryScanModelOptions,
                remoteModelOptions: remoteHistoryScanModelOptions,
                selectedModelID: $selectedHistoryScanModelID,
                draftPrompt: $suggestionFilterDraft.prompt,
                historyScanProgress: historyScanProgress,
                statusText: historyScanStatusText,
                cancellationText: historyScanCancellationText,
                actionMessage: suggestionActionMessage,
                onRestoreDefaultPrompt: restoreSuggestionIngestPromptToDefault,
                onSave: saveSuggestionIngestSettings,
                onStart: startSuggestionIngestFromDialog,
                onCancelRunning: requestSuggestionIngestCancellation
            )
        }
        .sheet(isPresented: $showDictionarySearchDialog) {
            SettingsSearchDialog(
                title: localized("Search Dictionary"),
                placeholder: localized("Search dictionary terms, aliases, or groups"),
                query: $dictionarySearchText,
                isPresented: $showDictionarySearchDialog
            )
        }
        .onAppear(perform: reloadContentAsync)
        .onChange(of: selectedTab) { _, newValue in
            if newValue == .replacements {
                selectedHotwordCategoryID = nil
                reloadReplacementEntries(reset: true)
            }
        }
        .onChange(of: dictionarySearchText) { _, _ in
            reloadReplacementEntries(reset: true)
        }
        .onReceive(dictionaryStore.$entries) { _ in
            guard selectedTab == .replacements else {
                return
            }
            reloadReplacementEntries(reset: true)
        }
        .alert(
            localized("Delete Dictionary Category?"),
            isPresented: Binding(
                get: { pendingDeleteCategory != nil },
                set: { if !$0 { pendingDeleteCategory = nil } }
            )
        ) {
            Button(localized("Move Terms to Default"), role: .destructive) {
                if let pendingDeleteCategory {
                    dictionaryStore.deleteCategory(id: pendingDeleteCategory.id, deleteEntries: false)
                }
                pendingDeleteCategory = nil
                reloadReplacementEntries(reset: true)
            }
            Button(localized("Delete Terms Too"), role: .destructive) {
                if let pendingDeleteCategory {
                    dictionaryStore.deleteCategory(id: pendingDeleteCategory.id, deleteEntries: true)
                }
                pendingDeleteCategory = nil
                reloadReplacementEntries(reset: true)
            }
            Button(localized("Cancel"), role: .cancel) {
                pendingDeleteCategory = nil
            }
        } message: {
            Text(localized("You can move this category's terms to the default category or delete them together."))
        }
    }

    private func scrollToNavigationTargetIfNeeded(using proxy: ScrollViewProxy) {
        guard let navigationRequest,
              navigationRequest.target.tab == .dictionary,
              let section = navigationRequest.target.section
        else {
            return
        }

        DispatchQueue.main.async {
            withAnimation(.easeInOut(duration: 0.18)) {
                proxy.scrollTo(section.rawValue, anchor: .top)
            }
        }
    }

    private func refreshPendingHistoryScanCountAsync() {
        let checkpoint = dictionarySuggestionStore.historyScanCheckpoint
        pendingHistoryScanCount = historyStore.pendingDictionaryHistoryEntryCount(after: checkpoint)
    }

    private var dictionaryListCard: some View {
        DictionaryEntriesCard(
            selectedTab: $selectedTab,
            selectedHotwordCategoryID: $selectedHotwordCategoryID,
            hotwordSections: dictionaryStore.hotwordEntriesByCategory(query: dictionarySearchText),
            replacementEntries: visibleReplacementEntries,
            searchText: dictionarySearchText,
            isLoadingEntries: selectedTab == .replacements && isLoadingReplacementEntries,
            onSearch: { showDictionarySearchDialog = true },
            onClearSearch: { dictionarySearchText = "" },
            onCreate: createDictionaryEntry,
            onCreateCategory: { categoryDialog = .create },
            onOpenIngest: openDictionaryIngestDialog,
            onOpenSettings: openDictionaryAdvancedSettings,
            onImport: importDictionary,
            onExport: exportDictionary,
            onCreateInCategory: { category in dialog = .create(categoryID: category.id, mode: .hotword) },
            onEditCategory: { category in categoryDialog = .edit(category) },
            onDeleteCategory: { category in pendingDeleteCategory = category },
            onEdit: { entry in dialog = .edit(entry) },
            onDelete: deleteDictionaryEntry
        )
    }

    @ViewBuilder
    private func dialogView(for dialog: DictionaryDialog) -> some View {
        DictionaryTermDialogView(
            dialog: dialog,
            availableCategories: dictionaryStore.categories,
            availableGroups: appBranchGroups,
            onCancel: {
                self.dialog = nil
            },
            onSave: { terms, replacementTerms, selectedCategoryID, selectedGroupID in
                try save(
                    dialog: dialog,
                    terms: terms,
                    replacementTerms: replacementTerms,
                    selectedCategoryID: selectedCategoryID,
                    selectedGroupID: selectedGroupID
                )
                self.dialog = nil
            }
        )
    }

    @ViewBuilder
    private func categoryDialogView(for dialog: DictionaryCategoryDialog) -> some View {
        DictionaryCategoryDialogView(
            dialog: dialog,
            onCancel: {
                categoryDialog = nil
            },
            onSave: { name in
                switch dialog {
                case .create:
                    _ = try dictionaryStore.createCategory(name: name)
                case .edit(let category):
                    try dictionaryStore.updateCategory(id: category.id, name: name)
                }
                categoryDialog = nil
            }
        )
    }

    private func save(
        dialog: DictionaryDialog,
        terms: [String],
        replacementTerms: [String],
        selectedCategoryID: UUID,
        selectedGroupID: UUID?
    ) throws {
        let selectedGroupName = groupNameSnapshot(
            for: selectedGroupID,
            in: dialog
        )
        switch dialog {
        case .create:
            for term in terms {
                try dictionaryStore.createManualEntry(
                    term: term,
                    replacementTerms: dialog.mode == .replacement ? replacementTerms : [],
                    categoryID: selectedCategoryID,
                    categoryNameSnapshot: dictionaryStore.categoryName(for: selectedCategoryID),
                    groupID: dialog.mode == .replacement ? selectedGroupID : nil,
                    groupNameSnapshot: dialog.mode == .replacement ? selectedGroupName : nil
                )
            }
        case .edit(let entry):
            guard let term = terms.first else {
                throw DictionaryStoreError.emptyTerm
            }
            let dialogMode = entry.replacementTerms.isEmpty ? DictionaryTermDialogMode.hotword : .replacement
            try dictionaryStore.updateEntry(
                id: entry.id,
                term: term,
                replacementTerms: dialogMode == .replacement ? replacementTerms : [],
                categoryID: selectedCategoryID,
                categoryNameSnapshot: dictionaryStore.categoryName(for: selectedCategoryID),
                groupID: dialogMode == .replacement ? selectedGroupID : nil,
                groupNameSnapshot: dialogMode == .replacement ? selectedGroupName : nil
            )
        }
        refreshDictionaryEntriesAfterMutation()
    }

    private func reloadContentAsync() {
        dictionarySuggestionStore.reloadAsync()
        refreshLocalContentState()
        reloadReplacementEntries(reset: true)
    }

    private func reloadReplacementEntries(reset: Bool) {
        let offset = reset ? 0 : visibleReplacementEntries.count
        guard reset || offset < totalReplacementEntryCount else { return }
        guard reset || !isLoadingReplacementEntries else { return }

        loadReplacementEntries(offset: offset, limit: entryPageSize, reset: reset)
    }

    private func loadReplacementEntries(offset: Int, limit: Int, reset: Bool) {
        replacementEntryPageGeneration += 1
        let generation = replacementEntryPageGeneration
        let query = dictionarySearchText
        isLoadingReplacementEntries = true

        dictionaryStore.loadEntries(
            requiringReplacementTerms: true,
            query: query,
            limit: limit,
            offset: offset
        ) { count, page in
            guard generation == replacementEntryPageGeneration else { return }
            totalReplacementEntryCount = count
            visibleReplacementEntries = reset ? page : visibleReplacementEntries + page
            isLoadingReplacementEntries = false
        }
    }

    private func createDictionaryEntry() {
        switch selectedTab {
        case .hotwords:
            dialog = .create(categoryID: selectedHotwordCategoryID, mode: .hotword)
        case .replacements:
            dialog = .create(categoryID: nil, mode: .replacement)
        }
    }

    private func deleteDictionaryEntry(_ entry: DictionaryEntry) {
        guard dictionaryStore.delete(id: entry.id) else {
            return
        }

        guard !entry.replacementTerms.isEmpty else { return }

        guard let removedIndex = visibleReplacementEntries.firstIndex(where: { $0.id == entry.id }) else {
            reloadReplacementEntries(reset: true)
            return
        }

        visibleReplacementEntries.remove(at: removedIndex)
        totalReplacementEntryCount = max(0, totalReplacementEntryCount - 1)

        guard visibleReplacementEntries.count < totalReplacementEntryCount else { return }
        loadReplacementEntries(offset: visibleReplacementEntries.count, limit: 1, reset: false)
    }

    private func refreshDictionaryEntriesAfterMutation() {
        guard selectedTab == .replacements else { return }
        let retainedVisibleCount = max(entryPageSize, visibleReplacementEntries.count)
        loadReplacementEntries(offset: 0, limit: retainedVisibleCount, reset: true)
    }

    private func refreshLocalContentState() {
        historyScanModelOptions = availableHistoryScanModels()
        automaticLearningPromptDraft = AppPromptDefaults.resolvedStoredText(
            storedAutomaticLearningPrompt,
            kind: .dictionaryAutoLearning
        )
        suggestionFilterDraft = dictionarySuggestionStore.filterSettings
        selectedHistoryScanModelID = resolvedDefaultHistoryScanModelID(from: historyScanModelOptions)
    }

    private func openDictionaryAdvancedSettings() {
        automaticLearningPromptDraft = AppPromptDefaults.resolvedStoredText(
            storedAutomaticLearningPrompt,
            kind: .dictionaryAutoLearning
        )
        showDictionaryAdvancedSettings = true
    }

    private func openDictionaryIngestDialog() {
        let options = availableHistoryScanModels()
        historyScanModelOptions = options
        suggestionFilterDraft = dictionarySuggestionStore.filterSettings
        selectedHistoryScanModelID = resolvedDefaultHistoryScanModelID(from: options)
        refreshPendingHistoryScanCountAsync()
        showDictionaryIngestDialog = true
    }

    private func requestSuggestionIngestCancellation() {
        suggestionActionMessage = nil
        onCancelIngestSuggestionsFromHistory()
    }

    private func runSuggestionIngest() {
        let options = availableHistoryScanModels()
        guard !options.isEmpty else {
            suggestionActionMessage = AppLocalization.localizedString(
                "No configured local or remote model is available for dictionary ingestion. Configure one in Model settings first."
            )
            return
        }

        historyScanModelOptions = options
        if !options.contains(where: { $0.id == selectedHistoryScanModelID }) {
            selectedHistoryScanModelID = resolvedDefaultHistoryScanModelID(from: options)
        }
        guard !selectedHistoryScanModelID.isEmpty else { return }

        suggestionActionMessage = nil
        saveSuggestionIngestSettings()
        onIngestSuggestionsFromHistory(
            DictionaryHistoryScanRequest(
                modelOptionID: selectedHistoryScanModelID,
                filterSettings: DictionarySuggestionFilterSettings(
                    prompt: suggestionFilterDraft.prompt,
                    batchSize: dictionarySuggestionStore.filterSettings.batchSize,
                    maxCandidatesPerBatch: dictionarySuggestionStore.filterSettings.maxCandidatesPerBatch
                ).sanitized()
            ),
            true
        )
    }

    private func startSuggestionIngestFromDialog() {
        saveSuggestionIngestSettings()
        runSuggestionIngest()
    }

    private func saveDictionaryAdvancedSettings() {
        let resolvedAutomaticLearningPrompt = AppPromptDefaults.resolvedStoredText(
            automaticLearningPromptDraft,
            kind: .dictionaryAutoLearning
        )
        automaticLearningPromptDraft = resolvedAutomaticLearningPrompt
        storedAutomaticLearningPrompt = AppPromptDefaults.canonicalStoredText(
            resolvedAutomaticLearningPrompt,
            kind: .dictionaryAutoLearning
        )
    }

    private func saveSuggestionIngestSettings() {
        let sanitized = DictionarySuggestionFilterSettings(
            prompt: suggestionFilterDraft.prompt,
            batchSize: dictionarySuggestionStore.filterSettings.batchSize,
            maxCandidatesPerBatch: dictionarySuggestionStore.filterSettings.maxCandidatesPerBatch
        ).sanitized()
        suggestionFilterDraft = sanitized
        dictionarySuggestionStore.saveFilterSettings(sanitized)

        if historyScanModelOptions.contains(where: { $0.id == selectedHistoryScanModelID }) {
            preferredHistoryScanModelID = selectedHistoryScanModelID
        }
    }

    private func restoreSuggestionIngestPromptToDefault() {
        suggestionFilterDraft.prompt = DictionarySuggestionFilterSettings.defaultPrompt
    }

    private func restoreAutomaticLearningPromptToDefault() {
        automaticLearningPromptDraft = AppPromptDefaults.text(for: .dictionaryAutoLearning)
    }

    private func resolvedDefaultHistoryScanModelID(from options: [DictionaryHistoryScanModelOption]) -> String {
        if options.contains(where: { $0.id == preferredHistoryScanModelID }) {
            return preferredHistoryScanModelID
        }
        if options.contains(where: { $0.id == selectedHistoryScanModelID }) {
            return selectedHistoryScanModelID
        }
        return options.first?.id ?? ""
    }

    private func exportDictionary() {
        let panel = NSSavePanel()
        panel.allowedContentTypes = [.json]
        panel.nameFieldStringValue = localized("Voxt-Dictionary.json")
        panel.canCreateDirectories = true

        guard panel.runModal() == .OK, let url = panel.url else { return }

        do {
            let text = try dictionaryStore.exportTransferJSONString()
            try text.write(to: url, atomically: true, encoding: .utf8)
            showDictionaryToast(localized("Dictionary exported successfully."))
        } catch {
            showDictionaryToast(AppLocalization.format(
                "Dictionary export failed: %@",
                error.localizedDescription
            ))
        }
    }

    private func importDictionary() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.canChooseFiles = true

        guard panel.runModal() == .OK, let url = panel.url else { return }

        do {
            let text = try String(contentsOf: url, encoding: .utf8)
            let result = try dictionaryStore.importTransferJSONString(text)
            refreshLocalContentState()
            reloadReplacementEntries(reset: true)
            showDictionaryToast(AppLocalization.format(
                "Imported %d terms and skipped %d duplicates.",
                result.addedCount,
                result.skippedCount
            ))
        } catch {
            showDictionaryToast(AppLocalization.format(
                "Dictionary import failed: %@",
                error.localizedDescription
            ))
        }
    }

    private func showDictionaryToast(_ message: String, duration: TimeInterval = 2.4) {
        dictionaryToastDismissTask?.cancel()
        dictionaryToastMessage = message
        dictionaryToastDismissTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(duration))
            guard !Task.isCancelled else { return }
            dictionaryToastMessage = ""
        }
    }

    private func dismissDictionaryToast() {
        dictionaryToastDismissTask?.cancel()
        dictionaryToastMessage = ""
    }

    private func groupName(for groupID: UUID?) -> String? {
        guard let groupID else { return nil }
        return appBranchGroups.first(where: { $0.id == groupID })?.name
    }

    private func groupNameSnapshot(
        for groupID: UUID?,
        in dialog: DictionaryDialog
    ) -> String? {
        guard let groupID else { return nil }
        if let currentName = groupName(for: groupID) {
            return currentName
        }
        guard case .edit(let entry) = dialog,
              entry.groupID == groupID
        else {
            return nil
        }
        return entry.groupNameSnapshot
    }

    private var historyScanStatusText: String {
        AppLocalization.format(
            "Scanned %d of %d history records. Added %d dictionary terms, skipped %d duplicates.",
            historyScanProgress.processedCount,
            historyScanProgress.totalCount,
            historyScanProgress.newSuggestionCount,
            historyScanProgress.duplicateCount
        )
    }

    private var historyScanCancellationText: String {
        AppLocalization.localizedString("Cancel requested. Stopping after the current batch.")
    }


}
