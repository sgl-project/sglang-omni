// AppEnhancementSettingsView.swift
// Provides App Enhancement Settings View for app enhancement settings.

import SwiftUI
import AppKit

struct AppEnhancementSettingsView: View {
    let navigationRequest: SettingsNavigationRequest?
    @AppStorage(AppPreferenceKey.interfaceLanguage) var interfaceLanguageRaw = AppInterfaceLanguage.system.rawValue

    @State var apps: [BranchApp] = []
    @State var urlItems: [BranchURLItem] = []
    @State var groups: [AppBranchGroup] = []

    @State var sourceTab: SourceTab = .apps
    @State var draggingAppID: String?
    @State var hoveredCardID: String?
    @State var appsRefreshRotation = 0.0

    @State var modal: AppBranchModal?
    @State var groupNameDraft = ""
    @State var groupPromptDraft = ""
    @State var groupAutoKeyPressEnabledDraft = false
    @State var groupAutoKeyPressHotkeyDraft = AppBranchGroup.defaultAutoKeyPressHotkey
    @State var urlDraft = ""
    @State var modalErrorMessage: String?

    var body: some View {
        ScrollViewReader { proxy in
            ScrollView {
                VStack(alignment: .leading, spacing: 10) {
                    sourceListCard
                        .settingsNavigationAnchor(.appBranchSources)
                    groupListCard
                        .settingsNavigationAnchor(.appBranchGroups)
                }
            }
            .onAppear {
                scrollToNavigationTargetIfNeeded(using: proxy)
            }
            .onChange(of: navigationRequest?.id) { _, _ in
                scrollToNavigationTargetIfNeeded(using: proxy)
            }
        }
        .onAppear(perform: handleOnAppear)
        .onReceive(NotificationCenter.default.publisher(for: NSWorkspace.didActivateApplicationNotification)) { _ in
            refreshApps()
        }
        .onChange(of: groups) { _, _ in
            saveGroups()
        }
        .onChange(of: urlItems) { _, _ in
            saveURLs()
        }
        .sheet(item: $modal) { currentModal in
            modalView(for: currentModal)
        }
        .id(interfaceLanguageRaw)
    }

    private func scrollToNavigationTargetIfNeeded(using proxy: ScrollViewProxy) {
        guard let navigationRequest,
              navigationRequest.target.featureTab == .appEnhancement ||
                navigationRequest.target.tab == .appEnhancement,
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
}
