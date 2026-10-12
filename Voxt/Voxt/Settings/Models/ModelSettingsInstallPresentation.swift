// ModelSettingsInstallPresentation.swift
// Provides Model Settings Install Presentation for model settings.

import SwiftUI

enum LocalModelInstallTarget: Hashable {
    case mlx(String)
    case customLLM(String)
    case ggufTranslation(GGUFTranslationModelID)
}

enum LocalModelInstallState: Equatable {
    case checking
    case installable(isEnabled: Bool)
    case downloading
    case paused
    case cancelling
    case installed
    case uninstalling
}

enum LocalModelInstallActionKind: Equatable {
    case inactive
    case use
    case install
    case pause
    case resume
    case cancel
    case uninstall
    case openLocation
    case configure
}

struct LocalModelInstallActionDescriptor: Equatable {
    let kind: LocalModelInstallActionKind
    let title: String
    var isEnabled: Bool = true
    var isDestructive: Bool = false
    var showsProgress: Bool = false
}

struct LocalModelInstallSnapshot: Equatable {
    let target: LocalModelInstallTarget
    let state: LocalModelInstallState
    let isInstalled: Bool
    let isCurrentSelection: Bool
    let statusText: String
    let badgeText: String?
    let downloadStatus: ModelDownloadStatusSnapshot?
    let canOpenLocation: Bool
    let canConfigure: Bool
    let configureActionTitle: String?
}

enum ModelSettingsInstallActionResolver {
    static func tableActions(
        for snapshot: LocalModelInstallSnapshot,
        perform: @escaping (LocalModelInstallTarget, LocalModelInstallActionKind) -> Void
    ) -> [ModelTableAction] {
        switch snapshot.state {
        case .checking:
            return [tableAction(for: .init(kind: .inactive, title: localized("Loading…"), isEnabled: false, showsProgress: true), target: snapshot.target, perform: perform)]
        case .uninstalling:
            return [tableAction(for: .init(kind: .inactive, title: localized("Uninstalling…"), isEnabled: false), target: snapshot.target, perform: perform)]
        case .downloading:
            return [
                tableAction(for: .init(kind: .pause, title: localized("Pause")), target: snapshot.target, perform: perform),
                tableAction(for: .init(kind: .cancel, title: localized("Cancel"), isDestructive: true), target: snapshot.target, perform: perform)
            ]
        case .paused:
            return [
                tableAction(for: .init(kind: .resume, title: localized("Continue")), target: snapshot.target, perform: perform),
                tableAction(for: .init(kind: .cancel, title: localized("Cancel"), isDestructive: true), target: snapshot.target, perform: perform)
            ]
        case .cancelling:
            return [
                tableAction(
                    for: .init(
                        kind: .inactive,
                        title: localized("Cancel"),
                        isEnabled: false,
                        showsProgress: true
                    ),
                    target: snapshot.target,
                    perform: perform
                )
            ]
        case .installed:
            return [
                tableAction(
                    for: .init(
                        kind: .use,
                        title: localized(snapshot.isCurrentSelection ? "Using" : "Use"),
                        isEnabled: !snapshot.isCurrentSelection
                    ),
                    target: snapshot.target,
                    perform: perform
                ),
                tableAction(
                    for: .init(kind: .uninstall, title: localized("Uninstall"), isDestructive: true),
                    target: snapshot.target,
                    perform: perform
                )
            ]
        case .installable(let isEnabled):
            return [
                tableAction(
                    for: .init(kind: .install, title: localized("Download"), isEnabled: isEnabled),
                    target: snapshot.target,
                    perform: perform
                )
            ]
        }
    }

    static func catalogPrimaryAction(
        for snapshot: LocalModelInstallSnapshot,
        perform: @escaping (LocalModelInstallTarget, LocalModelInstallActionKind) -> Void
    ) -> ModelTableAction? {
        let descriptor: LocalModelInstallActionDescriptor
        switch snapshot.state {
        case .checking:
            descriptor = .init(kind: .inactive, title: localized("Loading…"), isEnabled: false, showsProgress: true)
        case .uninstalling:
            descriptor = .init(kind: .inactive, title: localized("Uninstalling…"), isEnabled: false)
        case .downloading:
            descriptor = .init(kind: .pause, title: localized("Pause"))
        case .paused:
            descriptor = .init(kind: .resume, title: localized("Continue"))
        case .cancelling:
            descriptor = .init(
                kind: .inactive,
                title: localized("Cancel"),
                isEnabled: false,
                showsProgress: true
            )
        case .installed:
            descriptor = .init(kind: .uninstall, title: localized("Uninstall"), isDestructive: true)
        case .installable(let isEnabled):
            descriptor = .init(kind: .install, title: localized("Install"), isEnabled: isEnabled)
        }
        return tableAction(for: descriptor, target: snapshot.target, perform: perform)
    }

    static func catalogSecondaryActions(
        for snapshot: LocalModelInstallSnapshot,
        perform: @escaping (LocalModelInstallTarget, LocalModelInstallActionKind) -> Void
    ) -> [ModelTableAction] {
        var descriptors = [LocalModelInstallActionDescriptor]()

        switch snapshot.state {
        case .downloading, .paused:
            descriptors.append(.init(kind: .cancel, title: localized("Cancel"), isDestructive: true))
        case .installed:
            if snapshot.canOpenLocation {
                descriptors.append(.init(kind: .openLocation, title: localized("Open Location")))
            }
        case .checking, .installable, .cancelling, .uninstalling:
            break
        }

        if snapshot.canConfigure, let configureActionTitle = snapshot.configureActionTitle {
            descriptors.append(.init(kind: .configure, title: configureActionTitle))
        }

        return descriptors.map { tableAction(for: $0, target: snapshot.target, perform: perform) }
    }

    private static func tableAction(
        for descriptor: LocalModelInstallActionDescriptor,
        target: LocalModelInstallTarget,
        perform: @escaping (LocalModelInstallTarget, LocalModelInstallActionKind) -> Void
    ) -> ModelTableAction {
        ModelTableAction(
            title: descriptor.title,
            role: descriptor.isDestructive ? .destructive : nil,
            isEnabled: descriptor.isEnabled,
            showsProgress: descriptor.showsProgress
        ) {
            guard descriptor.isEnabled else { return }
            guard descriptor.kind != .inactive else { return }
            perform(target, descriptor.kind)
        }
    }

    private static func localized(_ key: String) -> String {
        AppLocalization.localizedString(key)
    }
}
