// LocalASRModelCatalogBuilder.swift
// Provides Local ASRModel Catalog Builder for model settings.

import SwiftUI

private func localizedModelCatalog(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

@MainActor
extension ModelCatalogBuilder {
    func dictationASREntry() -> ModelCatalogEntry {
        let decoration = catalogDecoration(
            base: [localizedModelCatalog("Local"), localizedModelCatalog("Built-in"), localizedModelCatalog("Fast")],
            installed: true,
            requiresConfiguration: false,
            configured: true,
            selectionID: .dictation
        )
        return ModelCatalogEntry(
            id: FeatureModelSelectionID.dictation.rawValue,
            title: localizedModelCatalog("Direct Dictation"),
            engine: localizedModelCatalog("System ASR"),
            sizeText: localizedModelCatalog("Built-in"),
            ratingText: "3.4",
            filterTags: decoration.filterTags,
            displayTags: decoration.displayTags,
            statusText: "",
            usageLocations: decoration.usageLocations,
            badgeText: nil,
            primaryAction: ModelTableAction(title: localizedModelCatalog("Settings")) {
                showASRHintTarget(.dictation)
            },
            secondaryActions: []
        )
    }

    func mlxASREntries() -> [ModelCatalogEntry] {
        mlxModelManager.displayModelsIncludingInstalled().map { model in
            let repo = MLXModelManager.canonicalModelRepo(model.id)
            let selectionID = FeatureModelSelectionID.mlx(repo)
            let installSnapshot = mlxInstallSnapshot(repo)
            let decoration = catalogDecoration(
                base: [localizedModelCatalog("Local")] + mlxCatalogTags(for: repo),
                installed: installSnapshot.isInstalled,
                requiresConfiguration: false,
                configured: true,
                selectionID: selectionID
            )

            return ModelCatalogEntry(
                id: "mlx:\(repo)",
                title: mlxModelManager.displayTitle(for: repo),
                engine: MLXWhisperMigrationSupport.isWhisperRepo(repo)
                    ? localizedModelCatalog("Whisper (MLX)")
                    : localizedModelCatalog("MLX Audio"),
                sizeText: mlxASRSizeText(repo: repo, isInstalled: installSnapshot.isInstalled),
                ratingText: MLXModelManager.ratingText(for: repo),
                filterTags: decoration.filterTags,
                displayTags: decoration.displayTags,
                statusText: installSnapshot.statusText,
                usageLocations: decoration.usageLocations,
                badgeText: installSnapshot.badgeText ?? ModelCatalogBadgeSupport.recommendedBadgeText(forMLXRepo: repo),
                primaryAction: catalogPrimaryAction(installSnapshot),
                secondaryActions: catalogSecondaryActions(installSnapshot)
            )
        }
    }

    private func mlxASRSizeText(repo: String, isInstalled: Bool) -> String {
        _ = isInstalled
        return mlxModelManager.remoteSizeText(repo: repo)
    }

}
