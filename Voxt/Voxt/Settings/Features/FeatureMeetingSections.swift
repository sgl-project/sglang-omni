// FeatureMeetingSections.swift
// Provides Feature Meeting Sections for feature settings.

import AppKit
import SwiftUI
import UniformTypeIdentifiers

extension FeatureSettingsView {
    var meetingContent: some View {
        featurePage(
            title: "",
            subtitle: "",
            systemImageName: nil,
            pills: meetingPills
        ) {
            FeatureSettingsCard(title: "") {
                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Speech Model"),
                        value: asrSelectionSummary(featureSettings.meeting.asrSelectionID),
                        action: { selectorSheet = .meetingASR }
                    )
                }

                FeatureSettingSection(title: "", detail: "") {
                    FeatureSelectorRow(
                        title: featureSettingsLocalized("Summary Model"),
                        value: llmSelectionSummary(featureSettings.meeting.summaryModelSelectionID),
                        action: { selectorSheet = .meetingSummary }
                    )
                }

                FeatureToggleRow(
                    title: featureSettingsLocalized("Hide Floating Window from Screen Sharing"),
                    detail: "",
                    isOn: binding(
                        get: { featureSettings.meeting.hideOverlayFromScreenSharing },
                        set: { featureSettings.meeting.hideOverlayFromScreenSharing = $0 }
                    )
                )

                FeatureToggleRow(
                    title: featureSettingsLocalized("Auto-generate Summary"),
                    detail: "",
                    isOn: binding(
                        get: { featureSettings.meeting.summaryAutoGenerate },
                        set: { featureSettings.meeting.summaryAutoGenerate = $0 }
                    )
                )

                meetingAdvancedSettingsSection
            }
        }
    }

    var meetingAdvancedSettingsSection: some View {
        GeneralAdvancedCard(isExpanded: $isMeetingAdvancedSettingsExpanded) {
            FeatureSettingSection(title: "", detail: "") {
                FeatureInlinePickerRow(
                    title: featureSettingsLocalized("Segmentation Mode"),
                    detail: meetingChunkingMode.wrappedValue.detail
                ) {
                    SettingsMenuPicker(
                        selection: meetingChunkingMode,
                        options: MeetingChunkingMode.allCases.map {
                            SettingsMenuOption(value: $0, title: $0.title)
                        },
                        selectedTitle: meetingChunkingMode.wrappedValue.title,
                        width: 220
                    )
                }

                FeatureInlinePickerRow(
                    title: featureSettingsLocalized("Silero Speech Detection"),
                    detail: meetingSileroVADSensitivity.wrappedValue.detail
                ) {
                    SettingsMenuPicker(
                        selection: meetingSileroVADSensitivity,
                        options: MeetingSileroVADSensitivity.allCases.map {
                            SettingsMenuOption(value: $0, title: $0.title)
                        },
                        selectedTitle: meetingSileroVADSensitivity.wrappedValue.title,
                        width: 220
                    )
                }

                FeatureInlinePickerRow(
                    title: featureSettingsLocalized("Speaker Separation Model"),
                    detail: meetingSpeakerDiarizationModelDetailText
                ) {
                    HStack(spacing: 8) {
                        Text(MeetingDiarizationMode.sortformerV2.title)
                        meetingSpeakerDiarizationModelDownloadControl
                        if case .downloaded = meetingDiarizationModelManager.state {
                            Image(systemName: "checkmark.circle.fill")
                        }
                    }
                }
            }
        }
    }

    var meetingPills: [FeatureSummaryPill] {
        [
            FeatureSummaryPill(
                title: featureSettingsLocalized("Audio Model"),
                value: shortSummary(asrSelectionSummary(featureSettings.meeting.asrSelectionID), maxLength: 52)
            ),
            FeatureSummaryPill(
                title: featureSettingsLocalized("Summary Model"),
                value: shortSummary(llmSelectionSummary(featureSettings.meeting.summaryModelSelectionID), maxLength: 52)
            ),
            FeatureSummaryPill(
                title: featureSettingsLocalized("Speakers"),
                value: featureSettings.meeting.speakerDiarizationModel.title
            )
        ]
    }

    @ViewBuilder
    var meetingSpeakerDiarizationModelDownloadControl: some View {
        switch meetingDiarizationModelManager.state {
        case .downloaded:
            EmptyView()
        case .notDownloaded:
            ProgressView(value: 0)
                .frame(width: 60)
        case let .downloading(progress, _):
            ProgressView(value: max(0, min(progress, 1)))
                .frame(width: 60)
        case .error:
            Button(featureSettingsLocalized("Retry")) {
                meetingDiarizationModelManager.downloadSelectedModel()
            }
            .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 9, height: 28))
        }
    }

    var meetingSpeakerDiarizationModelDetailText: String {
        switch meetingDiarizationModelManager.state {
        case .downloaded:
            return AppLocalization.format(
                "%@ %@",
                MeetingDiarizationMode.sortformerV2.title,
                featureSettingsLocalized("model is ready.")
            )
        case .notDownloaded:
            return AppLocalization.format(
                "%@ %@",
                MeetingDiarizationMode.sortformerV2.detail,
                featureSettingsLocalized("Download the model before meeting details can use speaker separation.")
            )
        case let .downloading(_, detail):
            return detail ?? featureSettingsLocalized("Installing speaker separation model...")
        case let .error(message):
            return AppLocalization.format("Download failed: %@", message)
        }
    }

    var meetingChunkingMode: Binding<MeetingChunkingMode> {
        Binding(
            get: {
                featureSettings.meeting.chunkingMode
            },
            set: { mode in
                featureSettings.meeting.chunkingModeRawValue = mode.rawValue
                saveFeatureSettings()
            }
        )
    }

    var meetingSileroVADSensitivity: Binding<MeetingSileroVADSensitivity> {
        Binding(
            get: {
                featureSettings.meeting.sileroVADSensitivity
            },
            set: { sensitivity in
                featureSettings.meeting.sileroVADSensitivityRawValue = sensitivity.rawValue
                saveFeatureSettings()
            }
        )
    }

}
