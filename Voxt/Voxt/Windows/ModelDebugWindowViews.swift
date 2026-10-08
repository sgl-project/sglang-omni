// ModelDebugWindowViews.swift
// Provides Model Debug Window Views for window and overlay UI.

import AppKit
import SwiftUI
import AVFoundation
import Combine

struct ASRDebugWindowView: View {
    @ObservedObject var viewModel: ASRDebugViewModel
    @State private var isModelSelectorPresented = false
    @State private var isAudioSelectorPresented = false

    private let columns = [
        GridItem(.flexible(minimum: 280), spacing: 12),
        GridItem(.flexible(minimum: 280), spacing: 12)
    ]

    private var selectorEntries: [FeatureModelSelectorEntry] {
        viewModel.options.map(\.selectorEntry)
    }

    var body: some View {
        VStack(spacing: 8) {
            HStack(alignment: .bottom, spacing: 10) {
                ModelDebugHeaderBadge()

                Button {
                    isModelSelectorPresented = true
                } label: {
                    ModelDebugToolbarSelectorLabel(
                        title: modelDebugLocalized("Model"),
                        value: viewModel.selectedModelTitle
                    )
                }
                .buttonStyle(.plain)
                .layoutPriority(1)

                Button {
                    viewModel.toggleRecording()
                } label: {
                    Image(systemName: viewModel.isRecording ? "stop.fill" : "mic.fill")
                        .font(.system(size: 16, weight: .semibold))
                        .foregroundStyle(viewModel.isRecording ? Color.green : Color.primary)
                        .frame(width: 44, height: 44)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(DetailPanelUIStyle.controlFillColor)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(
                                    viewModel.isRecording ? Color.green.opacity(0.28) : DetailPanelUIStyle.borderColor,
                                    lineWidth: 1
                                )
                        )
                }
                .buttonStyle(.plain)
                .disabled(viewModel.isRunning)

                Button {
                    isAudioSelectorPresented = true
                } label: {
                    ModelDebugToolbarSelectorLabel(
                        title: modelDebugLocalized("Audio"),
                        value: viewModel.selectedClipTitle
                    )
                }
                .buttonStyle(.plain)
                .disabled(viewModel.clips.isEmpty)
                .layoutPriority(1)

                Button {
                    viewModel.generateSelectedClip()
                } label: {
                    HStack(spacing: 6) {
                        if viewModel.isRunning {
                            if viewModel.isModelInitializing {
                                ModelInitializingIconView()
                                    .frame(width: 14, height: 14)
                            } else {
                                ProgressView()
                                    .controlSize(.small)
                                    .tint(.white)
                                    .scaleEffect(0.75)
                            }
                        }
                        Text(modelDebugLocalized(viewModel.isModelInitializing ? "Initializing…" : "Generate"))
                            .font(.system(size: 12, weight: .semibold))
                            .lineLimit(1)
                    }
                    .frame(width: 92, height: 30)
                }
                .buttonStyle(DetailPrimaryButtonStyle())
                .disabled(viewModel.isRunning || viewModel.isRecording)
            }
            .frame(height: 54)

            ASRDebugVADSnapshotView(snapshot: viewModel.vadSnapshot)

            ScrollView {
                Group {
                    if viewModel.results.isEmpty {
                        VStack {
                            debugEmptyState(
                                title: modelDebugLocalized("No debug results yet"),
                                detail: modelDebugLocalized("Record audio once, then switch models or clips to compare transcription output.")
                            )
                        }
                        .frame(maxWidth: .infinity, minHeight: 360, alignment: .center)
                    } else {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(viewModel.results) { result in
                                ASRDebugResultCard(
                                    result: result,
                                    onClose: { viewModel.removeResult(result.id) }
                                )
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 2)
                    }
                }
                .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, 10)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(ModelDebugWindowBackground())
        .ignoresSafeArea(.container, edges: .top)
        .overlay(alignment: .top) {
            if !viewModel.toastMessage.isEmpty {
                ModelDebugToast(message: viewModel.toastMessage) {
                    viewModel.dismissToast()
                }
                .padding(.top, 54)
                .transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .sheet(isPresented: $isModelSelectorPresented) {
            FeatureModelSelectorDialog(
                title: modelDebugLocalized("Choose Transcription ASR"),
                entries: selectorEntries,
                selectedID: FeatureModelSelectionID(rawValue: viewModel.selectedModelID),
                onSelect: { selectionID in
                    viewModel.selectedModelID = selectionID.rawValue
                }
            )
        }
        .sheet(isPresented: $isAudioSelectorPresented) {
            ASRDebugAudioSelectorSheet(
                clips: viewModel.clips,
                selectedClipID: viewModel.selectedClipID,
                onSelect: { viewModel.selectedClipID = $0 },
                onDelete: { viewModel.deleteClip($0) }
            )
        }
    }
}

struct LLMDebugWindowView: View {
    @ObservedObject var viewModel: LLMDebugViewModel
    @State private var isModelSelectorPresented = false
    @State private var isPresetSelectorPresented = false
    @State private var isPromptSettingsPresented = false

    private var columns: [GridItem] {
        [
            GridItem(.flexible(minimum: 260, maximum: .infinity), spacing: 12, alignment: .top),
            GridItem(.flexible(minimum: 260, maximum: .infinity), spacing: 12, alignment: .top)
        ]
    }

    private var selectorEntries: [FeatureModelSelectorEntry] {
        viewModel.modelOptions.map(\.selectorEntry)
    }

    var body: some View {
        VStack(spacing: 12) {
            HStack(alignment: .bottom, spacing: 10) {
                ModelDebugHeaderBadge()

                Button {
                    isModelSelectorPresented = true
                } label: {
                    ModelDebugToolbarSelectorLabel(
                        title: modelDebugLocalized("Model"),
                        value: viewModel.selectedModelTitle
                    )
                }
                .buttonStyle(.plain)
                .layoutPriority(1)

                Button {
                    isPresetSelectorPresented = true
                } label: {
                    ModelDebugToolbarSelectorLabel(
                        title: modelDebugLocalized("Preset"),
                        value: viewModel.selectedPresetTitle
                    )
                }
                .buttonStyle(.plain)
                .layoutPriority(1)

                Button {
                    isPromptSettingsPresented = true
                } label: {
                    Image(systemName: "slider.horizontal.3")
                        .font(.system(size: 14, weight: .semibold))
                        .frame(width: 44, height: 44)
                        .background(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .fill(DetailPanelUIStyle.controlFillColor)
                        )
                        .overlay(
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .strokeBorder(DetailPanelUIStyle.borderColor, lineWidth: 1)
                        )
                }
                .buttonStyle(.plain)
                .disabled(viewModel.selectedPreset == nil || viewModel.isRunning)

                Button {
                    viewModel.run()
                } label: {
                    HStack(spacing: 6) {
                        if viewModel.isRunning {
                            if viewModel.isModelInitializing {
                                ModelInitializingIconView()
                                    .frame(width: 14, height: 14)
                            } else {
                                ProgressView()
                                    .controlSize(.small)
                                    .tint(.white)
                                    .scaleEffect(0.75)
                            }
                        }
                        Text(modelDebugLocalized(viewModel.isModelInitializing ? "Initializing…" : "Generate"))
                            .font(.system(size: 12, weight: .semibold))
                            .lineLimit(1)
                    }
                    .frame(width: 92, height: 30)
                }
                .buttonStyle(DetailPrimaryButtonStyle())
                .disabled(viewModel.isRunning || viewModel.selectedPreset == nil || viewModel.selectedModelID.isEmpty)
            }
            .frame(height: 54)

            ScrollView {
                Group {
                    if viewModel.results.isEmpty {
                        VStack {
                            debugEmptyState(
                                title: modelDebugLocalized("No debug results yet"),
                                detail: modelDebugLocalized("Choose a preset, fill variables, and run different models to compare outputs.")
                            )
                        }
                        .frame(maxWidth: .infinity, minHeight: 360, alignment: .center)
                    } else {
                        LazyVGrid(columns: columns, spacing: 12) {
                            ForEach(viewModel.results) { result in
                                LLMDebugResultCard(
                                    result: result,
                                    onClose: { viewModel.removeResult(result.id) }
                                )
                            }
                        }
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.top, 2)
                    }
                }
                .frame(maxWidth: .infinity)
            }
        }
        .padding(.horizontal, 10)
        .padding(.top, 10)
        .padding(.bottom, 8)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(ModelDebugWindowBackground())
        .ignoresSafeArea(.container, edges: .top)
        .sheet(isPresented: $isModelSelectorPresented) {
            FeatureModelSelectorDialog(
                title: modelDebugLocalized("Choose Transcription LLM"),
                entries: selectorEntries,
                selectedID: FeatureModelSelectionID(rawValue: viewModel.selectedModelID),
                onSelect: { selectionID in
                    viewModel.selectedModelID = selectionID.rawValue
                }
            )
        }
        .sheet(isPresented: $isPresetSelectorPresented) {
            LLMDebugPresetSelectorSheet(
                presets: viewModel.presetOptions,
                selectedPresetID: viewModel.selectedPresetID,
                onSelect: { presetID in
                    viewModel.selectedPresetID = presetID
                    viewModel.presetDidChange()
                }
            )
        }
        .sheet(isPresented: $isPromptSettingsPresented) {
            if let selectedPreset = viewModel.selectedPreset {
                LLMDebugPromptSettingsSheet(
                    preset: selectedPreset,
                    variableValues: $viewModel.variableValues,
                    onApply: { prompt in
                        viewModel.applyPromptTemplate(prompt)
                    },
                    onSave: { prompt in
                        viewModel.savePromptTemplate(prompt)
                    }
                )
            }
        }
    }
}
