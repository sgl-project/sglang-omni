// TranscriptionTestSectionView.swift
// Provides Transcription Test Section View for settings screens.

import SwiftUI

private func localized(_ key: String) -> String {
    AppLocalization.localizedString(key)
}

struct TranscriptionTestSectionView: View {
    @State private var currentTestCaseIndex = 0
    @State private var testInputText = ""

    private let sampleTexts: [String] = [
        "Hi team, this is the weekly update. We completed the onboarding fix, reduced crash rate by 12 percent, and will validate the payment flow before Friday.",
        "Please call me at 415-867-2309 after 6:30 PM, and send the confirmation code 482917 to my backup number 650-301-7788.",
        "今天我们已经完成了语音识别模型的升级，整体识别准确率提升明显，请在下午三点前提交测试反馈。",
        "我们计划在 Thursday 之前完成 API 对接，当前进度是 78 percent，预计还需要 2 天来处理 edge case 和回归测试。",
        "The weather whether report said their plan was sound, but the team still heard here that the whole idea might be too weak this week."
    ]

    var body: some View {
        GroupBox {
            VStack(alignment: .leading, spacing: 12) {
                Text(localized("Test"))
                    .font(.headline)

                Text(localized("Switch sample text with dots, then paste your transcription below to compare differences."))
                    .font(.caption)
                    .foregroundStyle(.secondary)

                ScrollView {
                    diffedSampleTextView
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
                .frame(maxHeight: 120)
                .padding(8)
                .background(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(Color(nsColor: .windowBackgroundColor).opacity(0.8))
                )
                .overlay(
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .strokeBorder(.quaternary, lineWidth: 1)
                )

                HStack(spacing: 8) {
                    ForEach(sampleTexts.indices, id: \.self) { index in
                        Button {
                            currentTestCaseIndex = index
                        } label: {
                            Circle()
                                .fill(index == currentTestCaseIndex ? Color.accentColor : Color.secondary.opacity(0.35))
                                .frame(width: 8, height: 8)
                        }
                        .buttonStyle(.plain)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .center)

                ZStack(alignment: .bottomTrailing) {
                    TextEditor(text: $testInputText)
                        .settingsPromptEditor(height: 120, contentPadding: 6)

                    Button(localized("Clean")) {
                        testInputText = ""
                    }
                    .controlSize(.small)
                    .padding(8)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var selectedSampleText: String {
        sampleTexts[currentTestCaseIndex]
    }

    private var diffedSampleTextView: some View {
        let trimmedInput = testInputText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedInput.isEmpty else {
            return Text(selectedSampleText)
                .font(.body)
        }

        let segments = diffSegments(reference: selectedSampleText, input: trimmedInput)
        var attributed = AttributedString("")
        for segment in segments {
            var part = AttributedString(segment.text)
            if segment.isDifferent {
                part.backgroundColor = .yellow.opacity(0.45)
            }
            attributed.append(part)
        }
        return Text(attributed).font(.body)
    }

    private func diffSegments(reference: String, input: String) -> [(text: String, isDifferent: Bool)] {
        let ref = Array(reference)
        let inp = Array(input)
        let n = ref.count
        let m = inp.count
        guard n > 0 else { return [] }

        var dp = Array(repeating: Array(repeating: 0, count: m + 1), count: n + 1)
        if n > 0, m > 0 {
            for i in stride(from: n - 1, through: 0, by: -1) {
                for j in stride(from: m - 1, through: 0, by: -1) {
                    if ref[i] == inp[j] {
                        dp[i][j] = dp[i + 1][j + 1] + 1
                    } else {
                        dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])
                    }
                }
            }
        }

        var matched = Array(repeating: false, count: n)
        var i = 0
        var j = 0
        while i < n, j < m {
            if ref[i] == inp[j] {
                matched[i] = true
                i += 1
                j += 1
            } else if dp[i + 1][j] >= dp[i][j + 1] {
                i += 1
            } else {
                j += 1
            }
        }

        var result: [(String, Bool)] = []
        var current = ""
        var currentFlag = !matched[0]

        for idx in 0..<n {
            let flag = !matched[idx]
            if idx == 0 || flag == currentFlag {
                current.append(ref[idx])
            } else {
                result.append((current, currentFlag))
                current = String(ref[idx])
                currentFlag = flag
            }
        }

        if !current.isEmpty {
            result.append((current, currentFlag))
        }

        return result
    }
}
