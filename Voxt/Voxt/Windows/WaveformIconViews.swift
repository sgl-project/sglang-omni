// WaveformIconViews.swift
// Provides Waveform Icon Views for window and overlay UI.

import SwiftUI
import AppKit

struct AnswerIconView: View {
    var body: some View {
        ZStack {
            SVGPathShape(pathData: WaveformIconPathData.answerBackground)
                .fill(.white.opacity(0.4))

            SVGPathShape(pathData: WaveformIconPathData.answerSignal)
                .fill(.white)

            SVGPathShape(pathData: WaveformIconPathData.answerSparkle)
                .fill(.white)
        }
    }
}

struct TranscriptionModeIconView: View {
    var color: Color = .white

    private let viewport = CGSize(width: 392, height: 392)

    var body: some View {
        ZStack {
            SVGPathShape(
                pathData: WaveformIconPathData.transcriptionStem,
                viewport: viewport
            )
            .fill(color)

            SVGPathShape(
                pathData: WaveformIconPathData.transcriptionMicrophone,
                viewport: viewport
            )
            .fill(color)
        }
    }
}

struct NoteModeIconView: View {
    private let viewport = CGSize(width: 392, height: 392)

    var body: some View {
        ZStack {
            SVGPathShape(
                pathData: WaveformIconPathData.noteStem,
                viewport: viewport
            )
            .fill(.white)

            SVGPathShape(
                pathData: WaveformIconPathData.noteMicrophone,
                viewport: viewport
            )
            .fill(.white)
        }
    }
}

struct TranslationModeIconView: View {
    private let viewport = CGSize(width: 392, height: 392)

    var body: some View {
        ZStack {
            SVGPathShape(
                pathData: WaveformIconPathData.translationStem,
                viewport: viewport
            )
            .fill(.white)

            SVGPathShape(
                pathData: WaveformIconPathData.translationLanguage,
                viewport: viewport
            )
            .fill(.white)
        }
    }
}

struct RewriteModeIconView: View {
    private let viewport = CGSize(width: 392, height: 392)

    var body: some View {
        ZStack {
            SVGPathShape(
                pathData: WaveformIconPathData.rewriteStem,
                viewport: viewport
            )
            .fill(.white)

            SVGPathShape(
                pathData: WaveformIconPathData.rewriteMagic,
                viewport: viewport
            )
            .fill(.white)
        }
    }
}

struct OverlayCompactLeadingIconView: View {
    let image: NSImage

    var body: some View {
        Image(nsImage: image)
            .resizable()
            .interpolation(.high)
            .antialiased(true)
            .scaledToFit()
            .clipShape(RoundedRectangle(cornerRadius: 3.5, style: .continuous))
    }
}

struct CompactModeIconView: View {
    let sessionIconMode: OverlaySessionIconMode

    var body: some View {
        Group {
            switch sessionIconMode {
            case .transcription:
                TranscriptionModeIconView()
            case .note:
                NoteModeIconView()
            case .translation:
                TranslationModeIconView()
            case .rewrite:
                RewriteModeIconView()
            }
        }
        .frame(width: 14, height: 14)
        .opacity(0.92)
    }
}

struct WaveformCompactLeadingStatusIconView: View {
    let isCompleting: Bool
    let showsInitializationIcon: Bool
    let shouldAnimate: Bool
    let compactLeadingIconImage: NSImage?
    let sessionIconMode: OverlaySessionIconMode
    let displayMode: OverlayDisplayMode

    var body: some View {
        Group {
            if isCompleting {
                Image(systemName: "checkmark.circle.fill")
                    .font(.system(size: 13, weight: .semibold))
                    .foregroundStyle(.white.opacity(0.95))
            } else if showsInitializationIcon {
                ModelInitializingIconView(isAnimating: shouldAnimate)
            } else if let compactLeadingIconImage,
                      sessionIconMode == .transcription,
                      displayMode == .processing {
                OverlayCompactLeadingIconView(image: compactLeadingIconImage)
            } else {
                CompactModeIconView(sessionIconMode: sessionIconMode)
            }
        }
        .frame(width: 14, height: 14, alignment: .center)
    }
}

struct LoadingSpinnerIconView: View {
    var isAnimating: Bool

    @State private var rotationDegrees = 0.0

    var body: some View {
        Circle()
            .trim(from: 0.18, to: 0.88)
            .stroke(
                .white.opacity(0.95),
                style: StrokeStyle(lineWidth: 1.8, lineCap: .round)
            )
            .rotationEffect(.degrees(rotationDegrees))
            .padding(1)
            .onAppear {
                updateAnimationState()
            }
            .onChange(of: isAnimating) {
                updateAnimationState()
            }
    }

    private func updateAnimationState() {
        if isAnimating {
            rotationDegrees = 0
            withAnimation(.linear(duration: 0.9).repeatForever(autoreverses: false)) {
                rotationDegrees = 360
            }
        } else {
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                rotationDegrees = 0
            }
        }
    }
}

struct WaveformProcessingLoaderView: View {
    var isAnimating = true
    var itemCount = 5
    var itemSize = CGSize(width: 6, height: 6)
    var spacing: CGFloat = 4
    var color: Color = .white

    var body: some View {
        Group {
            if isAnimating {
                TimelineView(.animation) { context in
                    loaderContent(at: context.date)
                }
            } else {
                loaderContent(at: Date(timeIntervalSinceReferenceDate: 0))
            }
        }
        .frame(height: max(itemSize.height * 2.1, itemSize.height))
    }

    private func loaderContent(at date: Date) -> some View {
        HStack(alignment: .center, spacing: spacing) {
            ForEach(0..<itemCount, id: \.self) { index in
                let progress = phaseProgress(for: index, at: date)
                Circle()
                    .fill(color)
                    .frame(width: itemSize.width, height: itemSize.height)
                    .scaleEffect(loaderScale(for: progress))
                    .opacity(loaderOpacity(for: progress))
            }
        }
    }

    private func phaseProgress(for index: Int, at date: Date) -> Double {
        guard isAnimating else { return 1 }

        let cycleDuration = 1.2
        let staggerDelay = 0.12
        let rawProgress = date.timeIntervalSinceReferenceDate - (Double(index) * staggerDelay)
        let wrapped = rawProgress.truncatingRemainder(dividingBy: cycleDuration)
        return wrapped >= 0 ? (wrapped / cycleDuration) : ((wrapped + cycleDuration) / cycleDuration)
    }

    private func loaderScale(for progress: Double) -> CGFloat {
        switch progress {
        case 0..<0.3:
            let t = progress / 0.3
            return 0.6 + (0.7 * t)
        case 0.3..<0.6:
            let t = (progress - 0.3) / 0.3
            return 1.3 - (0.7 * t)
        default:
            return 0.6
        }
    }

    private func loaderOpacity(for progress: Double) -> Double {
        switch progress {
        case 0..<0.3:
            let t = progress / 0.3
            return 0.15 + (0.85 * t)
        case 0.3..<0.6:
            let t = (progress - 0.3) / 0.3
            return 1.0 - (0.85 * t)
        default:
            return 0.15
        }
    }
}

struct ModelInitializingIconView: View {
    private let viewport = CGSize(width: 24, height: 24)
    var isAnimating = true

    @State private var animate = false

    var body: some View {
        ZStack {
            SVGPathShape(
                pathData: WaveformIconPathData.initializingBottom,
                viewport: viewport
            )
            .fill(.white.opacity(0.42))

            SVGPathShape(
                pathData: WaveformIconPathData.initializingMiddle,
                viewport: viewport
            )
            .fill(.white)

            SVGPathShape(
                pathData: WaveformIconPathData.initializingTop,
                viewport: viewport
            )
            .fill(.white)
        }
        .rotationEffect(.degrees(animate ? 360 : 0))
        .scaleEffect(animate ? 1.04 : 0.96)
        .opacity(animate ? 0.98 : 0.62)
        .onAppear {
            updateAnimationState()
        }
        .onChange(of: isAnimating) {
            updateAnimationState()
        }
    }

    private func updateAnimationState() {
        guard isAnimating else {
            var transaction = Transaction()
            transaction.disablesAnimations = true
            withTransaction(transaction) {
                animate = false
            }
            return
        }

        animate = false
        withAnimation(.easeInOut(duration: 1.25).repeatForever(autoreverses: true)) {
            animate = true
        }
    }
}

struct CopyIconView: View {
    var color: Color = .white

    var body: some View {
        ZStack {
            SVGPathShape(pathData: WaveformIconPathData.copyFront)
                .fill(color.opacity(0.4))

            SVGPathShape(pathData: WaveformIconPathData.copyBack)
                .fill(color)

            SVGPathShape(pathData: WaveformIconPathData.copyFold)
                .fill(color)
        }
    }
}

struct CopySuccessIconView: View {
    var color: Color = .white

    var body: some View {
        ZStack {
            SVGPathShape(pathData: WaveformIconPathData.copyFront)
                .fill(color)

            SVGPathShape(pathData: WaveformIconPathData.copySuccessBack)
                .fill(color)

            SVGPathShape(pathData: WaveformIconPathData.copySuccessFold)
                .fill(color)
        }
    }
}

struct InjectAnswerIconView: View {
    private let viewport = CGSize(width: 24, height: 24)

    var body: some View {
        ZStack {
            SVGPathShape(
                pathData: WaveformIconPathData.injectBaseline,
                viewport: viewport
            )
            .fill(.white)

            SVGPathShape(
                pathData: WaveformIconPathData.injectPenTip,
                viewport: viewport
            )
            .fill(.white)

            SVGPathShape(
                pathData: WaveformIconPathData.injectBody,
                viewport: viewport
            )
            .fill(.white)
        }
    }
}
