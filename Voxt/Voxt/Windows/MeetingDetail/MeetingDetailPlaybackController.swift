import Foundation
import AVFoundation
import Combine

@MainActor
final class MeetingDetailPlaybackController: ObservableObject {
    static let minimumPlaybackRate: Float = 1
    static let maximumPlaybackRate: Float = 2.5

    @Published var currentTime: TimeInterval = 0
    @Published var duration: TimeInterval = 0
    @Published var isPlaying = false
    @Published private(set) var playbackRate: Float = 1

    private var player: AVAudioPlayer?
    private var timer: Timer?

    init(audioURL: URL?) {
        guard let audioURL else { return }
        player = try? AVAudioPlayer(contentsOf: audioURL)
        player?.enableRate = true
        player?.rate = playbackRate
        player?.prepareToPlay()
        duration = player?.duration ?? 0
    }

    deinit {
        timer?.invalidate()
    }

    var isAvailable: Bool {
        player != nil && duration > 0
    }

    func togglePlayPause() {
        guard let player else { return }
        if player.isPlaying {
            pause()
        } else {
            player.play()
            isPlaying = true
            startTimer()
        }
    }

    func pause() {
        guard let player else { return }
        player.pause()
        isPlaying = false
        stopTimer()
    }

    func seek(to time: TimeInterval) {
        guard let player else { return }
        let clamped = max(0, min(time, duration))
        player.currentTime = clamped
        currentTime = clamped
    }

    func seek(by offset: TimeInterval) {
        seek(to: currentTime + offset)
    }

    func setPlaybackRate(_ rate: Float) {
        let clampedRate = min(max(rate, Self.minimumPlaybackRate), Self.maximumPlaybackRate)
        playbackRate = clampedRate
        player?.enableRate = true
        player?.rate = clampedRate
    }

    private func startTimer() {
        stopTimer()
        timer = Timer.scheduledTimer(withTimeInterval: 0.15, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, let player = self.player else { return }
                self.currentTime = player.currentTime
                if !player.isPlaying {
                    self.isPlaying = false
                    self.stopTimer()
                }
            }
        }
    }

    private func stopTimer() {
        timer?.invalidate()
        timer = nil
    }
}
