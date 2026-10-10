// SystemNotificationSupport.swift
// Posts macOS User Notifications for long-running background work.

import Foundation
import UserNotifications

extension Notification.Name {
    static let voxtFileTaskNotificationTapped = Notification.Name("voxt.file-task.notification-tapped")
}

enum SystemNotificationSupport {
    private static let center = UNUserNotificationCenter.current()
    private static let presentationDelegate = PresentationDelegate()

    static func configure() {
        center.delegate = presentationDelegate
    }

    /// Warm up the permission prompt before a long task so completion posts are likelier
    /// to already be authorized. `post` still waits for authorization when needed.
    static func requestAuthorizationIfNeeded() {
        center.getNotificationSettings { settings in
            guard settings.authorizationStatus == .notDetermined else { return }
            center.requestAuthorization(options: [.alert, .sound]) { _, _ in }
        }
    }

    static func post(
        title: String,
        body: String,
        identifier: String = UUID().uuidString,
        userInfo: [String: String] = [:]
    ) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        content.sound = .default
        content.userInfo = userInfo

        let request = UNNotificationRequest(
            identifier: identifier,
            content: content,
            trigger: nil
        )
        deliver(request)
    }

    static func postModelDownloadSucceeded(modelName: String) {
        post(
            title: AppLocalization.localizedString("Model download succeeded"),
            body: AppLocalization.format("%@ downloaded successfully.", modelName)
        )
    }

    static func postModelDownloadFailed(modelName: String, message: String) {
        let detail = message.trimmingCharacters(in: .whitespacesAndNewlines)
        let body = detail.isEmpty ? modelName : "\(modelName): \(detail)"
        post(
            title: AppLocalization.localizedString("Model download failed"),
            body: body
        )
    }

    private static func deliver(_ request: UNNotificationRequest) {
        center.getNotificationSettings { settings in
            switch settings.authorizationStatus {
            case .authorized, .provisional, .ephemeral:
                center.add(request)
            case .notDetermined:
                center.requestAuthorization(options: [.alert, .sound]) { granted, _ in
                    guard granted else { return }
                    center.add(request)
                }
            case .denied:
                return
            @unknown default:
                return
            }
        }
    }

    private final class PresentationDelegate: NSObject, UNUserNotificationCenterDelegate {
        func userNotificationCenter(
            _ center: UNUserNotificationCenter,
            willPresent notification: UNNotification
        ) async -> UNNotificationPresentationOptions {
            [.banner, .sound]
        }

        func userNotificationCenter(
            _ center: UNUserNotificationCenter,
            didReceive response: UNNotificationResponse
        ) async {
            guard let taskID = response.notification.request.content.userInfo["fileTaskID"] as? String else {
                return
            }
            await MainActor.run {
                NotificationCenter.default.post(
                    name: .voxtFileTaskNotificationTapped,
                    object: nil,
                    userInfo: ["fileTaskID": taskID]
                )
            }
        }
    }
}
