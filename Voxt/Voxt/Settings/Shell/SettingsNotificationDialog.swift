import SwiftUI
import AppKit

extension SettingsView {
    struct HomeNotificationDialogView: View {
        @ObservedObject var store: VoxtNotificationStore
        let onClose: () -> Void
        @State private var selectedNotificationID: String?
        @State private var isListExpanded = false

        private var selectedNotification: VoxtAppNotification? {
            if let selectedNotificationID,
               let notification = store.notifications.first(where: { $0.id == selectedNotificationID }) {
                return notification
            }
            return store.latestNotification
        }

        var body: some View {
            VStack(alignment: .leading, spacing: 14) {
                HStack(spacing: 8) {
                    Text(AppLocalization.localizedString("Notifications"))
                        .font(.title3.weight(.semibold))

                    Button {
                        withAnimation(.easeInOut(duration: 0.18)) {
                            isListExpanded.toggle()
                        }
                    } label: {
                        HStack(spacing: 4) {
                            HomeNotificationListToggleIcon(isExpanded: isListExpanded)
                                .frame(width: 13, height: 13)
                            Text(AppLocalization.localizedString("Notification History"))
                                .font(.system(size: 11, weight: .semibold))
                        }
                    }
                    .buttonStyle(SettingsPillButtonStyle(horizontalPadding: 9, height: 26))

                    if store.isLoading {
                        ProgressView()
                            .controlSize(.small)
                    }

                    Spacer(minLength: 0)
                }

                if let errorMessage = store.lastErrorMessage {
                    HStack(spacing: 8) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(.orange)
                        Text(errorMessage)
                            .font(.system(size: 12))
                            .foregroundStyle(.secondary)
                        Spacer(minLength: 8)
                        Button(AppLocalization.localizedString("Refresh")) {
                            Task {
                                await store.refresh(force: true)
                            }
                        }
                        .buttonStyle(SettingsPillButtonStyle())
                        .disabled(store.isLoading)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(SettingsUIStyle.groupedFillColor)
                    .clipShape(RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous))
                }

                HStack(spacing: 0) {
                    if isListExpanded {
                        notificationList
                            .frame(width: 220)
                            .frame(maxHeight: .infinity)
                            .background(SettingsUIStyle.groupedFillColor)
                            .transition(.move(edge: .leading).combined(with: .opacity))

                        Divider()
                            .transition(.opacity)
                    }

                    notificationDetail
                        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
                }
                .clipShape(RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous))
                .overlay(
                    RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous)
                        .strokeBorder(SettingsUIStyle.subtleBorderColor, lineWidth: 1)
                )
                .animation(.easeInOut(duration: 0.18), value: isListExpanded)

            }
            .settingsDialogChrome(width: isListExpanded ? 760 : 540, height: 500, onClose: onClose)
            .animation(.easeInOut(duration: 0.18), value: isListExpanded)
            .onAppear {
                selectedNotificationID = selectedNotification?.id
                markSelectedNotificationAsViewed()
                Task {
                    await store.refresh(force: true)
                }
            }
            .onChange(of: store.notifications) { _, notifications in
                guard !notifications.isEmpty else {
                    selectedNotificationID = nil
                    return
                }
                if selectedNotificationID == nil ||
                    !notifications.contains(where: { $0.id == selectedNotificationID }) {
                    selectedNotificationID = notifications.first?.id
                }
                markSelectedNotificationAsViewed()
            }
            .onChange(of: selectedNotificationID) { _, _ in
                markSelectedNotificationAsViewed()
            }
        }

        private var notificationList: some View {
            Group {
                if store.notifications.isEmpty {
                    VStack(spacing: 10) {
                        Image(systemName: "bell")
                            .font(.system(size: 24, weight: .medium))
                        Text(AppLocalization.localizedString("No Notifications"))
                            .font(.system(size: 12, weight: .medium))
                    }
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                } else {
                    List(store.notifications, id: \.id, selection: $selectedNotificationID) { notification in
                        HomeNotificationListRow(notification: notification)
                            .tag(notification.id)
                    }
                    .listStyle(.sidebar)
                    .scrollContentBackground(.hidden)
                }
            }
        }

        @ViewBuilder
        private var notificationDetail: some View {
            if let selectedNotification {
                ScrollView {
                    VStack(alignment: .leading, spacing: 14) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(selectedNotification.title)
                                .font(.system(size: 20, weight: .bold))
                                .foregroundStyle(.primary)
                                .fixedSize(horizontal: false, vertical: true)

                            HStack(spacing: 8) {
                                Text(formattedPublishedAt(selectedNotification.publishedAt))
                                Text("|")
                                Text(AppLocalization.format("Version %@", selectedNotification.version))
                            }
                            .font(.system(size: 12, weight: .medium))
                            .foregroundStyle(.secondary)
                        }

                        if let coverImageURL = selectedNotification.coverImageURL {
                            AsyncImage(url: coverImageURL) { phase in
                                switch phase {
                                case .success(let image):
                                    image
                                        .resizable()
                                        .scaledToFill()
                                case .failure:
                                    coverPlaceholder
                                case .empty:
                                    ProgressView()
                                        .controlSize(.small)
                                @unknown default:
                                    ProgressView()
                                        .controlSize(.small)
                                }
                            }
                            .frame(maxWidth: .infinity)
                            .frame(height: 200)
                            .clipShape(RoundedRectangle(cornerRadius: SettingsUIStyle.compactCornerRadius, style: .continuous))
                        }

                        Text(selectedNotification.content)
                            .font(.system(size: 13))
                            .lineSpacing(4)
                            .foregroundStyle(.primary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(18)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                }
            } else {
                VStack(spacing: 10) {
                    Image(systemName: "doc.text")
                        .font(.system(size: 28, weight: .medium))
                    Text(AppLocalization.localizedString("Select Notification"))
                        .font(.system(size: 13, weight: .medium))
                }
                .foregroundStyle(.secondary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }

        private var coverPlaceholder: some View {
            ZStack {
                SettingsUIStyle.groupedFillColor
                Image(systemName: "photo")
                    .font(.title3.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
        }

        private func formattedPublishedAt(_ date: Date) -> String {
            date.formatted(date: .abbreviated, time: .shortened)
        }

        private func markSelectedNotificationAsViewed() {
            store.markAsViewed(selectedNotification)
        }
    }

    private struct HomeNotificationListRow: View {
        let notification: VoxtAppNotification

        var body: some View {
            VStack(alignment: .leading, spacing: 4) {
                Text(notification.title)
                    .font(.system(size: 12, weight: .semibold))
                    .lineLimit(2)

                HStack(spacing: 6) {
                    Text(notification.version)
                    Text(formattedDate(notification.publishedAt))
                }
                .font(.system(size: 10, weight: .medium))
                .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        }

        private func formattedDate(_ date: Date) -> String {
            date.formatted(date: .numeric, time: .omitted)
        }
    }

    private struct HomeNotificationListToggleIcon: View {
        let isExpanded: Bool

        var body: some View {
            ZStack {
                ForEach(Array(paths.enumerated()), id: \.offset) { _, path in
                    SVGPathShape(pathData: path)
                        .fill(.primary)
                }
            }
            .frame(width: 13, height: 13)
        }

        private var paths: [String] {
            isExpanded ? collapsePaths : openPaths
        }

        private var openPaths: [String] {
            [
                "M14.97 22.75H8.96997C3.53997 22.75 1.21997 20.43 1.21997 15V9C1.21997 3.57 3.53997 1.25 8.96997 1.25H14.97C20.4 1.25 22.72 3.57 22.72 9V15C22.72 20.43 20.41 22.75 14.97 22.75ZM8.96997 2.75C4.35997 2.75 2.71997 4.39 2.71997 9V15C2.71997 19.61 4.35997 21.25 8.96997 21.25H14.97C19.58 21.25 21.22 19.61 21.22 15V9C21.22 4.39 19.58 2.75 14.97 2.75H8.96997Z",
                "M14.97 22.75C14.56 22.75 14.22 22.41 14.22 22V2C14.22 1.59 14.56 1.25 14.97 1.25C15.38 1.25 15.72 1.59 15.72 2V22C15.72 22.41 15.39 22.75 14.97 22.75Z",
                "M7.96991 15.3109C7.77991 15.3109 7.58991 15.2409 7.43991 15.0909C7.14991 14.8009 7.14991 14.3209 7.43991 14.0309L9.46991 12.0009L7.43991 9.97086C7.14991 9.68086 7.14991 9.20086 7.43991 8.91086C7.72991 8.62086 8.20991 8.62086 8.49991 8.91086L11.0599 11.4709C11.3499 11.7609 11.3499 12.2409 11.0599 12.5309L8.49991 15.0909C8.35991 15.2409 8.16991 15.3109 7.96991 15.3109Z",
            ]
        }

        private var collapsePaths: [String] {
            [
                "M14.97 22.75H8.96997C3.53997 22.75 1.21997 20.43 1.21997 15V9C1.21997 3.57 3.53997 1.25 8.96997 1.25H14.97C20.4 1.25 22.72 3.57 22.72 9V15C22.72 20.43 20.41 22.75 14.97 22.75ZM8.96997 2.75C4.35997 2.75 2.71997 4.39 2.71997 9V15C2.71997 19.61 4.35997 21.25 8.96997 21.25H14.97C19.58 21.25 21.22 19.61 21.22 15V9C21.22 4.39 19.58 2.75 14.97 2.75H8.96997Z",
                "M7.96997 22.75C7.55997 22.75 7.21997 22.41 7.21997 22V2C7.21997 1.59 7.55997 1.25 7.96997 1.25C8.37997 1.25 8.71997 1.59 8.71997 2V22C8.71997 22.41 8.38997 22.75 7.96997 22.75Z",
                "M14.9701 15.3109C14.7801 15.3109 14.5901 15.2409 14.4401 15.0909L11.8801 12.5309C11.5901 12.2409 11.5901 11.7609 11.8801 11.4709L14.4401 8.91086C14.7301 8.62086 15.2101 8.62086 15.5001 8.91086C15.7901 9.20086 15.7901 9.68086 15.5001 9.97086L13.4801 12.0009L15.5101 14.0309C15.8001 14.3209 15.8001 14.8009 15.5101 15.0909C15.3601 15.2409 15.1701 15.3109 14.9701 15.3109Z",
            ]
        }
    }
}
