import Foundation

enum RemoteASRErrorPresentation {
    static func message(for error: Error) -> String {
        if let conflictMessage = VoxtNetworkSession.directModeConflictMessage(for: error) {
            return conflictMessage
        }
        if let proxyUnavailableMessage = VoxtNetworkSession.activeProxyUnavailableMessage(for: error) {
            return proxyUnavailableMessage
        }
        let nsError = error as NSError
        let description = nsError.localizedDescription.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedDescription = description.lowercased()
        if nsError.domain == NSURLErrorDomain {
            switch nsError.code {
            case NSURLErrorNotConnectedToInternet:
                return AppLocalization.localizedString("Network appears to be offline. Check your connection and try again.")
            case NSURLErrorTimedOut:
                return AppLocalization.localizedString("Remote ASR timed out. Please try again.")
            case NSURLErrorCannotConnectToHost, NSURLErrorNetworkConnectionLost, NSURLErrorCannotFindHost:
                return AppLocalization.localizedString("Couldn't reach the remote ASR service. Check your network and proxy settings.")
            default:
                break
            }
        }
        switch nsError.code {
        case 401:
            return AppLocalization.localizedString("Remote ASR authentication failed. Check the provider credentials and try again.")
        case 402:
            return AppLocalization.localizedString("Remote ASR billing or quota is unavailable. Check the provider account balance and limits.")
        case 403:
            return AppLocalization.localizedString("Remote ASR access was denied. Check the provider permissions, region, or endpoint.")
        case 408, 504:
            return AppLocalization.localizedString("Remote ASR timed out. Please try again.")
        case 409, 429:
            return AppLocalization.localizedString("Remote ASR is busy or has reached its quota. Please wait a moment and try again.")
        case 500 ... 599:
            return AppLocalization.localizedString("Remote ASR is temporarily unavailable. Please try again later.")
        default:
            break
        }
        if normalizedDescription.contains("exceededconcurrentquota")
            || normalizedDescription.contains("quota")
            || normalizedDescription.contains("rate limit")
            || normalizedDescription.contains("too many requests")
            || normalizedDescription.contains("concurrent") {
            return AppLocalization.localizedString("Remote ASR is busy or has reached its quota. Please wait a moment and try again.")
        }
        if normalizedDescription.contains("billing")
            || normalizedDescription.contains("insufficient")
            || normalizedDescription.contains("balance")
            || normalizedDescription.contains("arrears")
            || normalizedDescription.contains("欠费")
            || normalizedDescription.contains("余额")
            || normalizedDescription.contains("费用") {
            return AppLocalization.localizedString("Remote ASR billing or quota is unavailable. Check the provider account balance and limits.")
        }
        if normalizedDescription.contains("unauthorized")
            || normalizedDescription.contains("forbidden")
            || normalizedDescription.contains("access token")
            || normalizedDescription.contains("api key")
            || normalizedDescription.contains("鉴权")
            || normalizedDescription.contains("权限") {
            return AppLocalization.localizedString("Remote ASR authentication failed. Check the provider credentials and try again.")
        }
        if normalizedDescription.contains("network")
            || normalizedDescription.contains("socket is not connected")
            || normalizedDescription.contains("proxy")
            || normalizedDescription.contains("vpn") {
            return AppLocalization.localizedString("Couldn't reach the remote ASR service. Check your network and proxy settings.")
        }
        return description.isEmpty
            ? AppLocalization.localizedString("Remote ASR request failed.")
            : description
    }
}
