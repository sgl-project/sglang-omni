import Foundation
import Testing
@testable import VoxtConnectionKit

@MainActor private final class Received { var commands: [ConnectorTestCommand] = [] }

@MainActor @Test func hooksStayOffWithoutTheLaunchVariable() {
    let center = NotificationCenter(), received = Received()
    for environment in [[:], ["VOXT_CONNECTOR_TEST_HOOKS": "0"], ["VOXT_CONNECTOR_TEST_HOOKS": "yes"]] {
        let hooks = ConnectorTestHooks.install(environment: environment, center: center) { received.commands.append($0) }
        #expect(hooks == nil)
    }
    for command in ConnectorTestCommand.allCases {
        center.post(name: ConnectorTestHooks.notificationName(command), object: nil)
    }
    #expect(received.commands.isEmpty)
}

@MainActor @Test func eachCommandReachesTheHandlerWhenLaunchedForTests() async {
    let center = NotificationCenter(), received = Received()
    let hooks = ConnectorTestHooks.install(environment: ["VOXT_CONNECTOR_TEST_HOOKS": "1"], center: center) {
        received.commands.append($0)
    }
    #expect(hooks != nil)
    for command in ConnectorTestCommand.allCases {
        center.post(name: ConnectorTestHooks.notificationName(command), object: nil)
    }
    #expect(received.commands == ConnectorTestCommand.allCases)
    #expect(ConnectorTestHooks.notificationName(.showInDockOff).rawValue == "com.voxt.VoxtConnector.test.showInDockOff")
    _ = hooks
}
