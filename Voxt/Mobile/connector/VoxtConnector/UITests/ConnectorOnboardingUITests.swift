import XCTest

/// Run only in an approved app-launch slot, against an owned loopback fixture:
/// `tools/connector-health-fixture.py PORT`, then pass TEST_RUNNER_CONNECTOR_FIXTURE_PORT
/// and TEST_RUNNER_CONNECTOR_UI_LAUNCH_APPROVED=1 to xcodebuild.
@MainActor
final class ConnectorOnboardingUITests: XCTestCase {
    private var app: XCUIApplication!
    private var fixturePort: String!

    override func setUp() async throws {
        continueAfterFailure = false
        let environment = ProcessInfo.processInfo.environment
        guard environment["CONNECTOR_UI_LAUNCH_APPROVED"] == "1" else {
            throw XCTSkip("Connector first-launch permission must be approved")
        }
        fixturePort = try XCTUnwrap(environment["CONNECTOR_FIXTURE_PORT"])
        let port = try XCTUnwrap(Int(fixturePort))
        XCTAssertTrue((1...65535).contains(port))
        // The built test target, never an installed Connector, with its own empty phone list.
        app = XCUIApplication()
        app.launchEnvironment["VOXT_CONNECTOR_TEST_STORE"] = FileManager.default.temporaryDirectory
            .appending(path: "connector-ui-\(UUID().uuidString)/connections.json").path
        // No socket exists at this path, so the automatic request behaves as if Voxt were closed.
        app.launchEnvironment["VOXT_CONNECTOR_TEST_SOCKET"] = FileManager.default.temporaryDirectory
            .appending(path: "connector-ui-\(UUID().uuidString).sock").path
        app.launch()
    }

    override func tearDown() async throws {
        app?.terminate()
    }

    func testFirstRunOpensOneSettingsWindowWithoutAStoredCredential() {
        let window = app.windows["Voxt Connector"]
        XCTAssertTrue(window.waitForExistence(timeout: 5))
        XCTAssertEqual(app.windows.count, 1)
        XCTAssertTrue(window.textFields["connectorPort"].exists)
        XCTAssertTrue(window.secureTextFields["connectorToken"].exists)
        XCTAssertEqual(window.secureTextFields["connectorToken"].value as? String, "")
        let status = window.staticTexts["connectorStatus"]
        XCTAssertEqual(status.label, "Connect to Voxt on this Mac")
        XCTAssertTrue(status.isHittable, "status line must be visible without scrolling")
        XCTAssertTrue(window.frame.contains(status.frame))
        XCTAssertFalse(window.buttons["connectorAddPhone"].isEnabled)
        // Off by default; the test never turns it on, so the debug build isn't added to Login Items.
        let openAtLogin = window.descendants(matching: .any)["connectorOpenAtLogin"]
        XCTAssertTrue(openAtLogin.exists)
        XCTAssertEqual("\(openAtLogin.value ?? "")", "0")
        let showInDock = window.descendants(matching: .any)["connectorShowInDock"]
        XCTAssertTrue(showInDock.exists)
        XCTAssertEqual("\(showInDock.value ?? "")", "1")
    }

    /// The test build is unsigned, so it has no team to check Voxt against and never opens the socket.
    func testUnsignedBuildPointsToManualEntry() {
        app.buttons["connectorConnect"].click()
        assertStatus("Voxt doesn't accept this Connector build. Enter the port and Access token instead.")
        assertFeedback("Voxt doesn't accept this Connector build. Enter the port and Access token instead.")
        XCTAssertFalse(app.buttons["connectorAddPhone"].isEnabled)
    }

    func testInvalidPortShowsAnActionableError() {
        enterPort("0")
        let field = app.secureTextFields["connectorToken"]
        field.click()
        field.typeText("connector-owned-fixture")
        app.buttons["connectorConnect"].click()
        assertStatus("Enter the port shown in Voxt Settings.")
        assertFeedback("Enter the port shown in Voxt Settings.")
    }

    func testRejectedTokenNeverShowsReadyAndValidTokenConnects() {
        enterPort(fixturePort)
        let field = app.secureTextFields["connectorToken"]
        field.click()
        field.typeText("invalid-fixture-token")
        app.buttons["connectorConnect"].click()
        assertStatus("The access token was rejected. Copy it from Voxt Settings.")
        assertFeedback("The access token was rejected. Copy it from Voxt Settings.")
        XCTAssertFalse(app.buttons["connectorAddPhone"].isEnabled)
        field.click()
        field.typeKey("a", modifierFlags: .command)
        field.typeText("connector-owned-fixture")
        app.buttons["connectorConnect"].click()
        assertStatus("Voxt fixture is available")
        assertFeedback("Connected to Voxt fixture")
        XCTAssertTrue(app.buttons["connectorAddPhone"].isEnabled)
        XCTAssertTrue(app.staticTexts["connectorStatus"].isHittable)
    }

    private func enterPort(_ value: String) {
        let field = app.textFields["connectorPort"]
        XCTAssertTrue(field.waitForExistence(timeout: 5))
        field.click()
        field.typeKey("a", modifierFlags: .command)
        field.typeText(value)
    }

    private func assertFeedback(_ text: String) {
        let feedback = app.descendants(matching: .any)["connectorConnectFeedback"]
        let match = expectation(for: NSPredicate(format: "label == %@", text), evaluatedWith: feedback)
        wait(for: [match], timeout: 5)
    }

    private func assertStatus(_ text: String) {
        let predicate = NSPredicate(format: "label == %@", text)
        let match = expectation(for: predicate, evaluatedWith: app.staticTexts["connectorStatus"])
        wait(for: [match], timeout: 5)
    }
}
