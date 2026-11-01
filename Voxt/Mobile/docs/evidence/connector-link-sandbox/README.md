# Connector link under the App Sandbox

What this checks: a sandboxed Voxt (the release build) hands its port and token only to a Voxt Connector signed by the same team, found where both sides look for the socket.

The server is Voxt's own `Voxt/Core/LocalAPI/LocalAPIConnectorLink.swift` in a two-file app that has only the `com.apple.security.app-sandbox` entitlement. The client is Voxt Connector's own `ConnectorLink.swift` and `ConnectorIdentity.swift` (in `Mobile/connector/VoxtConnectionKit`). Both use test bundle IDs (`com.voxt.Voxt.linkcheck`, `com.voxt.VoxtConnector.linkcheck`) so a Voxt installed on the same Mac is not touched. No Voxt UI, microphone or model runs.

## Build and sign

`IDENTITY` is a Developer ID Application identity; both apps must use the same team.

```sh
L=../../../../Voxt/Core/LocalAPI K=../../../connector/VoxtConnectionKit/Sources/VoxtConnectionKit
mkdir -p LinkServer.app/Contents/MacOS srv cli
cp server-Info.plist LinkServer.app/Contents/Info.plist; cp server.swift srv/main.swift; cp client.swift cli/main.swift
swiftc -O -o LinkServer.app/Contents/MacOS/LinkServer $L/LocalAPIConnectorLink.swift srv/main.swift
swiftc -O -o linkclient $K/ConnectorLink.swift $K/ConnectorIdentity.swift cli/main.swift \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker client-Info.plist
codesign --force --options runtime --timestamp=none --entitlements sandbox.entitlements --sign "$IDENTITY" LinkServer.app
mkdir -p "LinkCheck-stage/Voxt LinkCheck.app/Contents/MacOS"
cp linkclient "LinkCheck-stage/Voxt LinkCheck.app/Contents/MacOS/"; cp client-Info.plist "LinkCheck-stage/Voxt LinkCheck.app/Contents/Info.plist"
codesign --force --options runtime --timestamp=none --sign "$IDENTITY" "LinkCheck-stage/Voxt LinkCheck.app"
codesign --force --options runtime --timestamp=none --identifier com.voxt.VoxtConnector.linkcheck --sign "$IDENTITY" linkclient
```

`run-applications.sh` copies the client bundle to `/Applications/Voxt LinkCheck.app` for one run and removes it on exit. `run-broken-fresh.sh` checks a bundle with a broken seal against a server that never saw the intact one.

## Results

Server start, inside the sandbox:

```
server sandbox container: com.voxt.Voxt.linkcheck
server socket: ~/Library/Containers/com.voxt.Voxt.linkcheck/Data/connector.sock (<n> bytes)
server requires connector: com.voxt.VoxtConnector.linkcheck
server listening
```

`./run-applications.sh`, client lines, then the server's log of each peer check:

```
== client in /Applications, signed
client approved port=47999 token=fixture-token alwaysAllowed=true
== same bundle, seal broken by an added resource
client failed: rejectedBySignature
== client outside /Applications
client failed: rejectedBySignature

server peer check start, guest lookup 0
server peer check true after 4 ms
server peer check start, guest lookup 0
server peer check false after 5 ms
server peer check start, guest lookup 100001
server peer check false after 0 ms
```

`./run-broken-fresh.sh`:

```
client failed: rejectedBySignature
server peer check start, guest lookup 0
server peer check false after 5 ms
```

What the results show:

- The socket at the top of the container fits the 103-byte limit of a socket path: 77 bytes for a seven-letter user name. Application Support inside Voxt's container would take 100 bytes for that name, and is too long for a user name of more than ten letters.
- The unsandboxed client, signed by the same team, connected into Voxt's container without a prompt.
- The sandbox lets Voxt look up a peer's code only when the peer's executable is somewhere the sandbox may read: the client in `/Applications` passes (lookup 0); the same client elsewhere is refused (lookup 100001, EPERM), and the Connector then uses manual entry.
- Before the on-disk check was added, the bundle with a file added after signing was approved by the running-code check alone. With `SecStaticCodeCheckValidity` (strict) against the same requirement it is refused.

## Voxt unit tests

From `Voxt/`, unsigned as in CI:

```sh
xcodebuild build-for-testing -project Voxt.xcodeproj -scheme Voxt -configuration TestDebug \
  -destination platform=macOS -derivedDataPath build/derived -skipPackagePluginValidation CODE_SIGNING_ALLOWED=NO
xcodebuild test-without-building -xctestrun build/derived/Build/Products/Voxt_*.xctestrun -destination platform=macOS \
  -only-testing:VoxtTests/LocalAPIComposePromptTests -only-testing:VoxtTests/LocalAPIConnectorLinkTests \
  -only-testing:VoxtTests/LocalAPIContractTests -only-testing:VoxtTests/LocalAPIDictationPipelineTests \
  -only-testing:VoxtTests/LocalAPIHTTPTests -only-testing:VoxtTests/LocalAPIRequestHandlerTests \
  -only-testing:VoxtTests/LocalAPIRouterTests -only-testing:VoxtTests/LocalAPIServerTests \
  -only-testing:VoxtTests/LocalAPISettingsTests -only-testing:VoxtTests/ASRHintSettingsTests \
  -only-testing:VoxtTests/EnhancementLanguageGuardTests -only-testing:VoxtTests/LLMExecutionPlanCompilerTests
```

```
** TEST BUILD SUCCEEDED **
Executed 228 tests, with 0 failures (0 unexpected)
** TEST EXECUTE SUCCEEDED **
```

An app built with `CODE_SIGNING_ALLOWED=NO` is only ad-hoc linker-signed, with no team and no entitlements (`codesign -dv`: `flags=0x20002(adhoc,linker-signed)`, `TeamIdentifier=not set`), so it runs without the sandbox and does not open the Connector link. The sandbox needs a signed build, as above.
