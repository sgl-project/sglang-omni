// BrowserAutomationScriptBuilder.swift
// Provides Browser Automation Script Builder for shared utilities.

import Foundation

enum BrowserAutomationScriptBuilder {
    /// Source tags prefixed onto the AppleScript JS return value:
    /// `F` form control, `E` contenteditable/textbox, `P` page selection.
    ///
    /// Uses a printable separator — control characters like SOH are often stripped
    /// by the Apple Event / Chromium JS bridge, which previously made real form
    /// selections look "untagged" and get rejected when AX focus was unavailable.
    static let selectionSourceForm = "F"
    static let selectionSourceEditor = "E"
    static let selectionSourcePage = "P"
    static let selectionSourceSeparator = "::V::"

    /// Full probe: form / editor / page selection.
    static var selectionJavaScript: String {
        selectionJavaScript(allowPageSelection: true)
    }

    /// AX-blind probe: only form / editor selection. Never returns page
    /// `getSelection()` text (avoids residual page ranges in empty inputs).
    static var formOrEditorSelectionJavaScript: String {
        selectionJavaScript(allowPageSelection: false)
    }

    /// Reads the editing selection for the deep active element.
    ///
    /// Critical rules (note / translation / rewrite / dictionary share this probe):
    /// - `INPUT` / `TEXTAREA`: use `selectionStart/End` only. Never fall back to
    ///   page `getSelection()` — residual page ranges are a common false positive
    ///   while the caret sits in a focused field with nothing selected.
    /// - `contenteditable` / `role=textbox`: accept only a non-collapsed selection
    ///   anchored inside the focused editor.
    /// - Otherwise (page body / article text): optionally accept non-collapsed
    ///   `getSelection()` when `allowPageSelection` is true.
    ///
    /// Return format: `<source>::V::<text>` where source is F/E/P.
    /// Uses single-quoted JS only so the snippet stays safe inside AppleScript
    /// double-quoted `execute javascript "..."` / `do JavaScript "..."` strings.
    static func selectionJavaScript(allowPageSelection: Bool) -> String {
        let pageReturn = allowPageSelection
            ? "return 'P'+sep+String(sel.toString()||'');"
            : "return 'P'+sep;"
        return (
            "(" +
            "function(){" +
            "var sep='::V::';" +
            "function deepActive(doc){" +
            "var el=doc.activeElement;var d=doc;" +
            "for(var i=0;i<24&&el;i++){" +
            "if(el.shadowRoot&&el.shadowRoot.activeElement){el=el.shadowRoot.activeElement;continue;}" +
            "if(el.tagName==='IFRAME'){" +
            "try{" +
            "var idoc=el.contentDocument||(el.contentWindow&&el.contentWindow.document);" +
            "if(idoc&&idoc.activeElement){d=idoc;el=idoc.activeElement;continue;}" +
            "}catch(e){return {el:null,doc:d};}" +
            "}" +
            "break;" +
            "}" +
            "return {el:el,doc:d};" +
            "}" +
            "var ctx=deepActive(document);var el=ctx.el;var doc=ctx.doc||document;" +
            "var tag=el&&el.tagName?String(el.tagName).toUpperCase():'';" +
            "if(tag==='INPUT'||tag==='TEXTAREA'){" +
            "try{" +
            "var a=el.selectionStart,b=el.selectionEnd;" +
            "if(typeof a==='number'&&typeof b==='number'){" +
            "return 'F'+sep+(b>a?String(el.value).substring(a,b):'');" +
            "}" +
            "}catch(e){}" +
            "return 'F'+sep;" +
            "}" +
            "var editable=!!(el&&(el.isContentEditable||(el.getAttribute&&(el.getAttribute('contenteditable')==='true'||el.getAttribute('role')==='textbox'))));" +
            "var sel=(doc.getSelection&&doc.getSelection())||window.getSelection();" +
            "if(!sel||!sel.rangeCount||sel.isCollapsed){" +
            "return (editable?'E':'P')+sep;" +
            "}" +
            "if(editable){" +
            "try{" +
            "var node=sel.anchorNode;" +
            "if(node&&el&&!el.contains(node.nodeType===1?node:node.parentNode))return 'E'+sep;" +
            "}catch(e){return 'E'+sep;}" +
            "return 'E'+sep+String(sel.toString()||'');" +
            "}" +
            pageReturn +
            "}" +
            ")()"
        )
    }

    /// AppleScript candidates that read the current page selection via JavaScript.
    /// Requires "Allow JavaScript from Apple Events" in the target browser.
    ///
    /// Production always uses `allowPageSelection == true`. Form-first JS still
    /// returns caret-only `F` empty inside inputs, so page ranges do not leak.
    /// `allowPageSelection == false` remains for focused form/editor-only tests.
    static func selectionScripts(
        bundleID: String,
        displayName: String?,
        allowPageSelection: Bool = true
    ) -> [String] {
        let js = selectionJavaScript(allowPageSelection: allowPageSelection)
        let name = displayName ?? bundleID
        switch bundleID {
        case "com.apple.Safari", "com.apple.SafariTechnologyPreview":
            return [
                "tell application id \"\(bundleID)\" to tell front window to do JavaScript \"\(js)\" in current tab",
                "tell application id \"\(bundleID)\" to do JavaScript \"\(js)\" in front document",
                "tell application \"\(name)\" to tell front window to do JavaScript \"\(js)\" in current tab"
            ]
        case "com.google.Chrome",
             "com.microsoft.edgemac",
             "com.brave.Browser",
             "company.thebrowser.Browser":
            return chromiumSelectionScripts(bundleID: bundleID, displayName: name, javaScript: js)
        default:
            // Custom browsers: try Chromium then Safari-style forms.
            return chromiumSelectionScripts(bundleID: bundleID, displayName: name, javaScript: js) + [
                "tell application id \"\(bundleID)\" to tell front window to do JavaScript \"\(js)\" in current tab",
                "tell application id \"\(bundleID)\" to do JavaScript \"\(js)\" in front document",
                "tell application \"\(name)\" to tell front window to do JavaScript \"\(js)\" in current tab"
            ]
        }
    }

    private static func chromiumSelectionScripts(
        bundleID: String,
        displayName: String,
        javaScript: String
    ) -> [String] {
        [
            "tell application id \"\(bundleID)\" to tell active tab of front window to execute javascript \"\(javaScript)\"",
            "tell application id \"\(bundleID)\" to tell active tab of window 1 to execute javascript \"\(javaScript)\"",
            "tell application \"\(displayName)\" to tell active tab of front window to execute javascript \"\(javaScript)\""
        ]
    }

    static func customBrowserPermissionProbeScripts(bundleID: String, displayName: String) -> [String] {
        // Permission probes run inside Settings, where avoiding UI freezes is
        // more important than preserving runtime ordering. Try the tab-based
        // variants first because some Chromium-like browsers return faster on
        // those forms during permission checks.
        [
            "tell application id \"\(bundleID)\" to get the URL of active tab of front window",
            "tell application id \"\(bundleID)\" to get the URL of active tab of window 1",
            "tell application \"\(displayName)\" to get the URL of active tab of front window",
            "tell application id \"\(bundleID)\" to get URL of front document",
            "tell application id \"\(bundleID)\" to get URL of current tab of front window",
            "tell application \"\(displayName)\" to get URL of front document"
        ]
    }

    static func customBrowserRuntimeScripts(bundleID: String, displayName: String) -> [String] {
        // Runtime URL reads happen on the app's normal path, so keep the more
        // conservative ordering that favors the historically successful
        // front-document / current-tab forms before falling back to others.
        [
            "tell application id \"\(bundleID)\" to get URL of front document",
            "tell application id \"\(bundleID)\" to get URL of current tab of front window",
            "tell application id \"\(bundleID)\" to get the URL of active tab of front window",
            "tell application id \"\(bundleID)\" to get the URL of active tab of window 1",
            "tell application \"\(displayName)\" to get URL of front document",
            "tell application \"\(displayName)\" to get the URL of active tab of front window"
        ]
    }
}
