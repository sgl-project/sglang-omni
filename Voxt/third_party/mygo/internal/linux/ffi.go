//go:build linux && (amd64 || arm64)

// Package linux implements the Linux backend (GTK 3 + WebKitGTK) in pure Go:
// the libraries are loaded at run time with purego, no cgo.
package linux

import (
	"errors"
	"fmt"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"
)

type ptr = uintptr

// Library handles.
var (
	libGLib, libGObject, libGIO, libGDK, libGTK, libWebKit, libJSC, libSoup, libCairo, libPixbuf, libIndicator ptr
)

// cs returns a NUL terminated copy of s for C functions.
func cs(s string) *byte {
	b := make([]byte, len(s)+1)
	copy(b, s)
	return &b[0]
}

// optCS is cs that maps "" to NULL.
func optCS(s string) *byte {
	if s == "" {
		return nil
	}
	return cs(s)
}

// goStr copies a C string.
func goStr(p ptr) string {
	if p == 0 {
		return ""
	}
	base := *(*unsafe.Pointer)(unsafe.Pointer(&p))
	return string(unsafe.Slice((*byte)(base), cStrlen(base)))
}

// strlenFn is libc's strlen, which measures long strings, such as the
// messages of pages, much faster than a loop.
var strlenFn, _ = purego.Dlsym(purego.RTLD_DEFAULT, "strlen")

func cStrlen(s unsafe.Pointer) int {
	for n := 0; n < 64 || strlenFn == 0; n++ {
		if *(*byte)(unsafe.Add(s, n)) == 0 {
			return n
		}
	}
	n, _, _ := purego.SyscallN(strlenFn, uintptr(s))
	return int(n)
}

// takeStr copies and frees a C string allocated by GLib.
func takeStr(p ptr) string {
	s := goStr(p)
	if p != 0 {
		gFree(p)
	}
	return s
}

// gErr converts and frees a GError.
func gErr(e ptr) error {
	if e == 0 {
		return nil
	}
	// struct GError { GQuark domain; gint code; gchar *message; }
	msg := goStr(*(*ptr)(unsafe.Add(*(*unsafe.Pointer)(unsafe.Pointer(&e)), 8)))
	gErrorFree(e)
	return errors.New(msg)
}

func open(names ...string) (ptr, error) {
	var errs []string
	for _, n := range names {
		h, err := purego.Dlopen(n, purego.RTLD_NOW|purego.RTLD_GLOBAL)
		if err == nil {
			return h, nil
		}
		errs = append(errs, err.Error())
	}
	return 0, fmt.Errorf("cannot load %s: %s", names[0], strings.Join(errs, "; "))
}

// bind registers fn from lib; missing symbols leave fn nil.
func bind(lib ptr, fn any, name string) bool {
	sym, err := purego.Dlsym(lib, name)
	if err != nil || sym == 0 {
		return false
	}
	purego.RegisterFunc(fn, sym)
	return true
}

// mustBind is bind for required symbols.
func mustBind(lib ptr, fn any, name string) {
	if !bind(lib, fn, name) {
		panic("mygo: missing symbol " + name)
	}
}

// GLib, GObject and GIO.
var (
	gFree                          func(p ptr)
	gErrorFree                     func(e ptr)
	gStrfreev                      func(v ptr)
	gFilenameFromURI               func(uri string, hostname, gerr ptr) ptr
	gErrorNewLiteral               func(domain uint32, code int32, msg *byte) ptr
	gQuarkFromString               func(s *byte) uint32
	gIdleAddFull                   func(priority int32, fn ptr, data ptr, notify ptr) uint32
	gTimeoutAdd                    func(ms uint32, fn ptr, data ptr) uint32
	gMainContextIteration          func(ctx ptr, mayBlock bool) bool
	gMainContextWakeup             func(ctx ptr)
	gMainLoopNew                   func(ctx ptr, running bool) ptr
	gMainLoopRun                   func(loop ptr)
	gMainLoopQuit                  func(loop ptr)
	gMainLoopUnref                 func(loop ptr)
	gSignalConnectData             func(instance ptr, signal *byte, handler ptr, data ptr, destroy ptr, flags int32) uint64
	gSignalHandlerDisconnect       func(instance ptr, id uint64)
	gObjectRef                     func(obj ptr) ptr
	gObjectRefSink                 func(obj ptr) ptr
	gObjectUnref                   func(obj ptr)
	gObjectSetBool                 func(obj ptr, name *byte, v bool, end ptr)
	gObjectGetPtr                  func(obj ptr, name *byte, out unsafe.Pointer, end ptr)
	gSlistFree                     func(list ptr)
	gListFree                      func(list ptr)
	gUnixInputStreamNew            func(fd int32, closeFD bool) ptr
	gInputStreamReadAll            func(stream ptr, buf unsafe.Pointer, count uintptr, read *uintptr, cancellable ptr, err *ptr) bool
	gAppInfoLaunchDefaultForURI    func(uri *byte, ctx ptr, err *ptr) bool
	gAppInfoGetDefaultForURIScheme func(scheme *byte) ptr
	gAppInfoGetID                  func(info ptr) ptr
	gFileNewForPath                func(path *byte) ptr
	gFileGetURI                    func(file ptr) ptr
	gFileTrash                     func(file ptr, cancellable ptr, err *ptr) bool
	gBusGetSync                    func(busType int32, cancellable ptr, err *ptr) ptr
	gDBusConnectionCallSync        func(conn ptr, name, path, iface, method *byte, params ptr, replyType ptr, flags int32, timeout int32, cancellable ptr, err *ptr) ptr
	gDBusConnectionSignalSubscribe func(conn ptr, sender, iface, member, path, arg0 *byte, flags int32, cb ptr, data ptr, free ptr) uint32
	gVariantNewString              func(s *byte) ptr
	gVariantNewUint32              func(v uint32) ptr
	gVariantNewInt32               func(v int32) ptr
	gVariantNewTuple               func(children unsafe.Pointer, n uintptr) ptr
	gVariantNewArray               func(typ ptr, children unsafe.Pointer, n uintptr) ptr
	gVariantTypeNew                func(s *byte) ptr
	gVariantGetChildValue          func(v ptr, i uintptr) ptr
	gVariantGetUint32              func(v ptr) uint32
	gVariantGetUint64              func(v ptr) uint64
	gVariantGetBoolean             func(v ptr) bool
	gVariantGetDouble              func(v ptr) float64
	gVariantGetString              func(v ptr, length *uintptr) ptr
	gVariantUnref                  func(v ptr)
	gVariantGetVariant             func(v ptr) ptr
	gVariantGetTypeString          func(v ptr) ptr
	gVariantNewBoolean             func(v bool) ptr
	gVariantNewDictEntry           func(key, value ptr) ptr
	gVariantNewVariant             func(v ptr) ptr
	gVariantNewDouble              func(v float64) ptr
	gVariantNewInt64               func(v int64) ptr
	gVariantNewObjectPath          func(s *byte) ptr
	gVariantLookupValue            func(dict ptr, key *byte, typ ptr) ptr
	gVariantNChildren              func(v ptr) uintptr
	gDBusConnectionEmitSignal      func(conn ptr, dest, path, iface, signal *byte, params ptr, err *ptr) bool
	gDBusConnectionGetUniqueName   func(conn ptr) ptr
)

// GIO's network monitor (network.go).
var (
	gNetworkMonitorGetDefault          func() ptr
	gNetworkMonitorGetNetworkAvailable func(monitor ptr) bool
	gNetworkMonitorGetNetworkMetered   func(monitor ptr) bool
)

// GDK and GdkPixbuf.
var (
	gdkScreenGetRGBAVisual      func(screen ptr) ptr
	gdkDisplayGetDefault        func() ptr
	gdkDisplayGetNMonitors      func(d ptr) int32
	gdkDisplayGetMonitor        func(d ptr, i int32) ptr
	gdkDisplayGetPrimaryMonitor func(d ptr) ptr
	gdkMonitorGetGeometry       func(m ptr, r *gdkRectangle)
	gdkMonitorGetWorkarea       func(m ptr, r *gdkRectangle)
	gdkMonitorGetScaleFactor    func(m ptr) int32
	gdkMonitorGetRefreshRate    func(m ptr) int32
	gdkDisplayGetMonitorAtWin   func(d, w ptr) ptr
	gdkMonitorGetModel          func(m ptr) ptr
	gdkDisplayGetDefaultSeat    func(d ptr) ptr
	gdkSeatGetPointer           func(s ptr) ptr
	gdkDeviceGetPosition        func(dev ptr, screen *ptr, x, y *int32)
	gdkWindowGetDevicePosition  func(w, dev ptr, x, y *int32, mask *uint32) ptr
	gdkWindowGetDisplay         func(w ptr) ptr
	gdkWindowGetFrameExtents    func(w ptr, r *gdkRectangle)
	gdkWindowGetOrigin          func(w ptr, x, y *int32) int32
	gdkWindowGetGeometry        func(w ptr, x, y, width, height *int32)
	gdkWindowGetCursor          func(w ptr) ptr
	gdkWindowSetCursor          func(w, cursor ptr)
	gdkCursorNewFromName        func(d ptr, name *byte) ptr
	gdkDisplayBeep              func(d ptr)
	gdkEventCopy                func(e ptr) ptr
	gdkEventFree                func(e ptr)
	gdkAtomIntern               func(name *byte, onlyIfExists bool) ptr
	gdkAtomName                 func(atom ptr) ptr
	gdkKeyvalFromName           func(name *byte) uint32
	gdkUnicodeToKeyval          func(r uint32) uint32
	gdkPixbufLoaderNew          func() ptr
	gdkPixbufLoaderWrite        func(l ptr, buf unsafe.Pointer, n uintptr, err *ptr) bool
	gdkPixbufLoaderClose        func(l ptr, err *ptr) bool
	gdkPixbufLoaderGetPixbuf    func(l ptr) ptr
	gdkPixbufSaveToBufferv      func(p ptr, buf *ptr, size *uintptr, typ *byte, keys, values ptr, err *ptr) bool
)

type gdkRectangle struct{ X, Y, Width, Height int32 }

type gdkRGBA struct{ R, G, B, A float64 }

type gdkGeometry struct {
	MinWidth, MinHeight, MaxWidth, MaxHeight   int32
	BaseWidth, BaseHeight, WidthInc, HeightInc int32
	MinAspect, MaxAspect                       float64
	WinGravity                                 int32
	_                                          int32
}

// GTK.
var (
	gtkInitCheck                        func(argc, argv ptr) bool
	gtkMain                             func()
	gtkMainQuit                         func()
	gtkGetCurrentEventTime              func() uint32
	gtkWindowNew                        func(typ int32) ptr
	gtkWindowSetTitle                   func(w ptr, title *byte)
	gtkWindowGetTitle                   func(w ptr) ptr
	gtkWindowSetDefaultSize             func(w ptr, width, height int32)
	gtkWindowResize                     func(w ptr, width, height int32)
	gtkWindowGetSize                    func(w ptr, width, height *int32)
	gtkWindowMove                       func(w ptr, x, y int32)
	gtkWindowGetPosition                func(w ptr, x, y *int32)
	gtkWindowSetPosition                func(w ptr, pos int32)
	gtkWindowSetResizable               func(w ptr, v bool)
	gtkWindowGetResizable               func(w ptr) bool
	gtkWindowSetDecorated               func(w ptr, v bool)
	gtkWindowSetDeletable               func(w ptr, v bool)
	gtkWindowGetDeletable               func(w ptr) bool
	gtkWindowSetKeepAbove               func(w ptr, v bool)
	gtkWindowSetSkipTaskbarHint         func(w ptr, v bool)
	gtkWindowSetTransientFor            func(w, parent ptr)
	gtkWindowSetModal                   func(w ptr, v bool)
	gtkWindowSetGeometryHints           func(w ptr, widget ptr, geom *gdkGeometry, mask int32)
	gtkWindowPresent                    func(w ptr)
	gtkWindowIconify                    func(w ptr)
	gtkWindowDeiconify                  func(w ptr)
	gtkWindowMaximize                   func(w ptr)
	gtkWindowUnmaximize                 func(w ptr)
	gtkWindowFullscreen                 func(w ptr)
	gtkWindowUnfullscreen               func(w ptr)
	gtkWindowIsActive                   func(w ptr) bool
	gtkWindowIsMaximized                func(w ptr) bool
	gtkWindowBeginMoveDrag              func(w ptr, button int32, rootX, rootY int32, timestamp uint32)
	gtkWindowBeginResizeDrag            func(w ptr, edge, button int32, rootX, rootY int32, timestamp uint32)
	gtkWindowAddAccelGroup              func(w, group ptr)
	gtkWindowRemoveAccelGroup           func(w, group ptr)
	gtkWindowSetUrgencyHint             func(w ptr, v bool)
	gtkWindowStick                      func(w ptr)
	gtkWindowUnstick                    func(w ptr)
	gtkWindowSetIcon                    func(w, pixbuf ptr)
	gtkWindowSetDefaultIcon             func(pixbuf ptr)
	gtkWidgetShowAll                    func(w ptr)
	gtkWidgetSetDirection               func(w ptr, dir int32)
	gtkWidgetSetDefaultDirection        func(dir int32)
	gtkWidgetShow                       func(w ptr)
	gtkWidgetHide                       func(w ptr)
	gtkWidgetDestroy                    func(w ptr)
	gtkWidgetGetVisible                 func(w ptr) bool
	gtkWidgetGetMapped                  func(w ptr) bool
	gtkWidgetSetOpacity                 func(w ptr, v float64)
	gtkWidgetGetOpacity                 func(w ptr) float64
	gtkWidgetGrabFocus                  func(w ptr)
	gtkWidgetSetVisual                  func(w, visual ptr)
	gtkWidgetSetAppPaintable            func(w ptr, v bool)
	gtkWidgetGetScreen                  func(w ptr) ptr
	gtkWidgetSetSensitive               func(w ptr, v bool)
	gtkWidgetIsSensitive                func(w ptr) bool
	gtkWidgetSetTooltipText             func(w ptr, s *byte)
	gtkWidgetAddAccelerator             func(w ptr, signal *byte, group ptr, key uint32, mods uint32, flags int32)
	gtkWidgetSetNoShowAll               func(w ptr, v bool)
	gtkWidgetSetVisible                 func(w ptr, v bool)
	gtkWidgetMnemonicActivate           func(w ptr, groupCycling bool) bool
	gtkBoxNew                           func(orientation int32, spacing int32) ptr
	gtkBoxPackStart                     func(box, child ptr, expand, fill bool, padding uint32)
	gtkBoxReorderChild                  func(box, child ptr, pos int32)
	gtkContainerAdd                     func(c, w ptr)
	gtkContainerRemove                  func(c, w ptr)
	gtkContainerCheckResize             func(c ptr)
	gtkMenuBarNew                       func() ptr
	gtkMenuNew                          func() ptr
	gtkMenuItemNewWithLabel             func(label *byte) ptr
	gtkMenuItemSetLabel                 func(item ptr, label *byte)
	gtkMenuItemSetSubmenu               func(item, menu ptr)
	gtkMenuItemGetSubmenu               func(item ptr) ptr
	gtkMenuItemGetLabel                 func(item ptr) ptr
	gtkMenuItemActivate                 func(item ptr)
	gtkContainerGetChildren             func(c ptr) ptr
	gtkCheckMenuItemNewWithLabel        func(label *byte) ptr
	gtkCheckMenuItemSetActive           func(item ptr, v bool)
	gtkCheckMenuItemSetDrawAsRadio      func(item ptr, v bool)
	gtkSeparatorMenuItemNew             func() ptr
	gtkMenuShellAppend                  func(shell, child ptr)
	gtkMenuPopupAtPointer               func(menu ptr, event ptr)
	gtkMenuPopupAtRect                  func(menu, window ptr, rect *gdkRectangle, rectAnchor, menuAnchor int32, event ptr)
	gtkMenuShellDeactivate              func(menu ptr)
	gtkWidgetGetWindow                  func(w ptr) ptr
	gtkWidgetRealize                    func(w ptr)
	gtkWidgetGetAllocation              func(w ptr, a *gdkRectangle)
	gtkAccelGroupNew                    func() ptr
	gtkFileChooserNativeNew             func(title *byte, parent ptr, action int32, accept, cancel *byte) ptr
	gtkNativeDialogRun                  func(d ptr) int32
	gtkNativeDialogHide                 func(d ptr)
	gtkFileChooserSetSelectMultiple     func(fc ptr, v bool)
	gtkFileChooserSetShowHidden         func(fc ptr, v bool)
	gtkFileChooserSetDoOverwriteConfirm func(fc ptr, v bool)
	gtkFileChooserSetCurrentFolder      func(fc ptr, path *byte) bool
	gtkFileChooserSetCurrentName        func(fc ptr, name *byte)
	gtkFileChooserGetFilenames          func(fc ptr) ptr
	gtkFileChooserGetFilename           func(fc ptr) ptr
	gtkFileChooserAddFilter             func(fc, filter ptr)
	gtkFileFilterNew                    func() ptr
	gtkFileFilterSetName                func(f ptr, name *byte)
	gtkFileFilterAddPattern             func(f ptr, pattern *byte)
	gtkMessageDialogNew                 func(parent ptr, flags, typ, buttons int32, format *byte, arg *byte) ptr
	gtkMessageDialogFormatSecondaryText func(d ptr, format *byte, arg *byte)
	gtkMessageDialogGetMessageArea      func(d ptr) ptr
	gtkDialogAddButton                  func(d ptr, text *byte, response int32) ptr
	gtkDialogSetDefaultResponse         func(d ptr, response int32)
	gtkDialogRun                        func(d ptr) int32
	gtkCheckButtonNewWithLabel          func(label *byte) ptr
	gtkToggleButtonGetActive            func(b ptr) bool
	gtkToggleButtonSetActive            func(b ptr, v bool)
	gtkClipboardGet                     func(atom ptr) ptr
	gtkClipboardSetText                 func(cb ptr, text *byte, n int32)
	gtkClipboardWaitForText             func(cb ptr) ptr
	gtkClipboardWaitForImage            func(cb ptr) ptr
	gtkClipboardSetImage                func(cb, pixbuf ptr)
	gtkClipboardWaitForContents         func(cb, target ptr) ptr
	gtkClipboardWaitForTargets          func(cb ptr, targets *ptr, n *int32) bool
	gtkClipboardClear                   func(cb ptr)
	gtkSelectionDataGetData             func(sd ptr) ptr
	gtkSelectionDataGetLength           func(sd ptr) int32
	gtkSelectionDataFree                func(sd ptr)
	gtkSelectionDataGetUris             func(sd ptr) ptr
	gtkSettingsGetDefault               func() ptr
	gtkOverlayNew                       func() ptr
	gtkOverlayAddOverlay                func(o, w ptr)
	gtkHeaderBarNew                     func() ptr
	gtkHeaderBarSetShowCloseButton      func(b ptr, v bool)
	gtkHeaderBarSetHasSubtitle          func(b ptr, v bool)
	gtkHeaderBarSetDecorationLayout     func(b ptr, layout *byte)
	gtkWidgetSetHalign                  func(w ptr, align int32)
	gtkWidgetSetValign                  func(w ptr, align int32)
	gtkWidgetSetSizeRequest             func(w ptr, width, height int32)
	gtkWidgetGetPreferredWidth          func(w ptr, min, natural *int32)
	gtkWidgetGetPreferredHeight         func(w ptr, min, natural *int32)
	gtkWidgetGetStyleContext            func(w ptr) ptr
	gtkStyleContextAddClass             func(c ptr, name *byte)
	gtkStyleContextAddProviderForScreen func(screen, provider ptr, priority uint32)
	gtkCssProviderNew                   func() ptr
	gtkCssProviderLoadFromData          func(p ptr, data *byte, length int, err *ptr) bool
	gtkAboutDialogNew                   func() ptr
	gtkAboutDialogSetProgramName        func(d ptr, s *byte)
	gtkAboutDialogSetVersion            func(d ptr, s *byte)
	gtkAboutDialogSetCopyright          func(d ptr, s *byte)
	gtkAboutDialogSetComments           func(d ptr, s *byte)
)

// WebKitGTK, JavaScriptCore, libsoup and cairo.
var (
	webkitWebContextGetDefault                        func() ptr
	webkitWebContextRegisterURIScheme                 func(ctx ptr, scheme *byte, cb ptr, data ptr, destroy ptr)
	webkitWebContextGetSecurityManager                func(ctx ptr) ptr
	webkitWebContextGetWebsiteDataManager             func(ctx ptr) ptr
	webkitWebContextSetPreferredLanguages             func(ctx ptr, languages **byte)
	webkitWebsiteDataManagerClear                     func(m ptr, types uint32, timespan int64, cancellable ptr, cb ptr, data ptr)
	webkitWebsiteDataManagerClearFinish               func(m, res ptr, err *ptr) bool
	webkitDownloadGetWebView                          func(d ptr) ptr
	webkitDownloadGetRequest                          func(d ptr) ptr
	webkitDownloadSetDestination                      func(d ptr, uri *byte)
	webkitDownloadSetAllowOverwrite                   func(d ptr, v bool)
	webkitDownloadCancel                              func(d ptr)
	webkitResponsePolicyDecisionGetResponse           func(d ptr) ptr
	webkitResponsePolicyDecisionIsMIMETypeSupported   func(d ptr) bool
	webkitPolicyDecisionDownload                      func(d ptr)
	webkitURIResponseGetURI                           func(r ptr) ptr
	webkitURIResponseGetHTTPHeaders                   func(r ptr) ptr
	webkitSecurityManagerRegisterSecure               func(m ptr, scheme *byte)
	webkitSecurityManagerRegisterCORS                 func(m ptr, scheme *byte)
	webkitUserContentManagerNew                       func() ptr
	webkitUserContentManagerRegisterHandler           func(ucm ptr, name *byte) bool
	webkitUserContentManagerUnregisterHandler         func(ucm ptr, name *byte)
	webkitUserContentManagerAddScript                 func(ucm, script ptr)
	webkitUserContentManagerRemoveAllScripts          func(ucm ptr)
	webkitUserScriptNew                               func(src *byte, frames, time int32, allow, block ptr) ptr
	webkitUserScriptUnref                             func(s ptr)
	webkitWebViewNewWithUserContentManager            func(ucm ptr) ptr
	webkitWebViewGetSettings                          func(wv ptr) ptr
	webkitSettingsSetEnableDeveloperExtras            func(s ptr, v bool)
	webkitSettingsSetUserAgent                        func(s ptr, ua *byte)
	webkitSettingsGetUserAgent                        func(s ptr) ptr
	webkitSettingsSetAllowFileAccessFromFileURLs      func(s ptr, v bool)
	webkitSettingsSetJavascriptCanAccessClipboard     func(s ptr, v bool)
	webkitWebViewLoadURI                              func(wv ptr, uri *byte)
	webkitWebViewLoadHTML                             func(wv ptr, html *byte, base *byte)
	webkitWebViewReload                               func(wv ptr)
	webkitWebViewReloadBypassCache                    func(wv ptr)
	webkitWebViewStopLoading                          func(wv ptr)
	webkitWebViewGoBack                               func(wv ptr)
	webkitWebViewGoForward                            func(wv ptr)
	webkitWebViewCanGoBack                            func(wv ptr) bool
	webkitWebViewCanGoForward                         func(wv ptr) bool
	webkitWebViewIsLoading                            func(wv ptr) bool
	webkitWebViewGetURI                               func(wv ptr) ptr
	webkitWebViewGetTitle                             func(wv ptr) ptr
	webkitWebViewSetZoomLevel                         func(wv ptr, z float64)
	webkitWebViewGetZoomLevel                         func(wv ptr) float64
	webkitWebViewSetBackgroundColor                   func(wv ptr, rgba *gdkRGBA)
	webkitWebViewEvaluateJavascript                   func(wv ptr, script *byte, length int, world, sourceURI *byte, cancellable, cb, data ptr)
	webkitWebViewRunJavascript                        func(wv ptr, script *byte, cancellable, cb, data ptr)
	webkitWebViewCallAsyncJavascriptFunction          func(wv ptr, body *byte, length int, args ptr, world, sourceURI *byte, cancellable, cb, data ptr)
	webkitWebViewCallAsyncJavascriptFunctionFinish    func(wv, res ptr, err *ptr) ptr
	webkitWebViewGetInspector                         func(wv ptr) ptr
	webkitWebInspectorShow                            func(i ptr)
	webkitWebInspectorClose                           func(i ptr)
	webkitWebInspectorGetWebView                      func(i ptr) ptr
	webkitWebViewGetSnapshot                          func(wv ptr, region, options int32, cancellable, cb, data ptr)
	webkitWebViewGetSnapshotFinish                    func(wv, res ptr, err *ptr) ptr
	webkitWebViewExecuteEditingCommand                func(wv ptr, cmd *byte)
	webkitPrintOperationNew                           func(wv ptr) ptr
	webkitPrintOperationRunDialog                     func(op, parent ptr) int32
	webkitPrintOperationSetPrintSettings              func(op, settings ptr)
	webkitPrintOperationSetPageSetup                  func(op, setup ptr)
	webkitPrintOperationPrint                         func(op ptr)
	webkitSettingsGetPrintBackgrounds                 func(s ptr) bool
	webkitSettingsSetPrintBackgrounds                 func(s ptr, v bool)
	webkitUserMediaPermissionRequestGetType           func() uintptr
	webkitGeolocationPermissionRequestGetType         func() uintptr
	webkitNotificationPermissionRequestGetType        func() uintptr
	webkitUserMediaPermissionIsForAudioDevice         func(req ptr) bool
	webkitUserMediaPermissionIsForVideoDevice         func(req ptr) bool
	webkitPermissionRequestAllow                      func(req ptr)
	webkitPermissionRequestDeny                       func(req ptr)
	gTypeCheckInstanceIsA                             func(instance ptr, typ uintptr) bool
	gtkPrintSettingsNew                               func() ptr
	gtkPrintSettingsSet                               func(s ptr, key, value *byte)
	gtkPageSetupNew                                   func() ptr
	gtkPageSetupSetPaperSize                          func(setup, size ptr)
	gtkPageSetupSetOrientation                        func(setup ptr, orientation int32)
	gtkPageSetupSetTopMargin                          func(setup ptr, margin float64, unit int32)
	gtkPageSetupSetBottomMargin                       func(setup ptr, margin float64, unit int32)
	gtkPageSetupSetLeftMargin                         func(setup ptr, margin float64, unit int32)
	gtkPageSetupSetRightMargin                        func(setup ptr, margin float64, unit int32)
	gtkPaperSizeNewCustom                             func(name, displayName *byte, width, height float64, unit int32) ptr
	gtkPaperSizeFree                                  func(size ptr)
	gtkRecentManagerGetDefault                        func() ptr
	gtkRecentManagerAddItem                           func(manager ptr, uri *byte) bool
	gtkRecentManagerGetItems                          func(manager ptr) ptr
	gtkRecentInfoGetURI                               func(info ptr) ptr
	gtkRecentInfoHasApplication                       func(info, app ptr) bool
	gtkRecentInfoUnref                                func(info ptr)
	gGetApplicationName                               func() ptr
	gtkRecentManagerRemoveItem                        func(manager ptr, uri *byte, gerr ptr) bool
	gtkRecentManagerHasItem                           func(manager ptr, uri *byte) bool
	webkitJavascriptResultGetJSValue                  func(r ptr) ptr
	webkitNavigationPolicyDecisionGetNavigationAction func(d ptr) ptr
	webkitNavigationPolicyDecisionGetFrameName        func(d ptr) ptr
	webkitNavigationActionGetRequest                  func(a ptr) ptr
	webkitNavigationActionGetNavigationType           func(a ptr) int32
	webkitNavigationActionIsUserGesture               func(a ptr) bool
	webkitURIRequestGetURI                            func(r ptr) ptr
	webkitPolicyDecisionUse                           func(d ptr)
	webkitPolicyDecisionIgnore                        func(d ptr)
	webkitURISchemeRequestGetURI                      func(r ptr) ptr
	webkitURISchemeRequestGetHTTPMethod               func(r ptr) ptr
	webkitURISchemeRequestGetHTTPHeaders              func(r ptr) ptr
	webkitURISchemeRequestGetHTTPBody                 func(r ptr) ptr
	webkitURISchemeRequestGetWebView                  func(r ptr) ptr
	webkitURISchemeRequestFinishWithResponse          func(r, resp ptr)
	webkitURISchemeRequestFinishError                 func(r, gerr ptr)
	webkitURISchemeRequestFinish                      func(r ptr, stream ptr, length int64, contentType *byte)
	webkitURISchemeResponseNew                        func(stream ptr, length int64) ptr
	webkitURISchemeResponseSetStatus                  func(resp ptr, status uint32, reason *byte)
	webkitURISchemeResponseSetContentType             func(resp ptr, ct *byte)
	webkitURISchemeResponseSetHTTPHeaders             func(resp, headers ptr)

	jscValueToString func(v ptr) ptr

	soupMessageHeadersNew     func(typ int32) ptr
	soupMessageHeadersAppend  func(h ptr, name, value *byte)
	soupMessageHeadersForeach func(h ptr, fn ptr, data ptr)
	soupMessageHeadersGetOne  func(h ptr, name *byte) ptr

	cairoSurfaceWriteToPNGStream func(surface ptr, write ptr, closure ptr) int32
	cairoSurfaceDestroy          func(s ptr)

	appIndicatorNew              func(id, icon *byte, category int32) ptr
	appIndicatorSetStatus        func(ind ptr, status int32)
	appIndicatorSetMenu          func(ind, menu ptr)
	appIndicatorSetIconFull      func(ind ptr, icon, desc *byte)
	appIndicatorSetIconThemePath func(ind ptr, path *byte)
	appIndicatorSetTitle         func(ind ptr, title *byte)
	appIndicatorSetLabel         func(ind ptr, label, guide *byte)
)

var webkitVersion string

func load() error {
	var err error
	if libGLib, err = open("libglib-2.0.so.0"); err != nil {
		return err
	}
	if libGObject, err = open("libgobject-2.0.so.0"); err != nil {
		return err
	}
	if libGIO, err = open("libgio-2.0.so.0"); err != nil {
		return err
	}
	if libGDK, err = open("libgdk-3.so.0"); err != nil {
		return err
	}
	if libGTK, err = open("libgtk-3.so.0"); err != nil {
		return err
	}
	if libCairo, err = open("libcairo.so.2"); err != nil {
		return err
	}
	if libPixbuf, err = open("libgdk_pixbuf-2.0.so.0"); err != nil {
		return err
	}
	libIndicator, _ = open("libayatana-appindicator3.so.1", "libappindicator3.so.1")

	g := libGLib
	mustBind(g, &gFree, "g_free")
	mustBind(g, &gErrorFree, "g_error_free")
	mustBind(g, &gStrfreev, "g_strfreev")
	mustBind(g, &gFilenameFromURI, "g_filename_from_uri")
	mustBind(g, &gErrorNewLiteral, "g_error_new_literal")
	mustBind(g, &gQuarkFromString, "g_quark_from_string")
	mustBind(g, &gIdleAddFull, "g_idle_add_full")
	mustBind(g, &gTimeoutAdd, "g_timeout_add")
	mustBind(g, &gMainContextIteration, "g_main_context_iteration")
	mustBind(g, &gMainContextWakeup, "g_main_context_wakeup")
	mustBind(g, &gMainLoopNew, "g_main_loop_new")
	mustBind(g, &gMainLoopRun, "g_main_loop_run")
	mustBind(g, &gMainLoopQuit, "g_main_loop_quit")
	mustBind(g, &gMainLoopUnref, "g_main_loop_unref")
	mustBind(g, &gSlistFree, "g_slist_free")
	mustBind(g, &gListFree, "g_list_free")
	mustBind(g, &gGetApplicationName, "g_get_application_name")
	mustBind(g, &gVariantNewString, "g_variant_new_string")
	mustBind(g, &gVariantNewUint32, "g_variant_new_uint32")
	mustBind(g, &gVariantNewInt32, "g_variant_new_int32")
	mustBind(g, &gVariantNewTuple, "g_variant_new_tuple")
	mustBind(g, &gVariantNewArray, "g_variant_new_array")
	mustBind(g, &gVariantTypeNew, "g_variant_type_new")
	mustBind(g, &gVariantGetChildValue, "g_variant_get_child_value")
	mustBind(g, &gVariantGetUint32, "g_variant_get_uint32")
	mustBind(g, &gVariantGetUint64, "g_variant_get_uint64")
	mustBind(g, &gVariantGetBoolean, "g_variant_get_boolean")
	mustBind(g, &gVariantGetDouble, "g_variant_get_double")
	mustBind(g, &gVariantGetString, "g_variant_get_string")
	mustBind(g, &gVariantUnref, "g_variant_unref")
	mustBind(g, &gVariantGetVariant, "g_variant_get_variant")
	mustBind(g, &gVariantGetTypeString, "g_variant_get_type_string")
	mustBind(g, &gVariantNewBoolean, "g_variant_new_boolean")
	mustBind(g, &gVariantNewDictEntry, "g_variant_new_dict_entry")
	mustBind(g, &gVariantNewVariant, "g_variant_new_variant")
	mustBind(g, &gVariantNewDouble, "g_variant_new_double")
	mustBind(g, &gVariantNewInt64, "g_variant_new_int64")
	mustBind(g, &gVariantNewObjectPath, "g_variant_new_object_path")
	mustBind(g, &gVariantLookupValue, "g_variant_lookup_value")
	mustBind(g, &gVariantNChildren, "g_variant_n_children")

	o := libGObject
	mustBind(o, &gSignalConnectData, "g_signal_connect_data")
	mustBind(o, &gSignalHandlerDisconnect, "g_signal_handler_disconnect")
	mustBind(o, &gObjectRef, "g_object_ref")
	mustBind(o, &gObjectRefSink, "g_object_ref_sink")
	mustBind(o, &gObjectUnref, "g_object_unref")
	mustBind(o, &gObjectSetBool, "g_object_set")
	mustBind(o, &gObjectGetPtr, "g_object_get")

	i := libGIO
	mustBind(i, &gUnixInputStreamNew, "g_unix_input_stream_new")
	mustBind(i, &gInputStreamReadAll, "g_input_stream_read_all")
	mustBind(i, &gAppInfoLaunchDefaultForURI, "g_app_info_launch_default_for_uri")
	mustBind(i, &gAppInfoGetDefaultForURIScheme, "g_app_info_get_default_for_uri_scheme")
	mustBind(i, &gAppInfoGetID, "g_app_info_get_id")
	mustBind(i, &gFileNewForPath, "g_file_new_for_path")
	mustBind(i, &gFileGetURI, "g_file_get_uri")
	mustBind(i, &gFileTrash, "g_file_trash")
	mustBind(i, &gBusGetSync, "g_bus_get_sync")
	mustBind(i, &gDBusConnectionEmitSignal, "g_dbus_connection_emit_signal")
	mustBind(i, &gDBusConnectionCallSync, "g_dbus_connection_call_sync")
	mustBind(i, &gDBusConnectionSignalSubscribe, "g_dbus_connection_signal_subscribe")
	mustBind(i, &gDBusConnectionGetUniqueName, "g_dbus_connection_get_unique_name")
	mustBind(i, &gNetworkMonitorGetDefault, "g_network_monitor_get_default")
	mustBind(i, &gNetworkMonitorGetNetworkAvailable, "g_network_monitor_get_network_available")
	mustBind(i, &gNetworkMonitorGetNetworkMetered, "g_network_monitor_get_network_metered")

	d := libGDK
	mustBind(d, &gdkScreenGetRGBAVisual, "gdk_screen_get_rgba_visual")
	mustBind(d, &gdkDisplayGetDefault, "gdk_display_get_default")
	mustBind(d, &gdkDisplayGetNMonitors, "gdk_display_get_n_monitors")
	mustBind(d, &gdkDisplayGetMonitor, "gdk_display_get_monitor")
	mustBind(d, &gdkDisplayGetPrimaryMonitor, "gdk_display_get_primary_monitor")
	mustBind(d, &gdkMonitorGetGeometry, "gdk_monitor_get_geometry")
	mustBind(d, &gdkMonitorGetWorkarea, "gdk_monitor_get_workarea")
	mustBind(d, &gdkMonitorGetScaleFactor, "gdk_monitor_get_scale_factor")
	mustBind(d, &gdkMonitorGetRefreshRate, "gdk_monitor_get_refresh_rate")
	mustBind(d, &gdkDisplayGetMonitorAtWin, "gdk_display_get_monitor_at_window")
	mustBind(d, &gdkMonitorGetModel, "gdk_monitor_get_model")
	mustBind(d, &gdkDisplayGetDefaultSeat, "gdk_display_get_default_seat")
	mustBind(d, &gdkSeatGetPointer, "gdk_seat_get_pointer")
	mustBind(d, &gdkDeviceGetPosition, "gdk_device_get_position")
	mustBind(d, &gdkWindowGetDevicePosition, "gdk_window_get_device_position")
	mustBind(d, &gdkWindowGetDisplay, "gdk_window_get_display")
	mustBind(d, &gdkWindowGetFrameExtents, "gdk_window_get_frame_extents")
	mustBind(d, &gdkWindowGetOrigin, "gdk_window_get_origin")
	mustBind(d, &gdkWindowGetGeometry, "gdk_window_get_geometry")
	mustBind(d, &gdkWindowGetCursor, "gdk_window_get_cursor")
	mustBind(d, &gdkWindowSetCursor, "gdk_window_set_cursor")
	mustBind(d, &gdkCursorNewFromName, "gdk_cursor_new_from_name")
	mustBind(d, &gdkDisplayBeep, "gdk_display_beep")
	mustBind(d, &gdkEventCopy, "gdk_event_copy")
	mustBind(d, &gdkEventFree, "gdk_event_free")
	mustBind(d, &gdkAtomIntern, "gdk_atom_intern")
	mustBind(d, &gdkAtomName, "gdk_atom_name")
	mustBind(d, &gdkKeyvalFromName, "gdk_keyval_from_name")
	mustBind(d, &gdkUnicodeToKeyval, "gdk_unicode_to_keyval")

	p := libPixbuf
	mustBind(p, &gdkPixbufLoaderNew, "gdk_pixbuf_loader_new")
	mustBind(p, &gdkPixbufLoaderWrite, "gdk_pixbuf_loader_write")
	mustBind(p, &gdkPixbufLoaderClose, "gdk_pixbuf_loader_close")
	mustBind(p, &gdkPixbufLoaderGetPixbuf, "gdk_pixbuf_loader_get_pixbuf")
	mustBind(p, &gdkPixbufSaveToBufferv, "gdk_pixbuf_save_to_bufferv")

	t := libGTK
	mustBind(t, &gtkInitCheck, "gtk_init_check")
	mustBind(t, &gtkMain, "gtk_main")
	mustBind(t, &gtkMainQuit, "gtk_main_quit")
	mustBind(t, &gtkGetCurrentEventTime, "gtk_get_current_event_time")
	mustBind(t, &gtkWindowNew, "gtk_window_new")
	mustBind(t, &gtkWindowSetTitle, "gtk_window_set_title")
	mustBind(t, &gtkWindowGetTitle, "gtk_window_get_title")
	mustBind(t, &gtkWindowSetDefaultSize, "gtk_window_set_default_size")
	mustBind(t, &gtkWindowResize, "gtk_window_resize")
	mustBind(t, &gtkWindowGetSize, "gtk_window_get_size")
	mustBind(t, &gtkWindowMove, "gtk_window_move")
	mustBind(t, &gtkWindowGetPosition, "gtk_window_get_position")
	mustBind(t, &gtkWindowSetPosition, "gtk_window_set_position")
	mustBind(t, &gtkWindowSetResizable, "gtk_window_set_resizable")
	mustBind(t, &gtkWindowGetResizable, "gtk_window_get_resizable")
	mustBind(t, &gtkWindowSetDecorated, "gtk_window_set_decorated")
	mustBind(t, &gtkWindowSetDeletable, "gtk_window_set_deletable")
	mustBind(t, &gtkWindowGetDeletable, "gtk_window_get_deletable")
	mustBind(t, &gtkWindowSetKeepAbove, "gtk_window_set_keep_above")
	mustBind(t, &gtkWindowSetSkipTaskbarHint, "gtk_window_set_skip_taskbar_hint")
	mustBind(t, &gtkWindowSetTransientFor, "gtk_window_set_transient_for")
	mustBind(t, &gtkWindowSetModal, "gtk_window_set_modal")
	mustBind(t, &gtkWindowSetGeometryHints, "gtk_window_set_geometry_hints")
	mustBind(t, &gtkWindowPresent, "gtk_window_present")
	mustBind(t, &gtkWindowIconify, "gtk_window_iconify")
	mustBind(t, &gtkWindowDeiconify, "gtk_window_deiconify")
	mustBind(t, &gtkWindowMaximize, "gtk_window_maximize")
	mustBind(t, &gtkWindowUnmaximize, "gtk_window_unmaximize")
	mustBind(t, &gtkWindowFullscreen, "gtk_window_fullscreen")
	mustBind(t, &gtkWindowUnfullscreen, "gtk_window_unfullscreen")
	mustBind(t, &gtkWindowIsActive, "gtk_window_is_active")
	mustBind(t, &gtkWindowIsMaximized, "gtk_window_is_maximized")
	mustBind(t, &gtkWindowBeginMoveDrag, "gtk_window_begin_move_drag")
	mustBind(t, &gtkWindowBeginResizeDrag, "gtk_window_begin_resize_drag")
	mustBind(t, &gtkWindowAddAccelGroup, "gtk_window_add_accel_group")
	mustBind(t, &gtkWindowRemoveAccelGroup, "gtk_window_remove_accel_group")
	mustBind(t, &gtkWindowSetUrgencyHint, "gtk_window_set_urgency_hint")
	mustBind(t, &gtkWindowStick, "gtk_window_stick")
	mustBind(t, &gtkWindowUnstick, "gtk_window_unstick")
	mustBind(t, &gtkWindowSetIcon, "gtk_window_set_icon")
	mustBind(t, &gtkWindowSetDefaultIcon, "gtk_window_set_default_icon")
	mustBind(t, &gtkWidgetShowAll, "gtk_widget_show_all")
	mustBind(t, &gtkWidgetSetDirection, "gtk_widget_set_direction")
	mustBind(t, &gtkWidgetSetDefaultDirection, "gtk_widget_set_default_direction")
	mustBind(t, &gtkWidgetShow, "gtk_widget_show")
	mustBind(t, &gtkWidgetHide, "gtk_widget_hide")
	mustBind(t, &gtkWidgetDestroy, "gtk_widget_destroy")
	mustBind(t, &gtkWidgetGetVisible, "gtk_widget_get_visible")
	mustBind(t, &gtkWidgetGetMapped, "gtk_widget_get_mapped")
	mustBind(t, &gtkWidgetSetOpacity, "gtk_widget_set_opacity")
	mustBind(t, &gtkWidgetGetOpacity, "gtk_widget_get_opacity")
	mustBind(t, &gtkWidgetGrabFocus, "gtk_widget_grab_focus")
	mustBind(t, &gtkWidgetSetVisual, "gtk_widget_set_visual")
	mustBind(t, &gtkWidgetSetAppPaintable, "gtk_widget_set_app_paintable")
	mustBind(t, &gtkWidgetGetScreen, "gtk_widget_get_screen")
	mustBind(t, &gtkWidgetSetSensitive, "gtk_widget_set_sensitive")
	mustBind(t, &gtkWidgetIsSensitive, "gtk_widget_is_sensitive")
	mustBind(t, &gtkWidgetSetTooltipText, "gtk_widget_set_tooltip_text")
	mustBind(t, &gtkWidgetAddAccelerator, "gtk_widget_add_accelerator")
	mustBind(t, &gtkWidgetSetNoShowAll, "gtk_widget_set_no_show_all")
	mustBind(t, &gtkWidgetSetVisible, "gtk_widget_set_visible")
	mustBind(t, &gtkWidgetMnemonicActivate, "gtk_widget_mnemonic_activate")
	mustBind(t, &gtkBoxNew, "gtk_box_new")
	mustBind(t, &gtkBoxPackStart, "gtk_box_pack_start")
	mustBind(t, &gtkBoxReorderChild, "gtk_box_reorder_child")
	mustBind(t, &gtkContainerAdd, "gtk_container_add")
	mustBind(t, &gtkContainerRemove, "gtk_container_remove")
	mustBind(t, &gtkContainerCheckResize, "gtk_container_check_resize")
	mustBind(t, &gtkMenuBarNew, "gtk_menu_bar_new")
	mustBind(t, &gtkMenuNew, "gtk_menu_new")
	mustBind(t, &gtkMenuItemNewWithLabel, "gtk_menu_item_new_with_label")
	mustBind(t, &gtkMenuItemSetLabel, "gtk_menu_item_set_label")
	mustBind(t, &gtkMenuItemSetSubmenu, "gtk_menu_item_set_submenu")
	mustBind(t, &gtkMenuItemGetSubmenu, "gtk_menu_item_get_submenu")
	mustBind(t, &gtkMenuItemGetLabel, "gtk_menu_item_get_label")
	mustBind(t, &gtkMenuItemActivate, "gtk_menu_item_activate")
	mustBind(t, &gtkContainerGetChildren, "gtk_container_get_children")
	mustBind(t, &gtkCheckMenuItemNewWithLabel, "gtk_check_menu_item_new_with_label")
	mustBind(t, &gtkCheckMenuItemSetActive, "gtk_check_menu_item_set_active")
	mustBind(t, &gtkCheckMenuItemSetDrawAsRadio, "gtk_check_menu_item_set_draw_as_radio")
	mustBind(t, &gtkSeparatorMenuItemNew, "gtk_separator_menu_item_new")
	mustBind(t, &gtkMenuShellAppend, "gtk_menu_shell_append")
	mustBind(t, &gtkMenuPopupAtPointer, "gtk_menu_popup_at_pointer")
	mustBind(t, &gtkMenuPopupAtRect, "gtk_menu_popup_at_rect")
	mustBind(t, &gtkMenuShellDeactivate, "gtk_menu_shell_deactivate")
	mustBind(t, &gtkWidgetGetWindow, "gtk_widget_get_window")
	mustBind(t, &gtkWidgetRealize, "gtk_widget_realize")
	mustBind(t, &gtkWidgetGetAllocation, "gtk_widget_get_allocation")
	mustBind(t, &gtkAccelGroupNew, "gtk_accel_group_new")
	mustBind(t, &gtkFileChooserNativeNew, "gtk_file_chooser_native_new")
	mustBind(t, &gtkNativeDialogRun, "gtk_native_dialog_run")
	mustBind(t, &gtkNativeDialogHide, "gtk_native_dialog_hide")
	mustBind(t, &gtkFileChooserSetSelectMultiple, "gtk_file_chooser_set_select_multiple")
	mustBind(t, &gtkFileChooserSetShowHidden, "gtk_file_chooser_set_show_hidden")
	mustBind(t, &gtkFileChooserSetDoOverwriteConfirm, "gtk_file_chooser_set_do_overwrite_confirmation")
	mustBind(t, &gtkFileChooserSetCurrentFolder, "gtk_file_chooser_set_current_folder")
	mustBind(t, &gtkFileChooserSetCurrentName, "gtk_file_chooser_set_current_name")
	mustBind(t, &gtkFileChooserGetFilenames, "gtk_file_chooser_get_filenames")
	mustBind(t, &gtkFileChooserGetFilename, "gtk_file_chooser_get_filename")
	mustBind(t, &gtkFileChooserAddFilter, "gtk_file_chooser_add_filter")
	mustBind(t, &gtkFileFilterNew, "gtk_file_filter_new")
	mustBind(t, &gtkFileFilterSetName, "gtk_file_filter_set_name")
	mustBind(t, &gtkFileFilterAddPattern, "gtk_file_filter_add_pattern")
	mustBind(t, &gtkMessageDialogNew, "gtk_message_dialog_new")
	mustBind(t, &gtkMessageDialogFormatSecondaryText, "gtk_message_dialog_format_secondary_text")
	mustBind(t, &gtkMessageDialogGetMessageArea, "gtk_message_dialog_get_message_area")
	mustBind(t, &gtkDialogAddButton, "gtk_dialog_add_button")
	mustBind(t, &gtkDialogSetDefaultResponse, "gtk_dialog_set_default_response")
	mustBind(t, &gtkDialogRun, "gtk_dialog_run")
	mustBind(t, &gtkCheckButtonNewWithLabel, "gtk_check_button_new_with_label")
	mustBind(t, &gtkToggleButtonGetActive, "gtk_toggle_button_get_active")
	mustBind(t, &gtkToggleButtonSetActive, "gtk_toggle_button_set_active")
	mustBind(t, &gtkClipboardGet, "gtk_clipboard_get")
	mustBind(t, &gtkClipboardSetText, "gtk_clipboard_set_text")
	mustBind(t, &gtkClipboardWaitForText, "gtk_clipboard_wait_for_text")
	mustBind(t, &gtkClipboardWaitForImage, "gtk_clipboard_wait_for_image")
	mustBind(t, &gtkClipboardSetImage, "gtk_clipboard_set_image")
	mustBind(t, &gtkClipboardWaitForContents, "gtk_clipboard_wait_for_contents")
	mustBind(t, &gtkClipboardWaitForTargets, "gtk_clipboard_wait_for_targets")
	mustBind(t, &gtkClipboardClear, "gtk_clipboard_clear")
	mustBind(t, &gtkSelectionDataGetData, "gtk_selection_data_get_data")
	mustBind(t, &gtkSelectionDataGetUris, "gtk_selection_data_get_uris")
	mustBind(t, &gtkSelectionDataGetLength, "gtk_selection_data_get_length")
	mustBind(t, &gtkSelectionDataFree, "gtk_selection_data_free")
	mustBind(t, &gtkSettingsGetDefault, "gtk_settings_get_default")
	mustBind(t, &gtkOverlayNew, "gtk_overlay_new")
	mustBind(t, &gtkOverlayAddOverlay, "gtk_overlay_add_overlay")
	mustBind(t, &gtkHeaderBarNew, "gtk_header_bar_new")
	mustBind(t, &gtkHeaderBarSetShowCloseButton, "gtk_header_bar_set_show_close_button")
	mustBind(t, &gtkHeaderBarSetHasSubtitle, "gtk_header_bar_set_has_subtitle")
	mustBind(t, &gtkHeaderBarSetDecorationLayout, "gtk_header_bar_set_decoration_layout")
	mustBind(t, &gtkWidgetSetHalign, "gtk_widget_set_halign")
	mustBind(t, &gtkWidgetSetValign, "gtk_widget_set_valign")
	mustBind(t, &gtkWidgetSetSizeRequest, "gtk_widget_set_size_request")
	mustBind(t, &gtkWidgetGetPreferredWidth, "gtk_widget_get_preferred_width")
	mustBind(t, &gtkWidgetGetPreferredHeight, "gtk_widget_get_preferred_height")
	mustBind(t, &gtkWidgetGetStyleContext, "gtk_widget_get_style_context")
	mustBind(t, &gtkStyleContextAddClass, "gtk_style_context_add_class")
	mustBind(t, &gtkStyleContextAddProviderForScreen, "gtk_style_context_add_provider_for_screen")
	mustBind(t, &gtkCssProviderNew, "gtk_css_provider_new")
	mustBind(t, &gtkCssProviderLoadFromData, "gtk_css_provider_load_from_data")
	mustBind(t, &gtkAboutDialogNew, "gtk_about_dialog_new")
	mustBind(t, &gtkAboutDialogSetProgramName, "gtk_about_dialog_set_program_name")
	mustBind(t, &gtkAboutDialogSetVersion, "gtk_about_dialog_set_version")
	mustBind(t, &gtkAboutDialogSetCopyright, "gtk_about_dialog_set_copyright")
	mustBind(t, &gtkAboutDialogSetComments, "gtk_about_dialog_set_comments")
	mustBind(libGObject, &gTypeCheckInstanceIsA, "g_type_check_instance_is_a")
	mustBind(t, &gtkPrintSettingsNew, "gtk_print_settings_new")
	mustBind(t, &gtkPrintSettingsSet, "gtk_print_settings_set")
	mustBind(t, &gtkPageSetupNew, "gtk_page_setup_new")
	mustBind(t, &gtkPageSetupSetPaperSize, "gtk_page_setup_set_paper_size")
	mustBind(t, &gtkPageSetupSetOrientation, "gtk_page_setup_set_orientation")
	mustBind(t, &gtkPageSetupSetTopMargin, "gtk_page_setup_set_top_margin")
	mustBind(t, &gtkPageSetupSetBottomMargin, "gtk_page_setup_set_bottom_margin")
	mustBind(t, &gtkPageSetupSetLeftMargin, "gtk_page_setup_set_left_margin")
	mustBind(t, &gtkPageSetupSetRightMargin, "gtk_page_setup_set_right_margin")
	mustBind(t, &gtkPaperSizeNewCustom, "gtk_paper_size_new_custom")
	mustBind(t, &gtkPaperSizeFree, "gtk_paper_size_free")
	mustBind(t, &gtkRecentManagerGetDefault, "gtk_recent_manager_get_default")
	mustBind(t, &gtkRecentManagerAddItem, "gtk_recent_manager_add_item")
	mustBind(t, &gtkRecentManagerGetItems, "gtk_recent_manager_get_items")
	mustBind(t, &gtkRecentInfoGetURI, "gtk_recent_info_get_uri")
	mustBind(t, &gtkRecentInfoHasApplication, "gtk_recent_info_has_application")
	mustBind(t, &gtkRecentInfoUnref, "gtk_recent_info_unref")
	mustBind(t, &gtkRecentManagerRemoveItem, "gtk_recent_manager_remove_item")
	mustBind(t, &gtkRecentManagerHasItem, "gtk_recent_manager_has_item")
	mustBind(libCairo, &cairoSurfaceWriteToPNGStream, "cairo_surface_write_to_png_stream")
	mustBind(libCairo, &cairoSurfaceDestroy, "cairo_surface_destroy")

	if libIndicator != 0 {
		bind(libIndicator, &appIndicatorNew, "app_indicator_new")
		bind(libIndicator, &appIndicatorSetStatus, "app_indicator_set_status")
		bind(libIndicator, &appIndicatorSetMenu, "app_indicator_set_menu")
		bind(libIndicator, &appIndicatorSetIconFull, "app_indicator_set_icon_full")
		bind(libIndicator, &appIndicatorSetIconThemePath, "app_indicator_set_icon_theme_path")
		bind(libIndicator, &appIndicatorSetTitle, "app_indicator_set_title")
		bind(libIndicator, &appIndicatorSetLabel, "app_indicator_set_label")
	}
	return nil
}

var (
	webKitOnce sync.Once
	errWebKit  error
)

// webKit loads WebKitGTK, JavaScriptCore and libsoup, the first time a
// window shows a web page or browsing data is cleared, and returns why they
// did not load. Windows that show Content need GTK alone: an app whose
// windows all do never loads them, which saves it some 75 libraries, 7 MB
// and 20 ms of startup.
func webKit() error {
	webKitOnce.Do(func() { errWebKit = loadWebKit() })
	if errWebKit == nil && theBackend != nil {
		cookiesOnce.Do(persistCookies)
	}
	return errWebKit
}

func loadWebKit() error {
	var err error
	if libWebKit, err = open("libwebkit2gtk-4.1.so.0", "libwebkit2gtk-4.0.so.37"); err != nil {
		return fmt.Errorf("%w (install WebKitGTK: libwebkit2gtk-4.1-0 on Debian/Ubuntu, webkit2gtk4.1 on Fedora)", err)
	}
	// 4.1 links libsoup 3 and 4.0 libsoup 2.4. Opening the loaded library
	// again returns the same handle, which tells the two apart.
	h41, _ := open("libwebkit2gtk-4.1.so.0")
	soup3 = h41 != 0 && h41 == libWebKit
	jsc, soup := "libjavascriptcoregtk-4.1.so.0", "libsoup-3.0.so.0"
	if !soup3 {
		jsc, soup = "libjavascriptcoregtk-4.0.so.18", "libsoup-2.4.so.1"
	}
	// SoupCookie pointers must never cross Soup ABI generations.
	if libJSC, err = open(jsc); err != nil {
		return err
	}
	if libSoup, err = open(soup); err != nil {
		return err
	}
	bindCookies()

	w := libWebKit
	mustBind(w, &webkitWebContextGetDefault, "webkit_web_context_get_default")
	mustBind(w, &webkitWebContextRegisterURIScheme, "webkit_web_context_register_uri_scheme")
	mustBind(w, &webkitWebContextGetSecurityManager, "webkit_web_context_get_security_manager")
	mustBind(w, &webkitWebContextGetWebsiteDataManager, "webkit_web_context_get_website_data_manager")
	bind(w, &webkitWebContextSetPreferredLanguages, "webkit_web_context_set_preferred_languages")
	mustBind(w, &webkitWebsiteDataManagerClear, "webkit_website_data_manager_clear")
	mustBind(w, &webkitWebsiteDataManagerClearFinish, "webkit_website_data_manager_clear_finish")
	mustBind(w, &webkitDownloadGetWebView, "webkit_download_get_web_view")
	mustBind(w, &webkitDownloadGetRequest, "webkit_download_get_request")
	mustBind(w, &webkitDownloadSetDestination, "webkit_download_set_destination")
	mustBind(w, &webkitDownloadSetAllowOverwrite, "webkit_download_set_allow_overwrite")
	mustBind(w, &webkitDownloadCancel, "webkit_download_cancel")
	mustBind(w, &webkitResponsePolicyDecisionGetResponse, "webkit_response_policy_decision_get_response")
	mustBind(w, &webkitResponsePolicyDecisionIsMIMETypeSupported, "webkit_response_policy_decision_is_mime_type_supported")
	mustBind(w, &webkitPolicyDecisionDownload, "webkit_policy_decision_download")
	mustBind(w, &webkitURIResponseGetURI, "webkit_uri_response_get_uri")
	mustBind(w, &webkitURIResponseGetHTTPHeaders, "webkit_uri_response_get_http_headers")
	mustBind(w, &webkitSecurityManagerRegisterSecure, "webkit_security_manager_register_uri_scheme_as_secure")
	mustBind(w, &webkitSecurityManagerRegisterCORS, "webkit_security_manager_register_uri_scheme_as_cors_enabled")
	mustBind(w, &webkitUserContentManagerNew, "webkit_user_content_manager_new")
	mustBind(w, &webkitUserContentManagerRegisterHandler, "webkit_user_content_manager_register_script_message_handler")
	mustBind(w, &webkitUserContentManagerUnregisterHandler, "webkit_user_content_manager_unregister_script_message_handler")
	mustBind(w, &webkitUserContentManagerAddScript, "webkit_user_content_manager_add_script")
	mustBind(w, &webkitUserContentManagerRemoveAllScripts, "webkit_user_content_manager_remove_all_scripts")
	mustBind(w, &webkitUserScriptNew, "webkit_user_script_new")
	mustBind(w, &webkitUserScriptUnref, "webkit_user_script_unref")
	mustBind(w, &webkitWebViewNewWithUserContentManager, "webkit_web_view_new_with_user_content_manager")
	mustBind(w, &webkitWebViewGetSettings, "webkit_web_view_get_settings")
	mustBind(w, &webkitSettingsSetEnableDeveloperExtras, "webkit_settings_set_enable_developer_extras")
	mustBind(w, &webkitSettingsSetUserAgent, "webkit_settings_set_user_agent")
	mustBind(w, &webkitSettingsGetUserAgent, "webkit_settings_get_user_agent")
	mustBind(w, &webkitSettingsSetAllowFileAccessFromFileURLs, "webkit_settings_set_allow_file_access_from_file_urls")
	mustBind(w, &webkitSettingsSetJavascriptCanAccessClipboard, "webkit_settings_set_javascript_can_access_clipboard")
	mustBind(w, &webkitWebViewLoadURI, "webkit_web_view_load_uri")
	mustBind(w, &webkitWebViewLoadHTML, "webkit_web_view_load_html")
	mustBind(w, &webkitWebViewReload, "webkit_web_view_reload")
	mustBind(w, &webkitWebViewReloadBypassCache, "webkit_web_view_reload_bypass_cache")
	mustBind(w, &webkitWebViewStopLoading, "webkit_web_view_stop_loading")
	mustBind(w, &webkitWebViewGoBack, "webkit_web_view_go_back")
	mustBind(w, &webkitWebViewGoForward, "webkit_web_view_go_forward")
	mustBind(w, &webkitWebViewCanGoBack, "webkit_web_view_can_go_back")
	mustBind(w, &webkitWebViewCanGoForward, "webkit_web_view_can_go_forward")
	mustBind(w, &webkitWebViewIsLoading, "webkit_web_view_is_loading")
	mustBind(w, &webkitWebViewGetURI, "webkit_web_view_get_uri")
	mustBind(w, &webkitWebViewGetTitle, "webkit_web_view_get_title")
	mustBind(w, &webkitWebViewSetZoomLevel, "webkit_web_view_set_zoom_level")
	mustBind(w, &webkitWebViewGetZoomLevel, "webkit_web_view_get_zoom_level")
	mustBind(w, &webkitWebViewSetBackgroundColor, "webkit_web_view_set_background_color")
	bind(w, &webkitWebViewEvaluateJavascript, "webkit_web_view_evaluate_javascript")
	bind(w, &webkitWebViewRunJavascript, "webkit_web_view_run_javascript")
	bind(w, &webkitWebViewCallAsyncJavascriptFunction, "webkit_web_view_call_async_javascript_function")
	bind(w, &webkitWebViewCallAsyncJavascriptFunctionFinish, "webkit_web_view_call_async_javascript_function_finish")
	mustBind(w, &webkitWebViewGetInspector, "webkit_web_view_get_inspector")
	mustBind(w, &webkitWebInspectorShow, "webkit_web_inspector_show")
	mustBind(w, &webkitWebInspectorClose, "webkit_web_inspector_close")
	mustBind(w, &webkitWebInspectorGetWebView, "webkit_web_inspector_get_web_view")
	mustBind(w, &webkitWebViewGetSnapshot, "webkit_web_view_get_snapshot")
	mustBind(w, &webkitWebViewGetSnapshotFinish, "webkit_web_view_get_snapshot_finish")
	mustBind(w, &webkitWebViewExecuteEditingCommand, "webkit_web_view_execute_editing_command")
	mustBind(w, &webkitPrintOperationNew, "webkit_print_operation_new")
	mustBind(w, &webkitPrintOperationRunDialog, "webkit_print_operation_run_dialog")
	mustBind(w, &webkitPrintOperationSetPrintSettings, "webkit_print_operation_set_print_settings")
	mustBind(w, &webkitPrintOperationSetPageSetup, "webkit_print_operation_set_page_setup")
	mustBind(w, &webkitPrintOperationPrint, "webkit_print_operation_print")
	mustBind(w, &webkitSettingsGetPrintBackgrounds, "webkit_settings_get_print_backgrounds")
	mustBind(w, &webkitSettingsSetPrintBackgrounds, "webkit_settings_set_print_backgrounds")
	mustBind(w, &webkitUserMediaPermissionRequestGetType, "webkit_user_media_permission_request_get_type")
	mustBind(w, &webkitGeolocationPermissionRequestGetType, "webkit_geolocation_permission_request_get_type")
	mustBind(w, &webkitNotificationPermissionRequestGetType, "webkit_notification_permission_request_get_type")
	mustBind(w, &webkitUserMediaPermissionIsForAudioDevice, "webkit_user_media_permission_is_for_audio_device")
	mustBind(w, &webkitUserMediaPermissionIsForVideoDevice, "webkit_user_media_permission_is_for_video_device")
	mustBind(w, &webkitPermissionRequestAllow, "webkit_permission_request_allow")
	mustBind(w, &webkitPermissionRequestDeny, "webkit_permission_request_deny")
	mustBind(w, &webkitJavascriptResultGetJSValue, "webkit_javascript_result_get_js_value")
	mustBind(w, &webkitNavigationPolicyDecisionGetNavigationAction, "webkit_navigation_policy_decision_get_navigation_action")
	bind(w, &webkitNavigationPolicyDecisionGetFrameName, "webkit_navigation_policy_decision_get_frame_name")
	mustBind(w, &webkitNavigationActionGetRequest, "webkit_navigation_action_get_request")
	mustBind(w, &webkitNavigationActionGetNavigationType, "webkit_navigation_action_get_navigation_type")
	mustBind(w, &webkitNavigationActionIsUserGesture, "webkit_navigation_action_is_user_gesture")
	mustBind(w, &webkitURIRequestGetURI, "webkit_uri_request_get_uri")
	mustBind(w, &webkitPolicyDecisionUse, "webkit_policy_decision_use")
	mustBind(w, &webkitPolicyDecisionIgnore, "webkit_policy_decision_ignore")
	mustBind(w, &webkitURISchemeRequestGetURI, "webkit_uri_scheme_request_get_uri")
	mustBind(w, &webkitURISchemeRequestGetWebView, "webkit_uri_scheme_request_get_web_view")
	mustBind(w, &webkitURISchemeRequestFinishError, "webkit_uri_scheme_request_finish_error")
	mustBind(w, &webkitURISchemeRequestFinish, "webkit_uri_scheme_request_finish")
	bind(w, &webkitURISchemeRequestGetHTTPMethod, "webkit_uri_scheme_request_get_http_method")
	bind(w, &webkitURISchemeRequestGetHTTPHeaders, "webkit_uri_scheme_request_get_http_headers")
	bind(w, &webkitURISchemeRequestGetHTTPBody, "webkit_uri_scheme_request_get_http_body")
	bind(w, &webkitURISchemeRequestFinishWithResponse, "webkit_uri_scheme_request_finish_with_response")
	bind(w, &webkitURISchemeResponseNew, "webkit_uri_scheme_response_new")
	bind(w, &webkitURISchemeResponseSetStatus, "webkit_uri_scheme_response_set_status")
	bind(w, &webkitURISchemeResponseSetContentType, "webkit_uri_scheme_response_set_content_type")
	bind(w, &webkitURISchemeResponseSetHTTPHeaders, "webkit_uri_scheme_response_set_http_headers")

	mustBind(libJSC, &jscValueToString, "jsc_value_to_string")
	mustBind(libSoup, &soupMessageHeadersNew, "soup_message_headers_new")
	mustBind(libSoup, &soupMessageHeadersGetOne, "soup_message_headers_get_one")
	mustBind(libSoup, &soupMessageHeadersAppend, "soup_message_headers_append")
	mustBind(libSoup, &soupMessageHeadersForeach, "soup_message_headers_foreach")
	return nil
}

// connect connects a signal handler with the given user data.
func connect(instance ptr, signal string, handler ptr, data ptr) uint64 {
	return gSignalConnectData(instance, cs(signal), handler, data, 0, 0)
}
