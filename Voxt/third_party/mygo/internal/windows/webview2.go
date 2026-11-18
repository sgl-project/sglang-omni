//go:build windows && (amd64 || arm64)

package windows

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/webview2"
)

// Vtable indices of the WebView2 interfaces used here, from WebView2.h
// (SDK 1.0.4191.47). Each interface extends the previous version, so the
// indices stay valid for newer runtimes.
const (
	// ICoreWebView2Environment
	envCreateController          = 3
	envCreateWebResourceResponse = 4

	// ICoreWebView2Controller, ICoreWebView2Controller2
	ctlPutIsVisible                      = 4
	ctlPutBounds                         = 6
	ctlGetZoomFactor                     = 7
	ctlPutZoomFactor                     = 8
	ctlMoveFocus                         = 12
	ctlAddAcceleratorKeyPressed          = 19
	ctlNotifyParentWindowPositionChanged = 23
	ctlClose                             = 24
	ctlGetCoreWebView2                   = 25
	ctl2PutDefaultBackgroundColor        = 27

	// ICoreWebView2
	wvGetSettings                            = 3
	wvGetSource                              = 4
	wvNavigate                               = 5
	wvNavigateToString                       = 6
	wvAddNavigationStarting                  = 7
	wvAddContentLoading                      = 9
	wvAddPermissionRequested                 = 23
	wvAddNavigationCompleted                 = 15
	wvAddProcessFailed                       = 25
	wvAddScriptToExecuteOnDocumentCreated    = 27
	wvRemoveScriptToExecuteOnDocumentCreated = 28
	wvExecuteScript                          = 29
	wvCapturePreview                         = 30
	wvReload                                 = 31
	wvAddWebMessageReceived                  = 34
	wvCallDevToolsProtocolMethod             = 36
	wvGetCanGoBack                           = 38
	wvGetCanGoForward                        = 39
	wvGoBack                                 = 40
	wvGoForward                              = 41
	wvStop                                   = 43
	wvAddNewWindowRequested                  = 44
	wvAddDocumentTitleChanged                = 46
	wvGetDocumentTitle                       = 48
	wvOpenDevToolsWindow                     = 51
	wvAddWebResourceRequested                = 55
	wvAddWebResourceRequestedFilter          = 57
	wvRemoveWebResourceRequestedFilter       = 58
	wvAddWindowCloseRequested                = 59

	// ICoreWebView2Settings, ICoreWebView2Settings2
	setPutIsScriptEnabled                = 4
	setPutIsWebMessageEnabled            = 6
	setPutAreDefaultScriptDialogsEnabled = 8
	setPutIsStatusBarEnabled             = 10
	setPutAreDevToolsEnabled             = 12
	setPutAreDefaultContextMenusEnabled  = 14
	setPutIsZoomControlEnabled           = 18
	set2GetUserAgent                     = 21
	set2PutUserAgent                     = 22

	// Event arguments and helpers.
	navStartingGetURI             = 3
	navStartingGetIsUserInitiated = 4
	navStartingPutCancel          = 8
	navStarting3GetNavigationKind = 12
	navCompletedGetIsSuccess      = 3
	navCompletedGetWebErrorStatus = 4
	msgTryGetWebMessageAsString   = 5
	msg2GetAdditionalObjects      = 6
	objectsGetCount               = 3
	objectsGetValueAtIndex        = 4
	fileGetPath                   = 3
	newWinGetURI                  = 3
	newWinPutHandled              = 6
	newWinGetWindowFeatures       = 10
	featGetHasPosition            = 3
	featGetHasSize                = 4
	featGetLeft                   = 5
	featGetTop                    = 6
	featGetHeight                 = 7
	featGetWidth                  = 8
	resReqGetRequest              = 3
	resReqPutResponse             = 5
	resReqGetDeferral             = 6
	resReqGetResourceContext      = 7
	reqGetURI                     = 3
	reqGetMethod                  = 5
	reqGetContent                 = 7
	reqGetHeaders                 = 9
	headersGetIterator            = 8
	iterGetCurrentHeader          = 3
	iterGetHasCurrentHeader       = 4
	iterMoveNext                  = 5
	deferralComplete              = 3
	procFailedGetKind             = 3
	wv4AddDownloadStarting        = 75  // ICoreWebView2_4
	wv13GetProfile                = 105 // ICoreWebView2_13
	profile2ClearBrowsingDataAll  = 12  // ICoreWebView2Profile2
	dlStartGetOperation           = 3
	dlStartPutCancel              = 5
	dlStartGetResultFilePath      = 6
	dlStartPutResultFilePath      = 7
	dlStartPutHandled             = 9
	dlOpAddStateChanged           = 7
	dlOpGetURI                    = 9
	dlOpGetState                  = 16
	dlOpGetInterruptReason        = 17
	permGetURI                    = 3
	permGetKind                   = 4
	permPutState                  = 7
	accelGetKeyEventKind          = 3
	accelGetVirtualKey            = 4
	accelPutHandled               = 8
)

var (
	iidICoreWebView2Controller2                  = guid("c979903e-d4ca-4228-92eb-47ee3fa96eab")
	iidICoreWebView2Settings2                    = guid("ee9a0f68-f46c-4e32-ac23-ef8cac224d2a")
	iidICoreWebView2WebMessageReceivedEventArgs2 = guid("06fc7ab7-c90c-4297-9389-33ca01cf6d5e")
	iidICoreWebView2NavigationStartingEventArgs3 = guid("ddffe494-4942-4bd2-ab73-35b8ff40e19f")
	iidICoreWebView2File                         = guid("f2c19559-6bc1-4583-a757-90021be9afec")
	iidICoreWebView2_4                           = guid("20d02d59-6df2-42dc-bd06-f98a694b1302")
	iidICoreWebView2_13                          = guid("f75f09a8-667e-4983-88d6-c8773f315e84")
	iidICoreWebView2Profile2                     = guid("fa740d4b-5eae-4344-a8ad-74be31925397")
)

const (
	hkeyCurrentUser = 0x80000001
	keyQueryValue   = 0x0001
)

// createEnvironment starts creating the WebView2 environment, in language,
// the app's (envOptions), or the system's for ""; done runs on the main
// thread with it, or with an error. A runtime that fails to create one with
// the options is asked for one without them.
func createEnvironment(userData, language string, done func(env uintptr, err error)) error {
	err := createEnvironmentWith(userData, language, func(env uintptr, err error) {
		if err != nil && language != "" {
			// A runtime that takes no options of ours: one without them,
			// in the language of the system.
			log.Printf("%v; creating it again without its language", err)
			if err := createEnvironmentWith(userData, "", done); err != nil {
				done(0, err)
			}
			return
		}
		done(env, err)
	})
	if err != nil && language != "" {
		return createEnvironmentWith(userData, "", done)
	}
	return err
}

// createEnvironmentWith creates the environment as createEnvironment does,
// with the options of language, or none for "".
func createEnvironmentWith(userData, language string, done func(env uintptr, err error)) error {
	completed := func(hr, env uintptr) {
		if failed(hr) || env == 0 {
			done(0, hresultError("creating the WebView2 environment", hr))
			return
		}
		addRef(env)
		done(env, nil)
	}
	dataDir := u16(userData)
	var options uintptr
	if language != "" {
		options = newEnvOptions(language)
		defer release(options) // the environment takes its own reference
	}

	// An official loader next to the executable wins.
	if proc := loaderProc(); proc != nil {
		hr := withHandler(completed, func(h uintptr) uintptr {
			r, _, _ := proc.Call(0, uintptr(unsafe.Pointer(dataDir)), options, h)
			return r
		})
		if failed(hr) {
			return hresultError("CreateCoreWebView2EnvironmentWithOptions", hr)
		}
		return nil
	}

	// Otherwise the runtime's client library, as the loader does: a fixed
	// version runtime named by WEBVIEW2_BROWSER_EXECUTABLE_FOLDER, else the
	// Evergreen runtime.
	dir, fixed, err := webview2.Find()
	if err != nil {
		return fmt.Errorf("mygo: %w", err)
	}
	client, runtimeType := webview2.ClientDLL(dir), uintptr(0)
	if fixed {
		runtimeType = 1
	}
	dll := syscall.NewLazyDLL(client)
	proc := dll.NewProc("CreateWebViewEnvironmentWithOptionsInternal")
	if err := proc.Find(); err != nil {
		return fmt.Errorf("mygo: unsupported WebView2 Runtime at %s: %w", client, err)
	}
	hr := withHandler(completed, func(h uintptr) uintptr {
		r, _, _ := proc.Call(1, runtimeType, uintptr(unsafe.Pointer(dataDir)), options, h)
		return r
	})
	if failed(hr) {
		return hresultError("creating the WebView2 environment", hr)
	}
	return nil
}

// loaderProc returns the official WebView2 loader next to the executable and
// its CreateCoreWebView2EnvironmentWithOptions, or nil when there is no
// loader to use. createEnvironment calls what it returns, and
// WebViewAvailable asks for it, so the two cannot disagree: a DLL of that
// name with no such export is no loader, and a runtime is looked for
// instead.
func loaderProc() *syscall.LazyProc {
	exe, err := os.Executable()
	if err != nil {
		return nil
	}
	path := filepath.Join(filepath.Dir(exe), "WebView2Loader.dll")
	if _, err := os.Stat(path); err != nil {
		return nil
	}
	proc := syscall.NewLazyDLL(path).NewProc("CreateCoreWebView2EnvironmentWithOptions")
	if proc.Find() != nil {
		return nil
	}
	return proc
}

// WebViewAvailable reports whether a window showing a web page can have
// one: the WebView2 Runtime is installed, or a loader next to the
// executable loads a fixed version runtime or the installed one itself. It
// starts nothing; once a window showing a page created the environment, it
// says whether that worked. It reads the registry and the disk only, so any
// goroutine may call it, before Init too.
func (b *Backend) WebViewAvailable() bool {
	switch b.envResult.Load() {
	case 1:
		return true
	case 2:
		return false
	}
	if loaderProc() != nil {
		return true
	}
	_, _, err := webview2.Find()
	return err == nil
}

// regDWORD reads a DWORD value, reporting whether it exists.
func regDWORD(root uintptr, path, name string) (uint32, bool) {
	var key uintptr
	if r, _, _ := procRegOpenKeyExW.Call(root, uintptr(unsafe.Pointer(u16(path))), 0, keyQueryValue, uintptr(unsafe.Pointer(&key))); r != 0 {
		return 0, false
	}
	defer procRegCloseKey.Call(key)
	var typ, v uint32
	size := uint32(4)
	if r, _, _ := procRegQueryValueExW.Call(key, uintptr(unsafe.Pointer(u16(name))), 0, uintptr(unsafe.Pointer(&typ)), uintptr(unsafe.Pointer(&v)), uintptr(unsafe.Pointer(&size))); r != 0 {
		return 0, false
	}
	return v, true
}
