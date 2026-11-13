//go:build windows

package main

import (
	"errors"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const maxBrowserElements = 500

var procBrowserIsWindowVisible = user32.NewProc("IsWindowVisible")

func (server *helperServer) serveBrowserURL(message proto.Message) {
	var params proto.BrowserURLParams
	if err := message.DecodeParams(&params); err != nil {
		server.fail(message.ID, proto.CodeBadRequest, "Bad params: %v", err)
		return
	}
	var result proto.BrowserURLResult
	if !server.automation.call(func(automation *comObject) {
		result = readBrowserURL(automation, params)
	}) {
		server.fail(message.ID, proto.CodeInternal, "The UI Automation thread stopped")
		return
	}
	server.result(message.ID, result)
}

func readBrowserURL(automation *comObject, params proto.BrowserURLParams) proto.BrowserURLResult {
	started := time.Now()
	result := proto.BrowserURLResult{Via: proto.URLViaNone}
	app, err := browserApp(params)
	if err == nil {
		result.BundleID = app.BundleID
		var windowHandle uintptr
		windowHandle, err = browserWindow(app.PID)
		if err == nil {
			var address string
			address, err = browserAddress(automation, windowHandle)
			if address != "" {
				result.URL = address
				result.Via = proto.URLViaAXDocument
			}
		}
	}
	if result.URL == "" {
		if err != nil {
			result.Error = err.Error()
		} else {
			result.Error = "no address bar URL"
		}
	}
	result.MS = millisecondsSince(started)
	return result
}

func browserApp(params proto.BrowserURLParams) (proto.App, error) {
	if params.PID != 0 {
		return appForProcess(uint32(params.PID)), nil
	}
	if params.BundleID == "" {
		return frontmostApp()
	}
	apps, err := runningApps()
	if err != nil {
		return proto.App{}, err
	}
	for _, app := range apps {
		if strings.EqualFold(app.BundleID, params.BundleID) {
			return appForProcess(uint32(app.PID)), nil
		}
	}
	return proto.App{}, errors.New("browser is not running")
}

func browserWindow(processID int) (uintptr, error) {
	foreground, _, _ := procGetForegroundWindow.Call()
	if foreground != 0 {
		var foregroundProcessID uint32
		if _, err := windows.GetWindowThreadProcessId(windows.HWND(foreground), &foregroundProcessID); err == nil && foregroundProcessID == uint32(processID) {
			return foreground, nil
		}
	}
	var found uintptr
	callback := syscall.NewCallback(func(windowHandle uintptr, _ uintptr) uintptr {
		visible, _, _ := procBrowserIsWindowVisible.Call(windowHandle)
		if visible == 0 || found != 0 {
			return 1
		}
		var windowProcessID uint32
		if _, err := windows.GetWindowThreadProcessId(windows.HWND(windowHandle), &windowProcessID); err == nil && windowProcessID == uint32(processID) {
			found = windowHandle
		}
		return 1
	})
	if err := windows.EnumWindows(callback, nil); err != nil {
		return 0, err
	}
	if found == 0 {
		return 0, errors.New("no visible browser window")
	}
	return found, nil
}

func browserAddress(automation *comObject, windowHandle uintptr) (string, error) {
	root, err := automation.output(automationElementFromHandle, windowHandle)
	if err != nil {
		return "", err
	}
	defer root.release()
	condition, err := automation.output(automationCreateTrueCondition)
	if err != nil {
		return "", err
	}
	defer condition.release()
	elements, err := root.output(elementFindAll, treeScopeDescendants, uintptr(unsafe.Pointer(condition)))
	if err != nil {
		return "", err
	}
	defer elements.release()
	var count int32
	if result := elements.call(arrayLength, uintptr(unsafe.Pointer(&count))); int32(result) < 0 {
		return "", hresultError("IUIAutomationElementArray.Length", result)
	}
	for index := int32(0); index < min(count, maxBrowserElements); index++ {
		element, err := elements.output(arrayElement, uintptr(index))
		if err != nil {
			continue
		}
		var controlType int32
		element.call(elementCurrentControlType, uintptr(unsafe.Pointer(&controlType)))
		var nameAddress, automationIDAddress *uint16
		element.call(elementCurrentName, uintptr(unsafe.Pointer(&nameAddress)))
		element.call(elementCurrentAutomationID, uintptr(unsafe.Pointer(&automationIDAddress)))
		name := readBSTR(nameAddress)
		automationID := readBSTR(automationIDAddress)
		if !addressBarControl(controlType, name, automationID) {
			element.release()
			continue
		}
		pattern, err := currentPattern(element, patternValue, &interfaceValuePattern)
		if err != nil {
			element.release()
			continue
		}
		var valueAddress *uint16
		result := pattern.call(valuePatternCurrentValue, uintptr(unsafe.Pointer(&valueAddress)))
		pattern.release()
		element.release()
		if int32(result) >= 0 {
			if value := strings.TrimSpace(readBSTR(valueAddress)); value != "" {
				return value, nil
			}
		}
	}
	return "", nil
}

func addressBarControl(controlType int32, name, automationID string) bool {
	if controlType != controlTypeEdit && controlType != controlTypeComboBox {
		return false
	}
	lowerName := strings.ToLower(name)
	lowerAutomationID := strings.ToLower(automationID)
	return strings.Contains(lowerName, "address bar") || strings.Contains(lowerName, "address and search") || strings.Contains(lowerName, "enter address") ||
		strings.Contains(name, "地址栏") || strings.Contains(name, "地址和搜索") || strings.Contains(name, "输入地址") ||
		strings.Contains(lowerAutomationID, "urlbar") || strings.Contains(lowerAutomationID, "addressbar") || strings.Contains(lowerAutomationID, "address-bar") || automationID == "Ctrl+L"
}
