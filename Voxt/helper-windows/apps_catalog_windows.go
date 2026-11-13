//go:build windows

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

const appPathsKey = `SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths`

var (
	classShellLink       = windows.GUID{Data1: 0x00021401, Data4: [8]byte{0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	interfaceShellLink   = windows.GUID{Data1: 0x000214f9, Data4: [8]byte{0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
	interfacePersistFile = windows.GUID{Data1: 0x0000010b, Data4: [8]byte{0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x46}}
)

func listWindowsApps(installed bool) ([]proto.AppInfo, error) {
	running, err := runningApps()
	if err != nil {
		return nil, err
	}
	if installed {
		return mergeWindowsApps(running, installedApps()), nil
	}
	return mergeWindowsApps(running, nil), nil
}

func runningApps() ([]proto.AppInfo, error) {
	snapshot, err := windows.CreateToolhelp32Snapshot(windows.TH32CS_SNAPPROCESS, 0)
	if err != nil {
		return nil, err
	}
	defer windows.CloseHandle(snapshot)
	entry := windows.ProcessEntry32{Size: uint32(unsafe.Sizeof(windows.ProcessEntry32{}))}
	if err := windows.Process32First(snapshot, &entry); err != nil {
		return nil, err
	}
	apps := make([]proto.AppInfo, 0)
	for {
		path := processExecutablePath(entry.ProcessID)
		if path != "" {
			apps = append(apps, appInfoForPath(path, true, int(entry.ProcessID), ""))
		}
		if err := windows.Process32Next(snapshot, &entry); err != nil {
			break
		}
	}
	return apps, nil
}

func installedApps() []proto.AppInfo {
	apps := make([]proto.AppInfo, 0)
	for _, root := range []string{os.Getenv("ProgramData"), os.Getenv("AppData")} {
		if root == "" {
			continue
		}
		directory := filepath.Join(root, `Microsoft\Windows\Start Menu\Programs`)
		_ = filepath.WalkDir(directory, func(path string, entry fs.DirEntry, err error) error {
			if err != nil || entry.IsDir() || !strings.EqualFold(filepath.Ext(entry.Name()), ".lnk") {
				return nil
			}
			target, err := resolveShellLink(path)
			if err != nil {
				return nil
			}
			target, ok := installedExecutablePath(target)
			if ok {
				name := strings.TrimSuffix(entry.Name(), filepath.Ext(entry.Name()))
				apps = append(apps, appInfoForPath(target, false, 0, name))
			}
			return nil
		})
	}
	for _, root := range []registry.Key{registry.LOCAL_MACHINE, registry.CURRENT_USER} {
		apps = append(apps, registryAppPaths(root)...)
	}
	return apps
}

func registryAppPaths(root registry.Key) []proto.AppInfo {
	key, err := registry.OpenKey(root, appPathsKey, registry.ENUMERATE_SUB_KEYS|registry.QUERY_VALUE)
	if err != nil {
		return nil
	}
	defer key.Close()
	names, err := key.ReadSubKeyNames(-1)
	if err != nil {
		return nil
	}
	apps := make([]proto.AppInfo, 0, len(names))
	for _, name := range names {
		subkey, err := registry.OpenKey(key, name, registry.QUERY_VALUE)
		if err != nil {
			continue
		}
		value, _, err := subkey.GetStringValue("")
		subkey.Close()
		if err != nil {
			continue
		}
		path, ok := installedExecutablePath(value)
		if ok {
			apps = append(apps, appInfoForPath(path, false, 0, ""))
		}
	}
	return apps
}

func appInfoForPath(path string, running bool, pid int, name string) proto.AppInfo {
	if name == "" {
		name = strings.TrimSuffix(filepath.Base(path), filepath.Ext(path))
	}
	return proto.AppInfo{BundleID: stableAppID(path), Name: name, Running: running, PID: pid, Path: path}
}

func stableAppID(path string) string {
	absolute, err := filepath.Abs(path)
	if err == nil {
		path = absolute
	}
	normalized := strings.ToLower(filepath.Clean(path))
	digest := sha256.Sum256([]byte(normalized))
	return "win32." + hex.EncodeToString(digest[:16])
}

func mergeWindowsApps(running, installed []proto.AppInfo) []proto.AppInfo {
	merged := make([]proto.AppInfo, 0, len(running)+len(installed))
	indices := make(map[string]int)
	for _, app := range append(append([]proto.AppInfo{}, running...), installed...) {
		key := strings.ToLower(app.BundleID)
		if key == "" {
			continue
		}
		if index, found := indices[key]; found {
			if merged[index].Path == "" {
				merged[index].Path = app.Path
			}
			continue
		}
		indices[key] = len(merged)
		merged = append(merged, app)
	}
	sort.Slice(merged, func(first, second int) bool {
		firstName := strings.ToLower(merged[first].Name)
		secondName := strings.ToLower(merged[second].Name)
		if firstName == secondName {
			return merged[first].BundleID < merged[second].BundleID
		}
		return firstName < secondName
	})
	return merged
}

func installedExecutablePath(command string) (string, bool) {
	command = strings.TrimSpace(strings.TrimPrefix(command, `"`))
	if index := strings.Index(command, `"`); index >= 0 {
		command = command[:index]
	}
	command = expandWindowsEnvironment(strings.TrimSpace(command))
	path, err := filepath.Abs(command)
	if err != nil || !strings.EqualFold(filepath.Ext(path), ".exe") {
		return "", false
	}
	details, err := os.Stat(path)
	if err != nil || details.IsDir() {
		return "", false
	}
	return filepath.Clean(path), true
}

func expandWindowsEnvironment(value string) string {
	source, err := windows.UTF16PtrFromString(value)
	if err != nil {
		return value
	}
	count, err := windows.ExpandEnvironmentStrings(source, nil, 0)
	if count == 0 || err != nil {
		return value
	}
	buffer := make([]uint16, count)
	count, err = windows.ExpandEnvironmentStrings(source, &buffer[0], uint32(len(buffer)))
	if count == 0 || err != nil {
		return value
	}
	return windows.UTF16ToString(buffer)
}

func resolveShellLink(path string) (string, error) {
	var link *comObject
	result, _, _ := procCoCreateInstance.Call(
		uintptr(unsafe.Pointer(&classShellLink)), 0, comClassContextAll,
		uintptr(unsafe.Pointer(&interfaceShellLink)), uintptr(unsafe.Pointer(&link)),
	)
	if int32(result) < 0 {
		return "", hresultError("CoCreateInstance(ShellLink)", result)
	}
	defer link.release()
	var persist *comObject
	result = link.call(0, uintptr(unsafe.Pointer(&interfacePersistFile)), uintptr(unsafe.Pointer(&persist)))
	if int32(result) < 0 {
		return "", hresultError("IShellLink.QueryInterface(IPersistFile)", result)
	}
	defer persist.release()
	pathPointer, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", err
	}
	result = persist.call(5, uintptr(unsafe.Pointer(pathPointer)), 0)
	if int32(result) < 0 {
		return "", hresultError("IPersistFile.Load", result)
	}
	buffer := make([]uint16, windows.MAX_LONG_PATH)
	result = link.call(3, uintptr(unsafe.Pointer(&buffer[0])), uintptr(len(buffer)), 0, 0)
	if int32(result) < 0 {
		return "", hresultError("IShellLinkW.GetPath", result)
	}
	return windows.UTF16ToString(buffer), nil
}
