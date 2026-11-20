//go:build windows && (amd64 || arm64)

package windows

import (
	"runtime"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

var (
	procSHGetKnownFolderItem = shell32.NewProc("SHGetKnownFolderItem")
	procCoUninitialize       = ole32.NewProc("CoUninitialize")

	folderIDAppsFolder = guid("1e87508d-89c2-42f0-8a7e-645a0f50ca58")
	bhidEnumItems      = guid("94f60519-2850-4924-aa5a-d15e84868039")
	iidIEnumShellItems = guid("70629033-e363-4a28-a567-0db78006e6d7")
)

const appsFolder = `shell:AppsFolder\`

// Applications lists the apps of the Start menu's All apps list, the
// shell's Applications folder (FOLDERID_AppsFolder): desktop apps by their
// shortcuts and packaged apps alike, as PowerToys' Command Palette lists
// them. An app's ID is its AppUserModelID, or for a desktop app without
// one, its path under a known folder; its Path is the shell item,
// shell:AppsFolder\<ID>, which FileIcon draws and OpenApplication starts.
// It runs on a thread of its own, in its own COM apartment.
func (b *Backend) Applications() ([]platform.Application, error) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if r, _, _ := procCoInitializeEx.Call(0, coinitApartmentThreaded); !failed(r) {
		defer procCoUninitialize.Call()
	}
	var folder uintptr
	if r, _, _ := procSHGetKnownFolderItem.Call(uintptr(unsafe.Pointer(&folderIDAppsFolder)), 0, 0, uintptr(unsafe.Pointer(&iidIShellItem)), uintptr(unsafe.Pointer(&folder))); failed(r) {
		return nil, hresultError("SHGetKnownFolderItem(AppsFolder)", r)
	}
	defer release(folder)
	var items uintptr
	if r := comCall(folder, 3, 0, uintptr(unsafe.Pointer(&bhidEnumItems)), uintptr(unsafe.Pointer(&iidIEnumShellItems)), uintptr(unsafe.Pointer(&items))); failed(r) { // BindToHandler
		return nil, hresultError("enumerating the Applications folder", r)
	}
	defer release(items)
	const (
		sigdnNormalDisplay         = 0
		sigdnParentRelativeParsing = 0x80018001
	)
	var apps []platform.Application
	for {
		var item uintptr
		var fetched uint32
		if r := comCall(items, 3, 1, uintptr(unsafe.Pointer(&item)), uintptr(unsafe.Pointer(&fetched))); r != 0 || fetched == 0 { // Next: S_FALSE at the end
			break
		}
		var name, id uintptr
		comCall(item, 5, sigdnNormalDisplay, uintptr(unsafe.Pointer(&name)))       // GetDisplayName
		comCall(item, 5, sigdnParentRelativeParsing, uintptr(unsafe.Pointer(&id))) // its ID in the folder
		release(item)
		a := platform.Application{Name: takeWstr(name), ID: takeWstr(id)}
		if a.Name == "" || a.ID == "" {
			continue
		}
		a.Path = appsFolder + a.ID
		apps = append(apps, a)
	}
	return apps, nil
}

// OpenApplication starts an app of the Applications folder by its shell
// item, with its default verb, or any program by its path.
func (shell) OpenApplication(path string) error { return shellExecute("", path, "") }
