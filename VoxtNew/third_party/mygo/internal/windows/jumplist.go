//go:build windows && (amd64 || arm64)

package windows

import (
	"errors"
	"os"
	"runtime"
	"strings"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// The app's jump list: its Recent category, which the shell fills from
// SHAddToRecentDocs, and its Tasks category. Both belong to the process's
// AppUserModelID, which MyGo leaves to the shell: it derives one from the
// executable's path, the one pinned shortcuts and notifications use too. An
// explicit ID would also have to be written into the installer's shortcuts,
// or the taskbar would show the running app apart from its pinned button.

// Identifiers of the jump list's objects, from shobjidl.h.
var (
	iidICustomDestinationList    = guid("6332debf-87b5-4670-90c0-5e57b408a49e")
	clsidDestinationList         = guid("77f10cf0-3db5-4966-b520-b7c54fd35ed6")
	iidIApplicationDestinations  = guid("12337d35-94c6-48a0-bce7-6a9c69d4d600")
	clsidApplicationDestinations = guid("86c14003-4d6b-4ef3-a7b4-0506663b2e68")

	iidIObjectArray       = guid("92ca9dcd-5622-4bba-a805-5e9f541bd8c9")
	iidIObjectCollection  = guid("5632b1a4-e38a-400a-928a-d4cd63230295")
	clsidObjectCollection = guid("2d3468c1-36a7-43b6-ac24-d3f02fd9607a") // EnumerableObjectCollection
	iidIShellLinkW        = guid("000214f9-0000-0000-c000-000000000046")
	clsidShellLink        = guid("00021401-0000-0000-c000-000000000046")
	iidIPropertyStore     = guid("886d8eeb-8cf2-4446-8d02-cdba1dbdcf99")
)

// Vtable indices, IUnknown (0, 1, 2) included, and constants.
const (
	cdlBeginList           = 4 // ICustomDestinationList
	cdlAppendKnownCategory = 6
	cdlAddUserTasks        = 7
	cdlCommitList          = 8
	cdlAbortList           = 11

	adRemoveAllDestinations = 5 // IApplicationDestinations

	oaGetCount  = 3 // IObjectArray
	oaGetAt     = 4
	ocAddObject = 5 // IObjectCollection

	slSetDescription      = 7 // IShellLinkW
	slSetWorkingDirectory = 9
	slSetArguments        = 11
	slSetIconLocation     = 17
	slSetPath             = 20

	psSetValue = 6 // IPropertyStore
	psCommit   = 7

	kdcRecent  = 2  // KDC_RECENT, the Recent category
	vtLPWSTR   = 31 // VT_LPWSTR, the type of System.Title
	shardPathW = 3  // SHARD_PATHW: SHAddToRecentDocs takes a path
)

// propertyKey is PROPERTYKEY.
type propertyKey struct {
	Format GUID
	ID     uint32
}

// pkeyTitle is System.Title, the label the jump list shows for a task (the
// shell link's description is its tooltip).
var pkeyTitle = propertyKey{Format: guid("f29f85e0-4ff9-1068-ab91-08002b27b3d9"), ID: 2}

// AddRecentDocument tells the shell the app opened path, which it lists in
// the Recent category of the app's jump list (for the file types the app
// is registered to open) and in the user's recent files.
func (a appController) AddRecentDocument(path string) error {
	procSHAddToRecentDocs.Call(shardPathW, uintptr(unsafe.Pointer(u16(path))))
	return nil
}

// ClearRecentDocuments removes the app's destinations, the Recent category
// of its jump list. SHAddToRecentDocs(NULL), what Electron calls, would
// clear the recent files of every app.
func (a appController) ClearRecentDocuments() error {
	ad, err := coCreate(clsidApplicationDestinations, iidIApplicationDestinations)
	if err != nil {
		return err
	}
	defer release(ad)
	if hr := comCall(ad, adRemoveAllDestinations); failed(hr) {
		return hresultError("IApplicationDestinations::RemoveAllDestinations", hr)
	}
	return nil
}

// SetJumpList commits the app's jump list: the Recent category, which a
// custom list shows only when it asks for it, and the tasks. CommitList
// replaces the whole list, which package mygo keeps.
func (a appController) SetJumpList(list platform.JumpList) error {
	tasks := list.Tasks
	dl, err := coCreate(clsidDestinationList, iidICustomDestinationList)
	if err != nil {
		return err
	}
	defer release(dl)
	var minSlots uint32
	var removed uintptr
	if hr := comCall(dl, cdlBeginList, uintptr(unsafe.Pointer(&minSlots)),
		uintptr(unsafe.Pointer(&iidIObjectArray)), uintptr(unsafe.Pointer(&removed))); failed(hr) {
		return hresultError("ICustomDestinationList::BeginList", hr)
	}
	release(removed) // destinations the user removed; tasks cannot be removed
	abort := func(err error) error {
		comCall(dl, cdlAbortList)
		return err
	}
	if hr := comCall(dl, cdlAppendKnownCategory, kdcRecent); failed(hr) {
		return abort(hresultError("ICustomDestinationList::AppendKnownCategory", hr))
	}
	if len(tasks) > 0 {
		oc, err := coCreate(clsidObjectCollection, iidIObjectCollection)
		if err != nil {
			return abort(err)
		}
		defer release(oc)
		if err := addTasks(oc, tasks); err != nil {
			return abort(err)
		}
		if hr := comCall(dl, cdlAddUserTasks, oc); failed(hr) {
			return abort(hresultError("ICustomDestinationList::AddUserTasks", hr))
		}
	}
	if hr := comCall(dl, cdlCommitList); failed(hr) {
		return abort(hresultError("ICustomDestinationList::CommitList", hr))
	}
	return nil
}

// addTasks adds a shell link per task to the object collection oc.
func addTasks(oc uintptr, tasks []platform.JumpListTask) error {
	exe, err := os.Executable()
	if err != nil {
		return err
	}
	for _, t := range tasks {
		path := t.Path
		if path == "" {
			path = exe
		}
		sl, err := coCreate(clsidShellLink, iidIShellLinkW)
		if err != nil {
			return err
		}
		err = func() error {
			if hr := comCall(sl, slSetPath, uintptr(unsafe.Pointer(u16(path)))); failed(hr) {
				return hresultError("IShellLinkW::SetPath", hr)
			}
			// The jump list shows only tasks with arguments, even empty ones.
			if hr := comCall(sl, slSetArguments, uintptr(unsafe.Pointer(u16(commandLine(t.Args))))); failed(hr) {
				return hresultError("IShellLinkW::SetArguments", hr)
			}
			if t.Description != "" {
				if hr := comCall(sl, slSetDescription, uintptr(unsafe.Pointer(u16(t.Description)))); failed(hr) {
					return hresultError("IShellLinkW::SetDescription", hr)
				}
			}
			if t.WorkingDir != "" {
				if hr := comCall(sl, slSetWorkingDirectory, uintptr(unsafe.Pointer(u16(t.WorkingDir)))); failed(hr) {
					return hresultError("IShellLinkW::SetWorkingDirectory", hr)
				}
			}
			if t.IconPath != "" {
				if hr := comCall(sl, slSetIconLocation, uintptr(unsafe.Pointer(u16(t.IconPath))), uintptr(t.IconIndex)); failed(hr) {
					return hresultError("IShellLinkW::SetIconLocation", hr)
				}
			}
			if err := setTitle(sl, t.Title); err != nil {
				return err
			}
			if hr := comCall(oc, ocAddObject, sl); failed(hr) {
				return hresultError("IObjectCollection::AddObject", hr)
			}
			return nil
		}()
		release(sl) // the collection holds its own reference
		if err != nil {
			return err
		}
	}
	return nil
}

// commandLine joins args into a command line that CommandLineToArgvW and
// Go's os.Args split back into them, as os/exec does.
func commandLine(args []string) string {
	quoted := make([]string, len(args))
	for i, a := range args {
		quoted[i] = syscall.EscapeArg(a)
	}
	return strings.Join(quoted, " ")
}

// setTitle sets the System.Title of a shell link.
func setTitle(sl uintptr, title string) error {
	ps := queryInterface(sl, &iidIPropertyStore)
	if ps == 0 {
		return errors.New("mygo: the shell link has no property store")
	}
	defer release(ps)
	// SetValue copies the string.
	s := utf16z(title)
	v := variant{VT: vtLPWSTR, Val: uint64(uintptr(unsafe.Pointer(&s[0])))}
	hr := comCall(ps, psSetValue, uintptr(unsafe.Pointer(&pkeyTitle)), uintptr(unsafe.Pointer(&v)))
	runtime.KeepAlive(s)
	if failed(hr) {
		return hresultError("IPropertyStore::SetValue", hr)
	}
	if hr := comCall(ps, psCommit); failed(hr) {
		return hresultError("IPropertyStore::Commit", hr)
	}
	return nil
}

// coCreate creates an in-process object of class clsid and returns its iid
// interface.
func coCreate(clsid, iid GUID) (uintptr, error) {
	var obj uintptr
	if hr, _, _ := procCoCreateInstance.Call(uintptr(unsafe.Pointer(&clsid)), 0,
		clsctxInprocServer, uintptr(unsafe.Pointer(&iid)), uintptr(unsafe.Pointer(&obj))); failed(hr) {
		return 0, hresultError("CoCreateInstance", hr)
	}
	return obj, nil
}
