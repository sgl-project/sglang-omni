//go:build linux && (amd64 || arm64)

package linux

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

type dialogs struct{ b *Backend }

const gtkResponseAccept = -3

func parentWindow(pw platform.Window) ptr {
	if w, ok := pw.(*window); ok && w != nil && !w.closed {
		return w.win
	}
	return 0
}

// slistStrings reads and frees a GSList of GLib allocated strings.
func slistStrings(list ptr) []string {
	var out []string
	for node := list; node != 0; node = field[ptr](node, 8) {
		out = append(out, takeStr(field[ptr](node, 0)))
	}
	if list != 0 {
		gSlistFree(list)
	}
	return out
}

func setupChooser(fc ptr, defaultPath string, filters []platform.FileFilter, hidden, save bool) {
	gtkFileChooserSetShowHidden(fc, hidden)
	if defaultPath != "" {
		if st, err := os.Stat(defaultPath); err == nil && st.IsDir() {
			gtkFileChooserSetCurrentFolder(fc, cs(defaultPath))
		} else {
			gtkFileChooserSetCurrentFolder(fc, cs(filepath.Dir(defaultPath)))
			if save {
				gtkFileChooserSetCurrentName(fc, cs(filepath.Base(defaultPath)))
			}
		}
	}
	for _, f := range filters {
		filter := gtkFileFilterNew()
		name := f.Name
		if name == "" {
			name = strings.Join(f.Extensions, ", ")
		}
		gtkFileFilterSetName(filter, cs(name))
		for _, e := range f.Extensions {
			e = strings.TrimPrefix(e, ".")
			if e == "*" {
				gtkFileFilterAddPattern(filter, cs("*"))
				continue
			}
			gtkFileFilterAddPattern(filter, cs("*."+e))
			if up := strings.ToUpper(e); up != e {
				gtkFileFilterAddPattern(filter, cs("*."+up))
			}
		}
		gtkFileChooserAddFilter(fc, filter)
	}
}

func (d dialogs) ShowOpenDialog(parent platform.Window, o *platform.OpenDialogOptions, cb func([]string, error)) {
	tx := platform.CurrentTexts()
	action := int32(0) // GTK_FILE_CHOOSER_ACTION_OPEN
	title := tx.Open
	if o.OpenDirectories {
		action, title = 2, tx.SelectFolder // SELECT_FOLDER
	}
	if o.Title != "" {
		title = o.Title
	}
	accept := "_" + tx.Open
	if o.ButtonLabel != "" {
		accept = o.ButtonLabel
	}
	fc := gtkFileChooserNativeNew(cs(title), parentWindow(parent), action, cs(accept), cs("_"+tx.Cancel))
	gtkFileChooserSetSelectMultiple(fc, o.Multiple)
	setupChooser(fc, o.DefaultPath, o.Filters, o.ShowHiddenFiles, false)
	var paths []string
	if d.b.runDialog(fc, true) == gtkResponseAccept {
		paths = slistStrings(gtkFileChooserGetFilenames(fc))
	}
	gObjectUnref(fc)
	cb(paths, nil)
}

func (d dialogs) ShowSaveDialog(parent platform.Window, o *platform.SaveDialogOptions, cb func(string, error)) {
	tx := platform.CurrentTexts()
	title := o.Title
	if title == "" {
		title = tx.Save
	}
	accept := "_" + tx.Save
	if o.ButtonLabel != "" {
		accept = o.ButtonLabel
	}
	fc := gtkFileChooserNativeNew(cs(title), parentWindow(parent), 1, cs(accept), cs("_"+tx.Cancel)) // SAVE
	gtkFileChooserSetDoOverwriteConfirm(fc, true)
	setupChooser(fc, o.DefaultPath, o.Filters, o.ShowHiddenFiles, true)
	path := ""
	if d.b.runDialog(fc, true) == gtkResponseAccept {
		path = takeStr(gtkFileChooserGetFilename(fc))
	}
	gObjectUnref(fc)
	cb(path, nil)
}

func (d dialogs) ShowMessageBox(parent platform.Window, o *platform.MessageBoxOptions, cb func(platform.MessageBoxResult, error)) {
	kind := int32(4) // GTK_MESSAGE_OTHER
	switch o.Type {
	case "info":
		kind = 0
	case "warning":
		kind = 1
	case "question":
		kind = 2
	case "error":
		kind = 3
	}
	message := o.Message
	if message == "" {
		message = o.Title
	}
	dlg := gtkMessageDialogNew(parentWindow(parent), 1|2, kind, 0, cs("%s"), cs(message)) // MODAL | DESTROY_WITH_PARENT, BUTTONS_NONE
	if o.Detail != "" {
		gtkMessageDialogFormatSecondaryText(dlg, cs("%s"), cs(o.Detail))
	}
	if o.Title != "" {
		gtkWindowSetTitle(dlg, cs(o.Title))
	}
	for i, label := range o.Buttons {
		gtkDialogAddButton(dlg, cs(label), int32(i))
	}
	gtkDialogSetDefaultResponse(dlg, int32(o.DefaultID))
	var check ptr
	if o.CheckboxLabel != "" {
		check = gtkCheckButtonNewWithLabel(cs(o.CheckboxLabel))
		gtkToggleButtonSetActive(check, o.CheckboxChecked)
		gtkContainerAdd(gtkMessageDialogGetMessageArea(dlg), check)
		gtkWidgetShow(check)
	}
	resp := d.b.runDialog(dlg, false)
	res := platform.MessageBoxResult{Response: int(resp)}
	if resp < 0 {
		// Escape or the window's close button.
		res.Response = o.CancelID
	}
	if check != 0 {
		res.CheckboxChecked = gtkToggleButtonGetActive(check)
	}
	gtkWidgetDestroy(dlg)
	cb(res, nil)
}
