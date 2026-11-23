//go:build linux && (amd64 || arm64)

package linux

import (
	"fmt"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

var (
	appsOnce sync.Once

	gAppInfoGetAll                func() ptr
	gAppInfoShouldShow            func(info ptr) bool
	gAppInfoGetDisplayName        func(info ptr) ptr
	gAppInfoLaunch                func(info, files, ctx ptr, gerr *ptr) bool
	gDesktopAppInfoGetFilename    func(info ptr) ptr
	gdkDisplayGetAppLaunchContext func(d ptr) ptr
)

func loadApps() {
	appsOnce.Do(func() {
		loadIcons() // g_desktop_app_info_new_from_filename
		mustBind(libGIO, &gAppInfoGetAll, "g_app_info_get_all")
		mustBind(libGIO, &gAppInfoShouldShow, "g_app_info_should_show")
		mustBind(libGIO, &gAppInfoGetDisplayName, "g_app_info_get_display_name")
		mustBind(libGIO, &gAppInfoLaunch, "g_app_info_launch")
		mustBind(libGIO, &gDesktopAppInfoGetFilename, "g_desktop_app_info_get_filename")
		mustBind(libGDK, &gdkDisplayGetAppLaunchContext, "gdk_display_get_app_launch_context")
	})
}

// Applications lists the applications of the desktop entries that menus
// show (g_app_info_should_show: not NoDisplay or Hidden, and meant for this
// desktop), with their names in the user's language, their desktop file
// IDs and the paths of their entries. GIO is thread-safe.
func (b *Backend) Applications() ([]platform.Application, error) {
	loadApps()
	var apps []platform.Application
	list := gAppInfoGetAll()
	for l := list; l != 0; l = field[ptr](l, 8) { // GList: data, next, prev
		info := field[ptr](l, 0)
		if gAppInfoShouldShow(info) {
			if path := goStr(gDesktopAppInfoGetFilename(info)); path != "" {
				apps = append(apps, platform.Application{
					Name: goStr(gAppInfoGetDisplayName(info)),
					ID:   goStr(gAppInfoGetID(info)),
					Path: path,
				})
			}
		}
		gObjectUnref(info)
	}
	gListFree(list)
	return apps, nil
}

// OpenApplication launches the application of a desktop entry, with the
// display's launch context, which gives it startup notification or the
// activation token that lets it take the focus.
func (shell) OpenApplication(path string) error {
	loadApps()
	info := gDesktopAppInfoNewFromFile(cs(path))
	if info == 0 {
		return fmt.Errorf("mygo: %q is not a desktop entry of an application", path)
	}
	defer gObjectUnref(info)
	ctx := gdkDisplayGetAppLaunchContext(gdkDisplayGetDefault())
	defer gObjectUnref(ctx)
	var gerr ptr
	if !gAppInfoLaunch(info, 0, ctx, &gerr) {
		return gErr(gerr)
	}
	return nil
}
