//go:build linux && (amd64 || arm64)

package linux

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

var (
	iconOnce sync.Once

	gFileQueryInfo             func(file ptr, attributes *byte, flags uint32, cancellable ptr, gerr *ptr) ptr
	gFileInfoGetIcon           func(info ptr) ptr
	gDesktopAppInfoNewFromFile func(path *byte) ptr
	gAppInfoGetIcon            func(info ptr) ptr
	gtkIconThemeGetDefault     func() ptr
	gtkIconThemeLookupByGicon  func(theme, icon ptr, size int32, flags uint32) ptr
	gtkIconInfoLoadIcon        func(info ptr, gerr *ptr) ptr
)

func loadIcons() {
	iconOnce.Do(func() {
		mustBind(libGIO, &gFileQueryInfo, "g_file_query_info")
		mustBind(libGIO, &gFileInfoGetIcon, "g_file_info_get_icon")
		mustBind(libGIO, &gDesktopAppInfoNewFromFile, "g_desktop_app_info_new_from_filename")
		mustBind(libGIO, &gAppInfoGetIcon, "g_app_info_get_icon")
		mustBind(libGTK, &gtkIconThemeGetDefault, "gtk_icon_theme_get_default")
		mustBind(libGTK, &gtkIconThemeLookupByGicon, "gtk_icon_theme_lookup_by_gicon")
		mustBind(libGTK, &gtkIconInfoLoadIcon, "gtk_icon_info_load_icon")
	})
}

// FileIcon returns the icon the file manager shows for a file or a folder,
// that of its content type (GIO's standard::icon), or the icon of the
// application of a desktop entry, from the icon theme at size by size
// pixels. Where the theme has no icon of the application, a desktop entry
// gets that of its content type, as other files do.
func (shell) FileIcon(path string, size int) ([]byte, error) {
	if _, err := os.Stat(path); err != nil {
		return nil, err
	}
	loadIcons()
	if strings.HasSuffix(path, ".desktop") {
		if app := gDesktopAppInfoNewFromFile(cs(path)); app != 0 {
			png := themeIcon(gAppInfoGetIcon(app), size)
			gObjectUnref(app)
			if png != nil {
				return png, nil
			}
		}
	}
	f := gFileNewForPath(cs(path))
	var gerr ptr
	info := gFileQueryInfo(f, cs("standard::icon"), 0, 0, &gerr)
	gObjectUnref(f)
	if info == 0 {
		return nil, gErr(gerr)
	}
	defer gObjectUnref(info)
	if png := themeIcon(gFileInfoGetIcon(info), size); png != nil {
		return png, nil
	}
	return nil, fmt.Errorf("mygo: the icon theme has no icon for %q", path)
}

// themeIcon draws a GIcon from the icon theme at size by size pixels, or
// returns nil where the theme has none of its names.
func themeIcon(icon ptr, size int) []byte {
	if icon == 0 {
		return nil
	}
	// GIO's themed icons list their generic names themselves, and GTK takes
	// GTK_ICON_LOOKUP_GENERIC_FALLBACK only with names of its own.
	const flags = 1 << 4 // GTK_ICON_LOOKUP_FORCE_SIZE
	info := gtkIconThemeLookupByGicon(gtkIconThemeGetDefault(), icon, int32(size), flags)
	if info == 0 {
		return nil
	}
	defer gObjectUnref(info)
	pix := gtkIconInfoLoadIcon(info, nil)
	if pix == 0 {
		return nil
	}
	defer gObjectUnref(pix)
	return pngFromPixbuf(pix)
}
