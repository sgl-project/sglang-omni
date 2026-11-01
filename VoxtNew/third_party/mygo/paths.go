package mygo

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// PathName names a well known directory for Application.Path.
type PathName string

// Well known directories.
const (
	PathHome PathName = "home"
	// PathAppData is the per-user application data directory:
	// ~/Library/Application Support, $XDG_CONFIG_HOME or %AppData%.
	PathAppData PathName = "appData"
	// PathUserData is PathAppData joined with the application name. It is
	// created on first use.
	PathUserData PathName = "userData"
	// PathCache is the per-user cache directory of this application. It is
	// created on first use.
	PathCache PathName = "cache"
	// PathLogs is where the application should write logs. It is created on
	// first use.
	PathLogs PathName = "logs"
	// PathResources is the directory of the files the app ships with: the
	// contents of the project's resources directory, with those of its
	// directories for the app's platform (such as resources/darwin-arm64)
	// merged in, and the resources listed in mygo.config.ts, which
	// `mygo dev` and `mygo build` copy there. It is Contents/Resources in a
	// macOS app bundle and the executable's directory elsewhere. Under
	// `go run` and `go test`, whose executables are temporary, it is the
	// resources directory in the working directory, as it is.
	PathResources PathName = "resources"
	PathTemp      PathName = "temp"
	PathExe       PathName = "exe"
	PathDesktop   PathName = "desktop"
	PathDocuments PathName = "documents"
	PathDownloads PathName = "downloads"
	PathMusic     PathName = "music"
	PathPictures  PathName = "pictures"
	PathVideos    PathName = "videos"
)

// Path returns a well known directory, for example where to store user
// data:
//
//	dir, err := mygo.App.Path(mygo.PathUserData)
func (a *Application) Path(name PathName) (string, error) {
	a.mu.Lock()
	override := a.paths[name]
	a.mu.Unlock()
	if override != "" {
		return override, nil
	}
	p, create, err := defaultPath(name)
	if err != nil {
		return "", err
	}
	if create {
		if err := os.MkdirAll(p, 0o755); err != nil {
			return "", err
		}
	}
	return p, nil
}

// SetPath overrides a directory returned by Path.
func (a *Application) SetPath(name PathName, path string) {
	a.mu.Lock()
	if a.paths == nil {
		a.paths = map[PathName]string{}
	}
	a.paths[name] = path
	a.mu.Unlock()
}

func defaultPath(name PathName) (path string, create bool, err error) {
	home, _ := os.UserHomeDir()
	switch name {
	case PathHome:
		return home, false, nil
	case PathAppData:
		dir, err := os.UserConfigDir()
		return dir, false, err
	case PathUserData:
		dir, err := os.UserConfigDir()
		return filepath.Join(dir, App.Name()), true, err
	case PathCache:
		dir, err := os.UserCacheDir()
		return filepath.Join(dir, App.Name()), true, err
	case PathLogs:
		if runtime.GOOS == "darwin" {
			return filepath.Join(home, "Library", "Logs", App.Name()), true, nil
		}
		dir, err := os.UserConfigDir()
		return filepath.Join(dir, App.Name(), "logs"), true, err
	case PathTemp:
		return os.TempDir(), false, nil
	case PathResources:
		exe, err := os.Executable()
		if err != nil {
			return "", false, err
		}
		wd, err := os.Getwd()
		return resourcesDir(exe, wd), false, err
	case PathExe:
		exe, err := os.Executable()
		return exe, false, err
	case PathDesktop, PathDocuments, PathDownloads, PathMusic, PathPictures, PathVideos:
		return userDir(home, name), false, nil
	}
	return "", false, fmt.Errorf("mygo: unknown path %q", name)
}

// resourcesDir returns the resource directory of the executable exe, given
// the working directory wd.
func resourcesDir(exe, wd string) string {
	if real, err := filepath.EvalSymlinks(exe); err == nil {
		exe = real
	}
	dir := filepath.Dir(exe)
	if filepath.Base(dir) == "MacOS" && filepath.Base(filepath.Dir(dir)) == "Contents" {
		return filepath.Join(filepath.Dir(dir), "Resources")
	}
	// go run and go test build into $WORK/b001/exe and $WORK/b001, where
	// $WORK is a temporary go-build directory.
	for d, i := dir, 0; i < 3; d, i = filepath.Dir(d), i+1 {
		if strings.HasPrefix(filepath.Base(d), "go-build") {
			return filepath.Join(wd, "resources")
		}
	}
	return dir
}

// userDir resolves the user's standard folders, honoring the XDG user dirs
// configuration on Linux.
func userDir(home string, name PathName) string {
	folder := map[PathName]string{
		PathDesktop: "Desktop", PathDocuments: "Documents", PathDownloads: "Downloads",
		PathMusic: "Music", PathPictures: "Pictures", PathVideos: "Videos",
	}[name]
	if runtime.GOOS == "darwin" && name == PathVideos {
		folder = "Movies"
	}
	if runtime.GOOS == "linux" {
		key := map[PathName]string{
			PathDesktop: "XDG_DESKTOP_DIR", PathDocuments: "XDG_DOCUMENTS_DIR", PathDownloads: "XDG_DOWNLOAD_DIR",
			PathMusic: "XDG_MUSIC_DIR", PathPictures: "XDG_PICTURES_DIR", PathVideos: "XDG_VIDEOS_DIR",
		}[name]
		config, err := os.UserConfigDir()
		if err == nil {
			if f, err := os.Open(filepath.Join(config, "user-dirs.dirs")); err == nil {
				defer f.Close()
				s := bufio.NewScanner(f)
				for s.Scan() {
					k, v, ok := strings.Cut(strings.TrimSpace(s.Text()), "=")
					if ok && k == key {
						v = strings.Trim(v, `"`)
						return strings.Replace(v, "$HOME", home, 1)
					}
				}
			}
		}
	}
	return filepath.Join(home, folder)
}
