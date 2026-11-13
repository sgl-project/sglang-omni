//go:build darwin

package registry

import "github.com/hehehai/actnowkit/internal/darwin"

type defaultSystem struct{}

func (defaultSystem) AppPaths(bundleID string) []string {
	paths, _ := darwin.AppURLs(bundleID)
	return paths
}

func (defaultSystem) Running(bundleID string) []Process {
	apps, _ := darwin.Running(bundleID)
	out := make([]Process, 0, len(apps))
	for _, a := range apps {
		out = append(out, Process{PID: a.PID, Path: a.Path})
	}
	return out
}

func (defaultSystem) BundleID(appPath string) (string, error) { return BundleIDAt(appPath) }

func teamID() string { return darwin.TeamID() }
