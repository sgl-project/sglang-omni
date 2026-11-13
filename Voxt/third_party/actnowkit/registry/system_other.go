//go:build !darwin

package registry

type defaultSystem struct{}

func (defaultSystem) AppPaths(string) []string { return nil }

func (defaultSystem) Running(string) []Process { return nil }

func (defaultSystem) BundleID(appPath string) (string, error) { return BundleIDAt(appPath) }

func teamID() string { return "" }
