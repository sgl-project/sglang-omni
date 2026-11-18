//go:build !linux

package mygo

import "os"

// fileWatchSubdirs returns the names of the directories in d. Reading a
// directory opened in a root stats every entry; only Linux, where a
// recursive watch enrolls every directory, reads the types alone.
func fileWatchSubdirs(d *os.Root) ([]string, error) {
	f, err := d.Open(".")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	entries, err := f.ReadDir(-1)
	var names []string
	for _, e := range entries {
		if e.IsDir() {
			names = append(names, e.Name())
		}
	}
	return names, err
}
