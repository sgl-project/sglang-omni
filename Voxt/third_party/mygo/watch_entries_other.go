//go:build !darwin

package mygo

// defaultWatchMaxEntries is how many files and directories a watch keeps a
// snapshot of when FileWatchOptions.MaxEntries is zero. Elsewhere a backend
// can take a handle for every entry (inotify's watch descriptors), so the
// limit keeps a watch from exhausting them.
const defaultWatchMaxEntries = 8192
