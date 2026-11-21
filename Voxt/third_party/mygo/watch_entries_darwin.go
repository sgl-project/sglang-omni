//go:build darwin

package mygo

// defaultWatchMaxEntries is how many files and directories a watch keeps a
// snapshot of when FileWatchOptions.MaxEntries is zero. macOS watches a
// whole tree with one FSEvents stream, so the limit is a bound on the
// memory of the snapshot, not on the descriptors a backend holds, and it
// can be high.
const defaultWatchMaxEntries = 65536
