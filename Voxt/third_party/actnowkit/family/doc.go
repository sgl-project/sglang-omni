// Package family is the base of the ActNow family protocol (spec §4–§6.1):
// where the shared directory is, the JSON file conventions every shared
// file follows (schema versions, unknown fields kept, atomic replacement,
// corrupt files moved aside), flock(2) locks, Darwin change notifications,
// and the table of known family apps.
//
// Every other actnowkit package builds on it; apps use it directly to
// create the directory at start-up and to watch for changes:
//
//	d := family.Default()
//	if _, err := d.Ensure("dev.hehehai.dicta"); err != nil { … }
//	w, _ := d.Watch(family.TopicSettings, family.TopicModels)
//	go func() { for topic := range w.C { reload(topic) } }()
//
// The package is pure Go: Darwin notifications go through libSystem with
// purego (no cgo); on other systems they are delivered inside the process
// only, which is what tests need.
package family
