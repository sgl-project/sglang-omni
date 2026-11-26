// Package cues is Voxt's own cue sounds: soft two-note pairs (the start
// rises, the stop falls), original synthesized audio that is part of this
// repository, so every build has them. Load puts them in one catalogue
// with the ActNow family's sounds an official build ships beside them
// (scripts/sync-sounds.sh).
package cues

import (
	"embed"
	"encoding/json"
	"io/fs"
	"path"
	"slices"
	"strings"

	"github.com/hehehai/actnowkit/sounds"
)

//go:embed catalog.json sounds/*.wav
var files embed.FS

// The folders of the merged catalogue's files.
const (
	ownDir    = "voxt"
	familyDir = "family"
)

// Load is the catalogue of Voxt's cues and of the family's sounds in
// family/dir (nil, or without a catalogue there: Voxt's cues alone). A
// family sound with the id of one of Voxt's is left out.
func Load(family fs.FS, dir string) (*sounds.Catalog, error) {
	b, err := files.ReadFile(sounds.CatalogFile)
	if err != nil {
		return nil, err
	}
	var cat sounds.Catalog
	if err := json.Unmarshal(b, &cat); err != nil {
		return nil, err
	}
	for i := range cat.Sounds {
		cat.Sounds[i].File = path.Join(ownDir, cat.Sounds[i].File)
	}
	mounted := mounts{ownDir: files}
	if fam, err := sounds.Load(family, dir); err == nil {
		if sub, err := fs.Sub(family, path.Clean(strings.TrimPrefix(dir, "/"))); err == nil {
			mounted[familyDir] = sub
			add(&cat, fam)
		}
	}
	if b, err = json.Marshal(&cat); err != nil {
		return nil, err
	}
	return sounds.Parse(b, mounted, ".")
}

// add puts fam's groups, tags, uses and sounds after cat's own.
func add(cat, fam *sounds.Catalog) {
	for _, g := range fam.Groups {
		if !slices.ContainsFunc(cat.Groups, func(o sounds.Group) bool { return o.ID == g.ID }) {
			cat.Groups = append(cat.Groups, g)
		}
	}
	for _, t := range fam.Tags {
		if !slices.ContainsFunc(cat.Tags, func(o sounds.Tag) bool { return o.ID == t.ID }) {
			cat.Tags = append(cat.Tags, t)
		}
	}
	for _, u := range fam.Uses {
		if !slices.ContainsFunc(cat.Uses, func(o sounds.Use) bool { return o.ID == u.ID }) {
			cat.Uses = append(cat.Uses, u)
		}
	}
	for _, s := range fam.Sounds {
		if !slices.ContainsFunc(cat.Sounds, func(o sounds.Sound) bool { return o.ID == s.ID }) {
			s.File = path.Join(familyDir, s.File)
			cat.Sounds = append(cat.Sounds, s)
		}
	}
	if fam.Notice != "" {
		cat.Notice = fam.Notice + "\n" + cat.Notice
	}
}

// mounts is a file system of folders, each another file system's root.
type mounts map[string]fs.FS

func (m mounts) Open(name string) (fs.File, error) {
	head, rest, _ := strings.Cut(name, "/")
	if sub, ok := m[head]; ok && rest != "" && fs.ValidPath(name) {
		return sub.Open(rest)
	}
	return nil, &fs.PathError{Op: "open", Path: name, Err: fs.ErrNotExist}
}
