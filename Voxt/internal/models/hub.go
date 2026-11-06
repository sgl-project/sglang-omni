package models

// The catalogue in the family's terms: a model as hfhub's download entry
// and as a claim (family spec §7.3), and where it is on disk.

import (
	"path/filepath"

	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/hfhub"
	fammodels "github.com/hehehai/actnowkit/models"
)

// Role is the family's word for what a kind of model does (a claim's
// role).
func (k Kind) Role() string {
	switch k {
	case ASR:
		return fammodels.RoleASR
	case LLM:
		return fammodels.RoleLLM
	case VAD:
		return fammodels.RoleVAD
	}
	return fammodels.RoleOther
}

// Entry is m as hfhub downloads it, at revision rev ("": the pinned one),
// with its auxiliary repos: only the files the catalogue lists, the LFS
// ones with their SHA-256 (the Hub names the others' blobs by their git
// ids, which hfhub checks).
func (m Model) Entry(rev string) hfhub.Entry {
	if rev == "" {
		rev = m.Revision
	}
	e := hfhub.Entry{ID: m.ID, Repo: m.Repo, Revision: rev, Family: m.Family, Role: m.Kind.Role(), Files: hubFiles(m.Files)}
	for _, a := range m.Aux {
		e.Aux = append(e.Aux, hfhub.Aux{Repo: a.Repo, Revision: a.Revision, Files: hubFiles(a.Files)})
	}
	return e
}

func hubFiles(files []File) []hfhub.File {
	var out []hfhub.File
	for _, f := range files {
		hf := hfhub.File{Path: f.Path, Size: f.Size}
		if f.LFS {
			hf.SHA256 = f.SHA256
		}
		out = append(out, hf)
	}
	return out
}

// Claim is VoxtNew's claim on m at rev ("": the pinned one), made with
// origin (fammodels.OriginReused…). At the pinned commit it carries the
// catalogue's sizes (not written to the claims file), so that Check finds
// a file cut short even without .actnow/blobs.json; another commit's
// files may have other sizes.
func (m Model) Claim(rev, origin string) fammodels.Claim {
	if rev == "" {
		rev = m.Revision
	}
	c := fammodels.Claim{Repo: m.Repo, Revision: rev, Model: m.ID, Family: m.Family, Role: m.Kind.Role(), Origin: origin}
	for _, f := range m.Files {
		c.Files = append(c.Files, f.Path)
		if rev == m.Revision && f.Size > 0 {
			if c.Sizes == nil {
				c.Sizes = map[string]int64{}
			}
			c.Sizes[f.Path] = f.Size
		}
	}
	return c
}

// AuxClaims are VoxtNew's claims on m's auxiliary repos, as hfhub makes
// them when it downloads m (origin as Claim's).
func (m Model) AuxClaims(origin string) []fammodels.Claim {
	var out []fammodels.Claim
	for _, a := range m.Aux {
		c := fammodels.Claim{Repo: a.Repo, Revision: a.Revision, Model: m.ID, Family: m.Family, Role: m.Kind.Role(), Aux: true, Origin: origin,
			Sizes: map[string]int64{}}
		for _, f := range a.Files {
			c.Files = append(c.Files, f.Path)
			c.Sizes[f.Path] = f.Size
		}
		out = append(out, c)
	}
	return out
}

// Ref is m's identity at rev ("": the pinned one).
func (m Model) Ref(rev string) fammodels.Ref {
	if rev == "" {
		rev = m.Revision
	}
	return fammodels.Ref{Repo: m.Repo, Revision: rev}
}

// find is the commit of m on disk under hub: the pinned one, else the
// first of Accept that is installed (as fammodels.FindInstalled, but each
// commit with its own claim: the pinned one's has the catalogue's sizes);
// "" when none is, or an auxiliary repo is not.
func find(hub string, m Model) (rev string, st fammodels.Status) {
	for _, c := range m.AuxClaims("") {
		if fammodels.Check(hub, c).State != fammodels.Installed {
			return "", fammodels.Status{}
		}
	}
	for _, r := range append([]string{m.Revision}, m.Accept...) {
		if st := fammodels.Check(hub, m.Claim(r, "")); st.State == fammodels.Installed {
			return r, st
		}
	}
	return "", fammodels.Status{}
}

// SnapshotDir is the folder a runtime loads m from when its commit rev is
// on disk under hub: $HUB/models--<owner>--<name>/snapshots/<rev>.
func SnapshotDir(hub string, m Model, rev string) string {
	return fammodels.SnapshotDir(hub, m.Ref(rev))
}

// Installed reports where m is installed under hub: its snapshot folder,
// "" when no commit it accepts is complete. Disk only; for tools and
// tests (the app asks its Manager, which keeps it in memory).
func Installed(hub string, m Model) string {
	rev, _ := find(hub, m)
	if rev == "" {
		return ""
	}
	return SnapshotDir(hub, m, rev)
}

// HubIn is VoxtNew's own model library for the family directory d, used
// when the family's settings name none: "SGLang Omni/models" beside d,
// i.e. ~/Library/Application Support/SGLang Omni/models. The local models
// are SGLang Omni's (its MLX runtime serves them), and the folder says so.
// Up to 2.0.0 it was the family's default library, ActNow/models/hub,
// which Manager.Import takes the models over from.
func HubIn(d family.Dir) string {
	return filepath.Join(filepath.Dir(d.Root), "SGLang Omni", "models")
}

// Hub is VoxtNew's default model library (HubIn of the family's default
// directory), for tools and tests; the app resolves it with the family's
// settings first (internal/actnow).
func Hub() string { return HubIn(family.Default()) }
