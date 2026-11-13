// Package hfhub downloads Hugging Face models into the family model
// library ($HUB, spec §7), reports what of them is on disk, verifies and
// deletes them, holding the family locks of package models and claiming
// what it downloaded.
//
// Experimental: the API may change before v1.0.
//
// Derived from Dicta's internal/hub, itself from AudioSmith v0.3.0
// (Apache-2.0); released here under Apache-2.0 by the copyright holder.
//
// # Layout
//
// Files go in the huggingface_hub cache layout, shared with the Python
// tools and swift-huggingface (spec §7.1):
//
//	$HUB/models--<owner>--<name>/blobs/<sha256 or git blob id>
//	$HUB/models--<owner>--<name>/snapshots/<commit>/<path> → ../../blobs/…
//	$HUB/models--<owner>--<name>/refs/main  (the commit "main" points at)
//
// with the spec's extensions:
//
//	snapshots/<commit>.incomplete/   a snapshot being linked, renamed in when complete
//	blobs/<blob>.incomplete          a blob being downloaded (huggingface_hub's name)
//	blobs/<blob>.incomplete.parts    which ranges of it are written (below)
//
// Only the holder of a blob's lock writes its .incomplete and
// .incomplete.parts files.
//
// # The .incomplete.parts file
//
// Files of at least two chunks download in ranges, in parallel, written at
// their offsets into blobs/<blob>.incomplete; blobs/<blob>.incomplete.parts
// records which ranges are written, replaced atomically after each one:
//
//	{"size": 10123, "chunk": 1024, "done": [true, true, false, …]}
//
// size is the blob's size in bytes; chunk the range size (the last range
// is shorter); done has one entry per range, ceil(size/chunk) of them,
// true once that range is written. A downloader resuming another's
// partial blob uses the file's chunk size, whatever its own. Without a
// parts file, an .incomplete file is a prefix written in order (as
// huggingface_hub writes it) and resumes from its length. A parts file
// that does not match the blob (other size, wrong count) makes both
// files untrustworthy: they are discarded. A parts file is written
// before any range goes into the .incomplete file, and removed after the
// blob is renamed in.
//
// # Downloads
//
// Client.Plan asks /api/models/<repo>/revision/<rev>?blobs=true for the
// commit and its files, racing the endpoints (huggingface.co,
// hf-mirror.com) and reusing the winner for a day; gated repos go to the
// official endpoint only, and only it ever sees the token. It refuses
// when the disk would keep less than Options.ReserveBytes free.
//
// Client.Download, for each repo of an entry (its auxiliary repos first):
//
//  1. takes the repo lock shared (spec §7.4), waiting for a deletion or a
//     snapshot being finished;
//  2. takes each missing blob's lock; when another app (or a Python tool)
//     holds it, reports Progress.WaitingFor and waits, then skips the
//     blob if it is complete by then. Blobs are fetched Parallel requests
//     at a time, big ones in ranges, resuming partial files, retrying
//     dropped connections, and checked (sha256 for LFS files, the git
//     blob id for others) before they are renamed in; each blob lock is
//     released once its blob is in place;
//  3. upgrades the repo lock to exclusive, checks the blobs again, links
//     the snapshot (relative links, built beside its place and renamed
//     in) and writes refs/<branch> last;
//  4. with Options.Store, claims the repo (models.Store.ClaimLocked) and,
//     once the main repo is done, the auxiliary repos; then posts the
//     family's models notification.
//
// Verify re-hashes and removes bad blobs (each holding its blob lock) so
// that any app's next Download repairs them. Delete follows the family's
// deletion rules through models.Store.Delete.
package hfhub
