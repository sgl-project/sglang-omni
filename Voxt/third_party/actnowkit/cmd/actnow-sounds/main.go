// Command actnow-sounds works with a catalogue of the family's sounds (a
// directory with catalog.json, as the private sound library and the apps'
// copies have):
//
//	actnow-sounds list   -dir DIR [-group G] [-tag T] [-q TEXT]
//	actnow-sounds check  -dir DIR
//	actnow-sounds subset -from DIR -to DIR [-groups G,…] [-uses U,…] [-ids ID,…] [-max-mb N]
//	actnow-sounds play   -dir DIR [-volume 0.1] ID|system:NAME …   (macOS)
//
// subset copies the sounds an app ships (in any of the groups, uses or ids
// given; all when none is) into DIR/sounds and writes DIR/catalog.json; it
// removes other .wav files there and fails when the copy would exceed
// -max-mb.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/hehehai/actnowkit/sounds"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	var err error
	switch os.Args[1] {
	case "list":
		err = list(os.Args[2:])
	case "check":
		err = check(os.Args[2:])
	case "subset":
		err = subset(os.Args[2:])
	case "play":
		err = play(os.Args[2:])
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "actnow-sounds:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: actnow-sounds list|check|subset|play [flags] (see go doc)")
	os.Exit(2)
}

func open(dir string) (*sounds.Catalog, error) {
	if dir == "" {
		return nil, fmt.Errorf("-dir is required")
	}
	return sounds.Load(os.DirFS(dir), ".")
}

func split(s string) []string {
	var out []string
	for _, v := range strings.Split(s, ",") {
		if v = strings.TrimSpace(v); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func list(args []string) error {
	fs := flag.NewFlagSet("list", flag.ExitOnError)
	dir := fs.String("dir", "", "catalogue directory")
	group := fs.String("group", "", "group")
	tag := fs.String("tag", "", "tags, comma-separated (all must match)")
	q := fs.String("q", "", "search text")
	fs.Parse(args)
	c, err := open(*dir)
	if err != nil {
		return err
	}
	for _, s := range c.Filter(sounds.Filter{Group: *group, Tags: split(*tag), Query: *q}) {
		fmt.Printf("%-10s %-9s %-8s %5.2fs  %s\n", s.ID, s.Group, s.Name.Zh, s.Duration, strings.Join(s.Tags, " "))
	}
	return nil
}

func check(args []string) error {
	fs := flag.NewFlagSet("check", flag.ExitOnError)
	dir := fs.String("dir", "", "catalogue directory")
	fs.Parse(args)
	c, err := open(*dir)
	if err != nil {
		return err
	}
	errs := c.Check()
	for _, e := range errs {
		fmt.Fprintln(os.Stderr, e)
	}
	if len(errs) > 0 {
		return fmt.Errorf("%d problems", len(errs))
	}
	fmt.Printf("%d sounds, %d groups: ok\n", len(c.Sounds), len(c.Groups))
	return nil
}

func subset(args []string) error {
	fs := flag.NewFlagSet("subset", flag.ExitOnError)
	from := fs.String("from", "", "the library's directory")
	to := fs.String("to", "", "the app's directory")
	groups := fs.String("groups", "", "groups to ship, comma-separated")
	uses := fs.String("uses", "", "uses to ship, comma-separated")
	ids := fs.String("ids", "", "sounds to ship, comma-separated")
	maxMB := fs.Float64("max-mb", 5, "fail above this size (MB); 0 = no limit")
	fs.Parse(args)
	c, err := open(*from)
	if err != nil {
		return err
	}
	if *to == "" {
		return fmt.Errorf("-to is required")
	}
	g, u, id := split(*groups), split(*uses), split(*ids)
	for _, want := range id {
		if _, ok := c.Sound(want); !ok {
			return fmt.Errorf("no sound %q", want)
		}
	}
	all := len(g)+len(u)+len(id) == 0
	sub := c.Subset(func(s sounds.Sound) bool {
		return all || slices.Contains(g, s.Group) || slices.Contains(id, s.ID) ||
			slices.ContainsFunc(u, func(x string) bool { return slices.Contains(s.Uses, x) })
	})
	var total int64
	for _, s := range sub.Sounds {
		st, err := os.Stat(filepath.Join(*from, s.File))
		if err != nil {
			return err
		}
		total += st.Size()
	}
	if *maxMB > 0 && float64(total) > *maxMB*1e6 {
		return fmt.Errorf("%d sounds are %.1f MB, above -max-mb %.1f", len(sub.Sounds), float64(total)/1e6, *maxMB)
	}
	keep := map[string]bool{}
	for _, s := range sub.Sounds {
		dst := filepath.Join(*to, filepath.FromSlash(s.File))
		if err := copyFile(filepath.Join(*from, filepath.FromSlash(s.File)), dst); err != nil {
			return err
		}
		keep[filepath.Clean(dst)] = true
	}
	stale, _ := filepath.Glob(filepath.Join(*to, "sounds", "*.wav"))
	for _, f := range stale {
		if !keep[filepath.Clean(f)] {
			os.Remove(f)
		}
	}
	b, err := sub.JSON()
	if err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(*to, sounds.CatalogFile), b, 0o644); err != nil {
		return err
	}
	fmt.Printf("%d sounds in %d groups, %.2f MB → %s\n", len(sub.Sounds), len(sub.Groups), float64(total)/1e6, *to)
	return nil
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

func play(args []string) error {
	fs := flag.NewFlagSet("play", flag.ExitOnError)
	dir := fs.String("dir", "", "catalogue directory (optional for system sounds)")
	volume := fs.Float64("volume", 0.1, "volume, 0–1")
	fs.Parse(args)
	var c *sounds.Catalog
	if *dir != "" {
		var err error
		if c, err = open(*dir); err != nil {
			return err
		}
	}
	p := sounds.NewPlayer(c, nil)
	for _, choice := range fs.Args() {
		t := sounds.Resolve(c, choice, "", int(*volume*100))
		if t.Kind == sounds.KindNone {
			return fmt.Errorf("%s: nothing to play", choice)
		}
		start := time.Now()
		if _, err := p.Play(t); err != nil {
			return fmt.Errorf("%s: %w", choice, err)
		}
		fmt.Printf("%s started in %v\n", choice, time.Since(start).Round(time.Microsecond))
		time.Sleep(max(t.Length(c), 300*time.Millisecond) + 100*time.Millisecond)
	}
	return nil
}
