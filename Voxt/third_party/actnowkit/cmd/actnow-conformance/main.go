// Command actnow-conformance inspects and exercises the ActNow family
// directory on a Mac, for developers of family apps:
//
//	actnow-conformance check [-family DIR] [-hub DIR]   problems in the family directory
//	actnow-conformance discover [-family DIR]           the family apps found, installed, running
//	actnow-conformance claims [-family DIR] [-hub DIR]  who claims which model, and what deleting it would do
//	actnow-conformance lock -repo R [-blob B] [-mode shared|exclusive] [-as BUNDLE] [-for 30s] [-hub DIR]
//	                                                    hold a model lock, to watch another app wait or refuse
//	actnow-conformance watch [-family DIR]              print family notifications as they arrive
//	actnow-conformance pins                             the family pins table
//
// DIR defaults follow the spec: $ACTNOW_FAMILY_DIR or ~/Library/Application
// Support/ActNow; the hub as models.ResolveHub resolves it.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/hehehai/actnowkit/conformance"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/settings"
)

func main() {
	if len(os.Args) < 2 {
		usage()
	}
	cmd, args := os.Args[1], os.Args[2:]
	fs := flag.NewFlagSet(cmd, flag.ExitOnError)
	famDir := fs.String("family", family.DefaultRoot(), "family directory")
	hubDir := fs.String("hub", "", "model library ($HUB); default: as the family settings and environment say")
	var err error
	switch cmd {
	case "check":
		fs.Parse(args)
		err = check(family.At(*famDir), hub(*famDir, *hubDir))
	case "discover":
		fs.Parse(args)
		err = discover(family.At(*famDir))
	case "claims":
		fs.Parse(args)
		err = claims(family.At(*famDir), hub(*famDir, *hubDir))
	case "lock":
		repo := fs.String("repo", "", "repo id (owner/name)")
		blob := fs.String("blob", "", "blob id: lock this blob instead of the repo")
		mode := fs.String("mode", "shared", "repo lock mode: shared or exclusive")
		as := fs.String("as", "dev.actnow.conformance", "bundle id written in the owner file")
		hold := fs.Duration("for", 30*time.Second, "how long to hold it")
		fs.Parse(args)
		err = lock(hub(*famDir, *hubDir), *repo, *blob, *mode, *as, *hold)
	case "watch":
		fs.Parse(args)
		err = watch(family.At(*famDir))
	case "pins":
		pins()
	default:
		usage()
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "actnow-conformance:", err)
		os.Exit(1)
	}
}

func usage() {
	fmt.Fprintln(os.Stderr, "usage: actnow-conformance check|discover|claims|lock|watch|pins [flags]")
	os.Exit(2)
}

func hub(famDir, flagHub string) string {
	if flagHub != "" {
		return flagHub
	}
	s, _, _ := settings.Read(family.At(famDir))
	root, _ := s.Get(settings.KeyModelsRoot)
	h, _ := models.ResolveHub(true, root, "")
	return h
}

func check(d family.Dir, hub string) error {
	problems := conformance.Check(d, hub, registry.DefaultSystem)
	errs := 0
	for _, p := range problems {
		fmt.Println(p)
		if p.Level == "error" {
			errs++
		}
	}
	fmt.Printf("%s (hub %s): %d problems, %d errors\n", d.Root, hub, len(problems), errs)
	if errs > 0 {
		return fmt.Errorf("%d errors", errs)
	}
	return nil
}

func discover(d family.Dir) error {
	found, errs := registry.Discover(d, registry.DefaultSystem)
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "BUNDLE ID\tVARIANT\tREGISTERED\tINSTALLED\tRUNNING\tPATH")
	for _, f := range found {
		pids := ""
		for _, p := range f.Processes {
			pids += fmt.Sprintf(" %d", p.PID)
		}
		fmt.Fprintf(w, "%s\t%s\t%v\t%v\t%v%s\t%s\n", f.BundleID, f.Variant, f.Manifest != nil, f.Installed, f.Running, pids, f.Path)
	}
	w.Flush()
	for _, e := range errs {
		fmt.Println("warning:", e)
	}
	return nil
}

func claims(d family.Dir, hub string) error {
	found, _ := registry.Discover(d, registry.DefaultSystem)
	states := registry.States(found)
	all, errs := models.ReadAllClaims(d)
	for _, e := range errs {
		fmt.Println("warning:", e)
	}
	refs := map[models.Ref][]string{}
	for bid, c := range all {
		for _, cl := range c.Claims {
			refs[cl.Ref()] = append(refs[cl.Ref()], bid)
		}
	}
	s := &models.Store{Family: d, Hub: hub, BundleID: "dev.actnow.conformance", Apps: func() map[string]models.AppState { return states }}
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "MODEL\tSTATE\tCLAIMED BY\tFREES\tIN USE")
	for ref, by := range refs {
		dec, err := s.Decide(ref)
		if err != nil {
			fmt.Fprintf(w, "%s\t%v\n", ref, err)
			continue
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%v\n", ref, dec.Status.State, strings.Join(by, ","), human(dec.FreeBytes), dec.InUse())
	}
	return w.Flush()
}

func human(n int64) string {
	switch {
	case n >= 1e9:
		return fmt.Sprintf("%.1f GB", float64(n)/1e9)
	case n >= 1e6:
		return fmt.Sprintf("%.0f MB", float64(n)/1e6)
	}
	return fmt.Sprintf("%d B", n)
}

func lock(hub, repo, blob, mode, as string, hold time.Duration) error {
	if !models.ValidRepo(repo) {
		return fmt.Errorf("-repo %q: want owner/name", repo)
	}
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	who := models.Owner{BundleID: as, Op: "conformance"}
	var l *models.HeldLock
	var err error
	if blob != "" {
		l, err = models.LockBlob(ctx, hub, repo, blob, who)
	} else {
		m := family.Shared
		if mode == "exclusive" {
			m = family.Exclusive
		}
		l, err = models.LockRepo(ctx, hub, repo, m, who)
	}
	if err != nil {
		return err
	}
	defer l.Release()
	fmt.Printf("holding %s (%s) for %s as %s; Ctrl-C to release\n", l.Path(), l.Mode(), hold, as)
	select {
	case <-time.After(hold):
	case <-ctx.Done():
	}
	return nil
}

func watch(d family.Dir) error {
	w, err := d.Watch(family.TopicApps, family.TopicSettings, family.TopicCredentials, family.TopicModels, family.TopicRuntime, family.TopicMemory, family.TopicAccount)
	if err != nil {
		return err
	}
	defer w.Close()
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt)
	defer stop()
	fmt.Printf("watching %s notifications (%s…); Ctrl-C to stop\n", d.Root, d.NotifyName(""))
	for {
		select {
		case t := <-w.C:
			fmt.Println(time.Now().Format(time.TimeOnly), t)
		case <-ctx.Done():
			return nil
		}
	}
}

func pins() {
	w := tabwriter.NewWriter(os.Stdout, 0, 4, 2, ' ', 0)
	fmt.Fprintln(w, "REPO\tREVISION\tFAMILY\tQUANT\tROLE\tAPPS")
	for _, p := range models.Pins() {
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", p.Repo, p.Revision[:12], p.Family, p.Quant, p.Role, strings.Join(p.Apps, ","))
	}
	w.Flush()
}
