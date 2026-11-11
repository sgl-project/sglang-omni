package main

import (
	"bufio"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"maps"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

func runDev(args []string) error {
	flags := newFlags("dev", "[flags] [dir]", `Develops the app with live reload. It writes the TypeScript client of a
frontend, runs devCommand from mygo.json (such as a Vite dev server),
waits for devUrl to answer, then builds a development app, which loads
devUrl in place of its built frontend, and launches it. Without devUrl,
the app serves frontendDist from disk. The development app is "<name> Dev" with the
identifier "<identifier>.dev", in .mygo/dev: a real bundle on macOS, and
on Windows an executable with the icon, manifest and version information
that mygo build embeds.

Changes to the Go code, mygo.json or mygo.config.ts, the icon, the
resources or the catalogs of messages rebuild the app, regenerate the
TypeScript client and the accessors of the catalogs, and relaunch it. A build that fails to compile keeps the previous one running; the next
change tries again. Frontend changes are left to the dev server. Quitting the app ends mygo dev.

On a terminal, type r and Enter to rebuild and restart the app, c to clear
the console, q to quit, h for help.

With -web, it develops an app of native UI in browsers instead: it builds
the app for them (GOOS=js GOARCH=wasm), serves it at http://localhost:<port>,
opens it, and rebuilds it as its code changes, which reloads the page; the
page shows why a build failed.

With -lang, the app runs in that language, a BCP 47 tag such as de or
zh-Hant, or a pseudo locale: en-XA (accented and longer text) or ar-XB
(right to left), which test layouts without translations. It does not
keep it: the app's own setting is left as it is.`)
	skipDevCommand := flags.Bool("skip-dev-command", false, "do not run devCommand")
	sign := flags.String("sign", "-", "macOS signing identity for the development app")
	web := flags.Bool("web", false, "develop the app in browsers")
	port := flags.Int("port", 5174, "with -web, the port to serve the app at, or the next free one")
	openPage := flags.Bool("open", true, "with -web, open the app in the browser")
	lang := flags.String("lang", "", "run the app in this language (as de, or the pseudo locales en-XA and ar-XB), without keeping it")
	if err := flags.Parse(args); err != nil {
		return err
	}
	c, err := loadConfig(dirArg(flags.Args()))
	if err != nil {
		return err
	}
	locale, err := devLocale(*lang)
	if err != nil {
		return err
	}
	if *web {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		con.startBanner("dev")
		return runWebDev(ctx, c, *port, *openPage, locale)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	s := &devSession{root: c.root, sign: *sign, name: devConfig(c).Name, env: []string{"MYGO_ENV=development"}}
	if locale != "" {
		// The app reads it as it starts, over the language it keeps.
		s.env = append(s.env, "MYGO_LOCALE="+locale)
	}
	var server atomic.Pointer[exec.Cmd] // the dev command
	go func() {
		// Ctrl+C again does not wait for the app to quit.
		<-ctx.Done()
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		s.killLive()
		kill(server.Load())
		con.interrupted()
		os.Exit(130)
	}()

	con.startBanner("dev")
	// The accessors of the catalogs and the TypeScript client come first:
	// the app and the frontend import them.
	if err := generateI18n(c); err != nil {
		return err
	}
	if err := writeClient(ctx, c, false); err != nil {
		if ctx.Err() != nil {
			return nil
		}
		return err
	}
	var starting *task
	var exited chan error // the dev command's
	var quitting atomic.Bool
	if c.DevCommand != "" && !*skipDevCommand {
		starting = con.start("Starting the dev server")
		cmd := shellCommand(c.root, c.DevCommand)
		out, errOut := labeledOutput(cmd, "web")
		if err := cmd.Start(); err != nil {
			starting.fail()
			return err
		}
		server.Store(cmd)
		// Stopping the command makes script runners complain; that is
		// noise.
		context.AfterFunc(ctx, func() {
			quitting.Store(true)
			out.close()
			errOut.close()
		})
		defer func() {
			quitting.Store(true)
			out.close()
			errOut.close()
			terminate(cmd)
		}()
		exited = make(chan error, 1)
		go func() {
			err := waited(cmd.Wait())
			out.flush()
			errOut.flush()
			exited <- err
		}()
	}
	switch {
	case c.DevURL != "":
		if starting == nil {
			starting = con.start("Waiting for the dev server")
		}
		starting.set(c.DevURL)
		wait, cancel := context.WithTimeout(ctx, 30*time.Second)
		err := waitForURL(wait, c.DevURL, exited)
		cancel()
		if errors.Is(err, context.Canceled) {
			starting.stop()
			return nil
		}
		if err := starting.end(err, "Dev server ready at "+cyan(c.DevURL)); err != nil {
			return err
		}
		s.env = append(s.env, "MYGO_DEV_URL="+c.DevURL)
	case starting != nil:
		starting.done("Started " + c.DevCommand)
	}
	if exited != nil {
		// From now on, the dev server exiting is news, unless Ctrl+C,
		// which it gets too, stopped it.
		go func() {
			err := <-exited
			select {
			case <-ctx.Done():
				return
			case <-time.After(250 * time.Millisecond):
			}
			if !quitting.Load() {
				msg := "The dev server exited"
				if err != nil {
					msg += " (" + err.Error() + ")"
				}
				warnf("%s", msg)
			}
		}()
	}
	if c.DevURL == "" && c.FrontendDist != "" {
		s.env = append(s.env, "MYGO_FRONTEND_DIST="+c.path(c.FrontendDist))
	}
	return s.run(ctx, c)
}

// devSession builds and runs development builds of an app.
type devSession struct {
	root string
	sign string
	name string   // of the development app, for messages
	env  []string // for the app

	// Used by one build at a time.
	i18nWarnings string // the warnings of the catalogs last shown
	iconKey      string
	icns         []byte
	mainFor      string // the Main that mainDir is the package directory of
	mainDir      string

	// Used by the run loop only.
	app *devProcess // the running build
	sum [32]byte    // fingerprint of the running build

	// live is app, which a build stops before it starts.
	liveMu sync.Mutex
	live   *devProcess
}

// setApp makes p the running build.
func (s *devSession) setApp(p *devProcess) {
	s.app = p
	s.liveMu.Lock()
	s.live = p
	s.liveMu.Unlock()
}

// stopLive stops the running build for the one about to start: builds
// never overlap, so the single instance lock, the web view's profile and
// any other state of the app's are free.
func (s *devSession) stopLive() {
	s.liveMu.Lock()
	p := s.live
	s.liveMu.Unlock()
	if p != nil {
		p.replaced.Store(true)
		p.stop()
	}
}

var errUnchanged = errors.New("unchanged")

// devRelaunchCode is the exit code of a build that mygo.App.Relaunch asks
// to start again (see login.go in package mygo).
const devRelaunchCode = 75

// run launches the app and rebuilds it on changes until the app quits or
// ctx is done.
func (s *devSession) run(ctx context.Context, c *Config) error {
	w := &watcher{}
	if in, err := listBuildInputs(c); err == nil {
		w.set(in)
	} else {
		warnf("%v", err)
	}
	changes := w.watch(ctx, 250*time.Millisecond)
	keys := readKeys()

	type result struct {
		p      *devProcess
		sum    [32]byte
		err    error
		inputs *buildInputs
	}
	results := make(chan result, 1)
	builds, cancel := context.WithCancel(ctx)
	building, pending, restart := false, false, false
	launched := false // a build has started
	importChecked := false
	var changed []string // the inputs that changed since the last build started
	start := func() {
		running := s.sum
		title, verb := "Rebuilding ", "Rebuilt "
		switch {
		case !launched:
			title, verb = "Building ", "Built "
		case restart:
			running, title, verb = [32]byte{}, "Restarting ", "Restarted "
		}
		// Why it builds: the step's title is plain, its line dims it.
		why := ""
		if len(changed) > 0 {
			why = "(" + changeSummary(s.root, changed) + ")"
		}
		building, pending, restart, changed = true, false, false, nil
		t := con.start(strings.TrimSpace(title + s.name + " " + why))
		go func() {
			p, sum, err := s.buildAndLaunch(builds, t, running)
			s.report(t, verb, why, err)
			// What the build reads may have changed, e.g. a new import.
			var inputs *buildInputs
			if c, cerr := loadConfig(s.root); cerr == nil {
				inputs, _ = listBuildInputs(c)
			}
			results <- result{p, sum, err, inputs}
		}()
	}
	var stopping sync.WaitGroup
	defer func() {
		cancel()
		var t *task
		if building || s.app != nil {
			t = con.start("Stopping " + s.name)
			if ctx.Err() != nil {
				t.set("Ctrl+C again to force")
			}
		}
		if building {
			if r := <-results; r.p != nil {
				r.p.stop()
			}
		}
		if s.app != nil {
			s.app.stop()
		}
		stopping.Wait()
		t.stop()
	}()

	start()
	for {
		var exited <-chan struct{}
		if s.app != nil {
			exited = s.app.done
		}
		select {
		case <-ctx.Done():
			return nil
		case <-exited:
			if s.app.replaced.Load() {
				s.setApp(nil) // stopped for the next build
				continue
			}
			err, exe := s.app.err, s.app.exe
			s.setApp(nil)
			var exit *exec.ExitError
			if errors.As(err, &exit) && exit.ExitCode() == devRelaunchCode {
				// mygo.App.Relaunch: start the same build again.
				p, err := s.launch(exe)
				if err != nil {
					con.println("  " + red(con.sym.fail) + " Relaunching " + s.name)
					con.details(err)
					s.hint("Save a change to start it again.")
					s.sum = [32]byte{}
					continue
				}
				logf("Relaunched %s", s.name)
				s.setApp(p)
				continue
			}
			s.sum = [32]byte{}
			if err == nil {
				logf("%s quit", s.name)
				return nil
			}
			con.println("  " + red(con.sym.fail) + " " + s.name + " exited (" + err.Error() + ")")
			s.hint("Save a change to start it again.")
		case paths := <-changes:
			for _, p := range paths {
				if !slices.Contains(changed, p) {
					changed = append(changed, p)
				}
			}
			if building {
				pending = true
			} else {
				start()
			}
		case key, ok := <-keys:
			switch {
			case !ok:
				keys = nil
			case key == "r":
				restart = true
				if building {
					pending = true
				} else {
					start()
				}
			case key == "c":
				con.clearScreen()
			case key == "q":
				return nil
			case key == "h":
				con.println(devHelp())
			}
		case r := <-results:
			building = false
			if r.inputs != nil {
				w.set(r.inputs)
				if !importChecked && r.err == nil {
					importChecked = true
					checkCatalogImported(c, r.inputs.sourceDirs)
				}
			}
			if r.err == nil {
				prev := s.app
				s.setApp(r.p)
				s.sum = r.sum
				if prev != nil {
					stopping.Add(1)
					go func() {
						defer stopping.Done()
						prev.stop()
					}()
				}
				if !launched {
					launched = true
					msg := "\n  " + cyan(con.sym.arrow) + " Watching for changes\n"
					if keys != nil {
						// A plain space after the arrow: Windows Terminal
						// colors the part of a glyph wider than its cell
						// as the next cell.
						msg += "  " + cyan(con.sym.arrow) + " " + dim("press ") + bold("h + enter") + dim(" to show help") + "\n"
					}
					con.println(msg)
				}
			}
			if pending {
				start()
			}
		}
	}
}

// report ends the step t of a build with how it went, and why it ran.
func (s *devSession) report(t *task, verb, why string, err error) {
	if why != "" {
		why = " " + dim(why)
	}
	switch {
	case err == nil:
		t.done(verb + s.name + why)
	case errors.Is(err, errUnchanged):
		t.done(dim("No change to "+s.name) + why)
	case errors.Is(err, context.Canceled):
		t.stop()
	default:
		t.fail()
		con.details(err)
		if s.liveRunning() {
			s.hint("The previous build keeps running: save a fix to try again.")
		} else {
			s.hint("Save a fix to try again.")
		}
	}
}

// hint tells what to do after a build failed, or the app crashed.
func (s *devSession) hint(msg string) {
	con.println("  " + dim(msg))
}

// killLive kills the running build at once.
func (s *devSession) killLive() {
	s.liveMu.Lock()
	p := s.live
	s.liveMu.Unlock()
	if p != nil {
		kill(p.cmd)
	}
}

// liveRunning reports whether a build runs that no next build stopped.
func (s *devSession) liveRunning() bool {
	s.liveMu.Lock()
	p := s.live
	s.liveMu.Unlock()
	if p == nil || p.replaced.Load() {
		return false
	}
	select {
	case <-p.done:
		return false
	default:
		return true
	}
}

// readKeys returns the lines typed on the terminal, which are shortcuts,
// as Vite has: a key, then Enter. It returns nil when standard input is
// not a terminal.
func readKeys() <-chan string {
	if !con.live || !isTerminal(os.Stdin) {
		return nil
	}
	ignoreTTIN()
	keys := make(chan string)
	go func() {
		defer close(keys)
		sc := bufio.NewScanner(os.Stdin)
		for sc.Scan() {
			keys <- strings.ToLower(strings.TrimSpace(sc.Text()))
		}
	}()
	return keys
}

func devHelp() string {
	var b strings.Builder
	b.WriteString("\n  " + bold("Shortcuts") + "\n")
	for _, k := range [][2]string{
		{"r", "rebuild and restart the app"},
		{"c", "clear the console"},
		{"q", "quit"},
	} {
		b.WriteString("  " + dim("press ") + bold(k[0]+" + enter") + dim(" to "+k[1]) + "\n")
	}
	return b.String()
}

// devConfig configures the development app: "<Name> Dev" with the
// identifier "<identifier>.dev", so that its data, preferences and single
// instance lock stay apart from the production app's.
func devConfig(c *Config) *Config {
	d := *c
	d.Name += " Dev"
	d.Identifier += ".dev"
	return &d
}

// buildAndLaunch builds the app and launches it once it differs from the
// running build, whose fingerprint is running, showing its progress on t.
// It returns the new process.
func (s *devSession) buildAndLaunch(ctx context.Context, t *task, running [32]byte) (p *devProcess, sum [32]byte, err error) {
	c, err := loadConfig(s.root)
	if err != nil {
		return nil, sum, err
	}
	dir := filepath.Join(s.root, ".mygo", "dev", runtime.GOOS+"-"+runtime.GOARCH)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, sum, err
	}
	stage, err := os.MkdirTemp(dir, ".staging-")
	if err != nil {
		return nil, sum, err
	}
	defer os.RemoveAll(stage)

	dc := devConfig(c)
	name := dc.executableName()
	switch runtime.GOOS {
	case "darwin":
	case "windows":
		name += ".exe"
	default:
		name = slugify(dc.Name)
	}
	bin := filepath.Join(stage, name)
	cleanup := func() {}
	if runtime.GOOS == "windows" {
		if cleanup, err = s.windowsResources(c, dc); err != nil {
			return nil, sum, err
		}
	}
	if err := s.updateI18n(c); err != nil {
		cleanup()
		return nil, sum, err
	}
	err = goBuild(ctx, c, t, bin, nil, "-ldflags", strings.TrimSpace(packageFlags(dc)))
	cleanup()
	if err != nil {
		return nil, sum, err
	}

	res, err := dc.appResources(runtime.GOOS, runtime.GOARCH)
	if err != nil {
		return nil, sum, err
	}

	// Fingerprint what the app is made of, to skip relaunching when a
	// change did not affect it.
	h := sha256.New()
	f, err := os.Open(bin)
	if err != nil {
		return nil, sum, err
	}
	_, err = io.Copy(h, f)
	f.Close()
	if err != nil {
		return nil, sum, err
	}
	if err := hashResources(h, res); err != nil {
		return nil, sum, err
	}
	var icns []byte
	if runtime.GOOS == "darwin" {
		if icns, err = s.icon(c); err != nil {
			return nil, sum, err
		}
		iconFile := ""
		if icns != nil {
			iconFile = bundleIcon
		}
		h.Write(icns)
		h.Write(infoPlist(dc, name, iconFile, ""))
		if err := hashEntitlements(h, dc); err != nil {
			return nil, sum, err
		}
	}
	copy(sum[:], h.Sum(nil))
	if sum == running {
		return nil, sum, errUnchanged
	}

	t.set("generating the TypeScript client")
	changed, err := generateBindings(c, bin)
	if err != nil {
		return nil, sum, err
	}
	if changed {
		logf("Wrote %s", cyan(relPathTo(c.root, c.path(c.Bindings))))
	}
	exe := filepath.Join(dir, name)
	if runtime.GOOS == "darwin" {
		t.set("signing")
		app, err := writeBundle(dc, stage, bin, icns, nil, res)
		if err != nil {
			return nil, sum, err
		}
		if err := codesign(dc, app, s.sign, false); err != nil {
			return nil, sum, err
		}
		final := filepath.Join(dir, filepath.Base(app))
		if err := replacePath(app, final); err != nil {
			return nil, sum, err
		}
		exe = bundleExecutable(final)
	} else if err := placeBuild(stage, dir, res); err != nil {
		return nil, sum, err
	}
	s.stopLive()
	if p, err = s.launch(exe); err != nil {
		return nil, sum, err
	}
	return p, sum, nil
}

// updateI18n writes the accessors of the catalogs where they changed, and
// shows their warnings when they changed too.
func (s *devSession) updateI18n(c *Config) error {
	out, err := writeI18n(c)
	if err != nil {
		return err
	}
	for _, f := range out.written {
		logf("Wrote %s", cyan(f))
	}
	var b strings.Builder
	for _, d := range out.warnings {
		b.WriteString(d.String() + "\n")
	}
	if w := b.String(); w != s.i18nWarnings {
		s.i18nWarnings = w
		for _, d := range out.warnings {
			warnf("%s", d)
		}
	}
	return nil
}

// hashEntitlements writes the entitlements that sign the app and code among
// its resources to w, to tell whether they changed.
func hashEntitlements(w io.Writer, c *Config) error {
	files := map[string]string{"": c.MacOS.Entitlements} // "": the app
	maps.Copy(files, c.MacOS.HelperEntitlements)
	for _, name := range slices.Sorted(maps.Keys(files)) {
		if files[name] == "" {
			continue
		}
		b, err := os.ReadFile(c.path(files[name]))
		if err != nil {
			return err
		}
		fmt.Fprintf(w, "%s\x00%d\x00%s", name, len(b), b)
	}
	return nil
}

// placeBuild moves the executable in stage and the resources res next to
// it into dir, and removes what an earlier build left there and this one
// does not have.
func placeBuild(stage, dir string, res []resource) error {
	if err := copyResources(res, stage); err != nil {
		return err
	}
	entries, err := os.ReadDir(stage)
	if err != nil {
		return err
	}
	placed := map[string]bool{}
	for _, e := range entries {
		if err := replacePath(filepath.Join(stage, e.Name()), filepath.Join(dir, e.Name())); err != nil {
			return err
		}
		placed[e.Name()] = true
	}
	entries, err = os.ReadDir(dir)
	if err != nil {
		return err
	}
	for _, e := range entries {
		if !placed[e.Name()] && !hiddenName(e.Name()) {
			_ = os.RemoveAll(filepath.Join(dir, e.Name()))
		}
	}
	return nil
}

// windowsResources puts the resources of the development app (icon,
// manifest, version) in the main package for one build, as mygo build does:
// its windows take the executable's icon. It returns the function that
// removes them.
func (s *devSession) windowsResources(c, dc *Config) (cleanup func(), err error) {
	if c.Main != s.mainFor || s.mainDir == "" {
		dir, err := packageDir(c)
		if err != nil {
			return nil, err
		}
		s.mainFor, s.mainDir = c.Main, dir
	}
	return windowsResources(dc, s.mainDir, runtime.GOARCH)
}

// icon returns the app icon as .icns, rendering it only when it changed.
func (s *devSession) icon(c *Config) ([]byte, error) {
	if c.Icon == "" {
		return nil, nil
	}
	info, err := os.Stat(c.path(c.Icon))
	if err != nil {
		return nil, err
	}
	key := fmt.Sprint(c.path(c.Icon), info.Size(), info.ModTime().UnixNano())
	if key != s.iconKey {
		icns, err := appIcon(c)
		if err != nil {
			return nil, err
		}
		s.iconKey, s.icns = key, icns
	}
	return s.icns, nil
}

// launch starts a build. MYGO_DEV=1 tells it that mygo dev runs it, which
// mygo.App.Relaunch asks to start it again.
func (s *devSession) launch(exe string) (*devProcess, error) {
	stdout, outDone, err := pipe(con.output("app", os.Stdout))
	if err != nil {
		return nil, err
	}
	stderr, errDone, err := pipe(con.output("app", os.Stderr))
	if err != nil {
		stdout.Close()
		return nil, err
	}
	cmd := exec.Command(exe)
	cmd.Dir = s.root
	cmd.Stdout, cmd.Stderr = stdout, stderr
	cmd.Env = append(append(os.Environ(), s.env...), "MYGO_DEV=1")
	setProcessGroup(cmd)
	err = cmd.Start()
	stdout.Close()
	stderr.Close()
	if err != nil {
		return nil, err
	}
	p := &devProcess{exe: exe, cmd: cmd, done: make(chan struct{})}
	go func() {
		p.err = cmd.Wait()
		// What the app printed last, as a panic, comes before what mygo
		// says of its exit, unless processes it started hold its output.
		timeout := time.After(250 * time.Millisecond)
		for _, copied := range []<-chan struct{}{outDone, errDone} {
			select {
			case <-copied:
			case <-timeout:
			}
		}
		close(p.done)
	}()
	return p, nil
}

// devProcess is a running development build.
type devProcess struct {
	exe  string
	cmd  *exec.Cmd
	done chan struct{} // closed when the process exited
	err  error         // how it exited, set before done is closed
	// replaced is set when the build is stopped for the next one.
	replaced atomic.Bool
}

// stopGrace is how long a stopped build may take to quit.
var stopGrace = 3 * time.Second

// stop asks the app to quit, like the Quit menu item, and kills it when it
// has not exited after stopGrace. Only the app is asked: it ends the
// processes it started, such as its web view's, which could lose what they
// had not written yet if asked along with it. Those it leaves are killed.
func (p *devProcess) stop() {
	select {
	case <-p.done:
		return
	default:
	}
	interrupt(p.cmd)
	select {
	case <-p.done:
	case <-time.After(stopGrace):
	}
	kill(p.cmd)
	<-p.done
}
