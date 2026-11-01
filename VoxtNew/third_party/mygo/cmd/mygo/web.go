package main

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Browsers run apps of native UI compiled to WebAssembly (GOOS=js
// GOARCH=wasm, package mygo's backend_js.go): mygo build -platform web makes
// the page, Go's loader and the app's WebAssembly, a static site to serve
// anywhere, and mygo dev -web serves them, rebuilding the app as its code
// changes and reloading the page.

// webPlatform is the platform of browsers, which -platform also takes as
// "web".
const webPlatform = "js/wasm"

// webPage returns the page loading the app from wasm: a spinner while it
// loads, then its windows, which fill the page. In development, it reloads
// as the app is rebuilt, and shows why a build failed.
func webPage(c *Config, wasm, icon string, dev bool, locale string) string {
	var b strings.Builder
	b.WriteString(`<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>` + html.EscapeString(c.Name) + `</title>
`)
	if icon != "" {
		b.WriteString(`    <link rel="icon" href="` + icon + `" />
`)
	}
	b.WriteString(`    <style>
      :root { color-scheme: light dark; }
      html, body { margin: 0; height: 100%; }
      body { display: flex; align-items: center; justify-content: center; color: #8e8e93;
        font: 13px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
      .spinner { width: 24px; height: 24px; animation: spin 1s steps(12) infinite; }
      .spinner line { stroke: currentColor; stroke-width: 2; stroke-linecap: round; }
      @keyframes spin { to { transform: rotate(360deg); } }
      .error { max-width: 80%; white-space: pre-wrap; font: 12px ui-monospace, Menlo, monospace; }
      #mygo-build-error { position: fixed; inset: 0; z-index: 2147483647; overflow: auto; margin: 0; padding: 24px;
        background: rgb(24 24 27 / 0.94); color: #fca5a5; white-space: pre-wrap; font: 12px/1.5 ui-monospace, Menlo, monospace; }
    </style>
  </head>
  <body>
    <svg class="spinner" viewBox="0 0 24 24" role="progressbar" aria-label="Loading">
      <g transform="translate(12 12)">
`)
	for i := range 12 {
		fmt.Fprintf(&b, `        <line y1="-5.5" y2="-10" opacity="%.2f" transform="rotate(%d)" />
`, max(1-0.08*float64(i), 0.2), -30*i)
	}
	b.WriteString(`      </g>
    </svg>
    <script src="wasm_exec.js"></script>
    <script>
      const go = new Go()
`)
	if dev {
		env := `MYGO_ENV: "development"`
		if locale != "" {
			// mygo dev -lang: the language the app runs in.
			env += `, MYGO_LOCALE: ` + strconv.Quote(locale)
		}
		b.WriteString(`      go.env = { ` + env + ` }
      // mygo dev: a new build reloads the page; one that failed shows why.
      const events = new EventSource("` + devEvents + `")
      events.addEventListener("reload", () => location.reload())
      events.addEventListener("build-error", (e) => {
        let pre = document.getElementById("mygo-build-error")
        if (!pre) {
          pre = document.createElement("pre")
          pre.id = "mygo-build-error"
          document.body.append(pre)
        }
        pre.textContent = JSON.parse(e.data)
      })
`)
	}
	b.WriteString(`      WebAssembly.instantiateStreaming(fetch("` + wasm + `"), go.importObject)
        .then((r) => go.run(r.instance))
        .catch((e) => {
          const p = document.createElement("p")
          p.className = "error"
          p.textContent = "The app did not load: " + e
          document.body.replaceChildren(p)
        })
    </script>
  </body>
</html>
`)
	return b.String()
}

// wasmLoader returns Go's loader of WebAssembly, wasm_exec.js, of the Go
// that compiles the app: they must match.
func wasmLoader() ([]byte, error) {
	root := goEnv("GOROOT")
	if root == "" {
		return nil, errors.New("cannot find Go's GOROOT")
	}
	return os.ReadFile(filepath.Join(root, "lib", "wasm", "wasm_exec.js"))
}

// webIcon returns the app's icon as the page's, if it has one.
func webIcon(c *Config) ([]byte, error) {
	if c.Icon == "" {
		return nil, nil
	}
	return os.ReadFile(c.path(c.Icon))
}

// compileWeb compiles the app for browsers into out.
func compileWeb(ctx context.Context, c *Config, t *task, out string, debug bool, overlay string) error {
	ldflags := "-s -w"
	if !debug {
		ldflags += " -X github.com/egoist/mygo.production=1"
	}
	flags := []string{"-trimpath", "-ldflags", ldflags}
	flags = append(flags, productionTags(debug)...)
	if overlay != "" {
		flags = append(flags, "-overlay", overlay)
	}
	return goBuild(ctx, c, t, out, []string{"GOOS=js", "GOARCH=wasm"}, flags...)
}

// buildWeb builds the app for browsers into <out>/web, a static site: the
// page, Go's loader, the app's WebAssembly named by its content, which
// browsers may keep for good, and the app's icon. It returns the files.
func buildWeb(c *Config, opts buildOptions) ([]string, error) {
	out := c.path(c.Out)
	if err := os.MkdirAll(out, 0o755); err != nil {
		return nil, err
	}
	stage, err := os.MkdirTemp(out, ".staging-web-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(stage)
	if err := os.Chmod(stage, 0o755); err != nil {
		return nil, err
	}
	if !opts.debug && keepInspector() {
		logf("Keeping the inspector of native UI (MYGO_INSPECTOR=1)")
	}
	wasm := filepath.Join(stage, "app.wasm")
	t := con.start("Compiling " + webPlatform)
	overlay, err := writeOverlay(filepath.Join(opts.work, "overlay.json"), opts.overlay)
	if err == nil {
		err = compileWeb(context.Background(), c, t, wasm, opts.debug, overlay)
	}
	if err := t.end(err, "Compiled "+webPlatform); err != nil {
		return nil, err
	}
	data, err := os.ReadFile(wasm)
	if err != nil {
		return nil, err
	}
	sum := sha256.Sum256(data)
	name := slugify(c.Name) + "." + hex.EncodeToString(sum[:6]) + ".wasm"
	if err := os.Rename(wasm, filepath.Join(stage, name)); err != nil {
		return nil, err
	}
	loader, err := wasmLoader()
	if err != nil {
		return nil, err
	}
	files := map[string][]byte{"wasm_exec.js": loader}
	icon, err := webIcon(c)
	if err != nil {
		return nil, err
	}
	iconName := ""
	if icon != nil {
		iconName = "icon" + filepath.Ext(c.Icon)
		files[iconName] = icon
	}
	files["index.html"] = []byte(webPage(c, name, iconName, false, ""))
	for file, b := range files {
		if err := os.WriteFile(filepath.Join(stage, file), b, 0o644); err != nil {
			return nil, err
		}
	}
	dir := filepath.Join(out, "web")
	if err := os.RemoveAll(dir); err != nil {
		return nil, err
	}
	if err := os.Rename(stage, dir); err != nil {
		return nil, err
	}
	artifacts := []string{filepath.Join(dir, "index.html"), filepath.Join(dir, name), filepath.Join(dir, "wasm_exec.js")}
	if iconName != "" {
		artifacts = append(artifacts, filepath.Join(dir, iconName))
	}
	return artifacts, nil
}

// devEvents is the path of the events the page of mygo dev -web hears of
// builds from.
const devEvents = "/__mygo/events"

// webDev serves the app to browsers while it is developed: the page, Go's
// loader and the last build that compiled, with the events that reload
// pages.
type webDev struct {
	c      *Config
	loader []byte
	icon   []byte
	locale string // of -lang, which the app reads as MYGO_LOCALE

	mu      sync.Mutex
	wasm    []byte
	failure string // why the last build failed, "" after one that compiled
	clients map[chan [2]string]bool
}

func (d *webDev) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Cache-Control", "no-store")
	switch r.URL.Path {
	case "/", "/index.html":
		icon := ""
		if d.icon != nil {
			icon = "icon" + filepath.Ext(d.c.Icon)
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		_, _ = w.Write([]byte(webPage(d.c, "app.wasm", icon, true, d.locale)))
	case "/wasm_exec.js":
		w.Header().Set("Content-Type", "text/javascript; charset=utf-8")
		_, _ = w.Write(d.loader)
	case "/app.wasm":
		d.mu.Lock()
		wasm := d.wasm
		d.mu.Unlock()
		w.Header().Set("Content-Type", "application/wasm")
		_, _ = w.Write(wasm)
	case devEvents:
		d.events(w, r)
	default:
		if d.icon != nil && r.URL.Path == "/icon"+filepath.Ext(d.c.Icon) {
			_, _ = w.Write(d.icon)
			return
		}
		http.NotFound(w, r)
	}
}

// events sends a page the events of builds: "reload" after one compiled,
// "build-error" with why one failed, and the last build's failure at once.
func (d *webDev) events(w http.ResponseWriter, r *http.Request) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/event-stream")
	ch := make(chan [2]string, 4)
	d.mu.Lock()
	d.clients[ch] = true
	failure := d.failure
	d.mu.Unlock()
	defer func() {
		d.mu.Lock()
		delete(d.clients, ch)
		d.mu.Unlock()
	}()
	send := func(event, data string) {
		fmt.Fprintf(w, "event: %s\ndata: %s\n\n", event, data)
		flusher.Flush()
	}
	send("hello", "{}")
	if failure != "" {
		b, _ := json.Marshal(failure)
		send("build-error", string(b))
	}
	for {
		select {
		case <-r.Context().Done():
			return
		case e := <-ch:
			send(e[0], e[1])
		}
	}
}

// broadcast sends an event to the pages open.
func (d *webDev) broadcast(event, data string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	for ch := range d.clients {
		select {
		case ch <- [2]string{event, data}:
		default:
		}
	}
}

// runWebDev develops the app in browsers: it serves it on port, or the
// next free one, opens it unless told not to, and rebuilds it as its code
// changes, which reloads the pages showing it.
func runWebDev(ctx context.Context, c *Config, port int, open bool, locale string) error {
	loader, err := wasmLoader()
	if err != nil {
		return err
	}
	icon, err := webIcon(c)
	if err != nil {
		return err
	}
	d := &webDev{c: c, loader: loader, icon: icon, locale: locale, clients: map[chan [2]string]bool{}}
	var ln net.Listener
	for p := port; p < port+20; p++ {
		if ln, err = net.Listen("tcp", "localhost:"+strconv.Itoa(p)); err == nil {
			break
		}
	}
	if err != nil {
		return err
	}
	srv := &http.Server{Handler: d, ReadHeaderTimeout: 10 * time.Second}
	go func() { _ = srv.Serve(ln) }()
	defer srv.Close()
	url := "http://localhost:" + strconv.Itoa(ln.Addr().(*net.TCPAddr).Port) + "/"

	work, err := os.MkdirTemp("", "mygo-web-dev-")
	if err != nil {
		return err
	}
	defer os.RemoveAll(work)
	wasm := filepath.Join(work, "app.wasm")
	// why is what changed, as "main.go changed", or "".
	build := func(why string) bool {
		title := "Building " + c.Name + " for browsers"
		if why != "" {
			title += " (" + why + ")"
		}
		t := con.start(title)
		var err error
		if c2, cerr := loadConfig(c.root); cerr != nil {
			err = cerr
		} else if out, ierr := writeI18n(c2); ierr != nil {
			err = ierr
		} else {
			for _, f := range out.written {
				t.set("wrote " + f)
			}
			err = compileWeb(ctx, c, t, wasm, true, "")
		}
		var data []byte
		if err == nil {
			data, err = os.ReadFile(wasm)
		}
		switch {
		case errors.Is(err, context.Canceled):
			t.stop()
			return false
		case err != nil:
			t.fail()
			con.details(err)
			con.println("  " + dim("The page shows why: save a fix to try again."))
			b, _ := json.Marshal(err.Error())
			d.mu.Lock()
			d.failure = err.Error()
			d.mu.Unlock()
			d.broadcast("build-error", string(b))
			return false
		}
		detail := formatSize(int64(len(data)))
		if why != "" {
			detail = why + ", " + detail
		}
		t.done("Built " + c.Name + " for browsers " + dim("("+detail+")"))
		d.mu.Lock()
		d.wasm, d.failure = data, ""
		d.mu.Unlock()
		d.broadcast("reload", "{}")
		return true
	}

	w := &watcher{}
	if in, err := listBuildInputs(c); err == nil {
		w.set(in)
	} else {
		warnf("%v", err)
	}
	changes := w.watch(ctx, 250*time.Millisecond)
	keys := readKeys()
	if err := generateI18n(c); err != nil {
		return err
	}
	build("")
	con.println("\n  " + cyan(con.sym.arrow) + " Serving at " + bold(url))
	if open {
		openBrowser(url)
	}
	msg := "  " + cyan(con.sym.arrow) + " Watching for changes\n"
	if keys != nil {
		msg += "  " + cyan(con.sym.arrow) + " " + dim("press ") + bold("h + enter") + dim(" to show help") + "\n"
	}
	con.println(msg)
	for {
		select {
		case <-ctx.Done():
			return nil
		case paths := <-changes:
			build(changeSummary(c.root, paths))
			if c2, err := loadConfig(c.root); err == nil {
				if in, err := listBuildInputs(c2); err == nil {
					w.set(in)
				}
			}
		case key, ok := <-keys:
			switch {
			case !ok:
				keys = nil
			case key == "r":
				build("")
			case key == "o":
				openBrowser(url)
			case key == "c":
				con.clearScreen()
			case key == "q":
				return nil
			case key == "h":
				con.println(webDevHelp())
			}
		}
	}
}

func webDevHelp() string {
	var b strings.Builder
	b.WriteString("\n  " + bold("Shortcuts") + "\n")
	for _, k := range [][2]string{
		{"r", "rebuild the app and reload its pages"},
		{"o", "open the app in the browser"},
		{"c", "clear the console"},
		{"q", "quit"},
	} {
		b.WriteString("  " + dim("press ") + bold(k[0]+" + enter") + dim(" to "+k[1]) + "\n")
	}
	return b.String()
}

// openBrowser opens url in the default browser.
func openBrowser(url string) {
	var cmd *exec.Cmd
	switch runtime.GOOS {
	case "darwin":
		cmd = exec.Command("open", url)
	case "windows":
		cmd = exec.Command("rundll32", "url.dll,FileProtocolHandler", url)
	default:
		cmd = exec.Command("xdg-open", url)
	}
	if err := cmd.Start(); err != nil {
		warnf("Cannot open the browser: %v", err)
		return
	}
	go func() { _ = cmd.Wait() }()
}

// isWebPlatform reports whether a -platform target is the browsers'.
func isWebPlatform(p string) bool { return slices.Contains([]string{"web", webPlatform}, p) }
