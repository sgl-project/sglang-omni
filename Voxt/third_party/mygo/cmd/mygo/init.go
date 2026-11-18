package main

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"io/fs"
	"math"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"strings"
	"text/template"
)

//go:embed all:template
var templateFS embed.FS

type templateData struct {
	Name       string
	Slug       string
	Module     string
	Identifier string
	// Runtime is the version of the mygo-runtime npm package.
	Runtime string
	// CLI is the version of the mygo-cli npm package, which runs mygo in
	// the scripts of package.json, and Mygo the command they run.
	CLI, Mygo string
	// ConfigImport is the module mygo.config.ts imports defineConfig from.
	ConfigImport string
}

// The templates of mygo init: a TypeScript frontend, and native UI.
const (
	webTemplate    = "web"
	nativeTemplate = "native"
)

// cliTool is the package of the CLI, which projects of native UI depend on
// as a tool of their module.
const cliTool = "github.com/egoist/mygo/cmd/mygo"

func runInit(args []string) error {
	flags := newFlags("init", "[flags] <dir>", "Creates a new MyGo project. The web template has a TypeScript frontend built\nwith Vite: Bun installs its dependencies, the mygo-cli package among them,\nand runs its scripts, bun run dev and bun run build. The native template\nshows a user interface MyGo draws itself, written in Go with package ui:\nthe module has the CLI as a tool, for go tool mygo dev and go tool mygo\nbuild.")
	name := flags.String("name", "", "application name (default: directory name)")
	module := flags.String("module", "", "Go module path (default: directory name)")
	local := flags.String("mygo", "", "path to a local checkout of MyGo to use via a replace directive")
	tmpl := flags.String("template", webTemplate, "the project: web (a TypeScript frontend) or native (native UI in Go)")
	if err := flags.Parse(args); err != nil {
		return err
	}
	if flags.NArg() != 1 {
		flags.Usage()
		return flag2Err("missing project directory")
	}
	if *tmpl != webTemplate && *tmpl != nativeTemplate {
		return flag2Err(fmt.Sprintf("unknown template %q: web or native", *tmpl))
	}
	dir, data, err := newProject(flags.Arg(0), *name, *module)
	if err != nil {
		return err
	}
	if *tmpl == nativeTemplate {
		return initNative(dir, data, *local)
	}
	data.Runtime, data.CLI, data.Mygo, data.ConfigImport = "^"+version, "^"+version, "mygo", "mygo-cli"
	if *local != "" {
		// The packages of the local checkout, like the Go module, and its
		// CLI, which go run builds from the replaced module.
		checkout, err := filepath.Abs(*local)
		if err != nil {
			return err
		}
		runtime := filepath.Join(checkout, "packages", "runtime")
		if _, err := os.Stat(filepath.Join(runtime, "dist", "index.js")); err != nil {
			return fmt.Errorf("the packages of %s are not built: run bun install && bun run build there first", checkout)
		}
		data.Runtime = "file:" + runtime
		data.CLI, data.Mygo = "", "go run github.com/egoist/mygo/cmd/mygo"
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
		data.ConfigImport = moduleSpecifier(dir, filepath.Join(checkout, "packages", "cli", "index.js"))
	}

	logf("Creating %s", dir)
	if err := writeProject(dir, webTemplate, data, *local, false); err != nil {
		return err
	}

	if _, err := exec.LookPath("bun"); err != nil {
		warnf("Bun not found: install it from https://bun.sh, then run bun install")
	} else {
		t := con.start("Installing the dependencies")
		if err := t.end(run(t, "bun", dir, "bun install"), "Installed the dependencies"); err != nil {
			return err
		}
	}
	if c, err := loadConfig(dir); err != nil {
		warnf("%v; run mygo generate once it can be read", err)
	} else {
		bin := tempBinary(c.executableName())
		defer os.Remove(bin)
		if err := buildBinary(c, bin, nil); err == nil {
			_, _ = generateBindings(c, bin)
		}
	}

	fmt.Printf("\nCreated %s. Next steps:\n\n  cd %s\n  bun run dev      # develop with live reload\n  bun run build    # package the app\n\n", data.Name, relPath(dir))
	return nil
}

// newProject returns the absolute directory of a new project, which must
// be empty or not exist, and what its template is filled with.
func newProject(path, name, module string) (string, templateData, error) {
	dir, err := filepath.Abs(path)
	if err != nil {
		return "", templateData{}, err
	}
	if entries, err := os.ReadDir(dir); err == nil && len(entries) > 0 {
		return "", templateData{}, fmt.Errorf("%s is not empty", dir)
	}
	data := templateData{Name: name, Module: module}
	if data.Name == "" {
		data.Name = filepath.Base(dir)
	}
	data.Slug = slugify(data.Name)
	if data.Module == "" {
		data.Module = data.Slug
	}
	data.Identifier = "com.example." + strings.ReplaceAll(data.Slug, "-", "")
	return dir, data, nil
}

// initNative creates a project of native UI in dir: a Go module, without
// a frontend.
func initNative(dir string, data templateData, local string) error {
	logf("Creating %s", dir)
	if err := writeProject(dir, nativeTemplate, data, local, true); err != nil {
		return err
	}
	fmt.Printf("\nCreated %s. Next steps:\n\n  cd %s\n  go tool mygo dev      # develop with live reload\n  go test               # test the view without a window\n  go tool mygo build    # package the app\n\n", data.Name, relPath(dir))
	return nil
}

// writeProject writes a template into dir with the app's icon and its Go
// module, which depends on MyGo, from local when it is a checkout, and,
// with tool, has the CLI as a tool.
func writeProject(dir, tmpl string, data templateData, local string, tool bool) error {
	if err := writeTemplate(dir, tmpl, data); err != nil {
		return err
	}
	if err := os.MkdirAll(filepath.Join(dir, resourcesDir), 0o755); err != nil {
		return err
	}
	if err := os.WriteFile(filepath.Join(dir, resourcesDir, "icon.png"), defaultIcon(), 0o644); err != nil {
		return err
	}
	gomod := fmt.Sprintf("module %s\n\ngo %s\n", data.Module, goVersion())
	if local != "" {
		abs, err := filepath.Abs(local)
		if err != nil {
			return err
		}
		if tool {
			gomod += "\ntool " + cliTool + "\n"
		}
		gomod += fmt.Sprintf("\nrequire github.com/egoist/mygo v0.0.0\n\nreplace github.com/egoist/mygo => %s\n", abs)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(gomod), 0o644); err != nil {
		return err
	}
	if local == "" {
		get := []string{"get", "github.com/egoist/mygo@latest"}
		if tool {
			get = []string{"get", "-tool", cliTool + "@latest"}
		}
		if err := goCommand(dir, nil, get...).Run(); err != nil {
			warnf("Could not fetch github.com/egoist/mygo: %v", err)
		}
	}
	if err := goCommand(dir, nil, "mod", "tidy").Run(); err != nil {
		warnf("go mod tidy failed: %v", err)
	}
	return nil
}

// relPath returns dir relative to the working directory when it is below
// it, for showing.
func relPath(dir string) string {
	if wd, err := os.Getwd(); err == nil {
		if r, err := filepath.Rel(wd, dir); err == nil && !strings.HasPrefix(r, "..") {
			return r
		}
	}
	return dir
}

// moduleSpecifier returns how a module in dir imports the file target: a
// relative path between their real locations, which runtimes resolve
// imports from, or a file URL where there is none, as between the drives
// of Windows.
func moduleSpecifier(dir, target string) string {
	if real, err := filepath.EvalSymlinks(dir); err == nil {
		dir = real
	}
	if real, err := filepath.EvalSymlinks(target); err == nil {
		target = real
	}
	rel, err := filepath.Rel(dir, target)
	if err != nil {
		path := filepath.ToSlash(target)
		if !strings.HasPrefix(path, "/") {
			path = "/" + path // C:/…
		}
		return (&url.URL{Scheme: "file", Path: path}).String()
	}
	rel = filepath.ToSlash(rel)
	if !strings.HasPrefix(rel, "../") {
		rel = "./" + rel
	}
	return rel
}

// templateFuncs are the functions of the templates: json renders a value
// as JSON, which is also a JavaScript literal.
var templateFuncs = template.FuncMap{
	"json": func(v any) (string, error) {
		b, err := json.Marshal(v)
		return string(b), err
	},
}

// writeTemplate installs the bundled skills and template/<tmpl> into dir.
func writeTemplate(dir, tmpl string, data templateData) error {
	if err := installSkills(dir); err != nil {
		return err
	}
	return writeTemplateDir(dir, tmpl, data)
}

func writeTemplateDir(dir, tmpl string, data templateData) error {
	root := "template/" + tmpl
	return fs.WalkDir(templateFS, root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, _ := filepath.Rel(root, path)
		target := filepath.Join(dir, rel)
		if d.IsDir() {
			return os.MkdirAll(target, 0o755)
		}
		content, err := templateFS.ReadFile(path)
		if err != nil {
			return err
		}
		if strings.HasSuffix(target, ".tmpl") {
			target = strings.TrimSuffix(target, ".tmpl")
			t, err := template.New(rel).Funcs(templateFuncs).Parse(string(content))
			if err != nil {
				return err
			}
			var b bytes.Buffer
			if err := t.Execute(&b, data); err != nil {
				return err
			}
			content = b.Bytes()
		}
		if filepath.Base(target) == "gitignore" {
			target = filepath.Join(filepath.Dir(target), ".gitignore")
		}
		return os.WriteFile(target, content, 0o644)
	})
}

var slugRe = regexp.MustCompile(`[^a-z0-9]+`)

func slugify(s string) string {
	s = strings.Trim(slugRe.ReplaceAllString(strings.ToLower(s), "-"), "-")
	if s == "" {
		return "app"
	}
	return s
}

func goVersion() string {
	v := strings.TrimPrefix(runtime.Version(), "go")
	if i := strings.IndexAny(v, " -"); i >= 0 {
		v = v[:i]
	}
	return v
}

type flagErr string

func (e flagErr) Error() string { return string(e) }

func flag2Err(s string) error { return flagErr(s) }

// defaultIcon draws a 1024x1024 app icon: a rounded square with a diagonal
// gradient and a ring.
func defaultIcon() []byte {
	const size = 1024
	img := image.NewNRGBA(image.Rect(0, 0, size, size))
	from := [3]float64{59, 130, 246} // blue
	to := [3]float64{168, 85, 247}   // violet
	inset, radius := 100.0, 185.0
	for y := range size {
		for x := range size {
			fx, fy := float64(x)+0.5, float64(y)+0.5
			// Signed distance to a rounded rectangle, for anti-aliasing.
			qx := math.Abs(fx-size/2) - (size/2 - inset - radius)
			qy := math.Abs(fy-size/2) - (size/2 - inset - radius)
			d := math.Hypot(math.Max(qx, 0), math.Max(qy, 0)) + math.Min(math.Max(qx, qy), 0) - radius
			alpha := math.Min(math.Max(0.5-d, 0), 1)
			if alpha == 0 {
				continue
			}
			t := (fx + fy) / (2 * size)
			c := [3]float64{}
			for i := range c {
				c[i] = from[i] + (to[i]-from[i])*t
			}
			// A white ring in the middle.
			r := math.Hypot(fx-size/2, fy-size/2)
			ring := math.Min(math.Max(0.5-math.Abs(r-230)+48, 0), 1)
			for i := range c {
				c[i] = c[i]*(1-ring) + 255*ring
			}
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(c[0]), G: uint8(c[1]), B: uint8(c[2]), A: uint8(alpha * 255)})
		}
	}
	var b bytes.Buffer
	_ = png.Encode(&b, img)
	return b.Bytes()
}
