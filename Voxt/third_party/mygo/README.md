<p align="center">
<img width="800" height="360" alt="MyGo!!!!!" src="https://github.com/user-attachments/assets/c3eee9bc-5f22-4032-95b6-776e8381910a" />
</p>

**Desktop apps in Go, with native UI or a web frontend.**

Each MyGo window shows one of two kinds of interface, and one app, or one
window, can mix them:

- **A web page**, in the webview the OS already has: WKWebView on macOS,
  WebKitGTK on Linux, WebView2 on Windows. Build the frontend with any web
  tools; it calls your Go code through a TypeScript client generated from it.
- **Native UI**, written in Go alone with package `ui` and
  drawn by MyGo itself on the GPU: no HTML, no JavaScript and no webview, so
  the window opens at once and takes little memory.

Either way, an app is a single Go binary of a few megabytes, focused on low
memory and CPU use.

- **Pure Go, no cgo**: build for every platform from any machine.
- **Typed IPC** for pages: bind Go services, stream values through channels
  and send typed events; the TypeScript client is generated from your Go
  code.
- **A Go UI toolkit** for native UI: flexbox and grid layout, widgets, text
  editing with input methods, virtualized lists, SVG icons, animations,
  screen reader support, and views you test without a window; Liquid Glass
  on every platform with the glass plugin; web pages inside native UI, under
  the menus and dialogs it shows over them.
- **Desktop APIs** for both: windows, menus, tray, dialogs, notifications,
  global shortcuts, deep links, file associations, recent documents, the
  system accent color and accessibility settings, watching files, and more.
- **Ready to ship**: app bundles and disk images, Windows installers, Debian
  packages and a Linux install script, code signing, notarization, and signed
  auto-updates with delta updates and an update window in the manner of
  Sparkle.

A window with a web page, whose frontend calls Go:

```go
type Greeter struct{}

// Greet returns a greeting.
func (Greeter) Greet(name string) string { return "Hello, " + name }

func main() {
	mygo.Bind(Greeter{})
	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{Title: "Hello", URL: "/"})
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
```

```ts
import { Greeter } from "./mygo"; // generated

document.body.textContent = await Greeter.greet("Ada");
```

A window with native UI, whose view is a Go function of the app's state:

```go
type counter struct{ n int }

func (s *counter) view(c *ui.Context) {
	ui.Column(c).Fill().Center().Gap(12).Children(func() {
		ui.Text(c, fmt.Sprint(s.n)).FontSize(40).Bold()
		if ui.PrimaryButton(c, "Increment").Clicked() {
			s.n++
		}
	})
}

func main() {
	s := &counter{}
	mygo.App.WhenReady(func() {
		mygo.NewWindow(mygo.WindowOptions{
			Title:   "Counter",
			Content: ui.View(s.view),
		})
	})
	if err := mygo.App.Run(); err != nil {
		log.Fatal(err)
	}
}
```

## Getting started

With [Go](https://go.dev/dl/) 1.27+ and [Bun](https://bun.sh), create an app
with a web frontend:

```sh
bunx mygo-cli init my-app      # or: npx mygo-cli init my-app
cd my-app
bun run dev
```

An app of native UI needs Go alone:

```sh
go run github.com/egoist/mygo/cmd/mygo@latest init -template native my-app
cd my-app
go tool mygo dev
```

Read the [documentation](docs/README.md).

## License

MIT
