# updater

The update window of [MyGo](https://github.com/egoist/mygo) apps, in the
manner of Sparkle: it checks for updates in the background, shows the
release notes of a new version and offers to install it, skip it or remind
the user later, downloads it with a progress bar and relaunches the app into
it. It is all Go, with no npm package.

```go
import "github.com/egoist/mygo/plugins/updater"

mygo.Use(updater.Plugin) // or updater.New(updater.Options{Interval: ..., Icon: ...})
```

and "Check for Updates…" in the app's menu:

```go
{Label: "My App", Submenu: []*mygo.MenuItem{
	{Role: mygo.RoleAbout},
	updater.MenuItem(),
	mygo.Separator(),
	{Role: mygo.RoleQuit},
}},
```

Apps whose windows all show native UI use `native.Plugin` (package
`github.com/egoist/mygo/plugins/updater/native`) instead: the same window
drawn in native UI, with no webview.

It speaks the user's language (English, Chinese, Dutch, French, German,
Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Turkish,
Ukrainian), and `Options.Strings` changes its texts or adds languages.

The app must be built with updates: see
[auto-updates](https://github.com/egoist/mygo/blob/main/docs/updates.md).
[The plugin's documentation](https://github.com/egoist/mygo/blob/main/docs/plugins/updater.md)
lists the options, the languages and the preferences
(`SetAutomaticChecks`, `SetAutomaticDownloads`, and `OnChange` to hear
them change) apps can offer.
