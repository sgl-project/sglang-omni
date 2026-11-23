package mygo

import (
	"fmt"
	"strings"
	"sync"
)

// Plugin adds a feature to an app in two halves: Go services, and a
// JavaScript package whose pages call them. The official plugins live in
// the repository's plugins directory, such as fetch, whose npm package
// @mygo-plugins/fetch gives pages a fetch that runs in Go:
//
//	mygo.Use(fetch.Plugin, websocket.Plugin)
//
// A plugin's services are bound like those of Bind, with the same rules
// for their methods, but under the name "plugin:<Name>" and outside the
// generated TypeScript client: its JavaScript package calls them with call
// of mygo-runtime, as in `call("plugin:fetch.Fetch", ...)`. Only trusted
// pages may call them, like every bound method.
type Plugin struct {
	// Name identifies the plugin: lowercase letters, digits and hyphens,
	// starting with a letter, e.g. "fetch". "mygo" is reserved.
	Name string
	// Service is bound as "plugin:<Name>", when not nil.
	Service any
	// Setup runs once, when Use is called with the plugin, to add
	// listeners, protocols and the like. An error makes Use panic.
	Setup func() error
}

var plugins struct {
	sync.Mutex
	used map[string]bool
}

// Use adds plugins to the app. Call it before App.Run, like Bind. Use
// panics when a plugin is invalid, already used, or its Setup fails.
func Use(ps ...Plugin) {
	for _, p := range ps {
		if err := use(p); err != nil {
			panic(err)
		}
	}
}

func use(p Plugin) error {
	if !isPluginName(p.Name) {
		return fmt.Errorf("mygo: plugin name %q must be lowercase letters, digits and hyphens, starting with a letter", p.Name)
	}
	if p.Name == "mygo" {
		return fmt.Errorf("mygo: plugin name %q is reserved", p.Name)
	}
	plugins.Lock()
	if plugins.used[p.Name] {
		plugins.Unlock()
		return fmt.Errorf("mygo: plugin %s is already used", p.Name)
	}
	if p.Service != nil {
		if err := bind(pluginPrefix+p.Name, p.Service, true); err != nil {
			plugins.Unlock()
			return fmt.Errorf("mygo: plugin %s: %s", p.Name, strings.TrimPrefix(err.Error(), "mygo: "))
		}
	}
	if plugins.used == nil {
		plugins.used = map[string]bool{}
	}
	plugins.used[p.Name] = true
	plugins.Unlock()
	// Unlocked: Setup may use the plugins it builds on.
	if p.Setup != nil {
		if err := p.Setup(); err != nil {
			return fmt.Errorf("mygo: plugin %s: %w", p.Name, err)
		}
	}
	return nil
}

const pluginPrefix = "plugin:"

func isPluginName(s string) bool {
	if s == "" || s[0] < 'a' || s[0] > 'z' {
		return false
	}
	for _, c := range s {
		if (c < 'a' || c > 'z') && (c < '0' || c > '9') && c != '-' {
			return false
		}
	}
	return true
}

// notBound explains why the method a page called is missing.
func notBound(name string) error {
	if rest, ok := strings.CutPrefix(name, pluginPrefix); ok {
		plugin, _, _ := strings.Cut(rest, ".")
		plugins.Lock()
		used := plugins.used[plugin]
		plugins.Unlock()
		if !used {
			return fmt.Errorf("the %s plugin is not used: call mygo.Use with it before App.Run", plugin)
		}
	}
	return fmt.Errorf("method %s is not bound", name)
}
