//go:build js

package mygo

import "github.com/egoist/mygo/internal/platform"

// Browsers show no web pages, for which `mygo generate` writes the client:
// builds for them leave out the TypeScript generator, with go/parser.

// GenerateTypeScript renders the typed TypeScript client, which browsers
// do not generate: it returns platform.ErrUnsupported.
func GenerateTypeScript() ([]byte, error) { return nil, platform.ErrUnsupported }

// WriteTypeScript writes the TypeScript client to path. Bind binds nothing
// in browsers, so there is no client and path is not written.
func WriteTypeScript(path string) error { return nil }

// writeClient writes nothing: there is no client in browsers.
func writeClient(path string) error { return nil }
