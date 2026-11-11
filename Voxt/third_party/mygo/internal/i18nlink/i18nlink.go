// Package i18nlink hands MyGo's own strings (internal/l10n) what package
// i18n does not export: a library's catalog that does not link the
// formatting of the dates of messages, which i18n.NewLibraryCatalog does
// for the catalogs of libraries and apps. MyGo's own strings have no
// dates, and every app formats them, so the patterns and names of dates
// of CLDR's locales are linked only into apps that format dates.
package i18nlink

import "io/fs"

// NewLibraryCatalog is i18n.NewLibraryCatalog, which returns an
// *i18n.Catalog, without linking the formatting of dates. Package i18n,
// which imports this one, sets it as it initializes.
var NewLibraryCatalog func(fsys fs.FS, source, namespace string) (any, error)
