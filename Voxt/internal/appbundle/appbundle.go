// Package appbundle reads what the app bundle on disk says about itself,
// for the time between an update replacing the bundle and the app
// restarting: MyGo's updater swaps VoxtNew.app while the old version
// still runs, so the running app compares its version with the one on
// disk (Updated) and, when they differ, asks to be restarted instead of
// starting the new version's helper.
//
// Imports: nothing internal.
package appbundle

import (
	"bytes"
	"encoding/xml"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// InfoPlist is the Info.plist of the bundle whose resources are in
// resources (Contents/Resources → Contents/Info.plist).
func InfoPlist(resources string) string {
	return filepath.Join(filepath.Dir(resources), "Info.plist")
}

// ErrNoVersion is returned for a property list without
// CFBundleShortVersionString.
var ErrNoVersion = errors.New("appbundle: no CFBundleShortVersionString")

// Version returns the CFBundleShortVersionString of the bundle whose
// resources are in resources. Outside a bundle (go run, go test) there is
// no Info.plist: an error satisfying os.IsNotExist.
func Version(resources string) (string, error) {
	data, err := os.ReadFile(InfoPlist(resources))
	if err != nil {
		return "", err
	}
	return ShortVersion(data)
}

// ShortVersion returns CFBundleShortVersionString from an XML property
// list (as mygo build writes it).
func ShortVersion(plist []byte) (string, error) {
	d := xml.NewDecoder(bytes.NewReader(plist))
	d.Strict = false
	depth := 0    // of the elements open: 1 plist, 2 the top dict, 3 its entries
	lastKey := "" // the top dict's last <key>
	var text []byte
	for {
		tok, err := d.Token()
		if err == io.EOF {
			return "", ErrNoVersion
		}
		if err != nil {
			return "", err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			depth++
			text = text[:0]
		case xml.CharData:
			text = append(text, t...)
		case xml.EndElement:
			if depth == 3 {
				switch t.Name.Local {
				case "key":
					lastKey = strings.TrimSpace(string(text))
					depth--
					continue
				case "string":
					if lastKey == "CFBundleShortVersionString" {
						return strings.TrimSpace(string(text)), nil
					}
				}
				lastKey = ""
			}
			depth--
		}
	}
}

// Updated reports whether the bundle on disk is another version than
// running, the version the app runs: an update was installed and the app
// has not restarted. Outside a bundle, or when either version is unknown,
// it is false.
func Updated(resources, running string) bool {
	if running == "" {
		return false
	}
	v, err := Version(resources)
	return err == nil && v != "" && v != running
}
