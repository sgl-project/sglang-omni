package conformance

import (
	"embed"
	"encoding/json"
	"io/fs"
)

//go:embed testdata/files testdata/cases.json
var testdata embed.FS

// Files are the sample files of every kind (testdata/files), the same
// files cases.json describes.
var Files, _ = fs.Sub(testdata, "testdata/files")

// File reads one sample file.
func File(name string) ([]byte, error) { return fs.ReadFile(Files, name) }

// Case is a row of testdata/cases.json: a sample file and what a
// conforming reader must make of it (spec §5.1).
type Case struct {
	File string `json:"file"`
	Kind string `json:"kind"` // schema kind expected: "settings", "models.claims"…
	// Result is "ok" (read and writable), "readOnly" (a newer major: read
	// what is understood, never write back), "corrupt" (treat as absent;
	// move aside before writing a new one) or "wrongKind".
	Result string `json:"result"`
	// Expect maps JSON pointers to the values a reader must find.
	Expect map[string]any `json:"expect,omitempty"`
	// Keep lists JSON pointers of members a read-modify-write must keep.
	Keep []string `json:"keep,omitempty"`
}

// Cases reads testdata/cases.json.
func Cases() ([]Case, error) {
	data, err := testdata.ReadFile("testdata/cases.json")
	if err != nil {
		return nil, err
	}
	var f struct {
		Cases []Case `json:"cases"`
	}
	err = json.Unmarshal(data, &f)
	return f.Cases, err
}
