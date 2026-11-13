package models

import (
	_ "embed"
	"encoding/json"
	"slices"
)

//go:embed pins.json
var pinsJSON []byte

// Pin is a row of the family pins table (spec §7.2): the commit every app
// should use for a repo. To move a repo to another commit, change the
// table first (an actnowkit release); the apps follow in their next
// versions, keeping the old commit in their catalogue's accept list
// meanwhile.
type Pin struct {
	Repo     string   `json:"repo"`
	Revision string   `json:"revision"`
	Family   string   `json:"family"`          // model series: "qwen3-asr"
	Quant    string   `json:"quant,omitempty"` // "4bit", "fp16", "Q4_K_M"…
	Role     string   `json:"role"`
	Bytes    int64    `json:"bytes,omitempty"`
	Files    []string `json:"files,omitempty"` // the files used, when not the whole repo
	Apps     []string `json:"apps"`            // app ids whose catalogue has it
	Aux      []Ref    `json:"aux,omitempty"`   // auxiliary repos it needs
	Note     string   `json:"note,omitempty"`
}

// Ref is the pin's identity.
func (p Pin) Ref() Ref { return Ref{Repo: p.Repo, Revision: p.Revision} }

var pins = func() []Pin {
	var f struct {
		Pins []Pin `json:"pins"`
	}
	if err := json.Unmarshal(pinsJSON, &f); err != nil {
		panic("models: pins.json: " + err.Error())
	}
	return f.Pins
}()

// Pins is the family pins table (a copy).
func Pins() []Pin { return slices.Clone(pins) }

// PinOf finds a repo's pin.
func PinOf(repo string) (Pin, bool) {
	for _, p := range pins {
		if p.Repo == repo {
			return p, true
		}
	}
	return Pin{}, false
}

// PinsOfFamily lists the pins of a model series ("gemma4"): the
// candidates for a catalogue entry's alternatives.
func PinsOfFamily(family string) []Pin {
	var out []Pin
	for _, p := range pins {
		if p.Family == family {
			out = append(out, p)
		}
	}
	return out
}

// PinsOfApp lists the pins whose Apps include appID: for the phase-0
// rule "do not delete what another installed app may use", until every
// app writes claims.
func PinsOfApp(appID string) []Pin {
	var out []Pin
	for _, p := range pins {
		if slices.Contains(p.Apps, appID) {
			out = append(out, p)
		}
	}
	return out
}
