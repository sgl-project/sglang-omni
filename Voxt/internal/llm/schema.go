package llm

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"slices"
	"strings"
)

// Schema is the subset of JSON Schema the skills need, and that
// structured-output modes commonly accept: type, properties, required,
// additionalProperties (false only), items, enum, min/maxItems,
// min/maxLength, minimum/maximum and description. It marshals to plain
// JSON Schema; Client.JSON validates replies against it.
type Schema struct {
	Type        string             `json:"type,omitempty"` // object, array, string, number, integer, boolean, null
	Description string             `json:"description,omitempty"`
	Properties  map[string]*Schema `json:"properties,omitempty"`
	Required    []string           `json:"required,omitempty"`
	// Closed rejects properties not in Properties (additionalProperties: false).
	Closed   bool     `json:"-"`
	Items    *Schema  `json:"items,omitempty"`
	Enum     []string `json:"enum,omitempty"` // string enums only
	MinItems *int     `json:"minItems,omitempty"`
	MaxItems *int     `json:"maxItems,omitempty"`
	MinLen   *int     `json:"minLength,omitempty"`
	MaxLen   *int     `json:"maxLength,omitempty"`
	Minimum  *float64 `json:"minimum,omitempty"`
	Maximum  *float64 `json:"maximum,omitempty"`
	// Nullable also accepts null (type: [T, "null"]).
	Nullable bool `json:"-"`
	// Name is the schema's name for OpenAI's json_schema mode (root only).
	Name string `json:"-"`
}

// MarshalJSON writes standard JSON Schema.
func (s *Schema) MarshalJSON() ([]byte, error) {
	type plain Schema
	m := map[string]any{}
	b, err := json.Marshal((*plain)(s))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	if s.Type == "object" {
		if s.Closed {
			m["additionalProperties"] = false
		}
		if s.Properties == nil {
			m["properties"] = map[string]any{}
		}
	}
	if s.Nullable && s.Type != "" {
		m["type"] = []string{s.Type, "null"}
	}
	return json.Marshal(m)
}

// Object, Array, String… build schemas tersely.
func Object(props map[string]*Schema, required ...string) *Schema {
	return &Schema{Type: "object", Properties: props, Required: required, Closed: true}
}
func Array(items *Schema) *Schema   { return &Schema{Type: "array", Items: items} }
func String() *Schema               { return &Schema{Type: "string"} }
func Number() *Schema               { return &Schema{Type: "number"} }
func Integer() *Schema              { return &Schema{Type: "integer"} }
func Boolean() *Schema              { return &Schema{Type: "boolean"} }
func Enum(values ...string) *Schema { return &Schema{Type: "string", Enum: values} }

// Range sets minimum and maximum and returns s.
func (s *Schema) Range(lo, hi float64) *Schema { s.Minimum, s.Maximum = &lo, &hi; return s }

// Count sets min/maxItems (max < 0: unbounded) and returns s.
func (s *Schema) Count(lo, hi int) *Schema {
	s.MinItems = &lo
	if hi >= 0 {
		s.MaxItems = &hi
	}
	return s
}

// Describe sets the description and returns s.
func (s *Schema) Describe(d string) *Schema { s.Description = d; return s }

// SchemaError is one validation failure.
type SchemaError struct {
	Path   string // JSON pointer-ish: "/lines/3/id"
	Detail string // English, for the repair prompt and logs
}

func (e SchemaError) String() string { return e.Path + ": " + e.Detail }

// Validate checks the JSON value v against s and returns every failure (at most 20).
func (s *Schema) Validate(v json.RawMessage) []SchemaError {
	var x any
	d := json.NewDecoder(bytes.NewReader(v))
	d.UseNumber()
	if err := d.Decode(&x); err != nil {
		return []SchemaError{{Path: "/", Detail: "not valid JSON: " + err.Error()}}
	}
	var errs []SchemaError
	s.validate("", x, &errs)
	return errs
}

func (s *Schema) validate(path string, x any, errs *[]SchemaError) {
	if len(*errs) >= 20 || s == nil {
		return
	}
	fail := func(format string, a ...any) {
		p := path
		if p == "" {
			p = "/"
		}
		*errs = append(*errs, SchemaError{Path: p, Detail: fmt.Sprintf(format, a...)})
	}
	if x == nil {
		if s.Nullable || s.Type == "null" || s.Type == "" {
			return
		}
		fail("must be %s, got null", s.Type)
		return
	}
	switch s.Type {
	case "object":
		obj, ok := x.(map[string]any)
		if !ok {
			fail("must be an object")
			return
		}
		for _, r := range s.Required {
			if _, ok := obj[r]; !ok {
				fail("missing required property %q", r)
			}
		}
		keys := make([]string, 0, len(obj))
		for k := range obj {
			keys = append(keys, k)
		}
		slices.Sort(keys)
		for _, k := range keys {
			sub, ok := s.Properties[k]
			if !ok {
				if s.Closed {
					fail("unexpected property %q", k)
				}
				continue
			}
			sub.validate(path+"/"+k, obj[k], errs)
		}
	case "array":
		arr, ok := x.([]any)
		if !ok {
			fail("must be an array")
			return
		}
		if s.MinItems != nil && len(arr) < *s.MinItems {
			fail("must have at least %d items, has %d", *s.MinItems, len(arr))
		}
		if s.MaxItems != nil && len(arr) > *s.MaxItems {
			fail("must have at most %d items, has %d", *s.MaxItems, len(arr))
		}
		for i, it := range arr {
			s.Items.validate(fmt.Sprintf("%s/%d", path, i), it, errs)
		}
	case "string":
		str, ok := x.(string)
		if !ok {
			fail("must be a string")
			return
		}
		if len(s.Enum) > 0 && !slices.Contains(s.Enum, str) {
			fail("must be one of %s, got %q", strings.Join(s.Enum, ", "), str)
		}
		n := len([]rune(str))
		if s.MinLen != nil && n < *s.MinLen {
			fail("must be at least %d characters", *s.MinLen)
		}
		if s.MaxLen != nil && n > *s.MaxLen {
			fail("must be at most %d characters", *s.MaxLen)
		}
	case "number", "integer":
		num, ok := x.(json.Number)
		if !ok {
			fail("must be a %s", s.Type)
			return
		}
		f, err := num.Float64()
		if err != nil {
			fail("must be a %s", s.Type)
			return
		}
		if s.Type == "integer" && f != math.Trunc(f) {
			fail("must be an integer")
		}
		if s.Minimum != nil && f < *s.Minimum {
			fail("must be ≥ %g", *s.Minimum)
		}
		if s.Maximum != nil && f > *s.Maximum {
			fail("must be ≤ %g", *s.Maximum)
		}
	case "boolean":
		if _, ok := x.(bool); !ok {
			fail("must be a boolean")
		}
	case "null":
		fail("must be null")
	}
}

// ExtractJSON finds the JSON value in a model's reply: the whole text,
// else the content of a ``` fence, else the first balanced {…} or […]
// that parses. It returns nil when there is none.
func ExtractJSON(text string) json.RawMessage {
	t := strings.TrimSpace(text)
	// Reasoning models sometimes leak a <think>…</think> preamble.
	if i := strings.Index(t, "</think>"); i >= 0 && strings.HasPrefix(t, "<think>") {
		t = strings.TrimSpace(t[i+len("</think>"):])
	}
	if json.Valid([]byte(t)) && t != "" {
		return json.RawMessage(t)
	}
	if i := strings.Index(t, "```"); i >= 0 {
		rest := t[i+3:]
		if nl := strings.IndexByte(rest, '\n'); nl >= 0 {
			rest = rest[nl+1:]
		}
		if j := strings.Index(rest, "```"); j >= 0 {
			if f := strings.TrimSpace(rest[:j]); json.Valid([]byte(f)) {
				return json.RawMessage(f)
			}
		}
	}
	for i := 0; i < len(t); i++ {
		if t[i] != '{' && t[i] != '[' {
			continue
		}
		if end := balanced(t, i); end > i {
			if c := t[i:end]; json.Valid([]byte(c)) {
				return json.RawMessage(c)
			}
		}
	}
	return nil
}

// balanced returns the end (exclusive) of the bracketed value starting at
// t[i], honouring strings, or -1.
func balanced(t string, i int) int {
	depth := 0
	inStr, esc := false, false
	for j := i; j < len(t); j++ {
		c := t[j]
		if inStr {
			switch {
			case esc:
				esc = false
			case c == '\\':
				esc = true
			case c == '"':
				inStr = false
			}
			continue
		}
		switch c {
		case '"':
			inStr = true
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				return j + 1
			}
		}
	}
	return -1
}
