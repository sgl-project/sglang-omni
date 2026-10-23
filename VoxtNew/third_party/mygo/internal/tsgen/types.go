package tsgen

import (
	"encoding"
	jsonv1 "encoding/json"
	"encoding/json/jsontext"
	"encoding/json/v2"
	"fmt"
	"reflect"
	"strings"
	"time"
)

var (
	timeType          = reflect.TypeFor[time.Time]()
	durationType      = reflect.TypeFor[time.Duration]()
	rawV1Type         = reflect.TypeFor[jsonv1.RawMessage]()
	rawV2Type         = reflect.TypeFor[jsontext.Value]()
	jsonMarshaler     = reflect.TypeFor[jsonv1.Marshaler]()
	jsonMarshalerTo   = reflect.TypeFor[json.MarshalerTo]()
	textMarshaler     = reflect.TypeFor[encoding.TextMarshaler]()
	textAppender      = reflect.TypeFor[encoding.TextAppender]()
	emptyStructFields = []field{}
)

func implements(t, iface reflect.Type) bool {
	return t.Implements(iface) || (t.Kind() != reflect.Pointer && reflect.PointerTo(t).Implements(iface))
}

// special returns the TypeScript type of Go types with a custom JSON
// representation.
func special(t reflect.Type) (string, bool) {
	switch t {
	case timeType:
		return "string", true
	case durationType:
		return "number", true
	case rawV1Type, rawV2Type:
		return "unknown", true
	}
	if implements(t, jsonMarshaler) || implements(t, jsonMarshalerTo) {
		return "unknown", true
	}
	if implements(t, textMarshaler) || implements(t, textAppender) {
		return "string", true
	}
	if (t.Kind() == reflect.Slice || t.Kind() == reflect.Array) && t.Elem().Kind() == reflect.Uint8 {
		// Byte slices and arrays are base64 strings.
		return "string", true
	}
	return "", false
}

func isNumber(k reflect.Kind) bool {
	switch k {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
		reflect.Float32, reflect.Float64:
		return true
	}
	return false
}

// Validate reports an error if values of type t cannot cross the IPC
// boundary as JSON.
func Validate(t reflect.Type) error {
	return validate(t, map[reflect.Type]bool{})
}

func validate(t reflect.Type, seen map[reflect.Type]bool) error {
	if seen[t] {
		return nil
	}
	seen[t] = true
	if _, ok := special(t); ok {
		return nil
	}
	switch t.Kind() {
	case reflect.Chan, reflect.Func, reflect.Complex64, reflect.Complex128, reflect.UnsafePointer:
		return fmt.Errorf("type %s cannot be encoded as JSON", t)
	case reflect.Pointer, reflect.Slice, reflect.Array:
		return validate(t.Elem(), seen)
	case reflect.Map:
		k := t.Key()
		if k.Kind() != reflect.String && !isNumber(k.Kind()) && !implements(k, textMarshaler) {
			return fmt.Errorf("map key type %s cannot be encoded as JSON", k)
		}
		return validate(t.Elem(), seen)
	case reflect.Struct:
		for _, f := range structFields(t) {
			if err := validate(f.typ, seen); err != nil {
				return fmt.Errorf("field %s: %w", f.goName, err)
			}
		}
	}
	return nil
}

// field is a JSON representable struct field.
type field struct {
	name     string
	typ      reflect.Type
	optional bool
	quoted   bool
	goName   string
	owner    reflect.Type
	depth    int
	tagged   bool
}

type tagOptions []string

func (o tagOptions) has(opt string) bool {
	for _, x := range o {
		if x == opt {
			return true
		}
	}
	return false
}

func parseTag(tag string) (string, tagOptions) {
	name, rest, _ := strings.Cut(tag, ",")
	if strings.HasPrefix(name, "'") && strings.HasSuffix(name, "'") && len(name) >= 2 {
		// json/v2 allows single quoted names.
		name = name[1 : len(name)-1]
	}
	if rest == "" {
		return name, nil
	}
	return name, strings.Split(rest, ",")
}

// structFields lists the JSON members of a struct following encoding/json/v2
// rules: embedded structs are inlined and shallower fields win conflicts.
func structFields(t reflect.Type) []field {
	var all []field
	var walk func(t reflect.Type, depth int, optional bool, visiting map[reflect.Type]bool)
	walk = func(t reflect.Type, depth int, optional bool, visiting map[reflect.Type]bool) {
		if visiting[t] {
			return
		}
		visiting[t] = true
		defer delete(visiting, t)
		for i := range t.NumField() {
			sf := t.Field(i)
			tag := sf.Tag.Get("json")
			if tag == "-" {
				continue
			}
			name, opts := parseTag(tag)
			// Go embedding implies JSON embedding unless the field is named;
			// the "embed" option requests it explicitly.
			if (sf.Anonymous && name == "") || opts.has("embed") {
				et, viaPtr := sf.Type, false
				if et.Kind() == reflect.Pointer && et.Name() == "" {
					et, viaPtr = et.Elem(), true
				}
				if et.Kind() == reflect.Struct {
					if _, custom := special(et); !custom {
						walk(et, depth+1, optional || viaPtr, visiting)
						continue
					}
				}
			}
			if !sf.IsExported() {
				continue
			}
			f := field{
				name:   name,
				typ:    sf.Type,
				goName: sf.Name,
				owner:  t,
				depth:  depth,
				tagged: name != "",
			}
			if f.name == "" {
				f.name = sf.Name
			}
			f.optional = optional || opts.has("omitempty") || opts.has("omitzero")
			f.quoted = opts.has("string")
			all = append(all, f)
		}
	}
	walk(t, 0, false, map[reflect.Type]bool{})
	if len(all) == 0 {
		return emptyStructFields
	}

	byName := map[string][]int{}
	for i, f := range all {
		byName[f.name] = append(byName[f.name], i)
	}
	keep := make([]bool, len(all))
	for _, idx := range byName {
		minDepth := all[idx[0]].depth
		for _, i := range idx {
			minDepth = min(minDepth, all[i].depth)
		}
		var shallow, tagged []int
		for _, i := range idx {
			if all[i].depth == minDepth {
				shallow = append(shallow, i)
				if all[i].tagged {
					tagged = append(tagged, i)
				}
			}
		}
		switch {
		case len(shallow) == 1:
			keep[shallow[0]] = true
		case len(tagged) == 1:
			keep[tagged[0]] = true
		}
	}
	out := make([]field, 0, len(all))
	for i, f := range all {
		if keep[i] {
			out = append(out, f)
		}
	}
	return out
}
