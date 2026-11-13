package registry

import (
	"bytes"
	"encoding/binary"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"unicode/utf16"
)

// BundleIDAt reads CFBundleIdentifier from <app>/Contents/Info.plist
// (XML or binary property list).
func BundleIDAt(appPath string) (string, error) {
	data, err := os.ReadFile(filepath.Join(appPath, "Contents", "Info.plist"))
	if err != nil {
		return "", err
	}
	if len(data) > 4<<20 {
		return "", errors.New("the Info.plist is too large")
	}
	strs, err := plistStrings(data)
	if err != nil {
		return "", err
	}
	id, ok := strs["CFBundleIdentifier"]
	if !ok {
		return "", errors.New("no CFBundleIdentifier in the Info.plist")
	}
	return id, nil
}

// plistStrings returns the string members of a property list's top-level
// dictionary.
func plistStrings(data []byte) (map[string]string, error) {
	if bytes.HasPrefix(data, []byte("bplist00")) {
		return binaryPlistStrings(data)
	}
	return xmlPlistStrings(data)
}

func xmlPlistStrings(data []byte) (map[string]string, error) {
	dec := xml.NewDecoder(bytes.NewReader(data))
	dec.Strict = false
	out := map[string]string{}
	depth := 0 // nesting of <dict>/<array> below the root dict
	inRoot := false
	key := ""
	var text bytes.Buffer
	for {
		tok, err := dec.Token()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, err
		}
		switch t := tok.(type) {
		case xml.StartElement:
			text.Reset()
			switch t.Name.Local {
			case "dict":
				if !inRoot {
					inRoot = true
				} else {
					depth++
				}
			case "array":
				depth++
			}
		case xml.CharData:
			text.Write(t)
		case xml.EndElement:
			switch t.Name.Local {
			case "dict", "array":
				if depth > 0 {
					depth--
				}
				key = ""
			case "key":
				if depth == 0 {
					key = text.String()
				}
			case "string":
				if depth == 0 && key != "" {
					out[key] = text.String()
				}
				key = ""
			default:
				if depth == 0 {
					key = ""
				}
			}
		}
	}
	if !inRoot {
		return nil, errors.New("not a property list dictionary")
	}
	return out, nil
}

// binaryPlistStrings reads a bplist00's top-level dictionary.
func binaryPlistStrings(data []byte) (map[string]string, error) {
	if len(data) < 40 {
		return nil, errors.New("binary plist too short")
	}
	tr := data[len(data)-32:]
	offSize, refSize := int(tr[6]), int(tr[7])
	numObjects := binary.BigEndian.Uint64(tr[8:16])
	top := binary.BigEndian.Uint64(tr[16:24])
	tableOff := binary.BigEndian.Uint64(tr[24:32])
	if offSize < 1 || offSize > 8 || refSize < 1 || refSize > 8 || numObjects > 1<<20 || top >= numObjects ||
		tableOff+numObjects*uint64(offSize) > uint64(len(data)) {
		return nil, errors.New("bad binary plist trailer")
	}
	readInt := func(b []byte) uint64 {
		var v uint64
		for _, c := range b {
			v = v<<8 | uint64(c)
		}
		return v
	}
	offset := func(obj uint64) (int, error) {
		if obj >= numObjects {
			return 0, errors.New("object out of range")
		}
		p := tableOff + obj*uint64(offSize)
		o := readInt(data[p : p+uint64(offSize)])
		if o >= uint64(len(data)) {
			return 0, errors.New("offset out of range")
		}
		return int(o), nil
	}
	// length reads an object's count: the low nibble, or a following int.
	length := func(pos int) (n int, start int, err error) {
		n = int(data[pos] & 0x0f)
		start = pos + 1
		if n != 0x0f {
			return n, start, nil
		}
		if start >= len(data) || data[start]>>4 != 0x1 {
			return 0, 0, errors.New("bad length")
		}
		size := 1 << (data[start] & 0x0f)
		if start+1+size > len(data) || size > 8 {
			return 0, 0, errors.New("bad length")
		}
		return int(readInt(data[start+1 : start+1+size])), start + 1 + size, nil
	}
	str := func(obj uint64) (string, bool, error) {
		pos, err := offset(obj)
		if err != nil {
			return "", false, err
		}
		kind := data[pos] >> 4
		if kind != 0x5 && kind != 0x6 {
			return "", false, nil
		}
		n, start, err := length(pos)
		if err != nil {
			return "", false, err
		}
		if kind == 0x5 {
			if start+n > len(data) {
				return "", false, errors.New("string out of range")
			}
			return string(data[start : start+n]), true, nil
		}
		if start+2*n > len(data) {
			return "", false, errors.New("string out of range")
		}
		u := make([]uint16, n)
		for i := range u {
			u[i] = binary.BigEndian.Uint16(data[start+2*i:])
		}
		return string(utf16.Decode(u)), true, nil
	}
	pos, err := offset(top)
	if err != nil {
		return nil, err
	}
	if data[pos]>>4 != 0xd {
		return nil, fmt.Errorf("top object is not a dictionary (0x%x)", data[pos])
	}
	n, start, err := length(pos)
	if err != nil {
		return nil, err
	}
	if start+2*n*refSize > len(data) {
		return nil, errors.New("dictionary out of range")
	}
	out := map[string]string{}
	for i := range n {
		kref := readInt(data[start+i*refSize : start+(i+1)*refSize])
		vref := readInt(data[start+(n+i)*refSize : start+(n+i+1)*refSize])
		k, ok, err := str(kref)
		if err != nil || !ok {
			continue
		}
		v, ok, err := str(vref)
		if err != nil || !ok {
			continue
		}
		out[k] = v
	}
	return out, nil
}
