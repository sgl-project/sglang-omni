package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"os"
	"slices"
	"strconv"
	"strings"
	"unicode/utf16"
)

// winresSyso builds a COFF object (.syso) with the Windows resources of the
// app: its icon, a manifest (per-monitor DPI awareness, common controls 6,
// Windows 10 and 11) and version information, which the backend reads to
// tell that the app was built by mygo build. The Go linker links the .syso
// files of the main package into the executable.
func winresSyso(c *Config, goarch string) ([]byte, error) {
	// The COFF machine and the relocation type of relative addresses.
	arch, ok := map[string]struct{ machine, relocation uint16 }{
		"amd64": {0x8664, 0x0003}, // IMAGE_REL_AMD64_ADDR32NB
		"arm64": {0xAA64, 0x0002}, // IMAGE_REL_ARM64_ADDR32NB
		"386":   {0x014C, 0x0007}, // IMAGE_REL_I386_DIR32NB
	}[goarch]
	if !ok {
		return nil, fmt.Errorf("no Windows resources for %s", goarch)
	}
	machine, relocation := arch.machine, arch.relocation

	const (
		rtIcon      = 3
		rtGroupIcon = 14
		rtVersion   = 16
		rtManifest  = 24
	)
	appLangs, err := c.languages()
	if err != nil {
		return nil, err
	}
	// The resources are in the app's own language, the first.
	langs := windowsLanguages(c, appLangs)
	res := map[uint16]map[uint16][]byte{
		rtVersion:  {1: versionInfo(c, langs)},
		rtManifest: {1: []byte(windowsManifest(c))},
	}
	if c.Icon != "" {
		src, err := os.ReadFile(c.path(c.Icon))
		if err != nil {
			return nil, err
		}
		images, err := iconImages(src)
		if err != nil {
			return nil, fmt.Errorf("icon: %w", err)
		}
		res[rtIcon] = map[uint16][]byte{}
		var group bytes.Buffer
		_ = binary.Write(&group, binary.LittleEndian, [3]uint16{0, 1, uint16(len(images))})
		for i, img := range images {
			id := uint16(i + 1)
			res[rtIcon][id] = img
			dim := byte(windowsIconSizes[i])
			if windowsIconSizes[i] >= 256 {
				dim = 0
			}
			group.Write([]byte{dim, dim, 0, 0})
			_ = binary.Write(&group, binary.LittleEndian, [2]uint16{1, 32})
			_ = binary.Write(&group, binary.LittleEndian, uint32(len(img)))
			_ = binary.Write(&group, binary.LittleEndian, id)
		}
		res[rtGroupIcon] = map[uint16][]byte{1: group.Bytes()}
	}

	// The resource section: three levels of directories (type, id,
	// language), the data entries, then the data. Directory entries point
	// to subdirectories by section offset; data entries hold relative
	// virtual addresses, which the linker computes from relocations.
	types := sortedKeys(res)
	dirSize := func(n int) int { return 16 + 8*n }
	size := dirSize(len(types))
	for _, t := range types {
		size += dirSize(len(res[t])) + len(res[t])*dirSize(1)
	}
	entriesAt := size
	count := 0
	for _, t := range types {
		count += len(res[t])
	}
	dataAt := entriesAt + 16*count

	raw := make([]byte, dataAt)
	le := binary.LittleEndian
	dir := func(at, n int) {
		le.PutUint16(raw[at+14:], uint16(n)) // NumberOfIdEntries
	}
	var relocs []uint32
	var data bytes.Buffer
	dir(0, len(types))
	next := dirSize(len(types))
	entry := entriesAt
	for i, t := range types {
		le.PutUint32(raw[16+8*i:], uint32(t))
		le.PutUint32(raw[16+8*i+4:], uint32(next)|0x80000000)
		ids := sortedKeys(res[t])
		idDir := next
		dir(idDir, len(ids))
		next += dirSize(len(ids))
		for j, id := range ids {
			le.PutUint32(raw[idDir+16+8*j:], uint32(id))
			le.PutUint32(raw[idDir+16+8*j+4:], uint32(next)|0x80000000)
			dir(next, 1)
			le.PutUint32(raw[next+16:], uint32(langs[0].id))
			le.PutUint32(raw[next+20:], uint32(entry))
			next += dirSize(1)

			for data.Len()%8 != 0 {
				data.WriteByte(0)
			}
			le.PutUint32(raw[entry:], uint32(dataAt+data.Len())) // OffsetToData
			le.PutUint32(raw[entry+4:], uint32(len(res[t][id])))
			relocs = append(relocs, uint32(entry))
			entry += 16
			data.Write(res[t][id])
		}
	}
	raw = append(raw, data.Bytes()...)
	for len(raw)%4 != 0 {
		raw = append(raw, 0)
	}

	var out bytes.Buffer
	const headerSize, sectionHeaderSize = 20, 40
	rawAt := headerSize + sectionHeaderSize
	relocAt := rawAt + len(raw)
	symbolsAt := relocAt + 10*len(relocs)
	_ = binary.Write(&out, le, struct {
		Machine, Sections                   uint16
		Time, Symbols, SymbolCount          uint32
		OptionalHeaderSize, Characteristics uint16
	}{machine, 1, 0, uint32(symbolsAt), 1, 0, 0})
	_ = binary.Write(&out, le, struct {
		Name                                        [8]byte
		VirtualSize, VirtualAddress, RawSize, RawAt uint32
		RelocAt, LineNumbersAt                      uint32
		Relocs, LineNumbers                         uint16
		Characteristics                             uint32
	}{[8]byte{'.', 'r', 's', 'r', 'c'}, 0, 0, uint32(len(raw)), uint32(rawAt), uint32(relocAt), 0, uint16(len(relocs)), 0, 0x40000040})
	out.Write(raw)
	for _, r := range relocs {
		_ = binary.Write(&out, le, r)         // VirtualAddress
		_ = binary.Write(&out, le, uint32(0)) // symbol 0: the section
		_ = binary.Write(&out, le, relocation)
	}
	// The section symbol, then an empty string table.
	out.Write([]byte{'.', 'r', 's', 'r', 'c', 0, 0, 0})
	_ = binary.Write(&out, le, uint32(0)) // Value
	_ = binary.Write(&out, le, uint16(1)) // SectionNumber
	_ = binary.Write(&out, le, uint16(0)) // Type
	out.Write([]byte{3, 0})               // IMAGE_SYM_CLASS_STATIC, no aux symbols
	_ = binary.Write(&out, le, uint32(4))
	return out.Bytes(), nil
}

func sortedKeys[V any](m map[uint16]V) []uint16 {
	keys := make([]uint16, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// windowsVersion turns a version such as "1.2.3" into four numbers.
func windowsVersion(v string) [4]uint16 {
	var out [4]uint16
	v, _, _ = strings.Cut(v, "-")
	for i, part := range strings.SplitN(v, ".", 4) {
		n, _ := strconv.Atoi(part)
		out[i] = uint16(n)
	}
	return out
}

func windowsManifest(c *Config) string {
	v := windowsVersion(c.Version)
	name := strings.Map(func(r rune) rune {
		if r >= 'a' && r <= 'z' || r >= 'A' && r <= 'Z' || r >= '0' && r <= '9' || r == '.' || r == '-' {
			return r
		}
		return -1
	}, c.Identifier)
	return fmt.Sprintf(`<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity type="win32" name="%s" version="%d.%d.%d.%d" processorArchitecture="*"/>
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*"/>
    </dependentAssembly>
  </dependency>
  <compatibility xmlns="urn:schemas-microsoft-com:compatibility.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
  <application xmlns="urn:schemas-microsoft-com:asm.v3">
    <windowsSettings>
      <dpiAware xmlns="http://schemas.microsoft.com/SMI/2005/WindowsSettings">true/pm</dpiAware>
      <dpiAwareness xmlns="http://schemas.microsoft.com/SMI/2016/WindowsSettings">PerMonitorV2</dpiAwareness>
      <activeCodePage xmlns="http://schemas.microsoft.com/SMI/2019/WindowsSettings">UTF-8</activeCodePage>
    </windowsSettings>
  </application>
</assembly>
`, name, v[0], v[1], v[2], v[3])
}

// versionInfo encodes a VS_VERSIONINFO resource, which holds the strings of
// the app in each of its languages, keyed by the language and the Unicode
// code page: Windows shows those of the system's language, else the first
// language's, the app's own.
func versionInfo(c *Config, langs []windowsLanguage) []byte {
	v := windowsVersion(c.Version)
	fixed := make([]byte, 52)
	le := binary.LittleEndian
	le.PutUint32(fixed[0:], 0xFEEF04BD)                    // dwSignature
	le.PutUint32(fixed[4:], 0x00010000)                    // dwStrucVersion
	le.PutUint32(fixed[8:], uint32(v[0])<<16|uint32(v[1])) // dwFileVersionMS
	le.PutUint32(fixed[12:], uint32(v[2])<<16|uint32(v[3]))
	copy(fixed[16:24], fixed[8:16])      // dwProductVersion
	le.PutUint32(fixed[24:], 0x3F)       // dwFileFlagsMask
	le.PutUint32(fixed[32:], 0x00040004) // dwFileOS: VOS_NT_WINDOWS32
	le.PutUint32(fixed[36:], 1)          // dwFileType: VFT_APP

	exe := c.executableName() + ".exe"
	var tables []verNode
	var translation []byte
	for _, l := range langs {
		var table []verNode
		for _, s := range [][2]string{
			{"FileDescription", l.name},
			{"FileVersion", c.Version},
			{"InternalName", exe},
			{"LegalCopyright", c.Copyright},
			{"OriginalFilename", exe},
			{"ProductName", l.name},
			{"ProductVersion", c.Version},
			{"MyGoIdentifier", c.Identifier},
		} {
			if s[1] == "" {
				continue
			}
			text := utf16.Encode([]rune(s[1] + "\x00"))
			value := make([]byte, 2*len(text))
			for i, u := range text {
				le.PutUint16(value[2*i:], u)
			}
			table = append(table, verNode{key: s[0], text: true, value: value, valueLen: len(text)})
		}
		// A table's key is its language and code page, 1200 for Unicode,
		// as 040904b0 for en-US.
		tables = append(tables, verNode{key: fmt.Sprintf("%04x%04x", l.id, 1200), text: true, children: table})
		translation = append(translation, byte(l.id), byte(l.id>>8), 0xB0, 0x04)
	}
	root := verNode{key: "VS_VERSION_INFO", value: fixed, valueLen: len(fixed), children: []verNode{
		{key: "StringFileInfo", text: true, children: tables},
		{key: "VarFileInfo", text: true, children: []verNode{{key: "Translation", value: translation, valueLen: len(translation)}}},
	}}
	return root.encode()
}

// verNode is a node of a version resource: a length, the length of its
// value, its type, a UTF-16 key, the value and child nodes, each aligned
// on 32 bits.
type verNode struct {
	key      string
	text     bool
	value    []byte
	valueLen int // in bytes for binary values, in characters for text
	children []verNode
}

func (n verNode) encode() []byte {
	b := make([]byte, 6)
	for _, u := range utf16.Encode([]rune(n.key + "\x00")) {
		b = binary.LittleEndian.AppendUint16(b, u)
	}
	pad := func() {
		for len(b)%4 != 0 {
			b = append(b, 0)
		}
	}
	pad()
	b = append(b, n.value...)
	for _, child := range n.children {
		pad()
		b = append(b, child.encode()...)
	}
	binary.LittleEndian.PutUint16(b[0:], uint16(len(b)))
	binary.LittleEndian.PutUint16(b[2:], uint16(n.valueLen))
	if n.text {
		binary.LittleEndian.PutUint16(b[4:], 1)
	}
	return b
}
