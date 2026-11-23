package main

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/md5"
	"encoding/xml"
	"fmt"
	"image/png"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"time"
)

// Linux configures Linux packaging.
type Linux struct {
	// Maintainer of the Debian package, "Name <email>" (default: the
	// author in package.json, else the app's name).
	Maintainer string `json:"maintainer"`
	// Comment is a short description of the app, for its desktop entry and
	// its package.
	Comment string `json:"comment"`
	// Categories of the desktop entry (default Utility).
	Categories []string `json:"categories"`
	// Depends lists Debian packages the app needs besides GTK and
	// WebKitGTK.
	Depends []string `json:"depends"`
	// Command is the name of a command that runs the app: a link in
	// /usr/bin from the Debian package, and in ~/.local/bin from
	// install.sh. Empty means none, and the app opens from the applications
	// menu.
	Command string `json:"command"`
}

// commandRe matches the names linux.command may take.
var commandRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._+-]*$`)

// debArch maps GOARCH to Debian architectures.
var debArch = map[string]string{"amd64": "amd64", "arm64": "arm64", "386": "i386", "arm": "armhf", "riscv64": "riscv64"}

// debVersion turns a semantic version into a Debian one: a pre-release
// sorts before its release with "~".
func debVersion(v string) string {
	v = strings.TrimPrefix(v, "v")
	if i := strings.IndexByte(v, '-'); i >= 0 {
		v = v[:i] + "~" + strings.ReplaceAll(v[i+1:], "-", ".")
	}
	return strings.ReplaceAll(v, "+", "~")
}

// writeDeb packages the Linux app in stage, the executable name and the
// files next to it, as a Debian package: the app in /opt/<name>, a link in
// /usr/bin named linux.command when there is one, its desktop entry and
// icons. It returns the package's path.
func writeDeb(c *Config, stage, name, goarch string, app []string) (string, error) {
	arch, ok := debArch[goarch]
	if !ok {
		return "", fmt.Errorf("no Debian architecture for %s", goarch)
	}
	data := newTarGz()
	opt := "opt/" + name
	for _, dir := range []string{"opt", opt, "usr", "usr/share", "usr/share/applications"} {
		data.dir(dir)
	}
	size := int64(0)
	md5sums := &bytes.Buffer{}
	for _, entry := range app {
		if strings.HasSuffix(entry, ".desktop") || entry == name+".png" || entry == name+".xml" {
			continue // installed where desktops look for them
		}
		err := filepath.WalkDir(filepath.Join(stage, entry), func(p string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}
			rel, _ := filepath.Rel(stage, p)
			target := opt + "/" + filepath.ToSlash(rel)
			info, err := d.Info()
			if err != nil {
				return err
			}
			switch {
			case d.IsDir():
				data.dir(target)
			case info.Mode()&fs.ModeSymlink != 0:
				link, err := os.Readlink(p)
				if err != nil {
					return err
				}
				data.link(target, link)
			default:
				b, err := os.ReadFile(p)
				if err != nil {
					return err
				}
				mode := int64(0o644)
				if info.Mode()&0o111 != 0 {
					mode = 0o755
				}
				data.file(target, b, mode)
				size += int64(len(b))
				fmt.Fprintf(md5sums, "%x  %s\n", md5.Sum(b), target)
			}
			return nil
		})
		if err != nil {
			return "", err
		}
	}
	if command := c.Linux.Command; command != "" {
		data.dir("usr/bin")
		data.link("usr/bin/"+command, "../../"+opt+"/"+name)
	}
	if xml := mimePackage(c); xml != "" {
		// Types the app defines; the shared-mime-info trigger registers them.
		for _, d := range []string{"usr/share/mime", "usr/share/mime/packages"} {
			data.dir(d)
		}
		data.file("usr/share/mime/packages/"+name+".xml", []byte(xml), 0o644)
		fmt.Fprintf(md5sums, "%x  usr/share/mime/packages/%s.xml\n", md5.Sum([]byte(xml)), name)
	}
	// The entry runs the app by its path, with or without a command.
	desktop := []byte(linuxDesktopEntry(c, "/"+opt+"/"+name, name))
	data.file("usr/share/applications/"+name+".desktop", desktop, 0o644)
	fmt.Fprintf(md5sums, "%x  usr/share/applications/%s.desktop\n", md5.Sum(desktop), name)
	if c.Icon != "" {
		src, err := os.ReadFile(c.path(c.Icon))
		if err != nil {
			return "", err
		}
		img, err := png.Decode(bytes.NewReader(src))
		if err != nil {
			return "", fmt.Errorf("icon %s: %w", c.Icon, err)
		}
		for _, px := range []int{256, 512} {
			dir := fmt.Sprintf("usr/share/icons/hicolor/%dx%d/apps", px, px)
			for _, d := range []string{"usr/share/icons", "usr/share/icons/hicolor", path.Dir(dir), dir} {
				data.dir(d)
			}
			var b bytes.Buffer
			if err := png.Encode(&b, resize(img, px)); err != nil {
				return "", err
			}
			data.file(dir+"/"+name+".png", b.Bytes(), 0o644)
			fmt.Fprintf(md5sums, "%x  %s/%s.png\n", md5.Sum(b.Bytes()), dir, name)
		}
	}

	depends := append([]string{"libgtk-3-0 | libgtk-3-0t64", "libwebkit2gtk-4.1-0"}, c.Linux.Depends...)
	description := c.Linux.Comment
	if description == "" {
		description = c.Name
	}
	control := fmt.Sprintf("Package: %s\nVersion: %s\nArchitecture: %s\nMaintainer: %s\nInstalled-Size: %d\nDepends: %s\nSection: utils\nPriority: optional\nDescription: %s\n %s is a desktop application.\n",
		name, debVersion(c.Version), arch, c.Linux.Maintainer, (size+1023)/1024, strings.Join(depends, ", "), description, c.Name)
	ctl := newTarGz()
	ctl.file("control", []byte(control), 0o644)
	ctl.file("md5sums", md5sums.Bytes(), 0o644)

	controlTar, err := ctl.bytes()
	if err != nil {
		return "", err
	}
	dataTar, err := data.bytes()
	if err != nil {
		return "", err
	}
	var deb bytes.Buffer
	deb.WriteString("!<arch>\n")
	for _, m := range []struct {
		name string
		data []byte
	}{{"debian-binary", []byte("2.0\n")}, {"control.tar.gz", controlTar}, {"data.tar.gz", dataTar}} {
		fmt.Fprintf(&deb, "%-16s%-12d%-6d%-6d%-8s%-10d`\n", m.name, time.Now().Unix(), 0, 0, "100644", len(m.data))
		deb.Write(m.data)
		if len(m.data)%2 == 1 {
			deb.WriteByte('\n')
		}
	}
	out := filepath.Join(stage, fmt.Sprintf("%s_%s_%s.deb", name, debVersion(c.Version), arch))
	return out, os.WriteFile(out, deb.Bytes(), 0o644)
}

// mimePackage returns the shared-mime-info package defining the MIME types
// of the file associations, or "" when they all have one.
func mimePackage(c *Config) string {
	var b strings.Builder
	for _, fa := range c.FileAssociations {
		t, defined := c.mimeType(fa)
		if !defined {
			continue
		}
		fmt.Fprintf(&b, "  <mime-type type=\"%s\">\n", t)
		if fa.Name != "" {
			fmt.Fprintf(&b, "    <comment>%s</comment>\n", xmlEscape(fa.Name))
		}
		for _, ext := range fa.Ext {
			fmt.Fprintf(&b, "    <glob pattern=\"*.%s\"/>\n", ext)
		}
		b.WriteString("  </mime-type>\n")
	}
	if b.Len() == 0 {
		return ""
	}
	return "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n" + b.String() + "</mime-info>\n"
}

func xmlEscape(s string) string {
	var b bytes.Buffer
	_ = xml.EscapeText(&b, []byte(s))
	return b.String()
}

// tarGz builds a gzip compressed tar archive in memory, with entries under
// "./" owned by root, as Debian packages have them.
type tarGz struct {
	buf  bytes.Buffer
	gz   *gzip.Writer
	tw   *tar.Writer
	dirs []string
	err  error
}

func newTarGz() *tarGz {
	t := &tarGz{}
	t.gz, _ = gzip.NewWriterLevel(&t.buf, gzip.BestCompression)
	t.tw = tar.NewWriter(t.gz)
	return t
}

func (t *tarGz) write(hdr *tar.Header, data []byte) {
	if t.err != nil {
		return
	}
	hdr.Name = "./" + hdr.Name
	hdr.ModTime = time.Now()
	hdr.Uname, hdr.Gname = "root", "root"
	if t.err = t.tw.WriteHeader(hdr); t.err == nil && data != nil {
		_, t.err = t.tw.Write(data)
	}
}

func (t *tarGz) dir(name string) {
	if slices.Contains(t.dirs, name) {
		return
	}
	t.dirs = append(t.dirs, name)
	t.write(&tar.Header{Typeflag: tar.TypeDir, Name: name + "/", Mode: 0o755}, nil)
}

func (t *tarGz) file(name string, data []byte, mode int64) {
	t.write(&tar.Header{Typeflag: tar.TypeReg, Name: name, Mode: mode, Size: int64(len(data))}, data)
}

func (t *tarGz) link(name, target string) {
	t.write(&tar.Header{Typeflag: tar.TypeSymlink, Name: name, Linkname: target, Mode: 0o777}, nil)
}

func (t *tarGz) bytes() ([]byte, error) {
	if t.err != nil {
		return nil, t.err
	}
	if err := t.tw.Close(); err != nil {
		return nil, err
	}
	if err := t.gz.Close(); err != nil {
		return nil, err
	}
	return t.buf.Bytes(), nil
}
