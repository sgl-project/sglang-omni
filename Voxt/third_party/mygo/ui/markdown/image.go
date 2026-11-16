package markdown

import (
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/egoist/mygo/internal/l10n"
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
)

// imageEntry is an image loading or loaded.
type imageEntry struct {
	mu   sync.Mutex
	src  ui.ImageSource
	err  error
	done bool
}

// FileImages returns a LoadImage for the files of directory dir, which
// paths without a scheme are relative to: PNG, JPEG, GIF, WebP, BMP and
// SVG. Other addresses do not load.
func FileImages(dir string) func(url string) (ui.ImageSource, error) {
	return func(url string) (ui.ImageSource, error) {
		if strings.Contains(url, "://") || strings.HasPrefix(url, "data:") {
			return nil, os.ErrNotExist
		}
		path := strings.TrimPrefix(url, "file:")
		if !filepath.IsAbs(path) {
			path = filepath.Join(dir, filepath.FromSlash(path))
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, err
		}
		if strings.EqualFold(filepath.Ext(path), ".svg") {
			return ui.ParseSVG(data)
		}
		return ui.DecodeBitmap(data)
	}
}

// image returns the image at url once loaded, starting to load it the
// first time.
func (st *viewState) image(c *ui.Context, url string) ui.ImageSource {
	o := st.o
	if o.LoadImage == nil || url == "" {
		return nil
	}
	v, loaded := o.images.LoadOrStore(url, &imageEntry{})
	e := v.(*imageEntry)
	if !loaded {
		load, services := o.LoadImage, c.Services()
		go func() {
			src, err := load(url)
			e.mu.Lock()
			e.src, e.err, e.done = src, err, true
			e.mu.Unlock()
			services.Invalidate()
		}()
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.src
}

// images builds a paragraph of images alone: each as large as it is, at
// most as wide as the view, its description showing until it loads.
func (st *viewState) images(c *ui.Context, i int) ui.Element {
	th := &st.th
	in := st.doc.Inlines(i)
	row := ui.Row(c).Wrap().Gap(th.em(0.5)).AlignSelf(ui.Stretch)
	return row.Children(func() {
		for k := 0; k < len(in); k = in[k].Next {
			n := &in[k]
			if n.Kind != parser.Image {
				continue
			}
			alt := parser.PlainText(in[k+1 : n.Next])
			src := st.image(c, n.Dest)
			if src == nil || n.Pending {
				label := alt
				if label == "" {
					label = l10n.In(c.Locale()).Image()
				}
				ph := ui.Row(c).Padding(th.em(0.5), th.em(0.75)).Radius(th.em(0.3)).Background(th.CodeBackground)
				ph.Role(ui.RoleImage).Label(alt)
				ph.Children(func() { ui.Text(c, label).TextColor(th.Muted).FontSize(th.em(0.875)) })
				continue
			}
			img := ui.Image(c, src).Radius(th.em(0.3)).MaxWidthPercent(100).Label(alt)
			var w, h float32
			switch s := src.(type) {
			case *ui.Bitmap:
				iw, ih := s.Size()
				w, h = float32(iw), float32(ih)
			case *ui.SVG:
				w, h = s.Size()
			}
			if w > 0 && h > 0 {
				img.AspectRatio(w / h)
			}
			if n.Title != "" {
				img.Tooltip(n.Title)
			}
		}
	})
}
