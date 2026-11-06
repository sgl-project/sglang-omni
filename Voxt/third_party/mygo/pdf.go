package mygo

import "github.com/egoist/mygo/internal/platform"

// PDFOptions configures Page.PrintToPDF. The zero value prints Letter
// pages in portrait orientation with margins of 0.4 inch and without
// backgrounds, like a browser's print dialog.
type PDFOptions struct {
	// PageSize is the size of the paper; the zero value is PageLetter.
	PageSize  PageSize
	Landscape bool
	// Margins surround the content of every page; nil means 0.4 inch on
	// every side, &Margins{} none.
	Margins *Margins
	// Background prints background colors and images, which printing
	// leaves out by default.
	Background bool
}

// PageSize is the size of a sheet of paper in inches, in portrait
// orientation.
type PageSize struct{ Width, Height float64 }

// Paper sizes.
var (
	PageLetter  = PageSize{8.5, 11}
	PageLegal   = PageSize{8.5, 14}
	PageTabloid = PageSize{11, 17}
	PageA3      = PageSize{11.69, 16.54}
	PageA4      = PageSize{8.27, 11.69}
	PageA5      = PageSize{5.83, 8.27}
)

// Margins are page margins in inches.
type Margins struct{ Top, Right, Bottom, Left float64 }

// PrintToPDF renders the page as a PDF document, laid out for printing
// (the page's print style sheets apply) on pages of the given size:
//
//	pdf, err := win.Page().PrintToPDF(mygo.PDFOptions{PageSize: mygo.PageA4, Background: true})
func (p *Page) PrintToPDF(opts PDFOptions) ([]byte, error) {
	if p.w.content != nil {
		return nil, errNoPage
	}
	size := opts.PageSize
	if size.Width <= 0 || size.Height <= 0 {
		size = PageLetter
	}
	m := Margins{0.4, 0.4, 0.4, 0.4}
	if opts.Margins != nil {
		m = *opts.Margins
	}
	pdfOpts := platform.PDFOptions{
		Landscape: opts.Landscape, PageWidth: size.Width, PageHeight: size.Height,
		MarginTop: m.Top, MarginRight: m.Right, MarginBottom: m.Bottom, MarginLeft: m.Left,
		Background: opts.Background,
	}
	type result struct {
		pdf []byte
		err error
	}
	ch := make(chan result, 1)
	p.w.do(func(n platform.Window) {
		n.PrintToPDF(pdfOpts, func(pdf []byte, err error) { deliver(ch, result{pdf, err}) })
	})
	if p.w.IsDestroyed() {
		return nil, errDestroyed
	}
	r := await(ch)
	return r.pdf, r.err
}
