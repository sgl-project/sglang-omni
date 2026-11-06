package markdown

import (
	"github.com/egoist/mygo/ui"
	"github.com/egoist/mygo/ui/markdown/parser"
)

// tableState keeps a table's column widths, which only grow while the
// table is being written, so that its columns do not jump as its cells
// fill.
type tableState struct {
	widths []float32
	seen   bool
}

// table builds a table: rows of cells as wide as their column's widest,
// up to Theme.TableMaxColumn, which wrap beyond, scrolling sideways when
// wider than the view.
func (st *viewState) table(c *ui.Context, i int, base spanStyle) ui.Element {
	d := st.doc
	th := &st.th
	b := d.Block(i)
	if st.tables == nil {
		st.tables = map[int]*tableState{}
	}
	ts := st.tables[b.Start]
	if ts == nil {
		ts = &tableState{}
		st.tables[b.Start] = ts
	}
	ts.seen = true
	cols := len(b.Align)
	padX, padY := th.em(th.TableCellPadding[0]), th.em(th.TableCellPadding[1])
	maxW := th.em(th.TableMaxColumn)
	// The widths of the columns, from their cells' text.
	widths := make([]float32, cols)
	if b.Open {
		copy(widths, ts.widths)
	}
	for r := i + 1; r < d.Block(i).Next; r = d.Block(r).Next {
		row := d.Block(r)
		k := 0
		for cell := r + 1; cell < d.Block(r).Next; cell = d.Block(cell).Next {
			if k >= cols {
				break
			}
			s := base
			if row.Header {
				s.weight = th.StrongWeight
			}
			p := st.para(cell, s)
			if !p.measured {
				p.width, p.measured = 0, true
				for _, pc := range p.pieces {
					pw, _ := c.MeasureText(0, pc.spans...)
					p.width += pw
				}
			}
			widths[k] = max(widths[k], min(p.width, maxW)+2*padX+1)
			k++
		}
	}
	ts.widths = append(ts.widths[:0], widths...)
	scroll := ui.ScrollHorizontal(c).AlignSelf(ui.Stretch)
	return scroll.Children(func() {
		grid := ui.Column(c).Border(1, th.TableBorder).Radius(th.em(0.3)).Clip().Shrink(0).Role(ui.RoleTable)
		grid.Children(func() {
			n := 0
			for r := i + 1; r < d.Block(i).Next; r = d.Block(r).Next {
				row := d.Block(r)
				rowEl := ui.Row(c).Role(ui.RoleRow).AlignItems(ui.Stretch)
				if row.Header {
					rowEl.Background(th.TableHeader)
				} else if n%2 == 1 && th.TableStripe.A > 0 {
					rowEl.Background(th.TableStripe)
				}
				if n > 0 {
					rowEl.BorderWidth(1, 0, 0, 0).BorderColor(th.TableBorder)
				}
				if st.fade != nil && b.Open {
					if a := st.fade(row.Start); a < 1 {
						rowEl.Opacity(a)
					}
				}
				header := row.Header
				rowEl.Children(func() {
					k := 0
					for cell := r + 1; cell < d.Block(r).Next; cell = d.Block(cell).Next {
						if k >= cols {
							break
						}
						s := base
						role := ui.RoleCell
						if header {
							s.weight = th.StrongWeight
							role = ui.RoleColumnHeader
						}
						box := ui.Column(c).Width(widths[k]).Padding(padY, padX).Role(role)
						if k > 0 {
							box.BorderWidth(0, 0, 0, 1).BorderColor(th.TableBorder)
						}
						align := ui.Start
						switch b.Align[k] {
						case parser.AlignCenter:
							align = ui.Center
						case parser.AlignRight:
							align = ui.End
						}
						box.Children(func() { st.paragraph(c, cell, s).TextAlign(align) })
						k++
					}
				})
				n++
			}
		})
	})
}
