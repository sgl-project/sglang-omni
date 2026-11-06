package vt

import "unsafe"

// RGB is a GhosttyColorRgb.
type RGB struct{ R, G, B uint8 }

// styleColor is a GhosttyStyleColor: a tag and, after padding, a union of a
// palette index and an RGB color.
type styleColor struct {
	tag   int32
	_     int32
	value [8]byte
}

const (
	colorNone    = 0
	colorPalette = 1
	colorRGB     = 2
)

// style is a GhosttyStyle.
type style struct {
	size                                                                    uintptr
	fg, bg, underlineColor                                                  styleColor
	bold, italic, faint, blink, inverse, invisible, strikethrough, overline bool
	underline                                                               int32
}

// renderCursor is a GhosttyRenderStateCursor.
type renderCursor struct {
	size                                       uintptr
	viewportHasValue                           bool
	viewportX, viewportY                       uint16
	wideTail, visible, blinking, passwordInput bool
	visualStyle                                int32
}

// renderColors is a GhosttyRenderStateColors.
type renderColors struct {
	size                           uintptr
	background, foreground, cursor RGB
	cursorHasValue                 bool
	palette                        [256]RGB
}

// rowSelection is a GhosttyRenderStateRowSelection.
type rowSelection struct {
	size         uintptr
	startX, endX uint16
}

// cellsView is a GhosttyCellsView.
type cellsView struct {
	ptr, len uintptr
}

// cString is a GhosttyString.
type cString struct {
	ptr, len uintptr
}

// Scrollbar is a GhosttyTerminalScrollbar: the rows of the screen and its
// scrollback, the first row of the viewport and how many rows it shows.
type Scrollbar struct {
	Total, Offset, Len uint64
}

// scrollViewport is a GhosttyTerminalScrollViewport.
type scrollViewport struct {
	tag   int32
	_     int32
	value [2]uint64
}

// modeConfig is a GhosttyTerminalModeConfig.
type modeConfig struct {
	mode  uint16
	value bool
}

// GridRef is a GhosttyGridRef: a cell of the terminal, valid until the
// terminal next changes.
type GridRef struct {
	size uintptr
	node uintptr
	x, y uint16
}

// point is a GhosttyPoint whose value is a coordinate.
type point struct {
	tag int32
	_   int32
	x   uint16
	_   uint16
	y   uint32
	_   uint64
}

// Selection is a GhosttySelection: two cells of the terminal, inclusive,
// valid until the terminal next changes.
type Selection struct {
	size       uintptr
	start, end GridRef
	rectangle  bool
}

// selectionFormatOptions is a GhosttyTerminalSelectionFormatOptions.
type selectionFormatOptions struct {
	size      uintptr
	emit      int32
	unwrap    bool
	trim      bool
	selection uintptr
}

// formatterOptions is a GhosttyFormatterTerminalOptions, with its extras
// (GhosttyFormatterTerminalExtra and GhosttyFormatterScreenExtra).
type formatterOptions struct {
	size      uintptr
	emit      int32
	unwrap    bool
	trim      bool
	extra     formatterExtra
	selection uintptr
}

type formatterExtra struct {
	size                                                     uintptr
	palette, modes, scrollingRegion, tabstops, pwd, keyboard bool
	screen                                                   struct {
		size                                                          uintptr
		cursor, style, hyperlink, protection, kittyKeyboard, charsets bool
	}
}

// mouseEncoderSize is a GhosttyMouseEncoderSize.
type mouseEncoderSize struct {
	size                                                 uintptr
	screenWidth, screenHeight, cellWidth, cellHeight     uint32
	paddingTop, paddingBottom, paddingRight, paddingLeft uint32
}

// mousePosition is a GhosttyMousePosition.
type mousePosition struct{ x, y float32 }

// gestureGeometry is a GhosttySelectionGestureGeometry.
type gestureGeometry struct {
	columns, cellWidth, paddingLeft, screenHeight uint32
}

// surfacePosition is a GhosttySurfacePosition.
type surfacePosition struct{ x, y float64 }

// sizeReport is a GhosttySizeReportSize.
type sizeReport struct {
	rows, columns         uint16
	cellWidth, cellHeight uint32
}

// clipboardWrite is a GhosttyClipboardWrite.
type clipboardWrite struct {
	size                 uintptr
	location             int32
	contents             uintptr
	contentsLen          uintptr
	name                 cString
	granted, canRemember bool
	ctx, reply           uintptr
}

// clipboardContent is a GhosttyClipboardContent.
type clipboardContent struct {
	mime, data cString
}

// clipboardWriteReply is a GhosttyClipboardWriteReply.
type clipboardWriteReply struct {
	size     uintptr
	result   int32
	remember bool
}

// clipboardRead is a GhosttyClipboardRead.
type clipboardRead struct {
	size                 uintptr
	location             int32
	mimes                uintptr
	mimesLen             uintptr
	list                 bool
	name                 cString
	granted, canRemember bool
	ctx, reply           uintptr
}

// clipboardReadReply is a GhosttyClipboardReadReply.
type clipboardReadReply struct {
	size         uintptr
	result       int32
	contents     uintptr
	contentsLen  uintptr
	available    uintptr
	availableLen uintptr
	remember     bool
}

// paste is a GhosttyPaste: what a paste offers the program.
type paste struct {
	size        uintptr
	location    int32
	source      int32
	mimes       uintptr
	mimesLen    uintptr
	reader      mimeReader
	allowUnsafe bool
}

// mimeReader is a GhosttyMimeReader.
type mimeReader struct {
	read, userdata uintptr
}

// desktopNotification is a GhosttyTerminalDesktopNotification.
type desktopNotification struct {
	size        uintptr
	title, body cString
}

// progressReport is a GhosttyTerminalProgressReport.
type progressReport struct {
	size     uintptr
	state    int32
	progress int8
}

// deviceAttributes is a GhosttyDeviceAttributes.
type deviceAttributes struct {
	conformance                        uint16
	features                           [64]uint16
	numFeatures                        uintptr
	deviceType, firmware, romCartridge uint16
	unitID                             uint32
}

// RowID identifies a row across render state updates (a
// GhosttyRenderStateRowId).
type RowID struct{ bits [2]uint64 }

func layouts() map[string]layout {
	var (
		sc  styleColor
		s   style
		c   renderCursor
		rc  renderColors
		rs  rowSelection
		cv  cellsView
		sb  Scrollbar
		sv  scrollViewport
		mc  modeConfig
		gr  GridRef
		pt  point
		sel Selection
		sfo selectionFormatOptions
		mes mouseEncoderSize
		cw  clipboardWrite
		cc  clipboardContent
		cwr clipboardWriteReply
		cr  clipboardRead
		crr clipboardReadReply
		pa  paste
		dn  desktopNotification
		pr  progressReport
		da  deviceAttributes
		fo  formatterOptions
	)
	return map[string]layout{
		"GhosttyColorRgb":   {unsafe.Sizeof(RGB{}), map[string]uintptr{"r": 0, "g": 1, "b": 2}},
		"GhosttyStyleColor": {unsafe.Sizeof(sc), map[string]uintptr{"tag": 0, "value": unsafe.Offsetof(sc.value)}},
		"GhosttyStyle": {unsafe.Sizeof(s), map[string]uintptr{
			"fg_color": unsafe.Offsetof(s.fg), "bg_color": unsafe.Offsetof(s.bg), "underline_color": unsafe.Offsetof(s.underlineColor),
			"bold": unsafe.Offsetof(s.bold), "italic": unsafe.Offsetof(s.italic), "faint": unsafe.Offsetof(s.faint),
			"blink": unsafe.Offsetof(s.blink), "inverse": unsafe.Offsetof(s.inverse), "invisible": unsafe.Offsetof(s.invisible),
			"strikethrough": unsafe.Offsetof(s.strikethrough), "overline": unsafe.Offsetof(s.overline), "underline": unsafe.Offsetof(s.underline),
		}},
		"GhosttyRenderStateCursor": {unsafe.Sizeof(c), map[string]uintptr{
			"viewport_has_value": unsafe.Offsetof(c.viewportHasValue), "viewport_x": unsafe.Offsetof(c.viewportX),
			"viewport_y": unsafe.Offsetof(c.viewportY), "wide_tail": unsafe.Offsetof(c.wideTail), "visible": unsafe.Offsetof(c.visible),
			"blinking": unsafe.Offsetof(c.blinking), "password_input": unsafe.Offsetof(c.passwordInput), "visual_style": unsafe.Offsetof(c.visualStyle),
		}},
		"GhosttyRenderStateColors": {unsafe.Sizeof(rc), map[string]uintptr{
			"background": unsafe.Offsetof(rc.background), "foreground": unsafe.Offsetof(rc.foreground), "cursor": unsafe.Offsetof(rc.cursor),
			"cursor_has_value": unsafe.Offsetof(rc.cursorHasValue), "palette": unsafe.Offsetof(rc.palette),
		}},
		"GhosttyRenderStateRowSelection": {unsafe.Sizeof(rs), map[string]uintptr{"start_x": unsafe.Offsetof(rs.startX), "end_x": unsafe.Offsetof(rs.endX)}},
		"GhosttyCellsView":               {unsafe.Sizeof(cv), map[string]uintptr{"ptr": 0, "len": unsafe.Offsetof(cv.len)}},
		"GhosttyString":                  {unsafe.Sizeof(cString{}), map[string]uintptr{"ptr": 0, "len": unsafe.Offsetof(cString{}.len)}},
		"GhosttyTerminalScrollbar":       {unsafe.Sizeof(sb), map[string]uintptr{"total": 0, "offset": unsafe.Offsetof(sb.Offset), "len": unsafe.Offsetof(sb.Len)}},
		"GhosttyTerminalScrollViewport":  {unsafe.Sizeof(sv), map[string]uintptr{"tag": 0, "value": unsafe.Offsetof(sv.value)}},
		"GhosttyTerminalModeConfig":      {unsafe.Sizeof(mc), map[string]uintptr{"mode": 0, "value": unsafe.Offsetof(mc.value)}},
		"GhosttyGridRef":                 {unsafe.Sizeof(gr), map[string]uintptr{"node": unsafe.Offsetof(gr.node), "x": unsafe.Offsetof(gr.x), "y": unsafe.Offsetof(gr.y)}},
		"GhosttyPoint":                   {unsafe.Sizeof(pt), map[string]uintptr{"tag": 0, "value": unsafe.Offsetof(pt.x)}},
		"GhosttyPointCoordinate":         {8, map[string]uintptr{"x": unsafe.Offsetof(pt.x) - unsafe.Offsetof(pt.x), "y": unsafe.Offsetof(pt.y) - unsafe.Offsetof(pt.x)}},
		"GhosttySelection":               {unsafe.Sizeof(sel), map[string]uintptr{"start": unsafe.Offsetof(sel.start), "end": unsafe.Offsetof(sel.end), "rectangle": unsafe.Offsetof(sel.rectangle)}},
		"GhosttyTerminalSelectionFormatOptions": {unsafe.Sizeof(sfo), map[string]uintptr{
			"emit": unsafe.Offsetof(sfo.emit), "unwrap": unsafe.Offsetof(sfo.unwrap), "trim": unsafe.Offsetof(sfo.trim), "selection": unsafe.Offsetof(sfo.selection),
		}},
		"GhosttyMouseEncoderSize": {unsafe.Sizeof(mes), map[string]uintptr{
			"screen_width": unsafe.Offsetof(mes.screenWidth), "screen_height": unsafe.Offsetof(mes.screenHeight),
			"cell_width": unsafe.Offsetof(mes.cellWidth), "cell_height": unsafe.Offsetof(mes.cellHeight),
			"padding_top": unsafe.Offsetof(mes.paddingTop), "padding_left": unsafe.Offsetof(mes.paddingLeft),
		}},
		"GhosttyMousePosition":            {unsafe.Sizeof(mousePosition{}), map[string]uintptr{"x": 0, "y": 4}},
		"GhosttySelectionGestureGeometry": {unsafe.Sizeof(gestureGeometry{}), map[string]uintptr{"columns": 0, "cell_width": 4, "padding_left": 8, "screen_height": 12}},
		"GhosttySurfacePosition":          {unsafe.Sizeof(surfacePosition{}), map[string]uintptr{"x": 0, "y": 8}},
		"GhosttySizeReportSize":           {unsafe.Sizeof(sizeReport{}), map[string]uintptr{"rows": 0, "columns": 2, "cell_width": 4, "cell_height": 8}},
		"GhosttyClipboardWrite": {unsafe.Sizeof(cw), map[string]uintptr{
			"location": unsafe.Offsetof(cw.location), "contents": unsafe.Offsetof(cw.contents), "contents_len": unsafe.Offsetof(cw.contentsLen),
			"name": unsafe.Offsetof(cw.name), "granted": unsafe.Offsetof(cw.granted), "ctx": unsafe.Offsetof(cw.ctx), "reply": unsafe.Offsetof(cw.reply),
		}},
		"GhosttyClipboardContent":    {unsafe.Sizeof(cc), map[string]uintptr{"mime": 0, "data": unsafe.Offsetof(cc.data)}},
		"GhosttyClipboardWriteReply": {unsafe.Sizeof(cwr), map[string]uintptr{"result": unsafe.Offsetof(cwr.result), "remember": unsafe.Offsetof(cwr.remember)}},
		"GhosttyClipboardRead": {unsafe.Sizeof(cr), map[string]uintptr{
			"location": unsafe.Offsetof(cr.location), "mimes": unsafe.Offsetof(cr.mimes), "mimes_len": unsafe.Offsetof(cr.mimesLen),
			"list": unsafe.Offsetof(cr.list), "name": unsafe.Offsetof(cr.name), "granted": unsafe.Offsetof(cr.granted),
			"can_remember": unsafe.Offsetof(cr.canRemember), "ctx": unsafe.Offsetof(cr.ctx), "reply": unsafe.Offsetof(cr.reply),
		}},
		"GhosttyClipboardReadReply": {unsafe.Sizeof(crr), map[string]uintptr{
			"result": unsafe.Offsetof(crr.result), "contents": unsafe.Offsetof(crr.contents), "contents_len": unsafe.Offsetof(crr.contentsLen),
			"available": unsafe.Offsetof(crr.available), "available_len": unsafe.Offsetof(crr.availableLen), "remember": unsafe.Offsetof(crr.remember),
		}},
		"GhosttyPaste": {unsafe.Sizeof(pa), map[string]uintptr{
			"location": unsafe.Offsetof(pa.location), "source": unsafe.Offsetof(pa.source), "mimes": unsafe.Offsetof(pa.mimes),
			"mimes_len": unsafe.Offsetof(pa.mimesLen), "reader": unsafe.Offsetof(pa.reader), "allow_unsafe": unsafe.Offsetof(pa.allowUnsafe),
		}},
		"GhosttyMimeReader":                  {unsafe.Sizeof(mimeReader{}), map[string]uintptr{"read": 0, "userdata": unsafe.Offsetof(mimeReader{}.userdata)}},
		"GhosttyTerminalDesktopNotification": {unsafe.Sizeof(dn), map[string]uintptr{"title": unsafe.Offsetof(dn.title), "body": unsafe.Offsetof(dn.body)}},
		"GhosttyTerminalProgressReport":      {unsafe.Sizeof(pr), map[string]uintptr{"state": unsafe.Offsetof(pr.state), "progress": unsafe.Offsetof(pr.progress)}},
		"GhosttyDeviceAttributes":            {unsafe.Sizeof(da), map[string]uintptr{"primary": 0, "secondary": unsafe.Offsetof(da.deviceType), "tertiary": unsafe.Offsetof(da.unitID)}},
		"GhosttyRenderStateRowId":            {unsafe.Sizeof(RowID{}), map[string]uintptr{"bits": 0}},
		"GhosttyFormatterTerminalOptions": {unsafe.Sizeof(fo), map[string]uintptr{
			"emit": unsafe.Offsetof(fo.emit), "unwrap": unsafe.Offsetof(fo.unwrap), "trim": unsafe.Offsetof(fo.trim),
			"extra": unsafe.Offsetof(fo.extra), "selection": unsafe.Offsetof(fo.selection),
		}},
		"GhosttyFormatterTerminalExtra": {unsafe.Sizeof(fo.extra), map[string]uintptr{"keyboard": unsafe.Offsetof(fo.extra.keyboard), "screen": unsafe.Offsetof(fo.extra.screen)}},
		"GhosttyFormatterScreenExtra":   {unsafe.Sizeof(fo.extra.screen), map[string]uintptr{"charsets": unsafe.Offsetof(fo.extra.screen.charsets)}},
	}
}
