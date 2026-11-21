//go:build linux && (amd64 || arm64)

package text

import (
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"unicode"
	"unicode/utf8"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo/internal/scene"
)

// Pango lays out paragraphs: fontconfig finds the fonts and the fallbacks
// for what a font lacks, HarfBuzz shapes, and Pango breaks lines. Cairo
// rasterizes glyphs, with FreeType, in color for color fonts. They are the
// libraries GTK draws with, loaded with purego.

func newEngine() engine {
	e, err := newPango()
	if err != nil {
		log.Printf("mygo: %v: text will not show", err)
		return &stubEngine{}
	}
	return e
}

const (
	pangoScale       = 1024
	pangoGlyphEmpty  = 0x0FFFFFFF
	pangoGlyphUnknow = 0x10000000

	pangoWrapWord     = 0
	pangoWrapWordChar = 2
	pangoStyleNormal  = 0
	pangoStyleItalic  = 2
	pangoDirLTR       = 0
	pangoDirRTL       = 1

	cairoFormatARGB32          = 0
	cairoAntialiasDefault      = 0
	cairoAntialiasNone         = 1
	cairoAntialiasGray         = 2
	cairoAntialiasSubpixel     = 3
	cairoHintStyleDefault      = 0
	cairoHintStyleNone         = 1
	cairoHintStyleSlight       = 2
	cairoHintStyleMedium       = 3
	cairoHintStyleFull         = 4
	cairoHintMetricsOn         = 2
	cairoStatusSuccess         = 0
	fcResultMatch              = 0
	fcSetApplication           = 1
	gTrue, gFalse          int = 1, 0
)

// Pango and cairo structs, as their headers lay them out.
type (
	gSList struct {
		data unsafe.Pointer
		next *gSList
	}
	pangoLayoutLine struct {
		layout     uintptr
		startIndex int32
		length     int32
		runs       *gSList
		flags      uint32
	}
	pangoGlyphItem struct {
		item   *pangoItem
		glyphs *pangoGlyphString
	}
	pangoItem struct {
		offset, length, numChars int32
		_                        int32
		shapeEngine, langEngine  uintptr
		font                     uintptr
		level                    uint8
	}
	pangoGlyphString struct {
		numGlyphs   int32
		glyphs      *pangoGlyphInfo
		logClusters *int32
	}
	pangoGlyphInfo struct {
		glyph                   uint32
		width, xOffset, yOffset int32
		attr                    uint32
	}
	cairoMatrix struct{ xx, yx, xy, yy, x0, y0 float64 }
	cairoGlyph  struct {
		index uint64
		x, y  float64
	}
	cairoTextExtents struct {
		xBearing, yBearing, width, height, xAdvance, yAdvance float64
	}
	pangoRectangle struct{ x, y, width, height int32 }
	fcFontSet      struct {
		nfont, sfont int32
		fonts        *uintptr
	}
)

var pangoLib struct {
	gObjectRef   func(obj uintptr) uintptr
	gObjectUnref func(obj uintptr)
	gFree        func(p uintptr)

	fontMapListFamilies   func(fontMap uintptr, families **uintptr, n *int32)
	fontFamilyGetName     func(family uintptr) *byte
	fontFamilyIsMonospace func(family uintptr) bool

	fontMapNew                func() uintptr
	fontMapCreateContext      func(fontMap uintptr) uintptr
	fontMapLoadFont           func(fontMap, context, desc uintptr) uintptr
	contextSetFontOptions     func(context, options uintptr)
	contextSetBaseDir         func(context uintptr, dir int32)
	contextSetRoundPositions  func(context uintptr, round int32)
	fontGetScaledFont         func(font uintptr) uintptr
	fontGetMetrics            func(font, language uintptr) uintptr
	fontDescribeAbsolute      func(font uintptr) uintptr
	fontGetHBFont             func(font uintptr) uintptr
	hbNominalGlyph            func(font uintptr, r rune, glyph *uint32) int32
	hbGlyphHAdvance           func(font uintptr, glyph uint32) int32
	metricsGetAscent          func(metrics uintptr) int32
	metricsGetUnderlinePos    func(metrics uintptr) int32
	metricsGetUnderlineThick  func(metrics uintptr) int32
	metricsGetStrikePos       func(metrics uintptr) int32
	metricsGetStrikeThick     func(metrics uintptr) int32
	fontGetGlyphExtents       func(font uintptr, glyph uint32, ink, logical *pangoRectangle)
	metricsGetDescent         func(metrics uintptr) int32
	metricsGetHeight          func(metrics uintptr) int32
	metricsUnref              func(metrics uintptr)
	descNew                   func() uintptr
	descFree                  func(desc uintptr)
	descSetFamily             func(desc uintptr, family string)
	descSetWeight             func(desc uintptr, weight int32)
	descSetStyle              func(desc uintptr, style int32)
	descSetAbsoluteSize       func(desc uintptr, size float64)
	descGetSize               func(desc uintptr) int32
	layoutNew                 func(context uintptr) uintptr
	layoutSetText             func(layout uintptr, text *byte, length int32)
	layoutSetFontDescription  func(layout, desc uintptr)
	layoutSetWidth            func(layout uintptr, width int32)
	layoutSetWrap             func(layout uintptr, wrap int32)
	layoutSetAutoDir          func(layout uintptr, auto int32)
	layoutGetLineCount        func(layout uintptr) int32
	layoutGetLineReadonly     func(layout uintptr, line int32) *pangoLayoutLine
	layoutSetAttributes       func(layout, attrs uintptr)
	attrListNew               func() uintptr
	attrListUnref             func(list uintptr)
	attrListInsert            func(list, attr uintptr)
	attrLetterSpacingNew      func(spacing int32) uintptr
	attrFontFeaturesNew       func(features string) uintptr // Pango 1.38
	attrFamilyNew             func(family string) uintptr
	attrSizeAbsoluteNew       func(size int32) uintptr
	attrWeightNew             func(weight int32) uintptr
	attrStyleNew              func(style int32) uintptr
	attrShapeNew              func(ink, logical *pangoRectangle) uintptr
	attrLanguageNew           func(language uintptr) uintptr
	languageFromString        func(tag string) uintptr
	languageGetDefault        func() uintptr
	fontMapConfigChanged      func(fontMap uintptr)
	cairoFontOptionsCreate    func() uintptr
	cairoFontOptionsDestroy   func(options uintptr)
	cairoFontOptionsAntialias func(options uintptr, antialias int32)
	cairoFontOptionsHintMets  func(options uintptr, hint int32)
	cairoFontOptionsHintStyle func(options uintptr, style int32)
	cairoFontOptionsSubpixel  func(options uintptr, order int32)
	cairoScaledFontFace       func(font uintptr) uintptr
	cairoScaledFontMatrix     func(font uintptr, m *cairoMatrix)
	cairoScaledFontOptions    func(font, options uintptr)
	cairoScaledFontCreate     func(face uintptr, fontMatrix, ctm *cairoMatrix, options uintptr) uintptr
	cairoScaledFontStatus     func(font uintptr) int32
	cairoScaledFontDestroy    func(font uintptr)
	cairoScaledFontExtents    func(font uintptr, glyphs *cairoGlyph, n int32, extents *cairoTextExtents)
	cairoImageSurfaceCreate   func(format, width, height int32) uintptr
	cairoImageSurfaceData     func(surface uintptr) *byte
	cairoImageSurfaceStride   func(surface uintptr) int32
	cairoSurfaceFlush         func(surface uintptr)
	cairoSurfaceDestroy       func(surface uintptr)
	cairoCreate               func(surface uintptr) uintptr
	cairoDestroy              func(cr uintptr)
	cairoSetScaledFont        func(cr, font uintptr)
	cairoSetSourceRGBA        func(cr uintptr, r, g, b, a float64)
	cairoShowGlyphs           func(cr uintptr, glyphs *cairoGlyph, n int32)
	fcConfigGetCurrent        func() uintptr
	fcConfigAppFontAddFile    func(config uintptr, file string) int32
	fcConfigGetFonts          func(config uintptr, set int32) *fcFontSet
	fcConfigParseFromMemory   func(config uintptr, xml string, complain int32) int32
	fcPatternGetString        func(pattern uintptr, object string, n int32, out **byte) int32
}

func loadPango() error {
	open := func(names ...string) (uintptr, error) {
		var last error
		for _, n := range names {
			h, err := purego.Dlopen(n, purego.RTLD_NOW|purego.RTLD_GLOBAL)
			if err == nil {
				return h, nil
			}
			last = err
		}
		return 0, last
	}
	gobject, err := open("libgobject-2.0.so.0")
	if err != nil {
		return err
	}
	pango, err := open("libpango-1.0.so.0")
	if err != nil {
		return err
	}
	glib, err := open("libglib-2.0.so.0")
	if err != nil {
		return err
	}
	pangocairo, err := open("libpangocairo-1.0.so.0")
	if err != nil {
		return err
	}
	cairo, err := open("libcairo.so.2")
	if err != nil {
		return err
	}
	var missing []string
	bind := func(lib uintptr, fn any, name string) {
		sym, err := purego.Dlsym(lib, name)
		if err != nil || sym == 0 {
			missing = append(missing, name)
			return
		}
		purego.RegisterFunc(fn, sym)
	}
	l := &pangoLib
	bind(gobject, &l.gObjectRef, "g_object_ref")
	bind(gobject, &l.gObjectUnref, "g_object_unref")
	bind(glib, &l.gFree, "g_free")
	bind(pango, &l.fontMapListFamilies, "pango_font_map_list_families")
	bind(pango, &l.fontFamilyGetName, "pango_font_family_get_name")
	bind(pango, &l.fontFamilyIsMonospace, "pango_font_family_is_monospace")
	bind(pangocairo, &l.fontMapNew, "pango_cairo_font_map_new")
	bind(pangocairo, &l.contextSetFontOptions, "pango_cairo_context_set_font_options")
	bind(pangocairo, &l.fontGetScaledFont, "pango_cairo_font_get_scaled_font")
	bind(pango, &l.fontMapCreateContext, "pango_font_map_create_context")
	bind(pango, &l.fontMapLoadFont, "pango_font_map_load_font")
	bind(pango, &l.contextSetBaseDir, "pango_context_set_base_dir")
	bind(pango, &l.fontGetMetrics, "pango_font_get_metrics")
	bind(pango, &l.fontDescribeAbsolute, "pango_font_describe_with_absolute_size")
	bind(pango, &l.metricsGetAscent, "pango_font_metrics_get_ascent")
	bind(pango, &l.metricsGetUnderlinePos, "pango_font_metrics_get_underline_position")
	bind(pango, &l.metricsGetUnderlineThick, "pango_font_metrics_get_underline_thickness")
	bind(pango, &l.metricsGetStrikePos, "pango_font_metrics_get_strikethrough_position")
	bind(pango, &l.metricsGetStrikeThick, "pango_font_metrics_get_strikethrough_thickness")
	bind(pango, &l.fontGetGlyphExtents, "pango_font_get_glyph_extents")
	bind(pango, &l.metricsGetDescent, "pango_font_metrics_get_descent")
	bind(pango, &l.metricsUnref, "pango_font_metrics_unref")
	bind(pango, &l.descNew, "pango_font_description_new")
	bind(pango, &l.descFree, "pango_font_description_free")
	bind(pango, &l.descSetFamily, "pango_font_description_set_family")
	bind(pango, &l.descSetWeight, "pango_font_description_set_weight")
	bind(pango, &l.descSetStyle, "pango_font_description_set_style")
	bind(pango, &l.descSetAbsoluteSize, "pango_font_description_set_absolute_size")
	bind(pango, &l.descGetSize, "pango_font_description_get_size")
	bind(pango, &l.layoutNew, "pango_layout_new")
	bind(pango, &l.layoutSetText, "pango_layout_set_text")
	bind(pango, &l.layoutSetFontDescription, "pango_layout_set_font_description")
	bind(pango, &l.layoutSetWidth, "pango_layout_set_width")
	bind(pango, &l.layoutSetWrap, "pango_layout_set_wrap")
	bind(pango, &l.layoutSetAutoDir, "pango_layout_set_auto_dir")
	bind(pango, &l.layoutGetLineCount, "pango_layout_get_line_count")
	bind(pango, &l.layoutGetLineReadonly, "pango_layout_get_line_readonly")
	bind(pango, &l.layoutSetAttributes, "pango_layout_set_attributes")
	bind(pango, &l.attrListNew, "pango_attr_list_new")
	bind(pango, &l.attrListUnref, "pango_attr_list_unref")
	bind(pango, &l.attrListInsert, "pango_attr_list_insert")
	bind(pango, &l.attrLetterSpacingNew, "pango_attr_letter_spacing_new")
	bind(pango, &l.attrFamilyNew, "pango_attr_family_new")
	bind(pango, &l.attrSizeAbsoluteNew, "pango_attr_size_new_absolute")
	bind(pango, &l.attrWeightNew, "pango_attr_weight_new")
	bind(pango, &l.attrStyleNew, "pango_attr_style_new")
	bind(pango, &l.attrShapeNew, "pango_attr_shape_new")
	bind(pango, &l.attrLanguageNew, "pango_attr_language_new")
	bind(pango, &l.languageFromString, "pango_language_from_string")
	bind(pango, &l.languageGetDefault, "pango_language_get_default")
	bind(cairo, &l.cairoFontOptionsCreate, "cairo_font_options_create")
	bind(cairo, &l.cairoFontOptionsDestroy, "cairo_font_options_destroy")
	bind(cairo, &l.cairoFontOptionsAntialias, "cairo_font_options_set_antialias")
	bind(cairo, &l.cairoFontOptionsHintMets, "cairo_font_options_set_hint_metrics")
	bind(cairo, &l.cairoFontOptionsHintStyle, "cairo_font_options_set_hint_style")
	bind(cairo, &l.cairoFontOptionsSubpixel, "cairo_font_options_set_subpixel_order")
	bind(cairo, &l.cairoScaledFontFace, "cairo_scaled_font_get_font_face")
	bind(cairo, &l.cairoScaledFontMatrix, "cairo_scaled_font_get_font_matrix")
	bind(cairo, &l.cairoScaledFontOptions, "cairo_scaled_font_get_font_options")
	bind(cairo, &l.cairoScaledFontCreate, "cairo_scaled_font_create")
	bind(cairo, &l.cairoScaledFontStatus, "cairo_scaled_font_status")
	bind(cairo, &l.cairoScaledFontDestroy, "cairo_scaled_font_destroy")
	bind(cairo, &l.cairoScaledFontExtents, "cairo_scaled_font_glyph_extents")
	bind(cairo, &l.cairoImageSurfaceCreate, "cairo_image_surface_create")
	bind(cairo, &l.cairoImageSurfaceData, "cairo_image_surface_get_data")
	bind(cairo, &l.cairoImageSurfaceStride, "cairo_image_surface_get_stride")
	bind(cairo, &l.cairoSurfaceFlush, "cairo_surface_flush")
	bind(cairo, &l.cairoSurfaceDestroy, "cairo_surface_destroy")
	bind(cairo, &l.cairoCreate, "cairo_create")
	bind(cairo, &l.cairoDestroy, "cairo_destroy")
	bind(cairo, &l.cairoSetScaledFont, "cairo_set_scaled_font")
	bind(cairo, &l.cairoSetSourceRGBA, "cairo_set_source_rgba")
	bind(cairo, &l.cairoShowGlyphs, "cairo_show_glyphs")
	if len(missing) > 0 {
		return fmt.Errorf("Pango or cairo lacks %s", strings.Join(missing, ", "))
	}
	// Optional: Pango 1.44 and later, and what adding fonts needs.
	missing = nil
	bind(pango, &l.contextSetRoundPositions, "pango_context_set_round_glyph_positions")
	bind(pango, &l.attrFontFeaturesNew, "pango_attr_font_features_new")
	bind(pango, &l.metricsGetHeight, "pango_font_metrics_get_height")
	bind(pango, &l.fontGetHBFont, "pango_font_get_hb_font")
	if hb, err := open("libharfbuzz.so.0"); err == nil {
		bind(hb, &l.hbNominalGlyph, "hb_font_get_nominal_glyph")
		bind(hb, &l.hbGlyphHAdvance, "hb_font_get_glyph_h_advance")
	}
	if ft, err := open("libpangoft2-1.0.so.0"); err == nil {
		bind(ft, &l.fontMapConfigChanged, "pango_fc_font_map_config_changed")
	}
	if fc, err := open("libfontconfig.so.1"); err == nil {
		bind(fc, &l.fcConfigGetCurrent, "FcConfigGetCurrent")
		bind(fc, &l.fcConfigAppFontAddFile, "FcConfigAppFontAddFile")
		bind(fc, &l.fcConfigGetFonts, "FcConfigGetFonts")
		bind(fc, &l.fcConfigParseFromMemory, "FcConfigParseAndLoadFromMemory")
		bind(fc, &l.fcPatternGetString, "FcPatternGetString")
	}
	return nil
}

type pangoEngine struct {
	shapeScratch
	fontMap uintptr
	context uintptr

	descs  map[Style]uintptr // font descriptions, by style
	fonts  map[uintptr]*Font // by PangoFont
	scaled map[scaledKey]uintptr

	registered map[string]bool // files of registered fonts
	// names are the names the app's fonts are listed under: the one the
	// app gave, or their own; appFamily the own family of each, by
	// lowercased name.
	names     []string
	appFamily map[string]string
	// uiFamily is the desktop's interface font, which system-ui stands for
	// before fontconfig's default.
	uiFamily string
	// The cairo antialiasing, hint style and subpixel order of the
	// desktop's settings (setFontRendering), as GTK has its labels draw.
	antialias, hintStyle, subpixels int32
	// languages are the PangoLanguages of tags.
	languages map[string]uintptr
}

// setFontRendering takes the desktop's settings (platform.FontRendering)
// for the fonts made from now on.
func (e *pangoEngine) setFontRendering(antialias, hinting, subpixels string) {
	e.antialias = map[string]int32{"none": cairoAntialiasNone, "gray": cairoAntialiasGray, "subpixel": cairoAntialiasSubpixel}[antialias]
	e.hintStyle = map[string]int32{"none": cairoHintStyleNone, "slight": cairoHintStyleSlight, "medium": cairoHintStyleMedium, "full": cairoHintStyleFull}[hinting]
	e.subpixels = map[string]int32{"rgb": 1, "bgr": 2, "vrgb": 3, "vbgr": 4}[subpixels]
	e.applyOptions()
	for _, sf := range e.scaled {
		pangoLib.cairoScaledFontDestroy(sf)
	}
	clear(e.scaled)
}

// applyOptions gives the context GTK's font options for labels: the
// desktop's antialiasing, hint style and subpixel order, metrics hinted
// to whole pixels, and glyphs on whole pixels.
func (e *pangoEngine) applyOptions() {
	l := &pangoLib
	opts := l.cairoFontOptionsCreate()
	l.cairoFontOptionsAntialias(opts, e.antialias)
	l.cairoFontOptionsHintStyle(opts, e.hintStyle)
	l.cairoFontOptionsSubpixel(opts, e.subpixels)
	l.cairoFontOptionsHintMets(opts, cairoHintMetricsOn)
	l.contextSetFontOptions(e.context, opts)
	l.cairoFontOptionsDestroy(opts)
	if l.contextSetRoundPositions != nil {
		l.contextSetRoundPositions(e.context, int32(gTrue))
	}
}

func (e *pangoEngine) setUIFamily(family string) {
	if family == e.uiFamily {
		return
	}
	e.uiFamily = family
	for _, d := range e.descs {
		pangoLib.descFree(d)
	}
	clear(e.descs)
}

type scaledKey struct {
	font     *Font
	scale    float32
	subpixel bool
}

func newPango() (*pangoEngine, error) {
	if err := loadPango(); err != nil {
		return nil, err
	}
	l := &pangoLib
	e := &pangoEngine{
		descs:      map[Style]uintptr{},
		fonts:      map[uintptr]*Font{},
		scaled:     map[scaledKey]uintptr{},
		registered: map[string]bool{},
		appFamily:  map[string]string{},
		languages:  map[string]uintptr{},
	}
	// A font map of our own: the default one belongs to the thread that
	// asks for it, and goroutines move between threads.
	e.fontMap = l.fontMapNew()
	if e.fontMap == 0 {
		return nil, errors.New("pango_cairo_font_map_new failed")
	}
	e.context = l.fontMapCreateContext(e.fontMap)
	// Until the app has the desktop's settings, GDK's defaults on X11
	// without a settings daemon: grayscale antialiasing, medium hinting.
	e.antialias, e.hintStyle = cairoAntialiasGray, cairoHintStyleMedium
	e.applyOptions()
	return e, nil
}

// pangoFamily returns the family list of a style for Pango, where
// fontconfig knows the generic families, and system-ui is the desktop's
// interface font, ui, when the app knows it.
func pangoFamily(list, ui string) string {
	var out []string
	systemUI := []string{"system-ui", "sans-serif"}
	if ui != "" {
		systemUI = []string{ui, "system-ui", "sans-serif"}
	}
	for _, f := range familyList(list) {
		switch generic(f) {
		case "system-ui":
			out = append(out, systemUI...)
		case "":
			out = append(out, f)
		default:
			out = append(out, generic(f))
		}
	}
	if len(out) == 0 {
		out = systemUI
	}
	return strings.Join(out, ",")
}

func (e *pangoEngine) desc(style Style) uintptr {
	key := Style{Family: style.Family, Size: style.FontSize(), Weight: style.weight(), Italic: style.Italic}
	if d, ok := e.descs[key]; ok {
		return d
	}
	l := &pangoLib
	if len(e.descs) >= 256 {
		for _, d := range e.descs {
			l.descFree(d)
		}
		clear(e.descs)
	}
	d := l.descNew()
	l.descSetFamily(d, pangoFamily(key.Family, e.uiFamily))
	l.descSetWeight(d, int32(key.Weight))
	if key.Italic {
		l.descSetStyle(d, pangoStyleItalic)
	} else {
		l.descSetStyle(d, pangoStyleNormal)
	}
	l.descSetAbsoluteSize(d, float64(key.Size)*pangoScale)
	e.descs[key] = d
	return d
}

func (e *pangoEngine) font(style Style) *Font {
	font := pangoLib.fontMapLoadFont(e.fontMap, e.context, e.desc(style))
	if font == 0 {
		return nil
	}
	f := e.fontOf(font)
	pangoLib.gObjectUnref(font)
	return f
}

// fontOf returns the Font of a PangoFont.
func (e *pangoEngine) fontOf(font uintptr) *Font {
	if f, ok := e.fonts[font]; ok {
		return f
	}
	l := &pangoLib
	l.gObjectRef(font)
	f := &Font{native: font}
	if d := l.fontDescribeAbsolute(font); d != 0 {
		f.Size = float32(l.descGetSize(d)) / pangoScale
		l.descFree(d)
	}
	if m := l.fontGetMetrics(font, 0); m != 0 {
		f.Ascent = float32(l.metricsGetAscent(m)) / pangoScale
		f.Descent = float32(l.metricsGetDescent(m)) / pangoScale
		// Hinted metrics are whole pixels (DIPs), rounded up.
		f.underlineTop = float32(l.metricsGetUnderlinePos(m)) / pangoScale
		f.underlineThick = float32(l.metricsGetUnderlineThick(m)) / pangoScale
		f.strikeTop = float32(l.metricsGetStrikePos(m)) / pangoScale
		f.strikeThick = float32(l.metricsGetStrikeThick(m)) / pangoScale
		if l.metricsGetHeight != nil {
			f.LineGap = max(float32(l.metricsGetHeight(m))/pangoScale-f.Ascent-f.Descent, 0)
		}
		l.metricsUnref(m)
	}
	e.fonts[font] = f
	return f
}

func (e *pangoEngine) shape(text []rune, style Style, spans []Span, width float32, rtl, wholeWords bool) []shapedLine {
	if len(text) == 0 {
		return nil
	}
	l := &pangoLib
	utf8, index := utf8Text(text)
	dir := int32(pangoDirLTR)
	if rtl {
		dir = pangoDirRTL
	}
	l.contextSetBaseDir(e.context, dir)
	layout := l.layoutNew(e.context)
	if layout == 0 {
		return nil
	}
	defer l.gObjectUnref(layout)
	l.layoutSetAutoDir(layout, int32(gFalse))
	l.layoutSetFontDescription(layout, e.desc(style))
	l.layoutSetText(layout, &utf8[0], int32(len(utf8)))
	if attrs := e.attributes(style, spans, text); attrs != 0 {
		l.layoutSetAttributes(layout, attrs)
		l.attrListUnref(attrs)
	}
	if width > 0 {
		// Round up to whole Pango units: truncating can make the box a
		// thousandth of a dip narrower than the text measured within it.
		// Pango also applies the letter spacing after the text's last
		// glyph when wrapping but trims it from the measured line, so
		// the box needs the spacing back to hold that line; the other
		// engines keep the trailing spacing in their measurements.
		w := math.Ceil(float64(width) * pangoScale)
		if style.LetterSpacing > 0 {
			w += math.Round(float64(style.LetterSpacing) * pangoScale)
		}
		l.layoutSetWidth(layout, int32(min(w, math.MaxInt32)))
		if wholeWords {
			l.layoutSetWrap(layout, pangoWrapWord)
		} else {
			l.layoutSetWrap(layout, pangoWrapWordChar)
		}
	}
	at := func(b int32) int { return index[max(0, min(int(b), len(utf8)))] }
	n := l.layoutGetLineCount(layout)
	lines := make([]shapedLine, 0, n)
	for i := range n {
		line := l.layoutGetLineReadonly(layout, i)
		if line == nil {
			continue
		}
		sl := shapedLine{start: at(line.startIndex), end: at(line.startIndex + line.length)}
		x := int32(0)
		// The runs are in visual order, each with its glyphs.
		for r := line.runs; r != nil; r = r.next {
			run := (*pangoGlyphItem)(r.data)
			item, gs := run.item, run.glyphs
			f := e.fontOf(item.font)
			sr := shapedRun{font: f, start: at(item.offset), end: at(item.offset + item.length)}
			if gs.numGlyphs > 0 {
				infos := unsafe.Slice(gs.glyphs, gs.numGlyphs)
				clusters := unsafe.Slice(gs.logClusters, gs.numGlyphs)
				sr.glyphs = make([]Glyph, len(infos))
				for j, g := range infos {
					cluster := at(item.offset + clusters[j])
					// Pango empties the space ending a wrapped line; the
					// other engines keep it, as editors need.
					if g.glyph == pangoGlyphEmpty && cluster < len(text) && unicode.IsSpace(text[cluster]) {
						g.glyph, g.width = e.space(item.font, text[cluster])
					}
					sr.glyphs[j] = Glyph{
						Font: f, ID: g.glyph,
						X: float32(x+g.xOffset) / pangoScale, Y: float32(g.yOffset) / pangoScale,
						Advance: float32(g.width) / pangoScale,
						Cluster: cluster, RTL: item.level&1 != 0,
					}
					x += g.width
				}
			}
			sl.runs = append(sl.runs, sr)
		}
		lines = append(lines, sl)
	}
	return lines
}

// language returns the PangoLanguage of a tag, Pango's default one, of
// the user's locale, for "". Pango keeps them for good.
func (e *pangoEngine) language(tag string) uintptr {
	if tag == "" {
		return pangoLib.languageGetDefault()
	}
	if p, ok := e.languages[tag]; ok {
		return p
	}
	p := pangoLib.languageFromString(tag)
	e.languages[tag] = p
	return p
}

// attributes returns Pango's attributes of a style's letter spacing,
// OpenType features and language, over the whole text, and of its spans'
// styles, over their runes; 0 without any.
func (e *pangoEngine) attributes(style Style, spans []Span, text []rune) uintptr {
	l := &pangoLib
	var list uintptr
	// add inserts an attribute over bytes from to to, or the whole text
	// when to is 0.
	add := func(attr uintptr, from, to int) {
		if attr == 0 {
			return
		}
		if list == 0 {
			list = l.attrListNew()
		}
		if to > 0 {
			// PangoAttribute: start_index 8, end_index 12.
			p := *(*unsafe.Pointer)(unsafe.Pointer(&attr))
			*(*uint32)(unsafe.Add(p, 8)) = uint32(from)
			*(*uint32)(unsafe.Add(p, 12)) = uint32(to)
		}
		l.attrListInsert(list, attr)
	}
	features := func(list string) uintptr {
		fs := features(list)
		if len(fs) == 0 || l.attrFontFeaturesNew == nil {
			return 0
		}
		// HarfBuzz's syntax, which Pango hands over.
		var b strings.Builder
		for i, f := range fs {
			if i > 0 {
				b.WriteByte(',')
			}
			b.Write(f.tag[:])
			b.WriteByte('=')
			b.WriteString(strconv.FormatUint(uint64(f.value), 10))
		}
		return l.attrFontFeaturesNew(b.String())
	}
	spacing := func(v float32) uintptr {
		if v == 0 {
			return 0
		}
		return l.attrLetterSpacingNew(int32(math.Round(float64(v) * pangoScale)))
	}
	add(spacing(style.LetterSpacing), 0, 0)
	add(features(style.Features), 0, 0)
	if style.Lang != "" {
		add(l.attrLanguageNew(e.language(style.Lang)), 0, 0)
	}
	runs := langRuns(text, style.Lang, spans)
	if len(spans) == 0 && len(runs) == 0 {
		return list
	}
	// The byte each rune starts at.
	at := make([]int, len(text)+1)
	for i, r := range text {
		at[i+1] = at[i] + utf8.RuneLen(r)
	}
	from := 0
	for _, sp := range spans {
		to := max(from, min(sp.End, len(text)))
		b0, b1 := at[from], at[to]
		start := from
		from = to
		if b1 == b0 {
			continue
		}
		if sp.Family != "" {
			add(l.attrFamilyNew(pangoFamily(sp.Family, e.uiFamily)), b0, b1)
		}
		if sp.Size > 0 {
			add(l.attrSizeAbsoluteNew(int32(math.Round(float64(sp.Size)*pangoScale))), b0, b1)
		}
		if sp.Weight > 0 {
			add(l.attrWeightNew(int32(min(sp.Weight, 999))), b0, b1)
		}
		if sp.Italic {
			add(l.attrStyleNew(pangoStyleItalic), b0, b1)
		}
		add(spacing(sp.LetterSpacing), b0, b1)
		add(features(sp.Features), b0, b1)
		if sp.ObjectWidth > 0 {
			// Each rune an inline object: a shape of its size standing on
			// the baseline, in its own attribute.
			box := pangoRectangle{y: -int32(math.Round(float64(sp.ObjectHeight) * pangoScale)),
				width: int32(math.Round(float64(sp.ObjectWidth) * pangoScale)), height: int32(math.Round(float64(sp.ObjectHeight) * pangoScale))}
			for k := start; k < to; k++ {
				add(l.attrShapeNew(&box, &box), at[k], at[k+1])
			}
		}
	}
	// The runs in other languages, over the whole text's: an attribute
	// added later over a range wins.
	for _, r := range runs {
		add(l.attrLanguageNew(e.language(r.lang)), at[r.start], at[r.end])
	}
	return list
}

// space returns the glyph and advance, in Pango units, of a whitespace
// rune in a font, from HarfBuzz as Pango shapes it.
func (e *pangoEngine) space(font uintptr, r rune) (uint32, int32) {
	l := &pangoLib
	if l.fontGetHBFont == nil || l.hbNominalGlyph == nil || l.hbGlyphHAdvance == nil {
		return pangoGlyphEmpty, 0
	}
	hb := l.fontGetHBFont(font)
	var glyph uint32
	if hb == 0 || l.hbNominalGlyph(hb, r, &glyph) == 0 {
		return pangoGlyphEmpty, 0
	}
	return glyph, l.hbGlyphHAdvance(hb, glyph)
}

// scaledFont returns a cairo font drawing a Font's glyphs at scale pixels
// per DIP, its user space in pixels.
// scaledFont returns the cairo font of f at scale pixels per DIP, with the
// context's options, but grayscale antialiasing instead of subpixel unless
// subpixel.
func (e *pangoEngine) scaledFont(f *Font, scale float32, subpixel bool) uintptr {
	key := scaledKey{f, scale, subpixel}
	if sf, ok := e.scaled[key]; ok {
		return sf
	}
	l := &pangoLib
	if len(e.scaled) >= 256 {
		for _, sf := range e.scaled {
			l.cairoScaledFontDestroy(sf)
		}
		clear(e.scaled)
	}
	src := l.fontGetScaledFont(f.native)
	if src == 0 {
		return 0
	}
	var m cairoMatrix
	l.cairoScaledFontMatrix(src, &m)
	s := float64(scale)
	m.xx, m.yx, m.xy, m.yy = m.xx*s, m.yx*s, m.xy*s, m.yy*s
	identity := cairoMatrix{xx: 1, yy: 1}
	opts := l.cairoFontOptionsCreate()
	l.cairoScaledFontOptions(src, opts)
	if !subpixel && e.antialias == cairoAntialiasSubpixel {
		l.cairoFontOptionsAntialias(opts, cairoAntialiasGray)
	}
	sf := l.cairoScaledFontCreate(l.cairoScaledFontFace(src), &m, &identity, opts)
	l.cairoFontOptionsDestroy(opts)
	if sf == 0 || l.cairoScaledFontStatus(sf) != cairoStatusSuccess {
		return 0
	}
	e.scaled[key] = sf
	return sf
}

// textParams leaves coverage as it is: cairo blends glyphs in sRGB space,
// as renderers do, with subpixel antialiasing where the desktop's settings
// ask for it.
func (e *pangoEngine) textParams() (scene.TextParams, bool) {
	return scene.TextParams{}, e.antialias == cairoAntialiasSubpixel
}

// positions puts glyphs on whole pixels, as GTK does: Pango rounds their
// positions, and labels sit on whole pixels.
func (e *pangoEngine) positions(*Font, float32, bool) (int, bool) { return 1, false }

// decorate places underlines and strikethroughs as GTK draws a label's
// (PangoRenderer): at the positions and thicknesses of the run's font,
// whole pixels, from the label's baseline, an underline along the ink and
// advances of the run, a strikethrough along its ink.
func (e *pangoEngine) decorate(r decoRange) []Stroke {
	s := r.scale
	base := e.baseline(r.baseline() * s)
	var out []Stroke
	r.runs(func(f *Font, i, j int) {
		ink0, ink1 := float32(math.MaxFloat32), float32(-math.MaxFloat32)
		for _, g := range r.line.Glyphs[i:j] {
			if a, b, ok := e.ink(f, g.ID); ok {
				ink0, ink1 = min(ink0, g.X+a), max(ink1, g.X+b)
			}
		}
		x0, x1 := r.span(i, j)
		top, thick := f.strikeTop, f.strikeThick
		if r.d == Underline {
			top, thick = f.underlineTop, f.underlineThick
			x0, x1 = min(x0, ink0), max(x1, ink1)
		} else {
			if ink0 > ink1 {
				return
			}
			x0, x1 = ink0, ink1
		}
		st := Stroke{X0: (r.x + x0) * s, X1: (r.x + x1) * s, Top: base - top*s, Bottom: base - top*s + thick*s}
		// Runs whose lines meet make one, as PangoRenderer draws them.
		if n := len(out); n > 0 && out[n-1].Top == st.Top && out[n-1].Bottom == st.Bottom {
			out[n-1].X1 = max(out[n-1].X1, st.X1)
			return
		}
		out = append(out, st)
	})
	return out
}

// join: cairo adds the coverage of a run's glyphs before compositing it.
func (e *pangoEngine) join() bool { return true }

// ink returns how far left and right of its origin a glyph's ink reaches,
// in DIPs.
func (e *pangoEngine) ink(f *Font, id uint32) (a, b float32, ok bool) {
	if id == pangoGlyphEmpty || id&pangoGlyphUnknow != 0 {
		return 0, 0, false
	}
	var ink, logical pangoRectangle
	pangoLib.fontGetGlyphExtents(f.native, id, &ink, &logical)
	if ink.width <= 0 {
		return 0, 0, false
	}
	return float32(ink.x) / pangoScale, float32(ink.x+ink.width) / pangoScale, true
}

// baseline is the whole pixel at or above y, as a GTK label's.
func (e *pangoEngine) baseline(y float32) float32 { return float32(math.Floor(float64(y) + 1e-3)) }

func (e *pangoEngine) glyph(f *Font, id uint32, scale, dx float32, _ Shade, subpixel bool) bitmap {
	// Pango's empty glyphs, and the boxes it draws for missing ones, are
	// not the font's.
	if id == pangoGlyphEmpty || id&pangoGlyphUnknow != 0 {
		return bitmap{}
	}
	b := e.draw(f, id, scale, dx, false)
	if b.color || b.pix == nil || !subpixel || e.antialias != cairoAntialiasSubpixel {
		return b
	}
	return e.draw(f, id, scale, dx, true)
}

// draw rasterizes a glyph with cairo: with subpixel antialiasing if
// subpixel, else in grayscale, or in color for color fonts.
func (e *pangoEngine) draw(f *Font, id uint32, scale, dx float32, subpixel bool) bitmap {
	sf := e.scaledFont(f, scale, subpixel)
	if sf == 0 {
		return bitmap{}
	}
	l := &pangoLib
	g := cairoGlyph{index: uint64(id)}
	var ext cairoTextExtents
	l.cairoScaledFontExtents(sf, &g, 1, &ext)
	if ext.width <= 0 || ext.height <= 0 {
		return bitmap{}
	}
	// The LCD filter spreads subpixel glyphs a pixel further.
	pad := 1
	if subpixel {
		pad = 2
	}
	left := int(math.Floor(ext.xBearing+float64(dx))) - pad
	top := int(math.Floor(ext.yBearing)) - pad
	right := int(math.Ceil(ext.xBearing+ext.width+float64(dx))) + pad
	bottom := int(math.Ceil(ext.yBearing+ext.height)) + pad
	w, h := right-left, bottom-top
	if w > 2048 || h > 2048 {
		return bitmap{}
	}
	surface := l.cairoImageSurfaceCreate(cairoFormatARGB32, int32(w), int32(h))
	defer l.cairoSurfaceDestroy(surface)
	cr := l.cairoCreate(surface)
	l.cairoSetScaledFont(cr, sf)
	l.cairoSetSourceRGBA(cr, 1, 1, 1, 1)
	g.x, g.y = float64(dx)-float64(left), -float64(top)
	l.cairoShowGlyphs(cr, &g, 1)
	l.cairoDestroy(cr)
	l.cairoSurfaceFlush(surface)
	data := l.cairoImageSurfaceData(surface)
	stride := int(l.cairoImageSurfaceStride(surface))
	if data == nil {
		return bitmap{}
	}
	src := unsafe.Slice(data, stride*h)
	if subpixel {
		// White glyphs with subpixel antialiasing: the coverage of each
		// subpixel in R, G and B (B, G, R in memory).
		b := bitmap{left: left, top: top, w: w, h: h, pix: make([]byte, 4*w*h), subpixel: true}
		for y := range h {
			row := src[y*stride:]
			for x := range w {
				p, q := b.pix[4*(y*w+x):], row[4*x:]
				p[0], p[1], p[2] = q[2], q[1], q[0]
				p[3] = uint8((uint16(q[0]) + uint16(q[1]) + uint16(q[2]) + 1) / 3)
			}
		}
		return b
	}
	// White glyphs of a plain font come out gray: B, G, R and A (in
	// memory order) are the same, and the alpha is the coverage. Others
	// are colored.
	color := false
	for y := 0; y < h && !color; y++ {
		row := src[y*stride : y*stride+4*w]
		for x := 0; x < 4*w; x += 4 {
			if a := row[x+3]; row[x] != a || row[x+1] != a || row[x+2] != a {
				color = true
				break
			}
		}
	}
	b := bitmap{left: left, top: top, w: w, h: h, color: color}
	if color {
		b.pix = make([]byte, 4*w*h)
		for y := range h {
			row := src[y*stride:]
			for x := range w {
				p, q := b.pix[4*(y*w+x):], row[4*x:]
				p[0], p[1], p[2], p[3] = q[2], q[1], q[0], q[3]
			}
		}
		return b
	}
	b.pix = make([]byte, w*h)
	for y := range h {
		row := src[y*stride:]
		for x := range w {
			b.pix[y*w+x] = row[4*x+3]
		}
	}
	return b
}

// fontMapFamilies lists the font map's families as fontconfig knows
// them, with the app's under their own names, without the generic families
// Pango adds (Sans, Serif, Monospace, System-ui), which are aliases rather
// than fonts.
func (e *pangoEngine) fontMapFamilies() []FontFamily {
	l := &pangoLib
	var arr *uintptr
	var n int32
	l.fontMapListFamilies(e.fontMap, &arr, &n)
	if arr != nil {
		defer l.gFree(uintptr(unsafe.Pointer(arr)))
	}
	var out []FontFamily
	for _, fam := range unsafe.Slice(arr, int(n)) {
		name := cString(l.fontFamilyGetName(fam))
		switch strings.ToLower(name) {
		case "", "sans", "serif", "monospace", "system-ui":
			continue
		}
		out = append(out, FontFamily{Name: name, Monospaced: l.fontFamilyIsMonospace(fam)})
	}
	return out
}

// appFamilies lists the families of the app's fonts, monospaced as
// fontconfig says their own families are.
func (e *pangoEngine) appFamilies() []FontFamily {
	if len(e.names) == 0 {
		return nil
	}
	mono := map[string]bool{} // by lowercased name
	for _, f := range e.fontMapFamilies() {
		mono[strings.ToLower(f.Name)] = f.Monospaced
	}
	out := make([]FontFamily, len(e.names))
	for i, name := range e.names {
		out[i] = FontFamily{Name: name, Monospaced: mono[strings.ToLower(e.appFamily[strings.ToLower(name)])]}
	}
	return out
}

// systemFamilies lists the font map's families without the app's, which it
// holds under their own names, as it is: a Pango font map is not
// thread-safe, so it lists them under the lock, as quickly as fontconfig
// does.
func (e *pangoEngine) systemFamilies() func() []FontFamily {
	app := map[string]bool{}
	for _, own := range e.appFamily {
		app[strings.ToLower(own)] = true
	}
	var out []FontFamily
	for _, f := range e.fontMapFamilies() {
		if !app[strings.ToLower(f.Name)] {
			out = append(out, f)
		}
	}
	return func() []FontFamily { return out }
}

// register adds a font file to fontconfig's fonts for the app, from the
// user's cache directory, and an alias for family.
func (e *pangoEngine) register(data []byte, family string) error {
	l := &pangoLib
	if l.fcConfigAppFontAddFile == nil || l.fontMapConfigChanged == nil {
		return errors.New("mygo: adding fonts needs fontconfig and PangoFT2")
	}
	sum := sha256.Sum256(data)
	dir, err := os.UserCacheDir()
	if err != nil {
		dir = os.TempDir()
	}
	dir = filepath.Join(dir, "mygo", "fonts")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mygo: cannot add font: %w", err)
	}
	path := filepath.Join(dir, hex.EncodeToString(sum[:16])+".font")
	if _, err := os.Stat(path); err != nil {
		tmp := path + ".tmp"
		if err := os.WriteFile(tmp, data, 0o644); err != nil {
			return fmt.Errorf("mygo: cannot add font: %w", err)
		}
		if err := os.Rename(tmp, path); err != nil {
			return fmt.Errorf("mygo: cannot add font: %w", err)
		}
	}
	config := l.fcConfigGetCurrent()
	if !e.registered[path] {
		if l.fcConfigAppFontAddFile(config, path) == 0 {
			return errors.New("mygo: cannot add font: not a font fontconfig reads")
		}
		e.registered[path] = true
	}
	names := fontFamilies(config, path)
	if family != "" {
		if len(names) == 0 {
			return errors.New("mygo: cannot add font: no family name")
		}
		if !strings.EqualFold(names[0], family) {
			xml := `<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><alias binding="same"><family>` +
				xmlEscape(family) + `</family><prefer><family>` + xmlEscape(names[0]) + `</family></prefer></alias></fontconfig>`
			if l.fcConfigParseFromMemory == nil || l.fcConfigParseFromMemory(config, xml, int32(gTrue)) == 0 {
				return errors.New("mygo: cannot add font under another family name")
			}
		}
		// Listed under the name the app gave it, which is the one it uses.
		e.appFamily[strings.ToLower(family)] = names[0]
		e.names = append(e.names, family)
	} else {
		for _, name := range names {
			e.appFamily[strings.ToLower(name)] = name
			e.names = append(e.names, name)
		}
	}
	l.fontMapConfigChanged(e.fontMap)
	// Fonts and descriptions found before may resolve differently now.
	for _, d := range e.descs {
		l.descFree(d)
	}
	clear(e.descs)
	return nil
}

// fontFamilies returns the families of the app's fonts from a file.
func fontFamilies(config uintptr, path string) []string {
	l := &pangoLib
	if l.fcConfigGetFonts == nil || l.fcPatternGetString == nil {
		return nil
	}
	set := l.fcConfigGetFonts(config, fcSetApplication)
	if set == nil || set.nfont <= 0 {
		return nil
	}
	var names []string
	for _, p := range unsafe.Slice(set.fonts, set.nfont) {
		var file, family *byte
		if l.fcPatternGetString(p, "file", 0, &file) != fcResultMatch || cString(file) != path {
			continue
		}
		if l.fcPatternGetString(p, "family", 0, &family) == fcResultMatch {
			names = append(names, cString(family))
		}
	}
	return names
}

func cString(p *byte) string {
	if p == nil {
		return ""
	}
	n := 0
	for *(*byte)(unsafe.Add(unsafe.Pointer(p), n)) != 0 {
		n++
	}
	return string(unsafe.Slice(p, n))
}

func xmlEscape(s string) string {
	return strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", `"`, "&quot;").Replace(s)
}

func (e *pangoEngine) fontCount() int { return len(e.fonts) }

// forgetFonts lets go of the Pango and cairo fonts of the Fonts made so
// far.
func (e *pangoEngine) forgetFonts() {
	l := &pangoLib
	for _, sf := range e.scaled {
		l.cairoScaledFontDestroy(sf)
	}
	clear(e.scaled)
	for _, f := range e.fonts {
		l.gObjectUnref(f.native)
	}
	clear(e.fonts)
}
