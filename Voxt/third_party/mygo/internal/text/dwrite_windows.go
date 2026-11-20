//go:build windows && (amd64 || arm64)

package text

import (
	"errors"
	"fmt"
	"log"
	"math"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"sync"
	"syscall"
	"unsafe"

	"github.com/ebitengine/purego"
	"github.com/egoist/mygo/internal/scene"
)

// DirectWrite lays out paragraphs with IDWriteTextLayout, which finds the
// fonts, falls back to others for what a font lacks, shapes, and breaks
// lines, and hands its glyph runs to a renderer (an IDWriteTextRenderer
// implemented here); IDWriteGlyphRunAnalysis rasterizes glyphs, layer by
// layer for color fonts such as Segoe UI Emoji. Fonts are those of the
// system's font collection, memory mapped and shared by every process,
// and those the app registers, in a collection of its own.
//
// Glyphs look as Direct2D draws them with the system's settings: in the
// rendering mode and grid fitting DirectWrite recommends for their font
// and size, with ClearType where the system smooths fonts with it (on
// opaque backgrounds), aliased where it does not smooth them, and blended
// with the gamma and contrast of the system's rendering parameters, which
// renderers apply (scene.TextParams).

func newEngine() engine {
	e, err := newDWrite()
	if err != nil {
		log.Printf("mygo: %v: text will not show", err)
		return &stubEngine{}
	}
	return e
}

var (
	dwriteDLL           = syscall.NewLazyDLL(systemDir() + `\dwrite.dll`)
	procDWriteCreate    = dwriteDLL.NewProc("DWriteCreateFactory")
	kernel32DLL         = syscall.NewLazyDLL(systemDir() + `\kernel32.dll`)
	procUserLocaleName  = kernel32DLL.NewProc("GetUserDefaultLocaleName")
	procSystemDirectory = syscall.NewLazyDLL("kernel32.dll").NewProc("GetSystemDirectoryW")
	procSystemParams    = syscall.NewLazyDLL(systemDir() + `\user32.dll`).NewProc("SystemParametersInfoW")
)

func systemDir() string {
	buf := make([]uint16, 260)
	n, _, _ := procSystemDirectory.Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	if n == 0 || int(n) > len(buf) {
		return `C:\Windows\System32`
	}
	return syscall.UTF16ToString(buf[:n])
}

type guid struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}

var (
	iidIUnknown             = guid{0x00000000, 0x0000, 0x0000, [8]byte{0xc0, 0, 0, 0, 0, 0, 0, 0x46}}
	iidIDWriteFactory       = guid{0xb859ee5a, 0xd838, 0x4b5b, [8]byte{0xa2, 0xe8, 0x1a, 0xdc, 0x7d, 0x93, 0xdb, 0x48}}
	iidIDWriteFactory2      = guid{0x0439fc60, 0xca44, 0x4994, [8]byte{0x8d, 0xee, 0x3a, 0x9a, 0xf7, 0xb7, 0x32, 0xec}}
	iidIDWriteFactory5      = guid{0x958db99a, 0xbe2a, 0x4f09, [8]byte{0xaf, 0x7d, 0x65, 0x18, 0x98, 0x03, 0xd1, 0xd3}}
	iidIDWriteFont1         = guid{0xacd16696, 0x8c14, 0x4f5d, [8]byte{0x87, 0x7e, 0xfe, 0x3f, 0xc1, 0xd3, 0x27, 0x38}}
	iidIDWriteFontFace2     = guid{0xd8b768ff, 0x64bc, 0x4e66, [8]byte{0x98, 0x2b, 0xec, 0x8e, 0x87, 0xf6, 0x93, 0xf7}}
	iidIDWriteFontFace3     = guid{0xd37d7598, 0x09be, 0x4222, [8]byte{0xa2, 0x36, 0x20, 0x81, 0x34, 0x1c, 0xc1, 0xf2}}
	iidIDWriteFactory3      = guid{0x9a1b41c3, 0xd3bb, 0x466a, [8]byte{0x87, 0xfc, 0xfe, 0x67, 0x55, 0x6a, 0x3b, 0x65}}
	iidIDWriteRendParams1   = guid{0x94413cf4, 0xa6fc, 0x4248, [8]byte{0x8b, 0x50, 0x66, 0x74, 0x34, 0x8f, 0xca, 0xd3}}
	iidIDWritePixelSnapping = guid{0xeaf3a2da, 0xecf4, 0x4d24, [8]byte{0xb6, 0x44, 0xb3, 0x4f, 0x68, 0x42, 0x02, 0x4b}}
	iidIDWriteTextRenderer  = guid{0xef8a8135, 0x5cc6, 0x45fe, [8]byte{0x88, 0x25, 0xc5, 0xa0, 0x72, 0x4e, 0xb8, 0x19}}
	iidIDWriteTextLayout1   = guid{0x9064d822, 0x80a7, 0x465c, [8]byte{0xa9, 0x86, 0xdf, 0x65, 0xf7, 0x8b, 0x8f, 0xeb}}
	iidIDWriteTextFormat1   = guid{0x5f174b49, 0x0d8b, 0x4cfb, [8]byte{0x8b, 0xca, 0xf1, 0xcc, 0xe9, 0xd0, 0x6c, 0x67}}
)

// Vtable indices, from the Windows SDK headers.
const (
	// IUnknown
	comQueryInterface = 0
	comAddRef         = 1
	comRelease        = 2

	// IDWriteFactory
	factoryGetSystemFontCollection = 3
	factoryRegisterFontFileLoader  = 13
	factoryCreateRenderingParams   = 10
	factoryCreateTextFormat        = 15
	factoryCreateTypography        = 16
	factoryCreateTextLayout        = 18
	factoryCreateGlyphRunAnalysis  = 23
	// IDWriteFactory2
	factory2GetSystemFontFallback  = 26
	factory2CreateFallbackBuilder  = 27
	factory2TranslateColorGlyphRun = 28
	factory2CreateGlyphRunAnalysis = 30
	fallbackAddMapping             = 3
	fallbackAddMappings            = 4
	fallbackCreate                 = 5
	format1SetFontFallback         = 34
	// IDWriteFactory3
	factory3CreateGlyphRunAnalysis          = 31
	factory3CreateFontCollectionFromFontSet = 37
	// IDWriteFactory5
	factory5CreateFontSetBuilder         = 43
	factory5CreateInMemoryFontFileLoader = 44

	// IDWriteFontCollection
	collectionGetFontFamilyCount = 3
	collectionGetFontFamily      = 4
	collectionFindFamilyName     = 5
	collectionGetFontFromFace    = 6
	// IDWriteFontFamily
	familyGetFamilyNames       = 6
	familyGetFirstMatchingFont = 7
	// IDWriteFont
	fontGetFontFamily  = 3
	fontCreateFontFace = 13
	// IDWriteFont1
	font1IsMonospacedFont = 17
	// IDWriteFontFace
	faceGetMetrics = 8
	// IDWriteFontFace2
	face2IsColorFont                 = 30
	face2GetRecommendedRenderingMode = 34
	// IDWriteFontFace3
	face3GetRecommendedRenderingMode = 44
	// IDWriteRenderingParams, and IDWriteRenderingParams1
	paramsGetGamma                      = 3
	paramsGetEnhancedContrast           = 4
	paramsGetClearTypeLevel             = 5
	paramsGetPixelGeometry              = 6
	paramsGetRenderingMode              = 7
	params1GetGrayscaleEnhancedContrast = 8
	// IDWriteLocalizedStrings
	stringsFindLocaleName  = 4
	stringsGetStringLength = 7
	stringsGetString       = 8

	// IDWriteTextFormat, and IDWriteTextLayout which extends it
	formatSetTextAlignment     = 3
	formatSetWordWrapping      = 5
	formatSetReadingDirection  = 6
	layoutSetFontCollection    = 30
	layoutSetFontFamilyName    = 31
	layoutSetFontWeight        = 32
	layoutSetFontStyle         = 33
	layoutSetFontSize          = 35
	layoutSetInlineObject      = 39
	layoutSetTypography        = 40
	layoutSetLocaleName        = 41
	layoutDraw                 = 58
	layoutGetLineMetrics       = 59
	layoutHitTestTextPosition  = 65
	layout1SetCharacterSpacing = 69
	typographyAddFontFeature   = 3
	glyphsGetAlphaTextureBound = 3
	glyphsCreateAlphaTexture   = 4
	colorRunsMoveNext          = 3
	colorRunsGetCurrentRun     = 4
	// IDWriteFontSetBuilder1
	builderCreateFontSet = 6
	builderAddFontFile   = 7
	// IDWriteInMemoryFontFileLoader
	loaderCreateInMemoryFontFileReference = 4
)

// DirectWrite enumerations.
const (
	fontStyleNormal   = 0
	fontStyleItalic   = 2
	fontStretchNormal = 5

	wordWrappingWrap      = 0
	wordWrappingNoWrap    = 1
	wordWrappingWholeWord = 3

	readingDirectionRTL   = 1
	textAlignmentTrailing = 1

	renderingModeAliased          = 1
	renderingModeGDIClassic       = 2
	renderingModeGDINatural       = 3
	renderingModeNatural          = 4
	renderingModeNaturalSymmetric = 5
	renderingModeOutline          = 6
	renderingModeDownsampled      = 7 // DWRITE_RENDERING_MODE1_NATURAL_SYMMETRIC_DOWNSAMPLED
	measuringModeNatural          = 0
	gridFitModeDefault            = 0
	antialiasModeClearType        = 0
	antialiasModeGrayscale        = 1
	outlineThresholdAntialiased   = 0
	outlineThresholdAliased       = 1
	textureAliased1x1             = 0
	textureClearType3x1           = 1
	pixelGeometryFlat             = 0
	pixelGeometryBGR              = 2

	spiGetFontSmoothing      = 0x004A
	spiGetFontSmoothingType  = 0x200A
	feFontSmoothingClearType = 2

	errNoColor                     = 0x8898500C
	errNotSufficientBuffer         = 0x8007007A
	errNoInterface                 = 0x80004002
	factoryTypeShared              = 0
	paletteIndexForeground         = 0xFFFF
	maxLayoutHeight        float32 = 1 << 24
)

// DirectWrite structs.
type (
	dwGlyphRun struct {
		fontFace      uintptr
		fontEmSize    float32
		glyphCount    uint32
		glyphIndices  *uint16
		glyphAdvances *float32
		glyphOffsets  *dwGlyphOffset
		isSideways    int32
		bidiLevel     uint32
	}
	dwGlyphOffset struct {
		advanceOffset, ascenderOffset float32
	}
	dwGlyphRunDescription struct {
		localeName   *uint16
		text         *uint16
		textLength   uint32
		clusterMap   *uint16
		textPosition uint32
	}
	dwColorGlyphRun struct {
		glyphRun                         dwGlyphRun
		glyphRunDescription              uintptr
		baselineOriginX, baselineOriginY float32
		runColor                         [4]float32
		paletteIndex                     uint16
	}
	dwLineMetrics struct {
		length, trailingWhitespaceLength, newlineLength uint32
		height, baseline                                float32
		isTrimmed                                       int32
	}
	dwHitTestMetrics struct {
		textPosition, length     uint32
		left, top, width, height float32
		bidiLevel                uint32
		isText, isTrimmed        int32
	}
	dwFontMetrics struct {
		designUnitsPerEm, ascent, descent uint16
		lineGap                           int16
		capHeight, xHeight                uint16
		underlinePosition                 int16
		underlineThickness                uint16
		strikethroughPosition             int16
		strikethroughThickness            uint16
	}
	dwRect struct{ left, top, right, bottom int32 }
)

// ptr converts an address from DirectWrite to a pointer.
func ptr(p uintptr) unsafe.Pointer { return *(*unsafe.Pointer)(unsafe.Pointer(&p)) }

func vtable(obj uintptr, i int) uintptr {
	return *(*uintptr)(unsafe.Add(ptr(*(*uintptr)(ptr(obj))), i*int(unsafe.Sizeof(uintptr(0)))))
}

// call calls method i of COM object obj. Pointers converted to uintptr in
// the call stay valid until it returns (go:uintptrescapes).
//
//go:uintptrescapes
func call(obj uintptr, i int, args ...uintptr) uintptr {
	var a [16]uintptr
	a[0] = obj
	n := copy(a[1:], args)
	r, _, _ := syscall.SyscallN(vtable(obj, i), a[:n+1]...)
	return r
}

type methodKey struct {
	fn uintptr
	t  reflect.Type
}

var methods sync.Map // methodKey → func

// method returns method i of COM object obj as a function of type F, for
// methods with floating point arguments: syscall does not pass them on
// ARM64, purego does.
func method[F any](obj uintptr, i int) F {
	key := methodKey{vtable(obj, i), reflect.TypeFor[F]()}
	if f, ok := methods.Load(key); ok {
		return f.(F)
	}
	var f F
	purego.RegisterFunc(&f, key.fn)
	methods.Store(key, f)
	return f
}

func failed(hr uintptr) bool { return int32(uint32(hr)) < 0 }

func release(obj uintptr) {
	if obj != 0 {
		call(obj, comRelease)
	}
}

func queryInterface(obj uintptr, iid *guid) uintptr {
	var out uintptr
	if failed(call(obj, comQueryInterface, uintptr(unsafe.Pointer(iid)), uintptr(unsafe.Pointer(&out)))) {
		return 0
	}
	return out
}

func utf16z(s string) []uint16 {
	u, err := syscall.UTF16FromString(s)
	if err != nil {
		return []uint16{0}
	}
	return u
}

// dwGeneric are the families the generic families stand for, best first.
var dwGeneric = map[string][]string{
	"system-ui":  {"Segoe UI", "Tahoma", "Arial"},
	"sans-serif": {"Segoe UI", "Arial"},
	"serif":      {"Times New Roman", "Cambria", "Georgia"},
	"monospace":  {"Cascadia Mono", "Consolas", "Courier New"},
}

type dwrite struct {
	shapeScratch
	factory  uintptr // IDWriteFactory
	factory2 uintptr // IDWriteFactory2, from Windows 8.1
	factory3 uintptr // IDWriteFactory3, from Windows 10
	factory5 uintptr // IDWriteFactory5, from Windows 10 1703
	system   uintptr // the system font collection
	locale   []uint16

	exists  map[string]bool // whether the system has a family
	formats map[formatKey]uintptr
	// fallbacks are the font fallbacks of family lists, 0 for a list of
	// one family.
	fallbacks map[string]uintptr
	faces     map[faceKey]uintptr
	fonts     map[fontKey]*Font
	color     map[uintptr]bool // whether a font face has color glyphs
	thin      map[uintptr]bool // whether a font face is too thin for antialiasing

	// The system's settings for text (settings): its rendering parameters,
	// the correction renderers apply, whether it smooths fonts with
	// ClearType, on a panel of blue, green and red subpixels, at what
	// level, and whether it does not smooth them.
	settingsRead bool
	params       uintptr // IDWriteRenderingParams
	text         scene.TextParams
	clearType    bool
	bgr          bool
	clearLevel   float32
	aliased      bool

	// The fonts the app registers: their files, a collection of them,
	// the family name of each name they are registered under, and the
	// names they are listed under: the one the app gave, or their own.
	loader  uintptr
	files   []uintptr
	custom  uintptr
	aliases map[string]string
	names   []string

	scratch []byte
}

type formatKey struct {
	list   string // the family list, whose others the fallback tries
	family string
	custom bool
	weight int
	italic bool
	size   float32
	lang   string // Style.Lang, "" for the user's locale
}

type faceKey struct {
	family string
	custom bool
	weight int
	italic bool
}

type fontKey struct {
	face uintptr
	size float32
}

func newDWrite() (*dwrite, error) {
	if err := procDWriteCreate.Find(); err != nil {
		return nil, err
	}
	e := &dwrite{
		exists:    map[string]bool{},
		formats:   map[formatKey]uintptr{},
		fallbacks: map[string]uintptr{},
		faces:     map[faceKey]uintptr{},
		fonts:     map[fontKey]*Font{},
		color:     map[uintptr]bool{},
		thin:      map[uintptr]bool{},
		aliases:   map[string]string{},
	}
	hr, _, _ := procDWriteCreate.Call(factoryTypeShared, uintptr(unsafe.Pointer(&iidIDWriteFactory)), uintptr(unsafe.Pointer(&e.factory)))
	if failed(hr) || e.factory == 0 {
		return nil, fmt.Errorf("DWriteCreateFactory failed (HRESULT %#08x)", uint32(hr))
	}
	e.factory2 = queryInterface(e.factory, &iidIDWriteFactory2)
	e.factory3 = queryInterface(e.factory, &iidIDWriteFactory3)
	e.factory5 = queryInterface(e.factory, &iidIDWriteFactory5)
	if hr := call(e.factory, factoryGetSystemFontCollection, uintptr(unsafe.Pointer(&e.system)), 0); failed(hr) {
		return nil, fmt.Errorf("IDWriteFactory::GetSystemFontCollection failed (HRESULT %#08x)", uint32(hr))
	}
	buf := make([]uint16, 85)
	if n, _, _ := procUserLocaleName.Call(uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf))); n > 0 {
		e.locale = buf[:n]
	} else {
		e.locale = utf16z("en-US")
	}
	return e, nil
}

// has reports whether a collection has a family.
func has(coll uintptr, family string) bool {
	name := utf16z(family)
	var index uint32
	var exists int32
	hr := call(coll, collectionFindFamilyName, uintptr(unsafe.Pointer(&name[0])), uintptr(unsafe.Pointer(&index)), uintptr(unsafe.Pointer(&exists)))
	return !failed(hr) && exists != 0
}

func (e *dwrite) hasSystem(family string) bool {
	key := strings.ToLower(family)
	ok, seen := e.exists[key]
	if !seen {
		ok = has(e.system, family)
		e.exists[key] = ok
	}
	return ok
}

// family returns the first family of a list the app or the system has,
// and whether it is the app's.
func (e *dwrite) family(list string) (string, bool) {
	for _, f := range familyList(list) {
		if g := generic(f); g != "" {
			for _, name := range dwGeneric[g] {
				if e.hasSystem(name) {
					return name, false
				}
			}
			continue
		}
		if name, ok := e.aliases[strings.ToLower(f)]; ok {
			return name, true
		}
		if e.hasSystem(f) {
			return f, false
		}
	}
	for _, name := range dwGeneric["system-ui"] {
		if e.hasSystem(name) {
			return name, false
		}
	}
	return "Segoe UI", false
}

func (e *dwrite) collection(custom bool) uintptr {
	if custom {
		return e.custom
	}
	return e.system
}

func dwStyle(italic bool) uintptr {
	if italic {
		return fontStyleItalic
	}
	return fontStyleNormal
}

// localeName returns the locale name DirectWrite takes for a language, which
// text formats shape with and the font fallback picks fonts by: the
// language's tag, or for "" the user's locale of regional formats
// (GetUserDefaultLocaleName), as text without a language had.
func (e *dwrite) localeName(lang string) []uint16 {
	if lang == "" {
		return e.locale
	}
	return utf16z(lang)
}

func (e *dwrite) format(style Style) uintptr {
	family, custom := e.family(style.Family)
	key := formatKey{style.Family, family, custom, style.weight(), style.Italic, style.FontSize(), style.Lang}
	if f, ok := e.formats[key]; ok {
		return f
	}
	if len(e.formats) >= 256 {
		for _, f := range e.formats {
			release(f)
		}
		clear(e.formats)
	}
	name := utf16z(family)
	var format uintptr
	hr := method[func(this uintptr, family *uint16, coll uintptr, weight, style, stretch uintptr, size float32, locale *uint16, out *uintptr) uintptr](e.factory, factoryCreateTextFormat)(
		e.factory, &name[0], e.collection(custom), uintptr(key.weight), dwStyle(key.italic), fontStretchNormal, key.size, &e.localeName(key.lang)[0], &format)
	if failed(hr) {
		return 0
	}
	// The families of the list after the one it draws with come before the
	// system's for what that one lacks; IDWriteTextFormat1 came with
	// Windows 8.1.
	if fb := e.fallback(style.Family); fb != 0 {
		if f1 := queryInterface(format, &iidIDWriteTextFormat1); f1 != 0 {
			call(f1, format1SetFontFallback, fb)
			release(f1)
		}
	}
	e.formats[key] = format
	return format
}

// fallback returns a font fallback trying the families of a list after the
// first the app or the system has, before the system's own fallback; 0
// when there are none.
func (e *dwrite) fallback(list string) uintptr {
	if fb, ok := e.fallbacks[list]; ok {
		return fb
	}
	type family struct {
		name   string
		custom bool
	}
	var families []family
	first := true
	for _, f := range familyList(list) {
		if g := generic(f); g != "" {
			// The system's generic families fall back as the system does.
			first = first && !slices.ContainsFunc(dwGeneric[g], e.hasSystem)
			continue
		}
		var fam family
		if name, ok := e.aliases[strings.ToLower(f)]; ok {
			fam = family{name, true}
		} else if e.hasSystem(f) {
			fam = family{f, false}
		} else {
			continue
		}
		if first {
			first = false
			continue
		}
		families = append(families, fam)
	}
	var fb uintptr
	defer func() { e.fallbacks[list] = fb }()
	if len(families) == 0 || e.factory2 == 0 {
		return 0
	}
	var builder uintptr
	if failed(call(e.factory2, factory2CreateFallbackBuilder, uintptr(unsafe.Pointer(&builder)))) {
		return 0
	}
	defer release(builder)
	// DWRITE_UNICODE_RANGE: all of Unicode.
	all := [2]uint32{0, 0x10FFFF}
	for _, f := range families {
		name := utf16z(f.name)
		names := []*uint16{&name[0]}
		method[func(this uintptr, ranges *[2]uint32, nRanges uint32, names **uint16, nNames uint32, coll uintptr, locale, base *uint16, scale float32) uintptr](builder, fallbackAddMapping)(
			builder, &all, 1, &names[0], 1, e.collection(f.custom), nil, nil, 1)
		runtime.KeepAlive(name)
	}
	var system uintptr
	if !failed(call(e.factory2, factory2GetSystemFontFallback, uintptr(unsafe.Pointer(&system)))) {
		call(builder, fallbackAddMappings, system)
		release(system)
	}
	call(builder, fallbackCreate, uintptr(unsafe.Pointer(&fb)))
	return fb
}

func (e *dwrite) font(style Style) *Font {
	family, custom := e.family(style.Family)
	key := faceKey{family, custom, style.weight(), style.Italic}
	face, ok := e.faces[key]
	if !ok {
		face = e.matchFace(e.collection(custom), family, key.weight, key.italic)
		e.faces[key] = face
	}
	if face == 0 {
		return nil
	}
	return e.fontOf(face, style.FontSize())
}

// matchFace returns the face of a family that best matches a weight and
// style.
func (e *dwrite) matchFace(coll uintptr, family string, weight int, italic bool) uintptr {
	name := utf16z(family)
	var index uint32
	var exists int32
	if failed(call(coll, collectionFindFamilyName, uintptr(unsafe.Pointer(&name[0])), uintptr(unsafe.Pointer(&index)), uintptr(unsafe.Pointer(&exists)))) || exists == 0 {
		return 0
	}
	var fam, font, face uintptr
	if failed(call(coll, collectionGetFontFamily, uintptr(index), uintptr(unsafe.Pointer(&fam)))) {
		return 0
	}
	defer release(fam)
	if failed(call(fam, familyGetFirstMatchingFont, uintptr(weight), fontStretchNormal, dwStyle(italic), uintptr(unsafe.Pointer(&font)))) {
		return 0
	}
	defer release(font)
	if failed(call(font, fontCreateFontFace, uintptr(unsafe.Pointer(&face)))) {
		return 0
	}
	return face
}

// fontOf returns the Font of a face at a size.
func (e *dwrite) fontOf(face uintptr, size float32) *Font {
	key := fontKey{face, size}
	if f, ok := e.fonts[key]; ok {
		return f
	}
	call(face, comAddRef)
	var m dwFontMetrics
	call(face, faceGetMetrics, uintptr(unsafe.Pointer(&m)))
	em := size / float32(max(m.designUnitsPerEm, 1))
	f := &Font{Size: size, Ascent: float32(m.ascent) * em, Descent: float32(m.descent) * em, LineGap: float32(m.lineGap) * em, native: face, thin: e.isThin(face),
		underlineTop: float32(m.underlinePosition) * em, underlineThick: float32(m.underlineThickness) * em,
		strikeTop: float32(m.strikethroughPosition) * em, strikeThick: float32(m.strikethroughThickness) * em}
	e.fonts[key] = f
	return f
}

// The renderer IDWriteTextLayout.Draw hands its glyph runs to, which
// collects them in drawing: a static COM object whose methods are created
// once (callbacks are never freed).
var (
	rendererOnce sync.Once
	rendererVtbl [10]uintptr
	rendererObj  struct{ vtbl *[10]uintptr }
	drawMu       sync.Mutex
	drawing      []dwRun
)

func textRenderer() uintptr {
	rendererOnce.Do(func() {
		rendererVtbl = [10]uintptr{
			syscall.NewCallback(rendererQueryInterface),
			syscall.NewCallback(rendererAddRef),
			syscall.NewCallback(rendererAddRef), // Release
			syscall.NewCallback(rendererIsPixelSnappingDisabled),
			syscall.NewCallback(rendererGetCurrentTransform),
			syscall.NewCallback(rendererGetPixelsPerDip),
			syscall.NewCallback(rendererDrawGlyphRun),
			syscall.NewCallback(rendererDrawDecoration), // DrawUnderline
			syscall.NewCallback(rendererDrawDecoration), // DrawStrikethrough
			syscall.NewCallback(rendererDrawInlineObject),
		}
		rendererObj.vtbl = &rendererVtbl
	})
	return uintptr(unsafe.Pointer(&rendererObj))
}

func rendererQueryInterface(this, iid, out uintptr) uintptr {
	switch *(*guid)(ptr(iid)) {
	case iidIUnknown, iidIDWritePixelSnapping, iidIDWriteTextRenderer:
		*(*uintptr)(ptr(out)) = this
		return 0
	}
	*(*uintptr)(ptr(out)) = 0
	return errNoInterface
}

func rendererAddRef(this uintptr) uintptr { return 1 }

func rendererIsPixelSnappingDisabled(this, context, out uintptr) uintptr {
	*(*int32)(ptr(out)) = 1
	return 0
}

func rendererGetCurrentTransform(this, context, out uintptr) uintptr {
	*(*[6]float32)(ptr(out)) = [6]float32{1, 0, 0, 1, 0, 0}
	return 0
}

func rendererGetPixelsPerDip(this, context, out uintptr) uintptr {
	*(*float32)(ptr(out)) = 1
	return 0
}

// rendererDrawGlyphRun receives (this, context, FLOAT baselineOriginX,
// FLOAT baselineOriginY, measuringMode, glyphRun, glyphRunDescription,
// clientDrawingEffect). A Go callback cannot read the floats: on x64 they
// leave garbage in their argument slots, on ARM64 they take no integer
// register. The run's origin comes from hit testing instead.
func rendererDrawGlyphRun(this, context, a2, a3, a4, a5, a6, a7 uintptr) uintptr {
	run, desc := a5, a6
	if runtime.GOARCH == "arm64" {
		run, desc = a3, a4
	}
	collect((*dwGlyphRun)(ptr(run)), (*dwGlyphRunDescription)(ptr(desc)))
	return 0
}

func rendererDrawDecoration(this, context, a2, a3, a4, a5 uintptr) uintptr { return 0 }

func rendererDrawInlineObject(this, context, a2, a3, a4, a5, a6, a7 uintptr) uintptr { return 0 }

// dwRun is a glyph run of a layout, copied.
type dwRun struct {
	face     uintptr
	size     float32
	rtl      bool
	pos, n   uint32 // the run's code units
	glyphs   []uint16
	advances []float32
	offsets  []dwGlyphOffset
	clusters []uint16 // the first glyph of the cluster of each code unit
}

func collect(run *dwGlyphRun, desc *dwGlyphRunDescription) {
	n := int(run.glyphCount)
	r := dwRun{face: run.fontFace, size: run.fontEmSize, rtl: run.bidiLevel&1 != 0, pos: desc.textPosition, n: desc.textLength}
	if n > 0 {
		r.glyphs = slices.Clone(unsafe.Slice(run.glyphIndices, n))
		r.advances = slices.Clone(unsafe.Slice(run.glyphAdvances, n))
		if run.glyphOffsets != nil {
			r.offsets = slices.Clone(unsafe.Slice(run.glyphOffsets, n))
		}
	}
	if desc.clusterMap != nil && desc.textLength > 0 {
		r.clusters = slices.Clone(unsafe.Slice(desc.clusterMap, desc.textLength))
	}
	drawing = append(drawing, r)
}

func (e *dwrite) shape(text []rune, style Style, spans []Span, width float32, rtl, wholeWords bool) []shapedLine {
	format := e.format(style)
	if format == 0 || len(text) == 0 {
		return nil
	}
	u16, index := utf16Text(text)
	var layout uintptr
	hr := method[func(this uintptr, text *uint16, n uint32, format uintptr, width, height float32, out *uintptr) uintptr](e.factory, factoryCreateTextLayout)(
		e.factory, &u16[0], uint32(len(u16)), format, width, maxLayoutHeight, &layout)
	if failed(hr) {
		return nil
	}
	defer release(layout)
	wrapping := uintptr(wordWrappingWrap)
	switch {
	case width <= 0:
		wrapping = wordWrappingNoWrap
	case wholeWords:
		wrapping = wordWrappingWholeWord
	}
	call(layout, formatSetWordWrapping, wrapping)
	e.typeset(layout, style, spans, text)
	if rtl {
		// Trailing alignment keeps the lines at the left, as left-to-right
		// lines are.
		call(layout, formatSetReadingDirection, readingDirectionRTL)
		call(layout, formatSetTextAlignment, textAlignmentTrailing)
	}

	var count uint32
	if hr := call(layout, layoutGetLineMetrics, 0, 0, uintptr(unsafe.Pointer(&count))); failed(hr) && uint32(hr) != errNotSufficientBuffer || count == 0 {
		return nil
	}
	metrics := make([]dwLineMetrics, count)
	if failed(call(layout, layoutGetLineMetrics, uintptr(unsafe.Pointer(&metrics[0])), uintptr(count), uintptr(unsafe.Pointer(&count)))) {
		return nil
	}

	drawMu.Lock()
	drawing = drawing[:0]
	method[func(this, context, renderer uintptr, x, y float32) uintptr](layout, layoutDraw)(layout, 0, textRenderer(), 0, 0)
	runs := slices.Clone(drawing)
	drawMu.Unlock()

	lines := make([]shapedLine, len(metrics))
	ends := make([]uint32, len(metrics)) // the code unit ending each line
	pos := uint32(0)
	for i, m := range metrics {
		lines[i].start = index[min(int(pos), len(u16))]
		pos += m.length
		ends[i] = pos
		lines[i].end = index[min(int(pos), len(u16))]
	}
	for _, r := range runs {
		li, _ := slices.BinarySearch(ends, r.pos+1)
		if li >= len(lines) || int(r.pos+r.n) > len(u16) {
			continue
		}
		var x, y float32
		var hit dwHitTestMetrics
		call(layout, layoutHitTestTextPosition, uintptr(r.pos), 0, uintptr(unsafe.Pointer(&x)), uintptr(unsafe.Pointer(&y)), uintptr(unsafe.Pointer(&hit)))
		lines[li].runs = append(lines[li].runs, e.run(r, x, index))
	}
	if len(layoutObjects.objects) > 0 {
		units := make([]uint32, len(text)+1)
		for i, r := range text {
			units[i+1] = units[i] + 1
			if r >= 0x10000 {
				units[i+1]++
			}
		}
		objectRuns(layout, lines, ends, spans, units, index)
	}
	return lines
}

// typeset applies a style's letter spacing and OpenType features to a text
// layout of text, and the styles of its spans to their ranges.
func (e *dwrite) typeset(layout uintptr, style Style, spans []Span, text []rune) {
	// The code unit of each rune, and a DWRITE_TEXT_RANGE of runes, passed
	// in a register.
	units := make([]uint32, len(text)+1)
	for i, r := range text {
		units[i+1] = units[i] + 1
		if r >= 0x10000 {
			units[i+1]++
		}
	}
	textRange := func(from, to int) uintptr {
		return uintptr(units[from]) | uintptr(units[to]-units[from])<<32
	}
	all := textRange(0, len(text))
	e.spacing(layout, style.LetterSpacing, all)
	e.typography(layout, style.Features, all)
	from := 0
	for _, sp := range spans {
		to := max(from, min(sp.End, len(text)))
		r := textRange(from, to)
		from = to
		if r>>32 == 0 {
			continue
		}
		if sp.Family != "" {
			family, custom := e.family(sp.Family)
			name := utf16z(family)
			call(layout, layoutSetFontCollection, e.collection(custom), r)
			call(layout, layoutSetFontFamilyName, uintptr(unsafe.Pointer(&name[0])), r)
			runtime.KeepAlive(name)
		}
		if sp.Weight > 0 {
			call(layout, layoutSetFontWeight, uintptr(min(sp.Weight, 999)), r)
		}
		if sp.Italic {
			call(layout, layoutSetFontStyle, fontStyleItalic, r)
		}
		if sp.Size > 0 {
			method[func(this uintptr, size float32, r uintptr) uintptr](layout, layoutSetFontSize)(layout, sp.Size, r)
		}
		e.spacing(layout, sp.LetterSpacing, r)
		e.typography(layout, sp.Features, r)
	}
	// The runes in other languages than the format's: those of spans, and
	// Han characters in a language that does not write them, which keep
	// the user's locale.
	for _, run := range langRuns(text, style.Lang, spans) {
		name := e.localeName(run.lang)
		call(layout, layoutSetLocaleName, uintptr(unsafe.Pointer(&name[0])), textRange(run.start, run.end))
		runtime.KeepAlive(name)
	}
	setObjects(layout, spans, units)
}

// spacing adds letter spacing to a range of a text layout.
func (e *dwrite) spacing(layout uintptr, spacing float32, r uintptr) {
	if spacing == 0 {
		return
	}
	// IDWriteTextLayout1 came with Windows 8.
	if l1 := queryInterface(layout, &iidIDWriteTextLayout1); l1 != 0 {
		method[func(this uintptr, leading, trailing, minAdvance float32, r uintptr) uintptr](l1, layout1SetCharacterSpacing)(l1, 0, spacing, 0, r)
		release(l1)
	}
}

// typography sets OpenType features over a range of a text layout.
func (e *dwrite) typography(layout uintptr, list string, r uintptr) {
	fs := features(list)
	if len(fs) == 0 {
		return
	}
	var typography uintptr
	if failed(call(e.factory, factoryCreateTypography, uintptr(unsafe.Pointer(&typography)))) {
		return
	}
	defer release(typography)
	for _, f := range fs {
		// DWRITE_FONT_FEATURE{nameTag, parameter}, the tag in the byte order
		// of DWRITE_MAKE_OPENTYPE_TAG.
		tag := uint32(f.tag[0]) | uint32(f.tag[1])<<8 | uint32(f.tag[2])<<16 | uint32(f.tag[3])<<24
		call(typography, typographyAddFontFeature, uintptr(tag)|uintptr(f.value)<<32)
	}
	call(layout, layoutSetTypography, typography, r)
}

// run positions the glyphs of a run whose origin, on the right for a
// right-to-left run, is at x.
func (e *dwrite) run(r dwRun, x float32, index []int) shapedRun {
	f := e.fontOf(r.face, r.size)
	out := shapedRun{font: f, start: index[r.pos], end: index[r.pos+r.n], glyphs: make([]Glyph, len(r.glyphs))}
	// The code unit starting the cluster of each glyph.
	first := make([]uint32, len(r.glyphs))
	for k := 0; k < len(r.clusters); {
		g0 := int(r.clusters[k])
		k1 := k + 1
		for k1 < len(r.clusters) && int(r.clusters[k1]) == g0 {
			k1++
		}
		g1 := len(r.glyphs)
		if k1 < len(r.clusters) {
			g1 = int(r.clusters[k1])
		}
		for g := g0; g < min(g1, len(first)); g++ {
			first[g] = uint32(k)
		}
		k = k1
	}
	pen := x
	for i, id := range r.glyphs {
		var off dwGlyphOffset
		if r.offsets != nil {
			off = r.offsets[i]
		}
		g := Glyph{Font: f, ID: uint32(id), Advance: r.advances[i], Y: -off.ascenderOffset, Cluster: index[r.pos+first[i]], RTL: r.rtl}
		if r.rtl {
			pen -= g.Advance
			g.X = pen - off.advanceOffset
		} else {
			g.X = pen + off.advanceOffset
			pen += g.Advance
		}
		out.glyphs[i] = g
	}
	return out
}

// settings reads the system's settings for text once: the rendering
// parameters DirectWrite makes of them (gamma, contrast, ClearType level,
// pixel geometry) and how the system smooths fonts.
func (e *dwrite) settings() {
	if e.settingsRead {
		return
	}
	e.settingsRead = true
	e.clearLevel = 1
	if failed(call(e.factory, factoryCreateRenderingParams, uintptr(unsafe.Pointer(&e.params)))) {
		e.params = 0
		return
	}
	gamma := method[func(uintptr) float32](e.params, paramsGetGamma)(e.params)
	contrast := method[func(uintptr) float32](e.params, paramsGetEnhancedContrast)(e.params)
	gray := contrast
	if p1 := queryInterface(e.params, &iidIDWriteRendParams1); p1 != 0 {
		gray = method[func(uintptr) float32](p1, params1GetGrayscaleEnhancedContrast)(p1)
		release(p1)
	}
	e.clearLevel = method[func(uintptr) float32](e.params, paramsGetClearTypeLevel)(e.params)
	geometry := call(e.params, paramsGetPixelGeometry)
	e.text = scene.TextParams{GammaRatios: scene.GammaRatios(gamma), Contrast: gray, SubpixelContrast: contrast}
	e.bgr = geometry == pixelGeometryBGR
	var smooth int32
	var kind uint32
	procSystemParams.Call(spiGetFontSmoothing, 0, uintptr(unsafe.Pointer(&smooth)), 0)
	procSystemParams.Call(spiGetFontSmoothingType, 0, uintptr(unsafe.Pointer(&kind)), 0)
	e.aliased = smooth == 0 || call(e.params, paramsGetRenderingMode) == renderingModeAliased
	e.clearType = !e.aliased && kind == feFontSmoothingClearType && geometry != pixelGeometryFlat
}

func (e *dwrite) textParams() (scene.TextParams, bool) {
	e.settings()
	if e.aliased {
		return scene.TextParams{}, false
	}
	return e.text, e.clearType
}

// mode returns the rendering mode and grid fitting DirectWrite recommends
// for a font at scale pixels per DIP, as Direct2D draws it with: from
// Windows 10, with natural symmetric rendering downsampled at high
// resolutions (IDWriteFontFace3).
func (e *dwrite) mode(f *Font, scale float32) (mode, gridFit uintptr) {
	if e.aliased {
		return renderingModeAliased, gridFitModeDefault
	}
	mode, gridFit = renderingModeNaturalSymmetric, gridFitModeDefault
	face, index := uintptr(0), 0
	if e.factory3 != 0 {
		face, index = queryInterface(f.native, &iidIDWriteFontFace3), face3GetRecommendedRenderingMode
	}
	if face == 0 {
		face, index = queryInterface(f.native, &iidIDWriteFontFace2), face2GetRecommendedRenderingMode
	}
	if face == 0 {
		return mode, gridFit
	}
	defer release(face)
	var m, g uint32
	hr := method[func(this uintptr, size, dpiX, dpiY float32, transform uintptr, sideways int32, threshold, measuring, params uintptr, mode, gridFit *uint32) uintptr](face, index)(
		face, f.Size, 96*scale, 96*scale, 0, 0, outlineThresholdAntialiased, measuringModeNatural, e.params, &m, &g)
	if failed(hr) || m == 0 {
		return mode, gridFit
	}
	// Glyph run analyses draw no outlines, which large text recommends.
	if m == renderingModeOutline {
		m = renderingModeNaturalSymmetric
	}
	return uintptr(m), uintptr(g)
}

// positions are those DirectWrite rasterizes glyphs at in their rendering
// mode, the nearest to the pen: six a pixel for ClearType, eight in
// natural, four in natural symmetric and two in downsampled rendering, and
// whole pixels in aliased and GDI rendering.
func (e *dwrite) positions(f *Font, scale float32, subpixel bool) (int, bool) {
	e.settings()
	mode, _ := e.mode(f, scale)
	switch {
	case mode == renderingModeAliased || mode == renderingModeGDIClassic || mode == renderingModeGDINatural:
		return 1, true
	case subpixel && e.clearType:
		return 6, true
	case mode == renderingModeNatural:
		return 8, true
	case mode == renderingModeDownsampled:
		return 2, true
	}
	return 4, true
}

// decorate places underlines and strikethroughs as Direct2D draws a text
// layout's, run by run along their advances: an underline at its font's
// offset, rounded to whole pixels, from the baseline DirectWrite snaps, a
// strikethrough at its offset from the baseline as laid out, both as thick
// as the font says, with their edges where Direct2D's eight samples across
// and down a pixel put them.
func (e *dwrite) decorate(r decoRange) []Stroke {
	s := r.scale
	base := r.baseline() * s
	var out []Stroke
	r.runs(func(f *Font, i, j int) {
		x0, x1 := r.span(i, j)
		var top, thick float32
		if r.d == Underline {
			top = float32(math.Round(float64(base))) + float32(math.Round(float64(-f.underlineTop*s)))
			thick = f.underlineThick * s
		} else {
			top = base - f.strikeTop*s
			thick = f.strikeThick * s
		}
		out = append(out, Stroke{X0: eighths((r.x + x0) * s), X1: eighths((r.x + x1) * s), Top: eighths(top), Bottom: eighths(top + thick)})
	})
	return out
}

// eighths moves an edge to where Direct2D's coverage puts it: rasterizing
// with eight samples across and down each pixel, at (k+0.5)/8.
func eighths(v float32) float32 { return float32(math.Ceil(float64(v)*8-0.5)) / 8 }

// baseline is the nearest whole pixel, where DirectWrite snaps baselines.
func (e *dwrite) baseline(y float32) float32 { return float32(math.Round(float64(y))) }

func (e *dwrite) glyph(f *Font, id uint32, scale, dx float32, _ Shade, subpixel bool) bitmap {
	e.settings()
	index := uint16(id)
	var advance float32
	run := dwGlyphRun{fontFace: f.native, fontEmSize: f.Size * scale, glyphCount: 1, glyphIndices: &index, glyphAdvances: &advance}
	if e.isColor(f.native) {
		if b, ok := e.colorGlyph(&run, dx); ok {
			return b
		}
	}
	mode, gridFit := e.mode(f, scale)
	subpixel = subpixel && e.clearType
	a, r, ok := e.analyzeAs(&run, dx, 0, mode, gridFit, subpixel)
	if !ok {
		return bitmap{}
	}
	defer release(a)
	w, h := int(r.right-r.left), int(r.bottom-r.top)
	if subpixel {
		pix := e.subpixels(a, r)
		if pix == nil {
			return bitmap{}
		}
		return bitmap{left: int(r.left), top: int(r.top), w: w, h: h, pix: pix, subpixel: true}
	}
	alpha := e.alpha(a, r)
	if alpha == nil {
		return bitmap{}
	}
	return bitmap{left: int(r.left), top: int(r.top), w: w, h: h, pix: alpha}
}

// join: Direct2D draws the glyphs of a run from its glyph cache, each in
// turn.
func (e *dwrite) join() bool { return false }

// subpixels returns the ClearType coverage of an analyzed run's pixels in
// r, red, green, blue and their mean for each, at the system's ClearType
// level: none (0) is the mean of the three.
func (e *dwrite) subpixels(a uintptr, r dwRect) []byte {
	w, h := int(r.right-r.left), int(r.bottom-r.top)
	n := 3 * w * h
	if cap(e.scratch) < n {
		e.scratch = make([]byte, n)
	}
	buf := e.scratch[:n]
	if failed(call(a, glyphsCreateAlphaTexture, textureClearType3x1, uintptr(unsafe.Pointer(&r)), uintptr(unsafe.Pointer(&buf[0])), uintptr(n))) {
		return nil
	}
	out := make([]byte, 4*w*h)
	level := min(max(e.clearLevel, 0), 1)
	for i := range w * h {
		c := buf[3*i : 3*i+3]
		if e.bgr {
			c = []byte{c[2], c[1], c[0]}
		}
		mean := (float32(c[0]) + float32(c[1]) + float32(c[2])) / 3
		for k := range 3 {
			out[4*i+k] = uint8(mean + (float32(c[k])-mean)*level + 0.5)
		}
		out[4*i+3] = uint8(mean + 0.5)
	}
	return out
}

// isThin reports whether a face is of a family whose strokes are too thin
// for antialiasing, which Direct2D gives more contrast: those Windows
// Terminal lists, digitized from typewriters' typeballs.
func (e *dwrite) isThin(face uintptr) bool {
	t, ok := e.thin[face]
	if ok {
		return t
	}
	var font, fam, strs uintptr
	if !failed(call(e.system, collectionGetFontFromFace, face, uintptr(unsafe.Pointer(&font)))) {
		if !failed(call(font, fontGetFontFamily, uintptr(unsafe.Pointer(&fam)))) {
			if !failed(call(fam, familyGetFamilyNames, uintptr(unsafe.Pointer(&strs)))) {
				switch englishName(strs) {
				case "Courier New", "Fixed Miriam Transparent", "Miriam Fixed", "Rod", "Rod Transparent", "Simplified Arabic Fixed":
					t = true
				}
				release(strs)
			}
			release(fam)
		}
		release(font)
	}
	e.thin[face] = t
	return t
}

func (e *dwrite) isColor(face uintptr) bool {
	c, ok := e.color[face]
	if !ok {
		if f2 := queryInterface(face, &iidIDWriteFontFace2); f2 != 0 {
			c = call(f2, face2IsColorFont) != 0
			release(f2)
		}
		e.color[face] = c
	}
	return c
}

// analyze returns the grayscale glyph run analysis of a run drawn at
// (x, y), in natural symmetric rendering, and the bounds of its pixels.
func (e *dwrite) analyze(run *dwGlyphRun, x, y float32) (uintptr, dwRect, bool) {
	mode := uintptr(renderingModeNaturalSymmetric)
	if e.aliased {
		mode = renderingModeAliased
	}
	return e.analyzeAs(run, x, y, mode, gridFitModeDefault, false)
}

// analyzeAs returns the glyph run analysis of a run drawn at (x, y) in a
// rendering mode and grid fitting, with ClearType if clearType, and the
// bounds of its pixels.
func (e *dwrite) analyzeAs(run *dwGlyphRun, x, y float32, mode, gridFit uintptr, clearType bool) (uintptr, dwRect, bool) {
	var a uintptr
	var hr uintptr
	if e.factory2 != 0 {
		antialias := uintptr(antialiasModeGrayscale)
		if clearType {
			antialias = antialiasModeClearType
		}
		// IDWriteFactory3 takes the rendering modes of Windows 10 too.
		f, index := e.factory2, factory2CreateGlyphRunAnalysis
		if e.factory3 != 0 {
			f, index = e.factory3, factory3CreateGlyphRunAnalysis
		}
		hr = method[func(this uintptr, run *dwGlyphRun, transform, rendering, measuring, gridFit, antialias uintptr, x, y float32, out *uintptr) uintptr](f, index)(
			f, run, 0, mode, measuringModeNatural, gridFit, antialias, x, y, &a)
	} else {
		hr = method[func(this uintptr, run *dwGlyphRun, pixelsPerDip float32, transform, rendering, measuring uintptr, x, y float32, out *uintptr) uintptr](e.factory, factoryCreateGlyphRunAnalysis)(
			e.factory, run, 1, 0, mode, measuringModeNatural, x, y, &a)
	}
	if failed(hr) || a == 0 {
		return 0, dwRect{}, false
	}
	var r dwRect
	if failed(call(a, glyphsGetAlphaTextureBound, e.texture(mode, clearType), uintptr(unsafe.Pointer(&r)))) || r.right <= r.left || r.bottom <= r.top {
		release(a)
		return 0, dwRect{}, false
	}
	return a, r, true
}

// texture returns the kind of texture a glyph run analysis gives: aliased
// in aliased rendering, ClearType's three values a pixel for ClearType,
// and otherwise, with IDWriteFactory2, grayscale coverage. Without it
// analyses give ClearType only, which alpha averages.
func (e *dwrite) texture(mode uintptr, clearType bool) uintptr {
	if mode == renderingModeAliased || e.factory2 != 0 && !clearType {
		return textureAliased1x1
	}
	return textureClearType3x1
}

// alpha returns the coverage of an analyzed run's pixels in r.
func (e *dwrite) alpha(a uintptr, r dwRect) []byte {
	w, h := int(r.right-r.left), int(r.bottom-r.top)
	out := make([]byte, w*h)
	if e.factory2 != 0 || e.aliased {
		if failed(call(a, glyphsCreateAlphaTexture, textureAliased1x1, uintptr(unsafe.Pointer(&r)), uintptr(unsafe.Pointer(&out[0])), uintptr(len(out)))) {
			return nil
		}
		return out
	}
	n := 3 * w * h
	if cap(e.scratch) < n {
		e.scratch = make([]byte, n)
	}
	buf := e.scratch[:n]
	if failed(call(a, glyphsCreateAlphaTexture, textureClearType3x1, uintptr(unsafe.Pointer(&r)), uintptr(unsafe.Pointer(&buf[0])), uintptr(n))) {
		return nil
	}
	for i := range out {
		out[i] = uint8((uint16(buf[3*i]) + uint16(buf[3*i+1]) + uint16(buf[3*i+2]) + 1) / 3)
	}
	return out
}

// colorGlyph draws a glyph of a color font layer by layer, or reports
// that it has no color.
func (e *dwrite) colorGlyph(run *dwGlyphRun, dx float32) (bitmap, bool) {
	if e.factory2 == 0 {
		return bitmap{}, false
	}
	var layers uintptr
	hr := method[func(this uintptr, x, y float32, run *dwGlyphRun, desc, measuring, transform, palette uintptr, out *uintptr) uintptr](e.factory2, factory2TranslateColorGlyphRun)(
		e.factory2, dx, 0, run, 0, measuringModeNatural, 0, 0, &layers)
	if failed(hr) || layers == 0 {
		return bitmap{}, false
	}
	defer release(layers)
	type layer struct {
		r     dwRect
		alpha []byte
		color [4]float32
	}
	var ls []layer
	var box dwRect
	for {
		var more int32
		if failed(call(layers, colorRunsMoveNext, uintptr(unsafe.Pointer(&more)))) || more == 0 {
			break
		}
		var cr *dwColorGlyphRun
		if failed(call(layers, colorRunsGetCurrentRun, uintptr(unsafe.Pointer(&cr)))) || cr == nil {
			break
		}
		a, r, ok := e.analyze(&cr.glyphRun, cr.baselineOriginX, cr.baselineOriginY)
		if !ok {
			continue
		}
		alpha := e.alpha(a, r)
		release(a)
		if alpha == nil {
			continue
		}
		c := cr.runColor
		if cr.paletteIndex == paletteIndexForeground {
			c = [4]float32{0, 0, 0, 1}
		}
		if len(ls) == 0 {
			box = r
		} else {
			box = dwRect{min(box.left, r.left), min(box.top, r.top), max(box.right, r.right), max(box.bottom, r.bottom)}
		}
		ls = append(ls, layer{r, alpha, c})
	}
	if len(ls) == 0 {
		return bitmap{}, true
	}
	w, h := int(box.right-box.left), int(box.bottom-box.top)
	pix := make([]byte, 4*w*h)
	for _, l := range ls {
		lw := int(l.r.right - l.r.left)
		for y := range int(l.r.bottom - l.r.top) {
			row := (y+int(l.r.top-box.top))*w + int(l.r.left-box.left)
			for x := range lw {
				cov := float32(l.alpha[y*lw+x]) / 255 * l.color[3]
				if cov == 0 {
					continue
				}
				p := pix[4*(row+x):]
				for c := range 3 {
					p[c] = uint8(l.color[c]*cov*255 + float32(p[c])*(1-cov) + 0.5)
				}
				p[3] = uint8(cov*255 + float32(p[3])*(1-cov) + 0.5)
			}
		}
	}
	return bitmap{left: int(box.left), top: int(box.top), w: w, h: h, pix: pix, color: true}, true
}

func (e *dwrite) register(data []byte, family string) error {
	if e.factory5 == 0 {
		return errors.New("mygo: adding fonts needs Windows 10 version 1703 or later")
	}
	if len(data) == 0 {
		return errors.New("mygo: no font data")
	}
	if e.loader == 0 {
		var loader uintptr
		if hr := call(e.factory5, factory5CreateInMemoryFontFileLoader, uintptr(unsafe.Pointer(&loader))); failed(hr) {
			return fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
		}
		if hr := call(e.factory, factoryRegisterFontFileLoader, loader); failed(hr) {
			release(loader)
			return fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
		}
		e.loader = loader
	}
	// Without an owner, the loader copies the data.
	var file uintptr
	if hr := call(e.loader, loaderCreateInMemoryFontFileReference, e.factory, uintptr(unsafe.Pointer(&data[0])), uintptr(len(data)), 0, uintptr(unsafe.Pointer(&file))); failed(hr) {
		return fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
	}
	one, err := e.newCollection([]uintptr{file})
	if err != nil {
		release(file)
		return err
	}
	names := familyNames(one)
	release(one)
	if len(names) == 0 {
		release(file)
		return errors.New("mygo: cannot add font: not a font DirectWrite reads")
	}
	coll, err := e.newCollection(append(e.files, file))
	if err != nil {
		release(file)
		return err
	}
	e.files = append(e.files, file)
	release(e.custom)
	e.custom = coll
	for _, name := range names {
		e.aliases[strings.ToLower(name)] = name
	}
	if family != "" {
		e.aliases[strings.ToLower(family)] = names[0]
		// Listed under the name the app gave it, which is the one it uses.
		e.names = append(e.names, family)
	} else {
		e.names = append(e.names, names...)
	}
	for k, f := range e.formats {
		if k.custom {
			release(f)
			delete(e.formats, k)
		}
	}
	for k := range e.faces {
		if k.custom {
			delete(e.faces, k)
		}
	}
	return nil
}

// newCollection returns a font collection of font files.
func (e *dwrite) newCollection(files []uintptr) (uintptr, error) {
	var builder, set, coll uintptr
	if hr := call(e.factory5, factory5CreateFontSetBuilder, uintptr(unsafe.Pointer(&builder))); failed(hr) {
		return 0, fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
	}
	defer release(builder)
	for _, f := range files {
		if hr := call(builder, builderAddFontFile, f); failed(hr) {
			return 0, fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
		}
	}
	if hr := call(builder, builderCreateFontSet, uintptr(unsafe.Pointer(&set))); failed(hr) {
		return 0, fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
	}
	defer release(set)
	if hr := call(e.factory5, factory3CreateFontCollectionFromFontSet, set, uintptr(unsafe.Pointer(&coll))); failed(hr) {
		return 0, fmt.Errorf("mygo: cannot add font (HRESULT %#08x)", uint32(hr))
	}
	return coll, nil
}

// appFamilies lists the families of the app's fonts, monospaced when
// their regular font in the app's collection is.
func (e *dwrite) appFamilies() []FontFamily {
	if len(e.names) == 0 {
		return nil
	}
	mono := map[string]bool{} // by the family's own name
	for _, f := range collectionFamilies(e.custom) {
		mono[f.Name] = f.Monospaced
	}
	out := make([]FontFamily, len(e.names))
	for i, name := range e.names {
		out[i] = FontFamily{Name: name, Monospaced: mono[e.aliases[strings.ToLower(name)]]}
	}
	return out
}

// systemFamilies lists the system collection's families, which hold none
// of the app's, monospaced when their regular font IDWriteFont1 (Windows 8
// and later) calls so: DirectWrite's shared factory is thread-safe.
func (e *dwrite) systemFamilies() func() []FontFamily {
	coll := e.system
	if coll == 0 {
		return nil
	}
	call(coll, comAddRef)
	return func() []FontFamily {
		defer release(coll)
		return collectionFamilies(coll)
	}
}

// familyNames returns the English names of a collection's families.
func familyNames(coll uintptr) []string {
	var names []string
	eachFamily(coll, func(fam uintptr, name string) { names = append(names, name) })
	return names
}

// collectionFamilies returns the families of a collection, by their
// English names.
func collectionFamilies(coll uintptr) []FontFamily {
	var out []FontFamily
	eachFamily(coll, func(fam uintptr, name string) {
		out = append(out, FontFamily{Name: name, Monospaced: monospaced(fam)})
	})
	return out
}

// eachFamily calls fn with each family of a collection and its English
// name.
func eachFamily(coll uintptr, fn func(fam uintptr, name string)) {
	if coll == 0 {
		return
	}
	n := uint32(call(coll, collectionGetFontFamilyCount))
	for i := range n {
		var fam, strs uintptr
		if failed(call(coll, collectionGetFontFamily, uintptr(i), uintptr(unsafe.Pointer(&fam)))) {
			continue
		}
		if !failed(call(fam, familyGetFamilyNames, uintptr(unsafe.Pointer(&strs)))) {
			if name := englishName(strs); name != "" {
				fn(fam, name)
			}
			release(strs)
		}
		release(fam)
	}
}

// monospaced reports whether a family's regular font is monospaced.
func monospaced(fam uintptr) bool {
	var font uintptr
	if failed(call(fam, familyGetFirstMatchingFont, 400, fontStretchNormal, dwStyle(false), uintptr(unsafe.Pointer(&font)))) {
		return false
	}
	defer release(font)
	font1 := queryInterface(font, &iidIDWriteFont1)
	if font1 == 0 {
		return false
	}
	defer release(font1)
	return call(font1, font1IsMonospacedFont) != 0
}

// englishName returns the English name of localized strings, or their
// first.
func englishName(strs uintptr) string {
	var index uint32
	var exists int32
	en := utf16z("en-us")
	call(strs, stringsFindLocaleName, uintptr(unsafe.Pointer(&en[0])), uintptr(unsafe.Pointer(&index)), uintptr(unsafe.Pointer(&exists)))
	if exists == 0 {
		index = 0
	}
	var length uint32
	if failed(call(strs, stringsGetStringLength, uintptr(index), uintptr(unsafe.Pointer(&length)))) {
		return ""
	}
	buf := make([]uint16, length+1)
	if failed(call(strs, stringsGetString, uintptr(index), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))) {
		return ""
	}
	return syscall.UTF16ToString(buf)
}

func (e *dwrite) fontCount() int { return len(e.fonts) }

// forgetFonts lets go of the faces of the Fonts made so far; the faces of
// the families stay.
func (e *dwrite) forgetFonts() {
	for _, f := range e.fonts {
		release(f.native)
	}
	clear(e.fonts)
}
