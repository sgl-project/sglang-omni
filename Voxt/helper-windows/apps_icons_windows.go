//go:build windows

package main

import (
	"bytes"
	"fmt"
	"image"
	"image/png"
	"runtime"
	"strings"
	"unsafe"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"golang.org/x/sys/windows"
)

const (
	shellItemImageGetImage = 3
	shellImageIconOnly     = 0x4
	bitmapRGB              = 0
)

var (
	shell32                         = windows.NewLazySystemDLL("shell32.dll")
	gdi32                           = windows.NewLazySystemDLL("gdi32.dll")
	procSHCreateItemFromParsingName = shell32.NewProc("SHCreateItemFromParsingName")
	procGetObject                   = gdi32.NewProc("GetObjectW")
	procGetDIBits                   = gdi32.NewProc("GetDIBits")
	procDeleteObject                = gdi32.NewProc("DeleteObject")
	procGetDC                       = user32.NewProc("GetDC")
	procReleaseDC                   = user32.NewProc("ReleaseDC")
	interfaceShellItemImageFactory  = windows.GUID{Data1: 0xbcc18b79, Data2: 0xba16, Data3: 0x442f, Data4: [8]byte{0x80, 0xc4, 0x8a, 0x59, 0xc3, 0x0c, 0x46, 0x3b}}
)

type windowsBitmap struct {
	bitmapType   int32
	width        int32
	height       int32
	widthBytes   int32
	planes       uint16
	bitsPerPixel uint16
	bits         unsafe.Pointer
}

type bitmapInfoHeader struct {
	size          uint32
	width         int32
	height        int32
	planes        uint16
	bitCount      uint16
	compression   uint32
	sizeImage     uint32
	horizontalPPM int32
	verticalPPM   int32
	colorsUsed    uint32
	colorsNeeded  uint32
}

func makeAppIcons(params proto.AppIconsParams, catalog []proto.AppInfo, extract func(path string, size int) ([]byte, error)) (proto.AppIconsResult, error) {
	if len(params.Apps) > proto.MaxIcons {
		return proto.AppIconsResult{}, fmt.Errorf("at most %d apps a request", proto.MaxIcons)
	}
	size := iconSide(params.Size)
	result := proto.AppIconsResult{Icons: make([]proto.AppIcon, len(params.Apps))}
	for index, ref := range params.Apps {
		result.Icons[index].BundleID = ref.BundleID
		path, found := resolveAppPath(ref, catalog)
		if !found {
			continue
		}
		encoded, err := extract(path, size)
		if err == nil {
			result.Icons[index].PNG = encoded
		}
	}
	return result, nil
}

func resolveAppPath(ref proto.AppRef, catalog []proto.AppInfo) (string, bool) {
	if ref.Path != "" {
		if path, found := installedExecutablePath(ref.Path); found {
			return path, true
		}
	}
	for _, app := range catalog {
		if strings.EqualFold(app.BundleID, ref.BundleID) && app.Path != "" {
			if path, found := installedExecutablePath(app.Path); found {
				return path, true
			}
		}
	}
	return "", false
}

func iconSide(size int) int {
	if size <= 0 {
		return proto.IconSize
	}
	return min(size, proto.MaxIconSize)
}

func extractAppIcon(path string, size int) ([]byte, error) {
	pathPointer, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return nil, err
	}
	var factory *comObject
	result, _, _ := procSHCreateItemFromParsingName.Call(
		uintptr(unsafe.Pointer(pathPointer)), 0,
		uintptr(unsafe.Pointer(&interfaceShellItemImageFactory)), uintptr(unsafe.Pointer(&factory)),
	)
	if int32(result) < 0 {
		return nil, hresultError("SHCreateItemFromParsingName", result)
	}
	defer factory.release()
	var bitmapHandle uintptr
	dimensions := uintptr(uint32(size)) | uintptr(uint32(size))<<32
	result = factory.call(shellItemImageGetImage, dimensions, shellImageIconOnly, uintptr(unsafe.Pointer(&bitmapHandle)))
	if int32(result) < 0 {
		return nil, hresultError("IShellItemImageFactory.GetImage", result)
	}
	if bitmapHandle == 0 {
		return nil, fmt.Errorf("shell returned no icon bitmap")
	}
	defer procDeleteObject.Call(bitmapHandle)
	var bitmap windowsBitmap
	result, _, _ = procGetObject.Call(bitmapHandle, unsafe.Sizeof(bitmap), uintptr(unsafe.Pointer(&bitmap)))
	if result == 0 || bitmap.width <= 0 || bitmap.height <= 0 || bitmap.width > 4096 || bitmap.height > 4096 {
		return nil, fmt.Errorf("invalid shell icon bitmap")
	}
	width := int(bitmap.width)
	height := int(bitmap.height)
	pixels := make([]byte, width*height*4)
	header := bitmapInfoHeader{
		size: uint32(unsafe.Sizeof(bitmapInfoHeader{})), width: bitmap.width, height: -bitmap.height,
		planes: 1, bitCount: 32, compression: bitmapRGB,
	}
	deviceContext, _, _ := procGetDC.Call(0)
	if deviceContext == 0 {
		return nil, fmt.Errorf("GetDC failed")
	}
	defer procReleaseDC.Call(0, deviceContext)
	rows, _, _ := procGetDIBits.Call(
		deviceContext, bitmapHandle, 0, uintptr(height), uintptr(unsafe.Pointer(&pixels[0])),
		uintptr(unsafe.Pointer(&header)), 0,
	)
	if rows == 0 {
		return nil, fmt.Errorf("GetDIBits failed")
	}
	converted, err := rgbaFromBGRA(width, height, pixels)
	if err != nil {
		return nil, err
	}
	var encoded bytes.Buffer
	if err := png.Encode(&encoded, converted); err != nil {
		return nil, err
	}
	return encoded.Bytes(), nil
}

func rgbaFromBGRA(width, height int, pixels []byte) (*image.RGBA, error) {
	if width <= 0 || height <= 0 || len(pixels) != width*height*4 {
		return nil, fmt.Errorf("invalid BGRA pixel buffer")
	}
	converted := image.NewRGBA(image.Rect(0, 0, width, height))
	for offset := 0; offset < len(pixels); offset += 4 {
		converted.Pix[offset] = pixels[offset+2]
		converted.Pix[offset+1] = pixels[offset+1]
		converted.Pix[offset+2] = pixels[offset]
		converted.Pix[offset+3] = pixels[offset+3]
	}
	return converted, nil
}

func runShellCOM(action func()) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	if err := windows.CoInitializeEx(0, windows.COINIT_APARTMENTTHREADED); err != nil {
		return err
	}
	defer windows.CoUninitialize()
	action()
	return nil
}
