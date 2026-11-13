//go:build windows

package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
)

func TestAddressBarControl(t *testing.T) {
	for _, test := range []struct {
		name         string
		controlType  int32
		elementName  string
		automationID string
		want         bool
	}{
		{name: "English Edge", controlType: controlTypeEdit, elementName: "Address and search bar", want: true},
		{name: "Chinese Chrome", controlType: controlTypeEdit, elementName: "地址和搜索栏", automationID: "Ctrl+L", want: true},
		{name: "Firefox", controlType: controlTypeComboBox, automationID: "urlbar-input", want: true},
		{name: "page field", controlType: controlTypeEdit, elementName: "Search", want: false},
		{name: "address form field", controlType: controlTypeEdit, elementName: "Home address", want: false},
		{name: "Chinese address form field", controlType: controlTypeEdit, elementName: "家庭地址", want: false},
		{name: "document", controlType: controlTypeDocument, elementName: "Address and search bar", want: false},
	} {
		t.Run(test.name, func(t *testing.T) {
			if got := addressBarControl(test.controlType, test.elementName, test.automationID); got != test.want {
				t.Fatalf("addressBarControl = %v, want %v", got, test.want)
			}
		})
	}
	if !slices.Contains(implementedMethods(), proto.MethodBrowserURL) {
		t.Fatal("browser.url is not advertised")
	}
}
