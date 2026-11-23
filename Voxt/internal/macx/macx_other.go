//go:build !darwin

package macx

import "time"

func activate(int, time.Duration, func(func())) (bool, time.Duration) { return false, 0 }

func frontmost() App { return App{} }

func systemProxies() Proxies { return Proxies{} }

func windowNumber(uintptr) int { return 0 }
