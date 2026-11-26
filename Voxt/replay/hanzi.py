"""Traditional to simplified Chinese, one character at a time, with ICU's
Hant-Hans transform in macOS's CoreFoundation."""

import ctypes
from functools import lru_cache

_UTF8 = 0x08000100
_P = ctypes.c_void_p
_CF = ctypes.CDLL("/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation")
for _name, _result, _args in (
    ("CFStringCreateWithCString", _P, [_P, ctypes.c_char_p, ctypes.c_uint32]),
    ("CFStringCreateMutableCopy", _P, [_P, ctypes.c_long, _P]),
    ("CFStringTransform", ctypes.c_bool, [_P, _P, _P, ctypes.c_bool]),
    (
        "CFStringGetCString",
        ctypes.c_bool,
        [_P, ctypes.c_char_p, ctypes.c_long, ctypes.c_uint32],
    ),
    ("CFRelease", None, [_P]),
):
    getattr(_CF, _name).restype = _result
    getattr(_CF, _name).argtypes = _args

# note (luojiaxuan): the Unified Ideographs blocks, extension A and the compatibility block, where
# Hant-Hans maps a character to one other character.
_HAN = ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2A6DF))


@lru_cache(maxsize=None)
def _simplified_char(char):
    name = _CF.CFStringCreateWithCString(None, b"Hant-Hans", _UTF8)
    source = _CF.CFStringCreateWithCString(None, char.encode(), _UTF8)
    target = _CF.CFStringCreateMutableCopy(None, 0, source)
    if not _CF.CFStringTransform(target, None, name, False):
        raise RuntimeError("Hant-Hans transform failed")
    out = ctypes.create_string_buffer(32)
    if not _CF.CFStringGetCString(target, out, len(out), _UTF8):
        raise RuntimeError("cannot read the transformed text")
    for ref in (name, source, target):
        _CF.CFRelease(ref)
    return out.value.decode()


def simplified(text):
    return "".join(
        _simplified_char(c) if any(lo <= ord(c) <= hi for lo, hi in _HAN) else c
        for c in text
    )
