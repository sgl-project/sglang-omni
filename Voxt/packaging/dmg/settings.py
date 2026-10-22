application = defines["app"]
background = defines["background"]
icon = defines["volume_icon"]

files = [(application, "Voxt.app")]
symlinks = {"Applications": "/Applications"}
hide_extensions = ["Voxt.app"]

format = "UDZO"
filesystem = "HFS+"

window_rect = ((200, 120), (720, 420))
default_view = "icon-view"
show_status_bar = False
show_tab_view = False
show_toolbar = False
show_pathbar = False
show_sidebar = False

arrange_by = None
label_pos = "bottom"
text_size = 14
icon_size = 112
icon_locations = {
    "Voxt.app": (170, 205),
    "Applications": (550, 205),
}
