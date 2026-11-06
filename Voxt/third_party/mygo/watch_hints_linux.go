package mygo

import (
	"os"
	"syscall"
	"unsafe"
)

// fileWatchSubdirs returns the names of the directories in d, from the
// types getdents gives, so that what isn't a directory is not stat'ed;
// an entry of unknown type is stat'ed, without following a symlink.
func fileWatchSubdirs(d *os.Root) ([]string, error) {
	f, err := d.Open(".")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	conn, err := f.SyscallConn()
	if err != nil {
		return nil, err
	}
	var names []string
	buf := make([]byte, 32<<10)
	for {
		var n int
		var errno error
		if err := conn.Read(func(fd uintptr) bool {
			n, errno = syscall.Getdents(int(fd), buf)
			return errno != syscall.EAGAIN
		}); err != nil {
			return nil, err
		}
		if errno != nil {
			return nil, &os.PathError{Op: "getdents", Path: f.Name(), Err: errno}
		}
		if n <= 0 {
			return names, nil
		}
		// struct linux_dirent64: ino, off, reclen, type, name.
		for b := buf[:n]; len(b) >= 19; {
			reclen := int(*(*uint16)(unsafe.Pointer(&b[16])))
			if reclen < 19 || reclen > len(b) {
				break
			}
			typ, name := b[18], b[19:reclen]
			for i, c := range name {
				if c == 0 {
					name = name[:i]
					break
				}
			}
			b = b[reclen:]
			s := string(name)
			if s == "." || s == ".." {
				continue
			}
			switch typ {
			case syscall.DT_DIR:
				names = append(names, s)
			case syscall.DT_UNKNOWN:
				if info, err := d.Lstat(s); err == nil && info.IsDir() {
					names = append(names, s)
				}
			}
		}
	}
}
