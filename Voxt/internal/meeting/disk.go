package meeting

// Disk thresholds for meeting capture (design §4.10; tests use smaller values).
const (
	diskMinStartBytes  = int64(1536 * 1024 * 1024) // 1.5 GiB
	diskMinRecordBytes = int64(966367641)          // 0.9 GiB (card-check fake)
)

// DiskMinStartBytes is the free space required before Start (tests may fake this).
func DiskMinStartBytes() int64 { return diskMinStartBytes }

// DiskMinRecordBytes is the free-space floor that auto-ends recording.
func DiskMinRecordBytes() int64 { return diskMinRecordBytes }
