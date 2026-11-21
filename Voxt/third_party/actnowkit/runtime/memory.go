package runtime

// PhysicalMemory is the machine's physical memory in bytes, 0 when unknown.
func PhysicalMemory() uint64 { return physicalMemory() }
