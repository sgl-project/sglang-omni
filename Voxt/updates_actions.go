package main

type updatesActions interface {
	checkForUpdates()
	setAutoCheck(on bool)
	setAutoInstall(on bool)
	setBeta(on bool)
	relaunch()
}

type releaseActions interface {
	checkReleasesNow()
	downloadRelease(address string)
}
