package actnow

// The model library moved to ~/Library/Application Support/ActNow/models/hub
// (family spec v1.0.9, actnowkit v0.8.0). Time Machine leaves it out
// unless the family's models.backup is on; VoxtNew's claims file, lost
// with the family folder, is written again when VoxtNew comes to the
// front (internal/models Manager.Migrate).

import (
	"context"
	"errors"
	"os"

	fammodels "github.com/hehehai/actnowkit/models"
	famsettings "github.com/hehehai/actnowkit/settings"
)

// Backup is the family's models.backup as last read: whether Time
// Machine backs up the model library (off by default). Cheap: views call it.
func (f *Family) Backup() bool { return f.backup.Load() }

// ReadBackup reads models.backup from settings.json.
func (f *Family) ReadBackup() bool {
	s, _, err := famsettings.Read(f.dir)
	on := err == nil && famsettings.ModelsBackup(s)
	f.backup.Store(on)
	return on
}

// SetBackup changes models.backup for every family app and applies it.
func (f *Family) SetBackup(ctx context.Context, on bool) error {
	if f.readOnly {
		return errors.New("the family folder is read only")
	}
	v := famsettings.BackupOff
	if on {
		v = famsettings.BackupOn
	}
	f.backup.Store(on) // the switch shows it at once
	if _, err := famsettings.Set(ctx, f.dir, f.bundleID, famsettings.KeyModelsBackup, v); err != nil {
		f.ReadBackup()
		return err
	}
	return fammodels.ApplyBackup(f.dir, f.Hub(), on)
}

// ApplyBackup marks the model library for Time Machine as models.backup
// says (at start, after downloads, when the family's settings change).
func (f *Family) ApplyBackup() {
	if f.readOnly || f.store == nil {
		return
	}
	if err := fammodels.ApplyBackup(f.dir, f.Hub(), f.ReadBackup()); err != nil {
		f.logf("family: Time Machine: %v", err)
	}
}

// ClaimsLost reports that VoxtNew's claims file is gone (a cleaner
// deleted the family folder or its models/claims): until VoxtNew claims
// its models again, other apps' deletes are unsure about them (spec
// §7.6, v1.0.9).
func (f *Family) ClaimsLost() bool {
	if f.readOnly {
		return false
	}
	_, err := os.Stat(f.dir.ClaimsFile(f.bundleID))
	return os.IsNotExist(err)
}
