//lint:file-ignore SA4006 the snippets mirror README.md, values unused on purpose

// Package readme keeps README.md's examples compiling.
package readme

import (
	"context"
	"errors"
	"io/fs"
	"testing"

	"github.com/hehehai/actnowkit/credentials"
	"github.com/hehehai/actnowkit/family"
	"github.com/hehehai/actnowkit/hfhub"
	"github.com/hehehai/actnowkit/models"
	"github.com/hehehai/actnowkit/redact"
	"github.com/hehehai/actnowkit/registry"
	famruntime "github.com/hehehai/actnowkit/runtime"
	"github.com/hehehai/actnowkit/settings"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/hehehai/actnowkit/sounds"
)

type ownSettings struct{ Language, Proxy, Endpoints, ModelsFolder string }

func readmeExamples(ctx context.Context) {
	const bundleID = "dev.hehehai.dicta"
	var (
		own        ownSettings
		follow     = settings.Follow{}
		version    = "1.0.0"
		hasModels  bool
		catalogue  []hfhub.Entry
		wanted     []models.Claim
		repo       = "mlx-community/Qwen3-ASR-0.6B-4bit"
		commit     = "313d850181767edf09f00a9c289becca70e58cd0"
		endpoints  = "auto"
		proxy      = ""
		key        = "sk-x"
		ownLimit   int64
		loaded     []famruntime.Loaded
		myManifest registry.Manifest
		reload     = func(string) {}
		claimsOf   = func([]models.Found) []models.Claim { return nil }
	)
	recordedKeys := "Fn"

	fam := family.Default()
	info, err := fam.Ensure(bundleID)
	if err != nil || info.ReadOnly() {
		return
	}
	fam.CleanAllTemps()
	famSettings, _, _ := settings.Read(fam)
	root, _ := famSettings.Get(settings.KeyModelsRoot)
	hub, _ := models.ResolveHub(follow.Of(settings.KeyModelsRoot), root, own.ModelsFolder)
	registry.Write(fam, registry.Manifest{
		App: "dicta", BundleID: bundleID, Name: "Dicta", Version: version,
		Path: registry.BundlePath(), TeamID: registry.TeamID(), URLScheme: "dicta",
		Capabilities: []string{registry.CapModelsHF, registry.CapASRLocal, registry.CapLLMLocal, registry.CapMeeting},
		Models:       &registry.ModelsInfo{Root: hub},
		Links:        map[string]string{registry.LinkModels: "dicta://models"},
	})
	found, _ := registry.Discover(fam, registry.DefaultSystem)
	store := &models.Store{
		Family: fam, Hub: hub, BundleID: bundleID, AppID: "dicta",
		Apps: func() map[string]models.AppState { return registry.States(found) },
	}
	store.ApplyTombstones()
	backup := settings.ModelsBackup(famSettings)
	store.Import(ctx, models.LegacyHubs(hub), backup, wanted...)
	store.Heal(ctx, wanted...)
	models.ApplyBackup(fam, hub, backup)
	_ = redact.Text("x")
	w, _ := fam.Watch(family.TopicApps, family.TopicSettings, family.TopicModels, family.TopicMemory)
	go func() {
		for topic := range w.C {
			reload(topic)
		}
	}()

	outcomes, _ := settings.Migrate(ctx, fam, bundleID, map[string]settings.Local{
		settings.KeyLanguage:    {Value: own.Language, IsDefault: own.Language == ""},
		settings.KeyProxy:       {Value: own.Proxy, IsDefault: own.Proxy == ""},
		settings.KeyHFEndpoints: {Value: own.Endpoints, IsDefault: own.Endpoints == "auto"},
		settings.KeyModelsRoot:  {Value: own.ModelsFolder, IsDefault: own.ModelsFolder == "", InUse: hasModels},
	})
	for _, o := range outcomes {
		follow[o.Key] = o.Follow
		_ = o.Conflict
	}
	for _, e := range catalogue {
		if models.Check(hub, models.Claim{Repo: e.Repo, Revision: e.Revision}).State == models.Installed {
			store.Claim(ctx, models.Claim{Repo: e.Repo, Revision: e.Revision, Model: e.ID, Role: models.RoleASR, Origin: models.OriginMigrated})
		}
	}

	foundModels := store.Survey(wanted...)
	store.Claim(ctx, claimsOf(foundModels)...)
	dl := hfhub.New(hfhub.Options{Store: store, Endpoints: settings.Endpoints(endpoints), Proxy: proxy})
	err = dl.Download(ctx, hfhub.Entry{ID: "qwen3-asr-0.6b-4bit", Repo: repo, Revision: commit}, func(p hfhub.Progress) {
		_ = p.WaitingFor
	})
	d, _ := store.Decide(models.Ref{Repo: repo, Revision: commit})
	if d.InUse() {
		_ = d.UserNames()
		_ = d.FreeBytes
	}
	res, err := store.Delete(ctx, d.Ref, models.ThisApp)
	_ = res.Deleted
	var busy *models.BusyError
	_ = errors.As(err, &busy)
	tok, tokCommit := "openai/whisper-large-v3", commit
	store.Claim(ctx, models.Claim{Repo: tok, Revision: tokCommit, Model: "whisper-turbo", Aux: true})
	res, err = store.DeleteModel(ctx, models.Ref{Repo: tok, Revision: tokCommit}, "whisper-turbo", models.ThisApp)
	_ = res.OwnModels
	usage, _ := store.Usage()
	_ = usage.Mine + usage.Shared
	orphans, _ := store.Orphans()
	_ = orphans
	_ = models.SnapshotDir(hub, d.Ref)

	cs := &credentials.Store{Family: fam, BundleID: bundleID}
	cs.Save(ctx, "deepseek", key, credentials.SaveOptions{})
	_ = credentials.Ref("deepseek")
	if e, ok := cs.Offer("deepseek"); ok {
		_ = e.Hint
	}
	key, err = cs.Use(ctx, "deepseek")
	key, err = cs.Secret(ctx, "deepseek")
	_, _ = key, err

	me := registry.HolderOf(myManifest)
	for _, c := range shortcuts.FindConflicts(me, registry.Holders(found)) {
		_ = c.Prompt
		_ = c.Key()
	}
	hits := shortcuts.UsedBy(recordedKeys, bundleID, registry.Holders(found))
	_ = hits

	cap := famruntime.FamilyCap(famruntime.PhysicalMemory())
	budget := famruntime.Budget(ownLimit, cap, famruntime.OthersLoadedBytes(fam, bundleID))
	famruntime.Write(fam, famruntime.File{BundleID: bundleID, BudgetBytes: budget, Loaded: loaded})
	famruntime.RequestMemory(fam)
	famruntime.Remove(fam, bundleID)
}

func TestReadmeCompiles(t *testing.T) { _ = readmeExamples }

// soundsExample mirrors README.md's "Cue sounds".
func soundsExample(resources fs.FS, row sounds.Sound) {
	cat, err := sounds.Load(resources, "sounds")
	_ = err
	player := sounds.NewPlayer(cat, nil)

	var recordStart = sounds.Event{ID: "record.start", Name: sounds.Text{En: "Start recording", Zh: "开始录音"},
		Default: "tally", Fallback: "Pop", Uses: []string{"record.start"}}

	prefs := sounds.DefaultPrefs()
	player.Preload(prefs.Choice(recordStart))
	player.PlayEvent(recordStart, prefs)
	t := prefs.Target(cat, recordStart)
	_ = t.Length(cat)

	f := sounds.Filter{Group: "notify", Tags: []string{"soft"}, Query: "bell"}
	rows := cat.Filter(f)
	_, _ = cat.GroupCounts(f), cat.TagCounts(f, true)
	suggested := cat.Suggested(recordStart)
	_, _ = rows, suggested
	player.Preview(row.ID, "", prefs.Volume)
	ch, left := player.Previewing()
	step := sounds.NearestVolumeStep(prefs.Volume)
	_, _, _ = ch, left, step
	prefs.Choose(recordStart, row.ID)
	player.Preview(prefs.Choice(recordStart), recordStart.Fallback, prefs.Volume)
}

func TestSoundsReadmeCompiles(t *testing.T) { _ = soundsExample }
