package sqlitestore

import (
	"context"
	"fmt"
	"strings"

	"github.com/egoist/mygo/plugins/sqlite"
	"github.com/sgl-project/sglang-omni/Voxt/internal/core"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func (r *dictionaryRepo) Profiles(ctx context.Context) ([]store.DictionaryProfile, error) {
	rows, err := r.db.Query(ctx, `SELECT id, name, active FROM dictionary_profiles ORDER BY name COLLATE NOCASE, id`)
	if err != nil {
		return nil, wrap("dictionary profiles", err)
	}
	profiles := make([]store.DictionaryProfile, 0, len(rows.Values))
	for _, row := range rows.Values {
		profiles = append(profiles, store.DictionaryProfile{ID: asString(row[0]), Name: asString(row[1]), Active: asInt(row[2]) != 0})
	}
	return profiles, nil
}

func (r *dictionaryRepo) CreateProfile(ctx context.Context, profile store.DictionaryProfile) error {
	if err := validID(profile.ID); err != nil {
		return err
	}
	name := strings.TrimSpace(profile.Name)
	if name == "" {
		return fmt.Errorf("dictionary profile %s: empty name: %w", profile.ID, core.ErrInvalid)
	}
	_, err := r.db.Execute(ctx, `INSERT INTO dictionary_profiles (id, name) VALUES (?, ?)`, profile.ID, name)
	return wrap("dictionary profile "+name, err)
}

func (r *dictionaryRepo) RenameProfile(ctx context.Context, id, name string) error {
	name = strings.TrimSpace(name)
	if name == "" {
		return fmt.Errorf("dictionary profile %s: empty name: %w", id, core.ErrInvalid)
	}
	res, err := r.db.Execute(ctx, `UPDATE dictionary_profiles SET name = ? WHERE id = ?`, name, id)
	if err != nil {
		return wrap("dictionary profile "+name, err)
	}
	if res.Changes == 0 {
		return fmt.Errorf("dictionary profile %s: %w", id, core.ErrNotFound)
	}
	return nil
}

func (r *dictionaryRepo) DeleteProfile(ctx context.Context, id string) error {
	rows, err := r.db.Query(ctx, `SELECT COUNT(*), EXISTS(SELECT 1 FROM dictionary_profiles WHERE id = ?) FROM dictionary_profiles`, id)
	if err != nil {
		return wrap("dictionary profile "+id, err)
	}
	if len(rows.Values) != 1 || asInt(rows.Values[0][1]) == 0 {
		return fmt.Errorf("dictionary profile %s: %w", id, core.ErrNotFound)
	}
	if asInt(rows.Values[0][0]) == 1 {
		return fmt.Errorf("dictionary profile %s: cannot delete the last one: %w", id, core.ErrInvalid)
	}
	results, err := r.db.Transaction(ctx, []sqlite.Statement{
		{SQL: `DELETE FROM dictionary_profiles WHERE id = ? AND (SELECT COUNT(*) FROM dictionary_profiles) > 1`, Args: []any{id}},
		{SQL: `UPDATE dictionary_profiles SET active = 1 WHERE id = (SELECT id FROM dictionary_profiles ORDER BY rowid LIMIT 1) AND NOT EXISTS(SELECT 1 FROM dictionary_profiles WHERE active = 1)`},
	})
	if err != nil {
		return wrap("dictionary profile "+id, err)
	}
	if results[0].Changes == 0 {
		return fmt.Errorf("dictionary profile %s: %w", id, core.ErrNotFound)
	}
	return nil
}

func (r *dictionaryRepo) ActivateProfile(ctx context.Context, id string) error {
	results, err := r.db.Transaction(ctx, []sqlite.Statement{
		{SQL: `UPDATE dictionary_profiles SET active = 0 WHERE active = 1 AND EXISTS(SELECT 1 FROM dictionary_profiles WHERE id = ?)`, Args: []any{id}},
		{SQL: `UPDATE dictionary_profiles SET active = 1 WHERE id = ?`, Args: []any{id}},
	})
	if err != nil {
		return wrap("dictionary profile "+id, err)
	}
	if results[1].Changes == 0 {
		return fmt.Errorf("dictionary profile %s: %w", id, core.ErrNotFound)
	}
	return nil
}
