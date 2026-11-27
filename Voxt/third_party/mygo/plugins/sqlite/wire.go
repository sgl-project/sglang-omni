package sqlite

import (
	"fmt"
	"math"
	"strconv"
)

// Every cell carries its SQLite storage class: JSON alone cannot distinguish
// BLOBs from text, or carry int64 without losing JavaScript precision.
type wireValue struct {
	Type    string  `json:"type"`
	Integer string  `json:"integer,omitzero"`
	Real    float64 `json:"real,omitzero"`
	Special string  `json:"special,omitzero"`
	Text    string  `json:"text,omitzero"`
	Blob    []byte  `json:"blob,omitzero"`
}
type wireStatement struct {
	SQL  string      `json:"sql"`
	Args []wireValue `json:"args"`
}
type wireResult struct {
	Changes      int64  `json:"changes"`
	LastInsertID string `json:"lastInsertId"`
}
type wireRows struct {
	Columns []string      `json:"columns"`
	Rows    [][]wireValue `json:"rows"`
}

func parameters(values []wireValue) ([]any, error) {
	args := make([]any, len(values))
	for i, value := range values {
		switch value.Type {
		case "null":
			args[i] = nil
		case "integer":
			n, err := strconv.ParseInt(value.Integer, 10, 64)
			if err != nil {
				return nil, fmt.Errorf("sqlite: parameter %d: invalid signed 64-bit integer", i+1)
			}
			args[i] = n
		case "real":
			if value.Special != "" || math.IsNaN(value.Real) || math.IsInf(value.Real, 0) {
				return nil, fmt.Errorf("sqlite: parameter %d: number must be finite", i+1)
			}
			args[i] = value.Real
		case "text":
			args[i] = value.Text
		case "blob":
			args[i] = value.Blob
		default:
			return nil, fmt.Errorf("sqlite: parameter %d: unknown value type %q", i+1, value.Type)
		}
	}
	return args, nil
}

func resultWire(result Result) wireResult {
	return wireResult{Changes: result.Changes, LastInsertID: strconv.FormatInt(result.LastInsertID, 10)}
}

func rowsWire(rows Rows) wireRows {
	result := wireRows{Columns: rows.Columns, Rows: make([][]wireValue, len(rows.Values))}
	for i, row := range rows.Values {
		result.Rows[i] = make([]wireValue, len(row))
		for j, cell := range row {
			value := wireValue{Type: "null"}
			switch v := cell.(type) {
			case int64:
				value.Type, value.Integer = "integer", strconv.FormatInt(v, 10)
			case float64:
				value.Type, value.Real = "real", v
				if math.IsInf(v, 0) || math.IsNaN(v) {
					value.Real = 0
					switch {
					case math.IsInf(v, 1):
						value.Special = "Infinity"
					case math.IsInf(v, -1):
						value.Special = "-Infinity"
					default:
						value.Special = "NaN"
					}
				}
			case string:
				value.Type, value.Text = "text", v
			case []byte:
				value.Type, value.Blob = "blob", v
			}
			result.Rows[i][j] = value
		}
	}
	return result
}
