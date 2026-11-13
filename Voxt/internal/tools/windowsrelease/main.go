// Command windowsrelease edits the Windows release configuration and checks its update manifest.
package main

import (
	"crypto/ed25519"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
)

func main() {
	if err := run(os.Args[1:], os.Stdout); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run(args []string, stdout io.Writer) error {
	if len(args) == 0 {
		return errors.New("usage: windowsrelease version|set-version|configure-updates|check-manifest ...")
	}
	switch args[0] {
	case "version":
		if len(args) != 2 {
			return errors.New("usage: windowsrelease version mygo.json")
		}
		config, err := readObject(args[1])
		if err != nil {
			return err
		}
		var version string
		if err := json.Unmarshal(config["version"], &version); err != nil {
			return fmt.Errorf("version: %w", err)
		}
		fmt.Fprintln(stdout, version)
		return nil
	case "set-version":
		if len(args) != 3 || args[2] == "" {
			return errors.New("usage: windowsrelease set-version mygo.json VERSION")
		}
		config, err := readObject(args[1])
		if err != nil {
			return err
		}
		version, _ := json.Marshal(args[2])
		config["version"] = version
		return writeObject(args[1], config)
	case "configure-updates":
		if len(args) != 5 {
			return errors.New("usage: windowsrelease configure-updates mygo.json PUBLIC_KEY PRIVATE_KEY_PATH NOTES_PATH")
		}
		publicKey, err := base64.StdEncoding.DecodeString(args[2])
		if err != nil || len(publicKey) != ed25519.PublicKeySize {
			return errors.New("the update public key must be a base64 Ed25519 public key")
		}
		config, err := readObject(args[1])
		if err != nil {
			return err
		}
		updates, _ := json.Marshal(map[string]string{
			"publicKey":  args[2],
			"privateKey": args[3],
			"changelog":  args[4],
			"github":     "sgl-project/sglang-omni",
			"tagPrefix":  "voxt-v",
		})
		config["updates"] = updates
		return writeObject(args[1], config)
	case "check-manifest":
		if len(args) != 6 {
			return errors.New("usage: windowsrelease check-manifest MANIFEST ARCHIVE VERSION URL PUBLIC_KEY")
		}
		return checkManifest(args[1], args[2], args[3], args[4], args[5])
	default:
		return fmt.Errorf("unknown command %q", args[0])
	}
}

func readObject(path string) (map[string]json.RawMessage, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var object map[string]json.RawMessage
	if err := json.Unmarshal(data, &object); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	return object, nil
}

func writeObject(path string, object map[string]json.RawMessage) error {
	data, err := json.MarshalIndent(object, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, append(data, '\n'), 0o644)
}

func checkManifest(manifestPath, archivePath, version, url, publicKeyText string) error {
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return err
	}
	var manifest struct {
		Version   string `json:"version"`
		URL       string `json:"url"`
		Size      int64  `json:"size"`
		Signature string `json:"signature"`
	}
	if err := json.Unmarshal(data, &manifest); err != nil {
		return fmt.Errorf("%s: %w", manifestPath, err)
	}
	archive, err := os.Open(archivePath)
	if err != nil {
		return err
	}
	digest := sha256.New()
	archiveSize, err := io.Copy(digest, archive)
	if closeErr := archive.Close(); err == nil {
		err = closeErr
	}
	if err != nil {
		return err
	}
	signature, signatureErr := base64.StdEncoding.DecodeString(manifest.Signature)
	publicKey, publicKeyErr := base64.StdEncoding.DecodeString(publicKeyText)
	switch {
	case manifest.Version != version:
		return fmt.Errorf("manifest version %q, want %q", manifest.Version, version)
	case manifest.URL != url:
		return fmt.Errorf("manifest URL %q, want %q", manifest.URL, url)
	case manifest.Size <= 0 || manifest.Size != archiveSize:
		return fmt.Errorf("manifest size %d, archive size %d", manifest.Size, archiveSize)
	case signatureErr != nil || len(signature) != ed25519.SignatureSize:
		return errors.New("manifest signature is not a base64 Ed25519 signature")
	case publicKeyErr != nil || len(publicKey) != ed25519.PublicKeySize:
		return errors.New("update public key is not a base64 Ed25519 public key")
	case !ed25519.Verify(ed25519.PublicKey(publicKey), digest.Sum(nil), signature):
		return errors.New("manifest signature does not verify the archive with the update public key")
	default:
		return nil
	}
}
