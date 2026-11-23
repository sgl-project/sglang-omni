// Command update-server serves a directory over HTTPS on the loopback
// interface, as the update feed of scripts/mac-update-e2e.sh and
// scripts/mac-beta-e2e.sh: MyGo's updater only downloads over HTTPS, and
// `updates.url` must be an https URL. It makes a self-signed certificate
// for 127.0.0.1, writes it to -cert (the test points SSL_CERT_FILE at it:
// Go 1.27 then trusts it alone, on macOS too, without touching the
// keychain) and the server's URL to -url, and logs each request.
//
//	go run ./scripts/update-server -dir feed -cert cert.pem -url url.txt
//
// With -site it is actnow.dev's update feed instead: dir is the bucket
// (actnow-releases, as -publish lays it out), served at
// /updates/<key> as the website serves it, only the files the website
// serves (update-*.json, install.sh, *.tar.gz, *.tgz, *.zip, *.delta,
// *.sig): a disk image, or anything else, is 404.
//
// With -publish BUILD it publishes a build into the bucket in dir instead,
// as actnow.dev's back end does after scripts/actnow-publish.sh uploads it
// (publish.go), and exits:
//
//	go run ./scripts/update-server -publish build/darwin-arm64 -dir bucket -app voxtnew
package main

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	"crypto/rand"
	"crypto/tls"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"flag"
	"fmt"
	"log"
	"math/big"
	"net"
	"net/http"
	"os"
	"path"
	"strings"
	"time"
)

func main() {
	dir := flag.String("dir", ".", "the directory to serve")
	addr := flag.String("addr", "127.0.0.1:0", "where to listen (loopback)")
	certOut := flag.String("cert", "cert.pem", "where to write the certificate")
	urlOut := flag.String("url", "url.txt", "where to write the URL")
	site := flag.Bool("site", false, "serve dir as actnow.dev serves its bucket: /updates/<key>, update files only")
	build := flag.String("publish", "", "publish this build into the bucket in dir, as actnow.dev does, and exit")
	app := flag.String("app", "voxtnew", "-publish: the app's directory in the bucket")
	flag.Parse()
	if *build != "" {
		if err := publish(*build, *dir, *app, os.Stdout); err != nil {
			log.Fatal(err)
		}
		return
	}

	cert, pemCert, err := selfSigned()
	if err != nil {
		log.Fatal(err)
	}
	if err := os.WriteFile(*certOut, pemCert, 0o644); err != nil {
		log.Fatal(err)
	}
	ln, err := net.Listen("tcp", *addr)
	if err != nil {
		log.Fatal(err)
	}
	if ip := ln.Addr().(*net.TCPAddr).IP; !ip.IsLoopback() {
		log.Fatalf("%s is not a loopback address", ip)
	}
	url := fmt.Sprintf("https://127.0.0.1:%d", ln.Addr().(*net.TCPAddr).Port)
	if err := os.WriteFile(*urlOut, []byte(url+"\n"), 0o644); err != nil {
		log.Fatal(err)
	}
	log.Printf("serving %s at %s", *dir, url)
	var files http.Handler = http.FileServer(http.Dir(*dir))
	if *site {
		files = actnowFeed(*dir)
	}
	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			log.Printf("%s %s", r.Method, r.URL.Path)
			files.ServeHTTP(w, r)
		}),
		TLSConfig:         &tls.Config{Certificates: []tls.Certificate{cert}},
		ReadHeaderTimeout: 10 * time.Second,
	}
	log.Fatal(srv.ServeTLS(ln, "", ""))
}

// selfSigned makes a certificate for 127.0.0.1 and localhost that is its
// own authority, for a day.
func selfSigned() (tls.Certificate, []byte, error) {
	key, err := ecdsa.GenerateKey(elliptic.P256(), rand.Reader)
	if err != nil {
		return tls.Certificate{}, nil, err
	}
	tmpl := &x509.Certificate{
		SerialNumber:          big.NewInt(time.Now().UnixNano()),
		Subject:               pkix.Name{CommonName: "VoxtNew update test"},
		NotBefore:             time.Now().Add(-time.Hour),
		NotAfter:              time.Now().Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageDigitalSignature | x509.KeyUsageCertSign,
		ExtKeyUsage:           []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		BasicConstraintsValid: true,
		IsCA:                  true,
		IPAddresses:           []net.IP{net.IPv4(127, 0, 0, 1)},
		DNSNames:              []string{"localhost"},
	}
	der, err := x509.CreateCertificate(rand.Reader, tmpl, tmpl, &key.PublicKey, key)
	if err != nil {
		return tls.Certificate{}, nil, err
	}
	return tls.Certificate{Certificate: [][]byte{der}, PrivateKey: key}, pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der}), nil
}

// actnowFeed serves the bucket in dir as actnow.dev does: GET and HEAD
// of /updates/<key>, for the update files only.
func actnowFeed(dir string) http.Handler {
	files := http.FileServer(http.Dir(dir))
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		key, ok := strings.CutPrefix(r.URL.Path, "/updates/")
		name := path.Base(key)
		served := strings.HasPrefix(name, "update-") && strings.HasSuffix(name, ".json") || name == "install.sh"
		for _, ext := range []string{".tar.gz", ".tgz", ".zip", ".delta", ".sig"} {
			served = served || strings.HasSuffix(name, ext)
		}
		if !ok || !served || r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.NotFound(w, r)
			return
		}
		r = r.Clone(r.Context())
		r.URL.Path = "/" + key
		files.ServeHTTP(w, r)
	})
}
