package main

import (
	"cmp"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/url"
	"os"
	"slices"
	"strings"
	"time"
)

// Uploads to Amazon S3 and compatible services, such as Cloudflare R2, in
// pure Go: objects are put with a single request signed with AWS Signature
// Version 4.

// S3 is a bucket of Amazon S3, or of a compatible service such as
// Cloudflare R2. The credentials come from AWS_ACCESS_KEY_ID,
// AWS_SECRET_ACCESS_KEY and AWS_SESSION_TOKEN.
type S3 struct {
	Bucket string `json:"bucket"`
	// Prefix is the directory of the files in the bucket, e.g. "my-app"
	// (default: its root).
	Prefix string `json:"prefix"`
	// Region is the region of the bucket (default: AWS_REGION, else
	// AWS_DEFAULT_REGION, else us-east-1).
	Region string `json:"region"`
	// Endpoint is the URL of a compatible service, e.g.
	// "https://<account>.r2.cloudflarestorage.com" (default: Amazon S3).
	Endpoint string `json:"endpoint"`
	// PathStyle names the bucket in the path of URLs rather than in the
	// host name (default: with Endpoint, or a bucket name with dots).
	PathStyle *bool `json:"pathStyle"`
}

func (s *S3) validate() error {
	if s.Bucket == "" {
		return errors.New("updates.s3 needs a bucket")
	}
	if s.Endpoint != "" {
		u, err := url.Parse(s.Endpoint)
		switch {
		case err != nil || (u.Scheme != "https" && u.Scheme != "http") || u.Host == "":
			return fmt.Errorf("updates.s3.endpoint %q is not an http(s) URL", s.Endpoint)
		case strings.Trim(u.Path, "/") != "" || u.RawQuery != "":
			return fmt.Errorf("updates.s3.endpoint %q has a path: the endpoint is the service's, and bucket names the bucket", s.Endpoint)
		}
	}
	return nil
}

// s3Client puts objects into a bucket.
type s3Client struct {
	bucket, prefix, region string
	endpoint               *url.URL
	pathStyle              bool
	accessKey, secretKey   string
	sessionToken           string
}

func newS3Client(s *S3) (*s3Client, error) {
	c := &s3Client{
		bucket:       s.Bucket,
		prefix:       strings.Trim(s.Prefix, "/"),
		region:       cmp.Or(s.Region, os.Getenv("AWS_REGION"), os.Getenv("AWS_DEFAULT_REGION"), "us-east-1"),
		pathStyle:    s.Endpoint != "" || strings.Contains(s.Bucket, "."),
		accessKey:    os.Getenv("AWS_ACCESS_KEY_ID"),
		secretKey:    os.Getenv("AWS_SECRET_ACCESS_KEY"),
		sessionToken: os.Getenv("AWS_SESSION_TOKEN"),
	}
	if c.accessKey == "" || c.secretKey == "" {
		return nil, errors.New("uploading to updates.s3 needs the credentials in AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY")
	}
	if s.PathStyle != nil {
		c.pathStyle = *s.PathStyle
	}
	var err error
	c.endpoint, err = url.Parse(cmp.Or(s.Endpoint, "https://s3."+c.region+".amazonaws.com"))
	return c, err
}

// key returns the key of the file named name.
func (c *s3Client) key(name string) string {
	if c.prefix == "" {
		return name
	}
	return c.prefix + "/" + name
}

// objectURL returns the URL of the object of key.
func (c *s3Client) objectURL(key string) string {
	host, path := c.endpoint.Host, "/"+s3Escape(key)
	if c.pathStyle {
		path = "/" + s3Escape(c.bucket) + path
	} else {
		host = c.bucket + "." + host
	}
	return c.endpoint.Scheme + "://" + host + path
}

// put uploads the file at path to key, with the headers of header.
func (c *s3Client) put(key, path string, header http.Header) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	sum := sha256.New()
	size, err := io.Copy(sum, f)
	if err != nil {
		return err
	}
	var body io.Reader = http.NoBody // an empty body of known length
	if size > 0 {
		body = io.NewSectionReader(f, 0, size)
	}
	req, err := http.NewRequest(http.MethodPut, c.objectURL(key), body)
	if err != nil {
		return err
	}
	req.ContentLength = size
	maps.Copy(req.Header, header)
	c.sign(req, hex.EncodeToString(sum.Sum(nil)), time.Now())
	req.Header.Set("User-Agent", "mygo/"+version)
	// S3 redirects requests for another region, which must not be followed
	// as a GET.
	client := &http.Client{Timeout: time.Hour, CheckRedirect: func(*http.Request, []*http.Request) error {
		return http.ErrUseLastResponse
	}}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 == 2 {
		return nil
	}
	msg := resp.Status
	// S3 describes errors in XML.
	var e struct{ Code, Message string }
	data, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
	if xml.Unmarshal(data, &e) == nil && e.Code != "" {
		msg += ": " + e.Code + ": " + e.Message
	}
	if region := resp.Header.Get("X-Amz-Bucket-Region"); region != "" && region != c.region {
		msg += " (the bucket is in " + region + ": set updates.s3.region)"
	}
	return fmt.Errorf("uploading to s3://%s/%s: %s", c.bucket, key, msg)
}

// sign signs req, which has no query, with AWS Signature Version 4 for S3,
// its headers with the host, and a payload of the SHA-256 payloadHash.
func (c *s3Client) sign(req *http.Request, payloadHash string, t time.Time) {
	t = t.UTC()
	date, amzDate := t.Format("20060102"), t.Format("20060102T150405Z")
	req.Header.Set("X-Amz-Date", amzDate)
	req.Header.Set("X-Amz-Content-Sha256", payloadHash)
	if c.sessionToken != "" {
		req.Header.Set("X-Amz-Security-Token", c.sessionToken)
	}
	headers := map[string]string{"host": cmp.Or(req.Host, req.URL.Host)}
	for name, values := range req.Header {
		trimmed := make([]string, len(values))
		for i, v := range values {
			trimmed[i] = strings.Join(strings.Fields(v), " ")
		}
		headers[strings.ToLower(name)] = strings.Join(trimmed, ",")
	}
	names := slices.Sorted(maps.Keys(headers))
	var canonical strings.Builder
	for _, name := range names {
		canonical.WriteString(name + ":" + headers[name] + "\n")
	}
	signed := strings.Join(names, ";")
	request := req.Method + "\n" + req.URL.EscapedPath() + "\n\n" + canonical.String() + "\n" + signed + "\n" + payloadHash
	scope := date + "/" + c.region + "/s3/aws4_request"
	hash := sha256.Sum256([]byte(request))
	key := hmacSHA256([]byte("AWS4"+c.secretKey), date)
	for _, s := range []string{c.region, "s3", "aws4_request"} {
		key = hmacSHA256(key, s)
	}
	signature := hmacSHA256(key, "AWS4-HMAC-SHA256\n"+amzDate+"\n"+scope+"\n"+hex.EncodeToString(hash[:]))
	req.Header.Set("Authorization", "AWS4-HMAC-SHA256 Credential="+c.accessKey+"/"+scope+
		", SignedHeaders="+signed+", Signature="+hex.EncodeToString(signature))
}

func hmacSHA256(key []byte, data string) []byte {
	h := hmac.New(sha256.New, key)
	h.Write([]byte(data))
	return h.Sum(nil)
}

// s3Escape escapes a key for the path of a URL as Signature Version 4
// wants it: every byte but letters, digits, "-._~" and "/".
func s3Escape(s string) string {
	var b strings.Builder
	for i := range len(s) {
		if ch := s[i]; 'A' <= ch && ch <= 'Z' || 'a' <= ch && ch <= 'z' || '0' <= ch && ch <= '9' || strings.IndexByte("-._~/", ch) >= 0 {
			b.WriteByte(ch)
		} else {
			fmt.Fprintf(&b, "%%%02X", ch)
		}
	}
	return b.String()
}
