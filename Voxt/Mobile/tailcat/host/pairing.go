package main

import (
 "context"
 "crypto/subtle"
 "encoding/json"
 "io"
 "net/http"
 "sync"
 "time"
 "tailscale.com/types/key"
)

type pairingHandler struct {
 mu sync.Mutex
 proof string
 expires time.Time
 now func()time.Time
 authorize func(context.Context,string)(string,error)
 bound, invitation string
}
func newPairingHandler(proof string, expires time.Time, now func()time.Time, authorize func(context.Context,string)(string,error))*pairingHandler {
 return &pairingHandler{proof:proof,expires:expires,now:now,authorize:authorize}
}
func (h *pairingHandler) ServeHTTP(w http.ResponseWriter,r *http.Request){
 w.Header().Set("Cache-Control","no-store")
 if r.URL.EscapedPath()!="/pair"||r.URL.RawQuery!=""{w.WriteHeader(403);return}
 if r.Method!="POST"{w.WriteHeader(405);return}
 if subtle.ConstantTimeCompare([]byte(r.Header.Get("Authorization")),[]byte("Bearer "+h.proof))!=1 {w.WriteHeader(401);return}
 body,err:=io.ReadAll(io.LimitReader(r.Body,4097));if err!=nil{w.WriteHeader(400);return};if len(body)>4096{w.WriteHeader(413);return}
 var req struct{PhoneKey string `json:"phoneKey"`}
 if json.Unmarshal(body,&req)!=nil{w.WriteHeader(400);return}
 var pub key.NodePublic
 if pub.UnmarshalText([]byte(req.PhoneKey))!=nil||pub.IsZero(){w.WriteHeader(400);return}
 phone:=pub.String()
 h.mu.Lock();defer h.mu.Unlock()
 if !h.now().Before(h.expires){w.WriteHeader(410);return}
 if h.bound!=""&&h.bound!=phone{w.WriteHeader(409);return}
 if h.invitation==""{
  invitation,err:=h.authorize(r.Context(),phone)
  if err!=nil||invitation==""{w.WriteHeader(503);return}
  h.bound=phone;h.invitation=invitation
 }
 w.Header().Set("Content-Type","application/json")
 _=json.NewEncoder(w).Encode(map[string]string{"invitation":h.invitation})
}
