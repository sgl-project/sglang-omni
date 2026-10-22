package main

import (
 "context"
 "errors"
 "net/http/httptest"
 "strings"
 "testing"
 "time"
 "tailscale.com/types/key"
)
func TestPairingHandlerControlsAndIdempotence(t *testing.T) {
 now:=time.Now(); calls:=0; phone:=key.NewNode().Public().String()
 h:=newPairingHandler("fixture-proof",now.Add(time.Minute),func()time.Time{return now},func(context.Context,string)(string,error){calls++;return "voxtmobile://pair?fixture",nil})
 send:=func(method,path,proof,body string)int{r:=httptest.NewRequest(method,path,strings.NewReader(body));r.Header.Set("Authorization","Bearer "+proof);w:=httptest.NewRecorder();h.ServeHTTP(w,r);return w.Code}
 body:=`{"phoneKey":"`+phone+`"}`
 for _,tc:=range []struct{method,path,proof,body string;code int}{
  {"POST","/v1/compose","fixture-proof",body,403},{"POST","/pair?url=/v1/health","fixture-proof",body,403},
  {"GET","/pair","fixture-proof",body,405},{"POST","/pair","wrong",body,401},
  {"POST","/pair","fixture-proof",`{"phoneKey":"bad"}`,400},
  {"POST","/pair","fixture-proof",strings.Repeat("x",5000),413},
  {"POST","/pair","fixture-proof",body,200},{"POST","/pair","fixture-proof",body,200},
  {"POST","/pair","fixture-proof",`{"phoneKey":"`+key.NewNode().Public().String()+`"}`,409},
 }{if got:=send(tc.method,tc.path,tc.proof,tc.body);got!=tc.code{t.Fatalf("%s got %d want %d",tc.path,got,tc.code)}}
 if calls!=1{t.Fatal("authorization not idempotent")};now=now.Add(2*time.Minute)
 if send("POST","/pair","fixture-proof",body)!=410{t.Fatal("expired session accepted")}
}
func TestPairingNeverReturnsInvitationWhenPersistenceFails(t *testing.T){
 h:=newPairingHandler("fixture-proof",time.Now().Add(time.Minute),time.Now,func(context.Context,string)(string,error){return "secret-invitation",errors.New("save denied")})
 r:=httptest.NewRequest("POST","/pair",strings.NewReader(`{"phoneKey":"`+key.NewNode().Public().String()+`"}`));r.Header.Set("Authorization","Bearer fixture-proof")
 w:=httptest.NewRecorder();h.ServeHTTP(w,r)
 if w.Code!=503||strings.Contains(w.Body.String(),"secret-invitation"){t.Fatal("persistence failure released invitation")}
}
