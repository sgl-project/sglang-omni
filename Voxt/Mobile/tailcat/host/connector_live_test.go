package main

import (
 "bufio"
 "context"
 "encoding/json"
 "io"
 "net"
 "net/http"
 "net/http/httptest"
 "net/url"
 "os"
 "strconv"
 "strings"
 "sync"
 "sync/atomic"
 "testing"
 "time"
 "github.com/tailscale/tailcat"
 "tailscale.com/types/key"
 "tailscale.com/types/logger"
)

// Exercises the production v2 runtime against an owned HTTP origin.
// Requires an explicit opt-in; never connects to the user's Voxt process.
func TestConnectorLivePairingAndForwarding(t *testing.T){
 if os.Getenv("VOXT_CONNECTOR_LIVE")!="1"{t.Skip("owned live transport opt-in")}
 var hits atomic.Int64
 origin:=httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter,r *http.Request){hits.Add(1);if r.Header.Get("Authorization")!="Bearer owned-api-token"{w.WriteHeader(401);return};io.WriteString(w,"owned-origin-ok")}));defer origin.Close()
 _,rawPort,_:=net.SplitHostPort(strings.TrimPrefix(origin.URL,"http://"));port,_:=strconv.Atoi(rawPort)
 input,write:=io.Pipe();read,output:=io.Pipe();var writeMu sync.Mutex
 send:=func(v any){writeMu.Lock();defer writeMu.Unlock();_ = json.NewEncoder(write).Encode(v)}
 events:=make(chan map[string]any,20);var persists atomic.Int64
 go func(){defer close(events);dec:=json.NewDecoder(read);for{var e map[string]any;if dec.Decode(&e)!=nil{return};if e["event"]=="persist"{persists.Add(1);send(map[string]any{"command":"ack","id":e["id"],"ok":true});continue};events<-e}}()
 done:=make(chan error,1);go func(){done<-RunConnector(input,output,Option{});output.Close()}()
 t.Cleanup(func(){write.Close();read.Close();select{case <-done:case <-time.After(10*time.Second):t.Error("runtime shutdown timeout")}})
 event:=func(want string)map[string]any{t.Helper();select{case e:=<-events:if e["event"]!=want{t.Fatalf("event type %v want %s",e["event"],want)};return e;case <-time.After(30*time.Second):t.Fatal("event timeout");return nil}}
 send(map[string]any{"version":2,"localPort":port,"allowedClients":[]string{},"token":"owned-api-token"});event("ready");send(map[string]any{"command":"pair"})
 e:=event("pairing");u,_:=url.Parse(e["url"].(string));q:=u.Query();var temporary key.NodePrivate
 if temporary.UnmarshalText([]byte(q.Get("key")))!=nil{t.Fatal("temporary key parse")}
 client:=func(address string,k key.NodePrivate)*tailcat.Client{c:=tailcat.NewClient(tailcat.Addr(address));c.Key=k;c.Logf=logger.Discard;t.Cleanup(func(){c.Close()});return c}
 temp:=client(q.Get("tailcat"),temporary)
 request:=func(c *tailcat.Client,port uint16,path,token,phone string)(int,[]byte){t.Helper();ctx,cancel:=context.WithTimeout(context.Background(),8*time.Second);defer cancel();conn,err:=c.DialTCPPort(ctx,port);if err!=nil{t.Fatalf("dial port %d failed",port)};defer conn.Close();conn.SetDeadline(time.Now().Add(15*time.Second));body,_:=json.Marshal(map[string]string{"phoneKey":phone});r,_:=http.NewRequest("POST","http://owned"+path,strings.NewReader(string(body)));r.Header.Set("Authorization","Bearer "+token);if r.Write(conn)!=nil{t.Fatal("request write failed")};resp,err:=http.ReadResponse(bufio.NewReader(conn),r);if err!=nil{t.Fatal("response failed")};data,_:=io.ReadAll(resp.Body);resp.Body.Close();t.Logf("path=%s status=%d",path,resp.StatusCode);return resp.StatusCode,data}
 phone:=key.NewNode()
 if status,_:=request(temp,pairingPort,"/v1/compose",q.Get("proof"),phone.Public().String());status!=403{t.Fatal("forbidden endpoint")}
 if hits.Load()!=0{t.Fatal("pairing reached normal origin")}
 if status,_:=request(temp,pairingPort,"/pair","wrong-proof",phone.Public().String());status!=401{t.Fatal("proof negative control")}
 status,body:=request(temp,pairingPort,"/pair",q.Get("proof"),phone.Public().String());if status!=200{t.Fatal("pairing failed")};event("paired")
 var response struct{Invitation string `json:"invitation"`};if json.Unmarshal(body,&response)!=nil{t.Fatal("invitation json")};inv,_:=url.Parse(response.Invitation);v:=inv.Query()
 normal:=client(v.Get("tailcat"),phone)
 if status,body:=request(normal,virtualPort,"/v1/health",v.Get("token"),"");status!=200||string(body)!="owned-origin-ok"{t.Fatal("normal positive control")}
 if status,_:=request(normal,virtualPort,"/v1/health","wrong-token","");status!=401{t.Fatal("normal negative control")}
 if status,_:=request(temp,pairingPort,"/pair",q.Get("proof"),phone.Public().String());status!=200||persists.Load()!=1{t.Fatal("same-phone retry changed authorization")}
 if status,_:=request(temp,pairingPort,"/pair",q.Get("proof"),key.NewNode().Public().String());status!=409{t.Fatal("second phone accepted")}
 reject:=func(c *tailcat.Client,port uint16){t.Helper();ctx,cancel:=context.WithTimeout(context.Background(),2*time.Second);defer cancel();conn,err:=c.DialTCPPort(ctx,port);if err==nil{conn.Close();t.Fatal("forbidden transport accepted")}}
 reject(client(q.Get("tailcat"),key.NewNode()),pairingPort)
 reject(client(v.Get("tailcat"),temporary),virtualPort)
 send(map[string]any{"command":"cancelPairing"});event("pairingClosed")
 reject(client(q.Get("tailcat"),temporary),pairingPort)
 if status,_:=request(normal,virtualPort,"/v1/health",v.Get("token"),"");status!=200{t.Fatal("normal node lost on pairing cancel")}
 t.Log("pairing+normal positive controls; proof/token/key/endpoint/second-phone negatives; persistence once; cancel isolation passed")
}
