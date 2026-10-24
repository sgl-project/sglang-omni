package main

import (
 "bufio"
 "context"
 "encoding/json"
 "io"
 "net"
 "net/http"
 "net/url"
 "strings"
 "sync/atomic"
 "testing"
 "time"
 "github.com/tailscale/tailcat"
 "tailscale.com/tailcfg"
 "tailscale.com/types/key"
)
type connectorFakeNode struct{closed atomic.Bool}
func(n *connectorFakeNode)Address()string{return "tc-owned-node"}
func(n *connectorFakeNode)Region()*tailcfg.DERPRegion{return fixtureRegion()}
func(n *connectorFakeNode)Close()error{n.closed.Store(true);return nil}
type nodeStart struct{spec Spec;node *connectorFakeNode}
func TestConnectorDualNodePersistenceAndIsolation(t *testing.T){
 input,write:=io.Pipe();read,output:=io.Pipe();defer write.Close();defer read.Close()
 events:=make(chan map[string]any,20);go func(){defer close(events);dec:=json.NewDecoder(read);for{var e map[string]any;if dec.Decode(&e)!=nil{return};events<-e}}()
 starts:=make(chan nodeStart,10);done:=make(chan error,1)
 go func(){done<-RunConnector(input,output,Option{Start:func(_ context.Context,s Spec)(Running,error){n:=&connectorFakeNode{};starts<-nodeStart{s,n};return n,nil}});output.Close()}()
 send:=func(v any){t.Helper();if json.NewEncoder(write).Encode(v)!=nil{t.Fatal("IPC write")}}
 event:=func(kind string)map[string]any{t.Helper();select{case e:=<-events:if e["event"]!=kind{t.Fatalf("event %v want %s",e["event"],kind)};return e;case <-time.After(3*time.Second):t.Fatal("event timeout");return nil}}
 send(map[string]any{"version":2,"localPort":18080,"allowedClients":[]string{},"token":"owned-api-token"});event("ready")
 select{case <-starts:t.Fatal("unpaired normal node started");default:}
 send(map[string]any{"command":"pair"});pair:=<-starts;e:=event("pairing")
 u,err:=url.Parse(e["url"].(string));if err!=nil{t.Fatal("QR parse")};q:=u.Query()
 var temporary key.NodePrivate;if temporary.UnmarshalText([]byte(q.Get("key")))!=nil{t.Fatal("QR key format")}
 if len(pair.spec.Allowed)!=1||pair.spec.Allowed[0]!=temporary.Public(){t.Fatal("temporary key allowlist")}
 if pair.spec.OnTCP(virtualPort)!=nil{t.Fatal("pair node exposes forwarding port")}
 if strings.Contains(e["url"].(string),"owned-api-token"){t.Fatal("token in QR")}
 phone:=key.NewNode().Public().String()
 request:=func(path,phone string)<-chan *http.Response{t.Helper();responses:=make(chan *http.Response,1);client,server:=net.Pipe();handler:=pair.spec.OnTCP(48932);if handler==nil{t.Fatal("pair port absent")};go handler(server)
  go func(){defer client.Close();client.SetDeadline(time.Now().Add(4*time.Second));r,_:=http.NewRequest("POST","http://owned"+path,strings.NewReader(`{"phoneKey":"`+phone+`"}`));r.Header.Set("Authorization","Bearer "+q.Get("proof"));if r.Write(client)!=nil{responses<-nil;return};resp,_:=http.ReadResponse(bufio.NewReader(client),r);if resp!=nil{io.Copy(io.Discard,resp.Body)};responses<-resp}();return responses}
 forbidden:=<-request("/v1/compose",phone);if forbidden==nil||forbidden.StatusCode!=403{t.Fatal("forbidden endpoint")}
 response:=request("/pair",phone);normal:=<-starts;persist:=event("persist")
 if len(normal.spec.Allowed)!=1||normal.spec.Allowed[0].String()!=phone{t.Fatal("normal allowlist")}
 if normal.spec.OnTCP(48932)!=nil{t.Fatal("normal node exposes pairing port")}
 select{case <-response:t.Fatal("response before persistence");case <-time.After(20*time.Millisecond):}
 send(map[string]any{"command":"ack","id":persist["id"],"ok":true});event("paired")
 if r:=<-response;r==nil||r.StatusCode!=200{t.Fatal("authorized phone failed")}
 if r:=<-request("/pair",phone);r==nil||r.StatusCode!=200{t.Fatal("same phone retry failed")}
 if r:=<-request("/pair",key.NewNode().Public().String());r==nil||r.StatusCode!=409{t.Fatal("second phone accepted")}
 send(map[string]any{"command":"cancelPairing"});event("pairingClosed")
 if !pair.node.closed.Load()||normal.node.closed.Load(){t.Fatal("cancel did not isolate node lifetimes")}
 write.Close();select{case <-done:case <-time.After(3*time.Second):t.Fatal("runtime did not stop on EOF")}
 if !normal.node.closed.Load(){t.Fatal("normal node leaked")}
}

func TestConnectorRejectedPersistenceClosesUncommittedNormalNode(t *testing.T){
 read,write:=io.Pipe();defer read.Close();defer write.Close()
 ctx,cancel:=context.WithCancel(context.Background());defer cancel()
 node:=&connectorFakeNode{}
 c:=&connectorRuntime{ctx:ctx,config:connectorConfig{LocalPort:18080},out:write,pending:map[string]chan bool{},private:key.NewNode(),psk:tailcat.NewPresharedKey(),start:func(context.Context,Spec)(Running,error){return node,nil}}
 done:=make(chan error,1);go func(){done<-c.replaceNormal(ctx,[]string{key.NewNode().Public().String()})}()
 var e map[string]any;if json.NewDecoder(read).Decode(&e)!=nil{t.Fatal("missing persist")}
 c.pendingMu.Lock();ack:=c.pending[e["id"].(string)];c.pendingMu.Unlock();ack<-false
 select{case err:=<-done:if err==nil{t.Fatal("failed persistence accepted")};case <-time.After(time.Second):t.Fatal("ack timeout")}
 if !node.closed.Load()||c.normal!=nil||len(c.allowed)!=0{t.Fatal("uncommitted authorization kept alive")}
}
