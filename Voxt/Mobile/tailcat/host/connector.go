package main

import (
 "bufio"
 "context"
 "crypto/rand"
 "encoding/base64"
 "encoding/json"
 "errors"
 "io"
 "net"
 "net/http"
 "net/url"
 "os"
 "os/signal"
 "strconv"
 "sort"
 "strings"
 "sync"
 "syscall"
 "time"
 "github.com/tailscale/tailcat"
 "tailscale.com/tailcfg"
 "tailscale.com/types/key"
)
const pairingPort=48932

type connectorConfig struct {
 Version int `json:"version"`
 LocalPort int `json:"localPort"`
 State json.RawMessage `json:"state"`
 AllowedClients []string `json:"allowedClients"`
 Token string `json:"token"`
 // Name is this Mac's name; the phone shows it in Settings.
 Name string `json:"name"`
}
type connectorCommand struct{Command string `json:"command"`;ID string `json:"id"`;OK bool `json:"ok"`}
type connectorRuntime struct{
 ctx context.Context
 config connectorConfig
 start func(context.Context,Spec)(Running,error)
 out io.Writer
 writeMu,opMu,pendingMu sync.Mutex
 pending map[string]chan bool
 serial uint64
 private key.NodePrivate
 psk tailcat.PresharedKey
 region *tailcfg.DERPRegion
 allowed []string
 normal Running
 normalConns *liveConns
 pairing *pairingNode
}
type pairingNode struct{node Running;server *http.Server;listener *connectionListener;cancel context.CancelFunc;once sync.Once}
func(p *pairingNode)close(){p.once.Do(func(){p.cancel();p.server.Close();p.listener.Close();p.node.Close()})}

// RunConnector is a separate protocol entry point: legacy host invocations keep v1.
// It never writes configuration or API tokens to logs; stdout is private typed IPC.
func RunConnector(in io.Reader,out io.Writer,opt Option)error{
 scanner:=bufio.NewScanner(in);scanner.Buffer(make([]byte,4096),64*1024)
 if !scanner.Scan(){return errors.New("invalid-request")}
 var cfg connectorConfig
 if json.Unmarshal(scanner.Bytes(),&cfg)!=nil||cfg.Version!=2||cfg.LocalPort<1024||cfg.LocalPort>65534||strings.TrimSpace(cfg.Token)==""{return errors.New("invalid-configuration")}
 if len(cfg.AllowedClients)>0{if _,code:=parseAllowlist(cfg.AllowedClients);code!=""{return errors.New(code)}}
 private,psk,region,code:=parseState(cfg.State);if code!=""{return errors.New(code)}
 if private.IsZero(){private=key.NewNode();psk=tailcat.NewPresharedKey()}
 ctx,cancel:=context.WithCancel(context.Background());defer cancel()
 start:=opt.Start;if start==nil{start=realStart}
 c:=&connectorRuntime{ctx:ctx,config:cfg,start:start,out:out,pending:map[string]chan bool{},private:private,psk:psk,region:region,allowed:append([]string(nil),cfg.AllowedClients...)}
 signals:=opt.Signals
 if signals==nil{ch:=make(chan os.Signal,1);signal.Notify(ch,syscall.SIGTERM,syscall.SIGINT);defer signal.Stop(ch);signals=ch}
 commands:=make(chan connectorCommand)
 go func(){defer close(commands);for scanner.Scan(){var cmd connectorCommand;if json.Unmarshal(scanner.Bytes(),&cmd)!=nil{cancel();return};select{case commands<-cmd:case <-ctx.Done():return}}}()
 var operations sync.WaitGroup
 tasks:=make(chan func()error,16)
 operations.Add(1)
 go func(){defer operations.Done();for{select{case <-ctx.Done():return;case fn:=<-tasks:
  c.opMu.Lock();if ctx.Err()==nil{if err:=fn();err!=nil{c.emit(map[string]any{"event":"error","code":"operation-failed"})}};c.opMu.Unlock()
 }}}()
 launch:=func(fn func()error){select{case tasks<-fn:default:c.emit(map[string]any{"event":"error","code":"busy"})}}
 launch(func()error{if len(c.allowed)>0{if err:=c.replaceNormal(ctx,c.allowed);err!=nil{return err}};return c.emit(map[string]any{"event":"ready","address":c.normalAddress()})})
 defer func(){cancel();operations.Wait();c.opMu.Lock();defer c.opMu.Unlock();if c.pairing!=nil{c.pairing.close()};c.stopNormal()}()
 for{select{
 case <-ctx.Done():return nil
 case <-signals:return nil
 case cmd,ok:=<-commands:
  if !ok{return nil}
  switch cmd.Command{
  case "ack":c.pendingMu.Lock();ch:=c.pending[cmd.ID];c.pendingMu.Unlock();if ch!=nil{select{case ch<-cmd.OK:default:}}
  case "pair":launch(c.beginPairing)
  case "cancelPairing":launch(func()error{if c.pairing!=nil{c.pairing.close();c.pairing=nil};return c.emit(map[string]any{"event":"pairingClosed"})})
  default:c.emit(map[string]any{"event":"error","code":"invalid-command"})
  }
 }}
}
func(c *connectorRuntime)emit(v any)error{c.writeMu.Lock();defer c.writeMu.Unlock();return json.NewEncoder(c.out).Encode(v)}
func(c *connectorRuntime)normalAddress()string{if c.normal==nil{return ""};return c.normal.Address()}
func(c *connectorRuntime)stopNormal(){if c.normalConns!=nil{c.normalConns.close();c.normalConns=nil};if c.normal!=nil{c.normal.Close();c.normal=nil}}
func(c *connectorRuntime)startNormal(keys []string)error{
 allowed,code:=parseAllowlist(keys);if code!=""{return errors.New(code)}
 conns,ctx:=newLiveConns(c.ctx)
 node,err:=c.start(ctx,Spec{Key:c.private,PSK:c.psk,Region:c.region,Allowed:allowed,OnTCP:serveVirtual(c.config.LocalPort,conns)})
 if err!=nil{conns.close();return err}
 if node.Region()==nil||node.Region().RegionID==0||!strings.HasPrefix(node.Address(),"tc"){node.Close();conns.close();return errors.New("invalid-node")}
 c.normal=node;c.normalConns=conns;c.region=node.Region();return nil
}
func(c *connectorRuntime)persist(ctx context.Context,keys []string)error{
 state,err:=marshalState(c.private,c.psk,c.region);if err!=nil{return err}
 c.serial++;id:=strconv.FormatUint(c.serial,10);ch:=make(chan bool,1)
 c.pendingMu.Lock();c.pending[id]=ch;c.pendingMu.Unlock()
 defer func(){c.pendingMu.Lock();delete(c.pending,id);c.pendingMu.Unlock()}()
 if err=c.emit(map[string]any{"event":"persist","id":id,"state":state,"allowedClients":keys});err!=nil{return err}
 timer:=time.NewTimer(10*time.Second);defer timer.Stop()
 select{case ok:=<-ch:if ok{return nil};return errors.New("persistence-rejected");case <-ctx.Done():return ctx.Err();case <-c.ctx.Done():return c.ctx.Err();case <-timer.C:return errors.New("persistence-timeout")}
}
func(c *connectorRuntime)replaceNormal(ctx context.Context,keys []string)error{
 previous:=append([]string(nil),c.allowed...);had:=c.normal!=nil
 c.stopNormal()
 err:=c.startNormal(keys)
 if err==nil{err=c.persist(ctx,keys)}
 if err!=nil{c.stopNormal();if had&&len(previous)>0{_ = c.startNormal(previous)};return err}
 c.allowed=append([]string(nil),keys...);return nil
}
func(c *connectorRuntime)beginPairing()error{
 if c.pairing!=nil{c.pairing.close();c.pairing=nil}
 client:=key.NewNode();serverKey:=key.NewNode();psk:=tailcat.NewPresharedKey();proofBytes:=make([]byte,32)
 if _,err:=rand.Read(proofBytes);err!=nil{return err};proof:=base64.RawURLEncoding.EncodeToString(proofBytes)
 expires:=time.Now().Add(120*time.Second);ctx,cancel:=context.WithDeadline(c.ctx,expires)
 listener:=newConnectionListener();p:=&pairingNode{listener:listener,cancel:cancel}
 handler:=newPairingHandler(proof,expires,time.Now,func(requestCtx context.Context,phone string)(string,error){
  c.opMu.Lock();defer c.opMu.Unlock()
  if c.pairing!=p||ctx.Err()!=nil{return "",errors.New("session-expired")}
  keys:=append([]string(nil),c.allowed...);found:=false;for _,k:=range keys{if k==phone{found=true}}
  if !found{keys=append(keys,phone)}
  if c.normal==nil||!found{if err:=c.replaceNormal(requestCtx,keys);err!=nil{return "",err}}
  if requestCtx.Err()!=nil||ctx.Err()!=nil{return "",errors.New("session-expired")}
  if err:=c.emit(map[string]any{"event":"paired","address":c.normalAddress(),"client":phone});err!=nil{return "",err}
  return pairLink(c.normalAddress(),strconv.Itoa(virtualPort),c.config.Token,phone,c.config.Name),nil
 })
 p.server=&http.Server{Handler:handler,ReadHeaderTimeout:5*time.Second,ReadTimeout:5*time.Second,WriteTimeout:15*time.Second,IdleTimeout:5*time.Second,MaxHeaderBytes:4096,BaseContext:func(net.Listener)context.Context{return ctx}}
 node,err:=c.start(ctx,Spec{Port:pairingPort,Key:serverKey,PSK:psk,Allowed:[]key.NodePublic{client.Public()},OnTCP:func(port uint16)func(net.Conn){if port!=pairingPort{return nil};return func(conn net.Conn){listener.offer(conn)}}})
 if err!=nil{cancel();listener.Close();return err};p.node=node;c.pairing=p
 go p.server.Serve(listener)
 go func(){<-ctx.Done();p.close()}()
 rawKey,_:=client.MarshalText()
 q:=url.Values{"v":{"1"},"tailcat":{node.Address()},"port":{strconv.Itoa(pairingPort)},"key":{string(rawKey)},"proof":{proof},"expires":{strconv.FormatInt(expires.Unix(),10)}}
 return c.emit(map[string]any{"event":"pairing","url":"voxtmobile://connect?"+queryEncode(q),"expiresAt":expires.Unix()})
}

type connectionListener struct{conns chan net.Conn;done chan struct{};once sync.Once}
func newConnectionListener()*connectionListener{return &connectionListener{conns:make(chan net.Conn),done:make(chan struct{})}}
func(l *connectionListener)offer(c net.Conn){select{case l.conns<-c:case <-l.done:c.Close()}}
func(l *connectionListener)Accept()(net.Conn,error){select{case c:=<-l.conns:return c,nil;case <-l.done:return nil,net.ErrClosed}}
func(l *connectionListener)Close()error{l.once.Do(func(){close(l.done)});return nil}
func(l *connectionListener)Addr()net.Addr{return &net.TCPAddr{IP:net.IPv4(127,0,0,1),Port:pairingPort}}

// pairLink is what a phone stores after pairing. name is optional and cut to 64 characters.
func pairLink(address,port,token,client,name string)string{
 q:=url.Values{"tailcat":{address},"port":{port},"token":{token},"client":{client}}
 if name=strings.TrimSpace(name);name!=""{
  if runes:=[]rune(name);len(runes)>64{name=string(runes[:64])}
  q.Set("name",name)
 }
 return "voxtmobile://pair?"+queryEncode(q)
}

// queryEncode is url.Values.Encode with RFC 3986 escaping: a space is %20 and '+' is %2B. The
// phone reads links with URLComponents, which keeps '+' as a plus (form encoding would turn a Mac's
// name into "…'s+Mac+mini").
func queryEncode(q url.Values)string{
 keys:=make([]string,0,len(q));for k:=range q{keys=append(keys,k)};sort.Strings(keys)
 var b strings.Builder
 for _,k:=range keys{for _,v:=range q[k]{
  if b.Len()>0{b.WriteByte('&')}
  b.WriteString(strings.ReplaceAll(url.QueryEscape(k),"+","%20"));b.WriteByte('=');b.WriteString(strings.ReplaceAll(url.QueryEscape(v),"+","%20"))
 }}
 return b.String()
}
