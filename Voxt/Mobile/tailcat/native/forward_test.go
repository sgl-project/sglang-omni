package main
import("bytes";"context";"io";"net";"net/http";"net/http/httptest";"testing";"time")
func TestForwardHTTPBodyAndBearer(t *testing.T){
 origin:=httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter,r *http.Request){
  if r.Header.Get("Authorization")!="Bearer fixture"{w.WriteHeader(401);return};io.Copy(w,r.Body)
 }));defer origin.Close()
 f,e:=startForward(func(ctx context.Context)(net.Conn,error){return (&net.Dialer{}).DialContext(ctx,"tcp",origin.Listener.Addr().String())});if e!=nil{t.Fatal(e)};defer f.Close()
 body:=bytes.Repeat([]byte("wave-data"),65536)
 c:=&http.Client{Timeout:time.Second}
 for _,auth:=range []bool{false,true}{
  req,_:=http.NewRequest("POST","http://"+f.Addr(),bytes.NewReader(body));if auth{req.Header.Set("Authorization","Bearer fixture")}
  res,e:=c.Do(req);if e!=nil{t.Fatal(e)};got,e:=io.ReadAll(res.Body);res.Body.Close();if e!=nil{t.Fatal(e)}
  if auth&&(res.StatusCode!=200||!bytes.Equal(body,got)){t.Fatal("body/auth forwarding corrupted")};if !auth&&res.StatusCode!=401{t.Fatal("negative authentication accepted")}
 }
 f.Close();f.Close()
 conn,e:=net.DialTimeout("tcp",f.Addr(),100*time.Millisecond);if e==nil{conn.Close();t.Fatal("closed listener still accepts")}
}
func TestCloseCancelsPendingDial(t *testing.T){
 entered:=make(chan struct{});exited:=make(chan struct{})
 f,e:=startForward(func(ctx context.Context)(net.Conn,error){close(entered);<-ctx.Done();close(exited);return nil,ctx.Err()});if e!=nil{t.Fatal(e)}
 c,e:=net.Dial("tcp",f.Addr());if e!=nil{t.Fatal(e)};defer c.Close()
 select{case <-entered:case <-time.After(time.Second):t.Fatal("dial never started")}
 f.Close()
 select{case <-exited:case <-time.After(time.Second):t.Fatal("dial not cancelled")}
}
func TestCloseInterruptsActiveStream(t *testing.T){
 remote,peer:=net.Pipe();defer peer.Close()
 f,e:=startForward(func(context.Context)(net.Conn,error){return remote,nil});if e!=nil{t.Fatal(e)}
 c,e:=net.Dial("tcp",f.Addr());if e!=nil{t.Fatal(e)};defer c.Close()
 c.Write([]byte("x"));b:=make([]byte,1);peer.SetReadDeadline(time.Now().Add(time.Second));if _,e=peer.Read(b);e!=nil{t.Fatal(e)}
 done:=make(chan struct{});go func(){f.Close();close(done)}()
 select{case <-done:case <-time.After(time.Second):t.Fatal("active stream prevents close")}
 c.SetReadDeadline(time.Now().Add(time.Second));if _,e=c.Read(b);e==nil{t.Fatal("closed stream readable")}
}
