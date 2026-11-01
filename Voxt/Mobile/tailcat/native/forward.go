package main

import("context";"io";"net";"sync";"time")

type Forward struct {
 listener net.Listener
 ctx context.Context
 cancel context.CancelFunc
 mu sync.Mutex
 closed bool
 active map[net.Conn]bool
 wg sync.WaitGroup
 once sync.Once
}
func startForward(dial func(context.Context)(net.Conn,error))(*Forward,error){
 ln,err:=net.Listen("tcp","127.0.0.1:0");if err!=nil{return nil,err}
 ctx,cancel:=context.WithCancel(context.Background())
 f:=&Forward{listener:ln,ctx:ctx,cancel:cancel,active:make(map[net.Conn]bool)}
 f.wg.Add(1)
 go func(){defer f.wg.Done();for{c,e:=ln.Accept();if e!=nil{return};if !f.track(c){c.Close();return};f.wg.Add(1);go f.serve(c,dial)}}()
 return f,nil
}
func(f *Forward)Addr()string{return f.listener.Addr().String()}
func(f *Forward)track(c net.Conn)bool{f.mu.Lock();defer f.mu.Unlock();if f.closed{return false};f.active[c]=true;return true}
func(f *Forward)untrack(c net.Conn){c.Close();f.mu.Lock();delete(f.active,c);f.mu.Unlock()}
func(f *Forward)serve(local net.Conn,dial func(context.Context)(net.Conn,error)){
 defer f.wg.Done();defer f.untrack(local)
 ctx,cancel:=context.WithTimeout(f.ctx,10*time.Second);defer cancel()
 remote,e:=dial(ctx);if e!=nil{return};if !f.track(remote){remote.Close();return};defer f.untrack(remote)
 var copying sync.WaitGroup;copying.Add(2)
 cp:=func(dst,src net.Conn){defer copying.Done();io.Copy(dst,src);if cw,ok:=dst.(interface{CloseWrite()error});ok{cw.CloseWrite()}else{dst.Close()}}
 go cp(remote,local);go cp(local,remote);copying.Wait()
}
func(f *Forward)Close(){f.once.Do(func(){f.cancel();f.listener.Close();f.mu.Lock();f.closed=true;for c:=range f.active{c.Close()};f.mu.Unlock();f.wg.Wait()})}
