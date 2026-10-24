package main

import (
 "encoding/json"
 "net/url"
 "os"
 "strings"
 "testing"
 "unicode/utf8"
)

func TestPairLinkCarriesTheMacName(t *testing.T){
 parse:=func(link string)url.Values{t.Helper();u,err:=url.Parse(link);if err!=nil||u.Scheme!="voxtmobile"||u.Host!="pair"{t.Fatalf("bad link %q",link)};return u.Query()}
 q:=parse(pairLink("tc-owned","48933","owned-token","nodekey:owned","Studio Mac & Co"))
 if q.Get("tailcat")!="tc-owned"||q.Get("port")!="48933"||q.Get("token")!="owned-token"||q.Get("client")!="nodekey:owned"{t.Fatal("pairing fields changed")}
 if q.Get("name")!="Studio Mac & Co"{t.Fatalf("name %q",q.Get("name"))}
 if _,ok:=parse(pairLink("tc","1","t","c","  "))["name"];ok{t.Fatal("blank name sent")}
 long:=parse(pairLink("tc","1","t","c",strings.Repeat("麦",80))).Get("name")
 if utf8.RuneCountInString(long)!=64{t.Fatalf("name not cut to 64 characters: %d",utf8.RuneCountInString(long))}
}

// The phone parses these links with URLComponents, which reads '+' as a plus, not a space
// (a Mac name would show as "…'s+Mac+mini"). Spaces must go out as %20 and '+' as %2B.
// testdata/pair-links.json is what the iPhone kit's tests parse; this test keeps it current.
var pairLinkNames=[]string{"Sam's Mac mini","C++ Mac","李的 Mac"}

func TestPairLinkEncodesNamesForThePhone(t *testing.T){
 var links []map[string]string
 for _,name:=range pairLinkNames{
  link:=pairLink("tc-owned","48931","owned+token","nodekey:"+strings.Repeat("a",64),name)
  query:=link[strings.Index(link,"?")+1:]
  if strings.Contains(query,"+"){t.Fatalf("form-encoded '+' in %q",link)}
  if !strings.Contains(query,"name=") || (strings.Contains(name," ")&&!strings.Contains(query,"%20")){t.Fatalf("space not %%20 in %q",link)}
  links=append(links,map[string]string{"name":name,"link":link})
 }
 want,_:=json.MarshalIndent(links,""," ");want=append(want,'\n')
 if os.Getenv("UPDATE_PAIR_LINKS")=="1"{if err:=os.WriteFile("testdata/pair-links.json",want,0o644);err!=nil{t.Fatal(err)}}
 got,err:=os.ReadFile("testdata/pair-links.json")
 if err!=nil||string(got)!=string(want){t.Fatal("testdata/pair-links.json is stale: rerun with UPDATE_PAIR_LINKS=1")}
}
