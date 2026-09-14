#include <algorithm>
#include <cstdint>
#include <vector>
#include <exception>
struct Event {int64_t x,y0,y1; int side; bool start;};
struct Fenwick{
 std::vector<int64_t> a;
 explicit Fenwick(size_t n):a(n+1,0){}
 void add(size_t i,int v){for(++i;i<a.size();i+=i&-i)a[i]+=v;}
 int64_t sum(size_t n)const{int64_t v=0;for(;n;n-=n&-n)v+=a[n];return v;}
};
extern "C" int count_overlaps(const int64_t* r,const int64_t* s,int64_t nr,int64_t ns,uint64_t* output){
 try{
  *output=0;if(nr==0||ns==0)return 0;
  std::vector<Event> events;std::vector<int64_t> ys;
  events.reserve(2*(nr+ns));ys.reserve(2*(nr+ns));
  for(int side=0;side<2;++side){
   const auto* p=side?s:r;int64_t n=side?ns:nr;
   for(int64_t i=0;i<n;++i,p+=7){
    if(p[2]>=p[4]||p[3]>=p[5])return 2;
    events.push_back({p[2],p[3],p[5],side,true});events.push_back({p[4],p[3],p[5],side,false});
    ys.push_back(p[3]);ys.push_back(p[5]);
   }
  }
  std::sort(ys.begin(),ys.end());ys.erase(std::unique(ys.begin(),ys.end()),ys.end());
  std::sort(events.begin(),events.end(),[](const Event&a,const Event&b){if(a.x!=b.x)return a.x<b.x;return a.start<b.start;});
  Fenwick lo[2]={Fenwick(ys.size()),Fenwick(ys.size())},hi[2]={Fenwick(ys.size()),Fenwick(ys.size())};
  for(const auto&e:events){
   auto y0=std::lower_bound(ys.begin(),ys.end(),e.y0)-ys.begin();
   auto y1=std::lower_bound(ys.begin(),ys.end(),e.y1)-ys.begin();
   if(e.start){
    auto end_le_y0=std::upper_bound(ys.begin(),ys.end(),e.y0)-ys.begin();
    auto count=lo[1-e.side].sum(y1)-hi[1-e.side].sum(end_le_y0);
    if(count<0)return 3;
    *output+=static_cast<uint64_t>(count);
   }
   lo[e.side].add(y0,e.start?1:-1);hi[e.side].add(y1,e.start?1:-1);
  }
  return 0;
 }catch(const std::exception&){return 1;}
}

