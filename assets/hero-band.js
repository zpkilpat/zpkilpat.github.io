// Quiet looping figure for the research page: a school with social discounting (same caricature as the demo).
(function(){
'use strict';
const canvas=document.getElementById('hero-band');if(!canvas)return;
const ctx=canvas.getContext('2d'),TAU=Math.PI*2,reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const P={th:1.6,mu:1,sig:.8,lam:3,alpha:.6,kick:.35,N:44,win:4.2};
const normal=()=>Math.sqrt(-2*Math.log(Math.max(1e-12,Math.random())))*Math.cos(TAU*Math.random());
let W=0,H=0,dpr=1,fish=[],trace=[],clock=0,first=null,pause=0,H1=true,n=0,nf=0;
function size(){const r=canvas.getBoundingClientRect();dpr=Math.min(2,window.devicePixelRatio||1);W=Math.max(300,Math.round(r.width));H=Math.max(110,Math.round(r.height));canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);layout();draw()}
const arenaH=()=>H*.66;
function layout(){const cx=W*.42,cy=arenaH()*.56,rx=Math.min(W*.2,260),ry=arenaH()*.3;fish=Array.from({length:P.N},(_,i)=>{const r=Math.sqrt((i+.5)/P.N),a=i*2.39996;const bx=cx+r*rx*Math.cos(a),by=cy+r*ry*Math.sin(a);return{bx,by,x:bx,y:by,vx:0,vy:0,h:Math.random()*TAU,b:0,fled:false,fx:0,fy:0}});newTrial()}
function newTrial(){n++;H1=n%4!==0;clock=0;first=null;pause=0;nf=0;trace=Array.from({length:8},()=>[]);fish.forEach(f=>{f.b=0;f.fled=false})}
function step(dt){
 for(const f of fish){
  if(f.fled){f.x+=f.fx*dt;f.y+=f.fy*dt;f.h=Math.atan2(f.fy,f.fx);continue}
  f.x+=(f.bx-f.x)*Math.min(1,2.2*dt);f.y+=(f.by-f.y)*Math.min(1,2.2*dt);
  f.vx+=normal()*14*dt;f.vy+=normal()*14*dt;f.vx*=.95;f.vy*=.95;f.x+=f.vx*dt*3;f.y+=f.vy*dt*3;f.h=Math.atan2(f.vy,f.vx)||f.h}
 if(pause>0){pause-=dt;if(pause<=0)newTrial();return}
 clock+=dt;const frac=1-nf/P.N,present=H1&&clock>.5;let nw=0;
 fish.forEach(f=>{if(f.fled)return;f.b+=((present?P.mu:0)-P.alpha*P.lam*frac*f.b)*dt+P.sig*Math.sqrt(dt)*normal();if(f.b>=P.th){f.fled=true;nw++;f.at=clock;const cx=W*.42,cy=arenaH()*.56,ang=Math.atan2(f.y-cy,f.x-cx)+normal()*.45;f.fx=Math.cos(ang)*(W*.1+Math.random()*W*.06);f.fy=Math.sin(ang)*(arenaH()*.2+Math.random()*arenaH()*.1)}});
 if(nw){if(first===null)first=clock;nf+=nw;fish.forEach(f=>{if(!f.fled)f.b+=P.kick*nw})}
 trace.forEach((tr,i)=>{const f=fish[i*5];if(f&&(!f.fled||f.at>=clock-.02))tr.push([clock,f.b])});
 if(first!==null&&clock>first+2.4)pause=.5;else if(clock>P.win+1)pause=.4}
function draw(){
 const c=ctx,AH=arenaH();c.clearRect(0,0,W,H);c.fillStyle='#1f3a31';c.fillRect(0,0,W,H);
 c.strokeStyle='rgba(55,84,74,.7)';c.lineWidth=1;for(let x=W*.08;x<W;x+=W*.12){c.beginPath();c.moveTo(x,10);c.lineTo(x,AH);c.stroke()}
 if(H1&&clock>.3){const u=Math.min(1,(clock-.3)/4.5),bx=-W*.05+u*W*1.1,by=AH*.3,r=Math.max(40,AH*.55),g=c.createRadialGradient(bx,by,2,bx,by,r);g.addColorStop(0,'rgba(8,18,14,.55)');g.addColorStop(1,'rgba(8,18,14,0)');c.fillStyle=g;c.fillRect(bx-r,by-r,r*2,r*2)}
 const s=Math.max(.8,Math.min(1.3,W/900));
 for(const f of fish){c.save();c.translate(f.x,f.y);c.rotate(f.h);c.fillStyle=f.fled?'#d8bd82':'#8fb8a6';c.globalAlpha=f.fled?.95:.6+Math.min(.35,Math.max(0,f.b)*.25);c.beginPath();c.moveTo(10*s,0);c.lineTo(-7*s,-4.8*s);c.lineTo(-4.5*s,0);c.lineTo(-7*s,4.8*s);c.closePath();c.fill();c.restore()}
 c.globalAlpha=1;
 // belief strip
 const x0=W*.62,x1=W-24,y0=AH+16,y1=H-12,yb=v=>y1-(Math.max(-.4,Math.min(2,v))+.4)/2.4*(y1-y0),xt=t=>x0+t/P.win*(x1-x0);
 c.font='11px Arial';c.fillStyle='#c3cfc4';c.fillText('SOCIAL ESCAPE',24,H-14);
 c.strokeStyle='#d8bd82';c.setLineDash([4,5]);c.beginPath();c.moveTo(x0,yb(P.th));c.lineTo(x1,yb(P.th));c.stroke();c.setLineDash([]);
 c.strokeStyle='#37544a';c.beginPath();c.moveTo(x0,yb(0));c.lineTo(x1,yb(0));c.stroke();
 trace.forEach((tr,i)=>{if(tr.length<2)return;const fl=fish[i*5]&&fish[i*5].fled;c.strokeStyle=fl?'#d8bd82':'#8fb8a6';c.globalAlpha=.85;c.lineWidth=1.4;c.beginPath();tr.forEach(([t,v],k)=>k?c.lineTo(xt(t),yb(v)):c.moveTo(xt(t),yb(v)));c.stroke()});c.globalAlpha=1}
size();let ro;try{ro=new ResizeObserver(size);ro.observe(canvas)}catch(e){addEventListener('resize',size)}
if(reduced){for(let i=0;i<260&&(first===null||clock<first+.7);i++)step(1/60);draw();return}
let last=0,vis=true,acc=0;try{new IntersectionObserver(e=>{vis=e[0].isIntersecting}).observe(canvas)}catch(e){}
(function frame(t){const dt=Math.min((t-last)/1000||0,.05);last=t;if(vis&&!document.hidden){acc+=dt;while(acc>=1/60){step(1/60);acc-=1/60}draw()}requestAnimationFrame(frame)})(0);
})();
