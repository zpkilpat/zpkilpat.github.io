(function(){
const {normal,circularError,patchProbability}=Models;
const $=id=>document.getElementById(id),val=id=>Number($(id).value);
// Timed delayed-estimation trials. Target hue is never exposed in response controls.
let target=0,phase='idle',timer=null,trials=0,totalError=0;
function swatch(hue){$('memory-display').style.background=`hsl(${hue},65%,50%)`;}
$('memory-start').addEventListener('click',()=>{if(phase==='cue'||phase==='delay'||phase==='response')return;clearTimeout(timer);target=Math.random()*360;phase='cue';$('delay').disabled=true;$('memory-start').disabled=true;$('memory-response').hidden=true;$('memory-display').textContent='';swatch(target);$('memory-status').textContent='Remember this color.';const delay=val('delay');timer=setTimeout(()=>{phase='delay';$('memory-display').style.background='#eeebe2';$('memory-display').textContent='+';$('memory-status').textContent=`Hold the color in memory for ${delay} seconds.`;timer=setTimeout(()=>{phase='response';$('recall').value=Math.floor(Math.random()*360);swatch(val('recall'));$('memory-display').textContent='';$('memory-response').hidden=false;$('memory-status').textContent='Adjust the slider to recreate the color, then submit.';$('recall').focus()},delay*1000)},1000)});
$('recall').addEventListener('input',()=>{if(phase==='response')swatch(val('recall'))});
$('memory-submit').addEventListener('click',()=>{if(phase!=='response')return;const error=circularError(val('recall'),target);trials++;totalError+=error;phase='feedback';$('memory-response').hidden=true;swatch(target);$('memory-status').textContent=`Error: ${error.toFixed(1)}°. The original color is shown again.`;$('memory-score').textContent=`${trials} ${trials===1?'trial':'trials'} · mean error ${(totalError/trials).toFixed(1)}°`;$('memory-start').disabled=false;$('memory-start').textContent='Next trial';$('delay').disabled=false});
// Cancel timed trials if the page is hidden, avoiding invalid memory delays.
document.addEventListener('visibilitychange',()=>{if(document.hidden&&(phase==='cue'||phase==='delay')){clearTimeout(timer);phase='idle';$('memory-display').style.background='#eeebe2';$('memory-display').textContent='Ready?';$('memory-status').textContent='Trial cancelled while the page was hidden. Start again when ready.';$('memory-start').disabled=false;$('delay').disabled=false}});
let source=Math.random()<.5?-1:1,observation=null,awaiting=false,correct=0,guesses=0,last=[];
const tc=$('triangles-canvas'),ctx=tc.getContext('2d');
function drawTriangles(reveal=false){ctx.clearRect(0,0,800,330);ctx.fillStyle='#f2efe6';ctx.fillRect(0,0,800,330);ctx.strokeStyle='#bfbfb4';ctx.beginPath();ctx.moveTo(40,175);ctx.lineTo(760,175);ctx.stroke();[-1,1].forEach(side=>{const x=side<0?240:560;ctx.fillStyle=reveal&&source===side?'#8a6b21':'#1b1c1c';ctx.beginPath();ctx.moveTo(x,64);ctx.lineTo(x-22,105);ctx.lineTo(x+22,105);ctx.closePath();ctx.fill();ctx.font='16px Arial';ctx.fillText(side<0?'LEFT':'RIGHT',x-21,135)});last.forEach((x,i)=>{ctx.globalAlpha=.1+.25*(i/Math.max(last.length,1));ctx.fillStyle='#40604f';ctx.beginPath();ctx.arc(Math.max(18,Math.min(782,x)),220,4,0,Math.PI*2);ctx.fill()});ctx.globalAlpha=1;if(observation!==null){const x=Math.max(18,Math.min(782,observation));ctx.fillStyle='#8a6b21';ctx.beginPath();ctx.arc(x,175,9,0,Math.PI*2);ctx.fill();if(observation<18||observation>782){ctx.font='24px Arial';ctx.fillText(observation<18?'←':'→',observation<18?8:760,160)}}ctx.font='13px Arial';ctx.fillStyle='#5b5f5e';ctx.fillText('Newest observation',315,275);if(last.length)ctx.fillText('Previous observations (faded)',310,310)}
function triangleControls(){ $('guess-left').disabled=!awaiting;$('guess-right').disabled=!awaiting;$('triangles-start').disabled=awaiting;}
$('triangles-start').addEventListener('click',()=>{if(awaiting)return;if(Math.random()<val('hazard'))source*=-1;if(observation!==null)last.push(observation);if(last.length>12)last.shift();observation=(source<0?240:560)+normal()*val('observation-noise');awaiting=true;$('hazard').disabled=true;$('observation-noise').disabled=true;drawTriangles();triangleControls();$('triangles-evidence').textContent=`Latest dot: ${((observation/800)*100).toFixed(1)}% of display width. Sources are at 30% (left) and 70% (right).`;$('triangles-status').textContent='Which source most likely generated this dot?';});
for(const side of [-1,1])$(side<0?'guess-left':'guess-right').addEventListener('click',()=>{if(!awaiting)return;awaiting=false;guesses++;const hit=side===source;if(hit)correct++;drawTriangles(true);triangleControls();$('triangles-status').textContent=`${hit?'Correct.':'Not this time.'} The source was ${source<0?'left':'right'}. Draw the next observation.`;$('triangles-score').textContent=`${correct} / ${guesses} correct (${Math.round(correct/guesses*100)}%)`;$('triangles-start').textContent='Next observation';});
$('triangles-reset').addEventListener('click',()=>{source=Math.random()<.5?-1:1;observation=null;awaiting=false;correct=guesses=0;last=[];$('hazard').disabled=false;$('observation-noise').disabled=false;drawTriangles();triangleControls();$('triangles-evidence').textContent='Draw an observation to begin.';$('triangles-status').textContent='Draw, decide, then see feedback.';$('triangles-score').textContent='0 / 0 correct';$('triangles-start').textContent='Draw observation'});drawTriangles();
let harvests=0,food=0,elapsed=0,patches=1,patchFood=0,hits=[],history=[];
const fc=$('foraging-canvas'),fx=fc.getContext('2d');
const FW=800,FH=345,FL=54,FR=18,FT=20,FB=110;
// Plots only what the forager can observe: the running intake rate inside this patch
// against the running average for the whole session. Their crossing is the marginal
// value theorem. The true success probability is deliberately never drawn.
function drawForaging(){
 const pw=FW-FL-FR,ph=FH-FT-FB;
 fx.clearRect(0,0,FW,FH);fx.fillStyle='#f6f4ee';fx.fillRect(0,0,FW,FH);
 const span=Math.max(hits.length,10),px=i=>FL+pw*(i/span),py=v=>FT+ph*(1-Math.min(v,1));
 fx.strokeStyle='#cfccc0';fx.lineWidth=1;
 fx.beginPath();fx.moveTo(FL,FT);fx.lineTo(FL,FT+ph);fx.lineTo(FL+pw,FT+ph);fx.stroke();
 fx.font='12px Arial';fx.fillStyle='#5b5f5e';fx.textAlign='right';
 for(const v of [0,.5,1]){fx.fillText(v.toFixed(1),FL-8,py(v)+4)}
 fx.textAlign='left';
 fx.save();fx.translate(14,FT+ph/2);fx.rotate(-Math.PI/2);fx.textAlign='center';
 fx.fillText('food / second',0,0);fx.restore();
 // session average: the habitat rate the theorem compares against
 const lr=elapsed?food/elapsed:0;
 fx.setLineDash([5,4]);fx.strokeStyle='#8a6b21';fx.lineWidth=1.5;
 fx.beginPath();fx.moveTo(FL,py(lr));fx.lineTo(FL+pw,py(lr));fx.stroke();fx.setLineDash([]);
 fx.fillStyle='#8a6b21';fx.textAlign='left';
 fx.fillText(`session average ${lr.toFixed(2)}`,FL+6,py(lr)-7);
 // observed intake rate inside this patch
 if(hits.length){
  fx.strokeStyle='#40604f';fx.lineWidth=2;fx.beginPath();
  let run=0;
  hits.forEach((h,k)=>{run+=h;const X=px(k+1),Y=py(run/(k+1));k?fx.lineTo(X,Y):fx.moveTo(X,Y)});
  fx.stroke();
  run=0;
  hits.forEach((h,k)=>{run+=h;fx.fillStyle='#40604f';fx.beginPath();
   fx.arc(px(k+1),py(run/(k+1)),3.5,0,Math.PI*2);fx.fill()});
  // per-harvest outcome ticks
  hits.forEach((h,k)=>{fx.fillStyle=h?'#40604f':'#c9c5b6';
   fx.fillRect(px(k+1)-3,FT+ph+8,6,h?12:5)});
 } else {
  fx.fillStyle='#5b5f5e';fx.textAlign='center';
  fx.fillText('Harvest to start plotting this patch.',FL+pw/2,FT+ph/2);fx.textAlign='left';
 }
 // legend, with the marks it describes
 const ly=FT+ph+40;
 fx.fillStyle='#40604f';fx.fillRect(FL,ly-9,6,12);
 fx.fillStyle='#5b5f5e';fx.fillText('found food',FL+12,ly);
 fx.fillStyle='#c9c5b6';fx.fillRect(FL+92,ly-2,6,5);
 fx.fillStyle='#5b5f5e';fx.fillText('nothing',FL+104,ly);
 fx.textAlign='right';fx.fillText(`harvests in patch ${patches}`,FL+pw,ly);fx.textAlign='left';
 // yields from patches already left
 if(history.length){
  const hy=FH-16;
  fx.fillStyle='#5b5f5e';fx.fillText('earlier patches',FL,hy-14);
  history.slice(-11).forEach((h,k)=>{const X=FL+k*62;
   fx.fillStyle='#e0ddd0';fx.fillRect(X,hy-8,50,9);
   fx.fillStyle='#40604f';fx.fillRect(X,hy-8,50*(h.harvests?h.food/h.harvests:0),9);
   fx.fillStyle='#5b5f5e';fx.fillText(`${h.food}/${h.harvests}`,X,hy+14)});
 }
}
function renderPatch(){$('food').textContent=food;$('elapsed').textContent=elapsed;
 $('rate').textContent=elapsed?(food/elapsed).toFixed(2):'0.00';drawForaging()}
function lockPatch(){$('travel').disabled=true;$('depletion').disabled=true}
$('harvest').addEventListener('click',()=>{lockPatch();
 const success=Math.random()<patchProbability(harvests,val('depletion'));
 if(success){food++;patchFood++}hits.push(success?1:0);harvests++;elapsed++;renderPatch();
 $('foraging-status').textContent=`${success?'Found one food item.':'No food this harvest.'} Patch ${patches} · ${harvests} harvests. Stay or move on?`});
$('leave').addEventListener('click',()=>{lockPatch();history.push({harvests,food:patchFood});
 elapsed+=val('travel');harvests=0;patchFood=0;hits=[];patches++;renderPatch();
 $('foraging-status').textContent=`Traveled for ${val('travel')} seconds. Patch ${patches} is fresh.`});
$('foraging-reset').addEventListener('click',()=>{harvests=food=elapsed=patchFood=0;patches=1;
 hits=[];history=[];$('travel').disabled=false;$('depletion').disabled=false;renderPatch();
 $('foraging-status').textContent='New session. Adjust the settings before your first action.'});
renderPatch();
document.documentElement.dataset.tasksReady='1';
})();
