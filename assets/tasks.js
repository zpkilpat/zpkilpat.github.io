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
document.documentElement.dataset.tasksReady='1';
})();
