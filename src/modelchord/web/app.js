'use strict';
const $ = (id) => document.getElementById(id);
const state = {catalog:null, current:null, inputs:null, busy:false, sampleDomain:null, pixels:[]};
const percent = (value) => Number.isFinite(value) ? `${(value*100).toFixed(1)}%` : '—';
const number = (value) => Number.isFinite(value) ? value.toLocaleString('en-US') : '—';
function node(tag, cls, text) { const el=document.createElement(tag); if(cls)el.className=cls; if(text!==undefined)el.textContent=text; return el; }
function title(value){return String(value).replace(/[_-]+/g,' ').replace(/\b\w/g,c=>c.toUpperCase());}
async function api(path,body){
  const options=body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)};
  const res=await fetch(path,options);
  const type=res.headers.get('content-type')||'';
  const data=type.includes('json')?await res.json():{error:await res.text()};
  if(!res.ok)throw new Error(data.error||data.detail||`Request failed (${res.status}).`);
  return data;
}
function message(id,text,kind=''){const el=$(id);el.textContent=text;el.classList.toggle('error',kind==='error');}
function option(select,value,label){const opt=document.createElement('option');opt.value=value;opt.textContent=label;select.append(opt);return opt;}
function setBusy(busy){state.busy=busy;$('build-button').disabled=busy||!state.catalog;$('build-button').textContent=busy?'Building and evaluating…':'Build static model ↗';$('build-form').setAttribute('aria-busy',String(busy));}
function floorInput(cap){
  const wrapper=node('span','floor-control');const input=document.createElement('input');input.type='number';input.min='0';input.max='100';input.step='0.5';input.value=(cap.floor*100).toFixed(1);input.dataset.capability=cap.name;input.setAttribute('aria-label',`${title(cap.name)} minimum accuracy in percent`);input.setAttribute('form','build-form');input.required=true;wrapper.append(input,node('span','','%'));return wrapper;
}
function renderCapabilities(caps){
  const rows=$('capability-rows');rows.replaceChildren();
  caps.forEach(cap=>{const tr=document.createElement('tr');tr.dataset.capability=cap.name;const name=node('td');name.append(node('span','cap-name',title(cap.name)));if(cap.protected)name.append(node('span','protected-tag','PROTECTED'));const floor=node('td');floor.append(floorInput(cap));const ordinary=node('td','ordinary-score metric-dash','—');const final=node('td','final-score metric-dash','—');const verdict=node('td');verdict.append(node('span','floor-verdict','Not evaluated'));tr.append(name,floor,ordinary,final,verdict);rows.append(tr);});
}
function renderCatalog(data){
  state.catalog=data;const base=data.base||{};const arch=base.architecture||{};const baseInfo=$('base-info');baseInfo.replaceChildren(node('span','small-label','COMMON BASE'),node('strong','',`${arch.kind||'Model'} · ${arch.input_dim||'?'} inputs · ${(arch.classes||[]).length} classes`));
  $('contract').textContent=JSON.stringify(base.contract||{},null,2);$('experts').replaceChildren();
  (data.experts||[]).forEach(expert=>{const label=node('label',`expert-option${expert.compatible===false?' incompatible':''}`);const input=document.createElement('input');input.type='checkbox';input.name='experts';input.value=expert.id;input.checked=expert.compatible!==false;input.disabled=expert.compatible===false;const words=node('span');words.append(node('strong','',expert.label||title(expert.id)),node('small','',expert.compatible===false?`Incompatible: ${expert.error||'contract mismatch'}`:`${title(expert.task||expert.id)} specialist · compatible`));label.append(input,words);$('experts').append(label);});
  const caps=data.capabilities||[];renderCapabilities(caps);
  const models=$('inference-model');models.replaceChildren();option(models,base.id||'base','Common base');(data.experts||[]).filter(e=>e.compatible!==false).forEach(e=>option(models,e.id,e.label||title(e.id)));(data.results||[]).forEach(r=>option(models,r.id,r.label||title(r.id)));models.disabled=false;
  const domains=$('sample-domain');domains.replaceChildren();caps.forEach(c=>option(domains,c.name,title(c.name)));domains.disabled=!caps.length;$('load-sample').disabled=!caps.length;
  const count=(data.experts||[]).filter(e=>e.compatible!==false).length;
  const s=data.study||{};const evaluation=s.evaluation_split||s.split||'selection data';
  message('catalog-status',`${count} compatible specialists available. Build checks use ${evaluation}. Parent experts are not needed by an exported model.`);$('catalog-status').classList.add('success');
  message('build-status','Choose specialists and floors, then build.');$('build-button').disabled=count===0;
}
function renderMetrics(result,requestedFloors){
  const metrics=result.metrics||{};const ordinary=result.ordinary_metrics||metrics;const report=result.report||{};let met=0,total=0,worst=0;
  $('capability-rows').querySelectorAll('tr[data-capability]').forEach(tr=>{const name=tr.dataset.capability;const score=metrics[name]?.accuracy;const before=ordinary[name]?.accuracy;const floor=requestedFloors[name];const effective=report.effective_selection_floors?.[name]??floor;const valid=Number.isFinite(score);const passed=valid&&score+1e-9>=effective;const deficit=valid?Math.max(0,effective-score):null;if(valid){total++;if(passed)met++;worst=Math.max(worst,deficit);}
    tr.querySelector('.ordinary-score').textContent=percent(before);tr.querySelector('.ordinary-score').classList.toggle('metric-dash',!Number.isFinite(before));const final=tr.querySelector('.final-score');final.textContent=percent(score);final.className=`final-score metric-final${valid&&!passed?' short':''}`;const verdict=tr.querySelector('.floor-verdict');verdict.className=`floor-verdict ${valid?(passed?'met':'short'):''}`;verdict.textContent=valid?(passed?'✓ Met':`−${(deficit*100).toFixed(1)} pt`):'Unavailable';if(effective>floor+1e-9)verdict.title=`Protected-task gate requires ${percent(effective)}.`;else verdict.removeAttribute('title');
  });
  const acceptable=total>0&&met===total;const tag=$('result-state');tag.textContent=acceptable?'All measured floors met':'Floors remain unmet';tag.className=`state-tag ${acceptable?'met':'deficit'}`;
  const summary=$('floor-summary');summary.className=`floor-summary ${acceptable?'met':'deficit'}`;summary.querySelector('p').textContent=acceptable?`${met}/${total} measured floors met. ${report.attempts===0?'The ordinary merge already meets this contract.':'Review the recorded recipe before exporting.'}`:`${met}/${total} measured floors met. Worst shortfall: ${(worst*100).toFixed(1)} percentage points. ${report.status==='no acceptable candidate found'?'No acceptable candidate found within this budget.':'This artifact does not satisfy the requested build contract.'}`;
}
function renderArtifact(result){
  const report=result.report||{};
  const hasRepair=Boolean(report.algorithm)||Number(report.attempts)>0;
  const hasCorrection=Number(report.rank_used_upper_bound)>0;
  const artifactLabel=result.label||(hasCorrection?'Merged model + bounded correction':hasRepair?'Merged model · no correction accepted':'Ordinary merged model');
  state.current=result;$('artifact-empty').hidden=true;$('artifact-result').hidden=false;$('artifact-name').textContent=artifactLabel;$('artifact-id').textContent=result.id;
  $('download').href=result.download_url||'#';$('download').hidden=!result.download_url;
  $('stored-bytes').textContent=number(result.model?.bytes);$('parameter-count').textContent=number(result.model?.parameters);$('rank-used').textContent=number(result.report?.rank_used_upper_bound??0);$('candidate-count').textContent=number(result.report?.attempts??0);$('recipe').textContent=JSON.stringify(result.recipe||result.report||{},null,2);
  const sel=$('inference-model');if(!Array.from(sel.options).some(o=>o.value===result.id))option(sel,result.id,'Current build');sel.value=result.id;
}
$('build-form').addEventListener('submit',async(event)=>{event.preventDefault();if(state.busy)return;const experts=Array.from(document.querySelectorAll('input[name=experts]:checked')).map(el=>el.value);if(!experts.length){message('build-status','Select at least one compatible specialist.','error');return;}
  const floors={};document.querySelectorAll('input[data-capability]').forEach(input=>floors[input.dataset.capability]=Number(input.value)/100);setBusy(true);message('build-status','Building static weights and evaluating every capability. This may take a moment.');
  try{const result=await api('/api/build',{experts,method:$('method').value,scale:Number($('scale').value),repair:$('repair').checked,budget:{rank:Number($('budget-rank').value),rounds:Number($('budget-rounds').value)},floors});renderMetrics(result,floors);renderArtifact(result);message('build-status',result.report?.status==='no acceptable candidate found'?'Build finished. No acceptable candidate found within the repair budget.':'Build complete. Measured results and export are ready.');}
  catch(err){message('build-status',`Build failed: ${err.message}`,'error');}
  finally{setBusy(false);}
});
$('scale').addEventListener('input',()=>{$('scale-value').textContent=Number($('scale').value).toFixed(2);});
const methodNotes={task_arithmetic:'Scales the summed specialist updates from the common base.',average:'Scales the mean specialist update from the common base.',ties:'Trims small updates, elects signs, then averages aligned values.',dare:'Drops and rescales update coordinates using a fixed recorded seed.',tsv:'Merges compressed task matrices using decorrelated singular vectors.',regmean:'Uses permitted calibration activations to merge linear layers. The scale control is unused.'};
$('method').addEventListener('change',()=>{$('method-note').textContent=methodNotes[$('method').value];const disabled=$('method').value==='regmean';$('scale').disabled=disabled;$('scale-value').textContent=disabled?'N/A':Number($('scale').value).toFixed(2);});
function paintPixels(){state.pixels.forEach((button,i)=>{const value=state.inputs?state.inputs[0][i]:0;const shade=Math.round(24+value*214);button.style.backgroundColor=`rgb(${Math.round(shade*.93)}, ${shade}, ${Math.round(shade*.91)})`;button.setAttribute('aria-label',`Row ${Math.floor(i/8)+1}, column ${i%8+1}, intensity ${Math.round(value*100)} percent`);});}
for(let i=0;i<64;i++){const button=node('button','pixel');button.type='button';button.tabIndex=i===0?0:-1;button.disabled=true;button.addEventListener('click',()=>{if(!state.inputs)return;state.inputs[0][i]=(Math.round(state.inputs[0][i]*4)+1)%5/4;paintPixels();$('prediction-output').hidden=true;message('inference-status','Input changed. Run inference to measure this version.');});button.addEventListener('keydown',event=>{const deltas={ArrowLeft:-1,ArrowRight:1,ArrowUp:-8,ArrowDown:8};if(!(event.key in deltas))return;event.preventDefault();const next=Math.max(0,Math.min(63,i+deltas[event.key]));state.pixels.forEach((b,j)=>b.tabIndex=j===next?0:-1);state.pixels[next].focus();});state.pixels.push(button);$('pixels').append(button);}paintPixels();
$('load-sample').addEventListener('click',async()=>{$('load-sample').disabled=true;message('inference-status','Loading a permitted sample…');try{const sample=await api(`/api/sample?domain=${encodeURIComponent($('sample-domain').value)}`);if(!sample.inputs||!Array.isArray(sample.inputs[0])||sample.inputs[0].length!==64)throw new Error('This workbench requires an 8 × 8 digit input.');state.inputs=sample.inputs;state.sampleDomain=sample.domain;state.pixels.forEach(button=>button.disabled=false);paintPixels();$('sample-label').textContent=`${title(sample.domain)} sample · labeled ${sample.label}`;$('predict').disabled=false;$('prediction-output').hidden=true;message('inference-status','Sample loaded. Change pixels or run inference.');}catch(err){message('inference-status',`Sample failed: ${err.message}`,'error');}finally{$('load-sample').disabled=false;}});
$('inference-model').addEventListener('change',()=>{$('prediction-output').hidden=true;if(state.inputs)message('inference-status','Artifact changed. Run inference to measure this model.');});
$('predict').addEventListener('click',async()=>{if(!state.inputs)return;$('predict').disabled=true;message('inference-status','Running local inference…');try{const result=await api('/api/predict',{artifact:$('inference-model').value,inputs:state.inputs});$('predicted-label').textContent=result.labels[0];$('probabilities').replaceChildren();const ranked=result.probabilities[0].map((p,i)=>({p,label:result.classes[i]})).sort((a,b)=>b.p-a.p).slice(0,3);ranked.forEach(({p,label})=>{const row=node('div','probability-row');const track=node('div','probability-track');const fill=node('div','probability-fill');fill.style.width=`${Math.max(0,Math.min(100,p*100))}%`;track.append(fill);row.append(node('span','',label),track,node('span','',percent(p)));$('probabilities').append(row);});$('prediction-output').hidden=false;message('inference-status','Inference complete. Scores reflect the current pixel values.');}catch(err){message('inference-status',`Inference failed: ${err.message}`,'error');}finally{$('predict').disabled=false;}});
api('/api/catalog').then(renderCatalog).catch(err=>{message('catalog-status',`Could not load the catalog: ${err.message} Start the local workbench with a valid artifact directory, then reload.`,'error');message('build-status','Catalog unavailable. See the message above.','error');});

$('repair').addEventListener('change',()=>{$('repair-budget').disabled=!$('repair').checked;});
