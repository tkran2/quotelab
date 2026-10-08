// Structural interaction checks for the generated dashboard, without third-party JS deps.
// This DOM harness exercises plotting/data bindings; it does not replace screenshot QA.
import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const input = process.argv[2] || 'reports/QuoteLab_Dashboard.html';
const baseUrl = process.argv[3];
const html = fs.readFileSync(input, 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
class Element {
  constructor(id) {
    this.id=id; this.innerHTML=''; this.textContent=''; this.value='';
    this.clientWidth=560; this.style={}; this.dataset={}; this.events={}; this.children={};
    this.classes=new Set(); this.disabled=false;
    this.classList={add:x=>this.classes.add(x),remove:x=>this.classes.delete(x),
      toggle:(x,on)=>{if(on)this.classes.add(x);else this.classes.delete(x)}};
  }
  addEventListener(name,fn){this.events[name]=fn}
  querySelector(key){return this.children[key]||(this.children[key]=new Element(key))}
  getBoundingClientRect(){return {left:0,top:0,width:560,height:260}}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id)};
for(const match of html.matchAll(/<input[^>]*id="([^"]+)"[^>]*value="([^"]+)"/g))get('#'+match[1]).value=match[2];
get('#policy').value='adaptive';
const tabs=['path','frontier','evidence','toxicity','refresh'].map(tab=>{const e=new Element(tab);e.dataset.tab=tab;return e});
const views=tabs.map(tab=>get('#'+tab.dataset.tab));for(const view of views)view.id=view.id.slice(1);
const context={document:{querySelector:get,querySelectorAll:s=>s==='.tabs button'?tabs:s==='.view'?views:[]},
  window:{addEventListener(){}},setTimeout,clearTimeout,
  fetch:(url,opts)=>fetch(new URL(url,baseUrl),opts)};
vm.runInNewContext(script,context,{timeout:10000});
let checks=0;
for(const scenario of ['benign','toxic','volatility_burst','jump_shock','thin_market','high_cost']){
  get('#scenario').value=scenario;
  for(const policy of ['fixed','inventory','volatility','adaptive']){
    get('#policy').value=policy;
    for(const tab of tabs){
      tab.events.click();
      for(const [id,el] of elements){
        if(el.innerHTML.includes('<svg'))assert(!/NaN|Infinity|undefined/.test(el.innerHTML),'invalid chart in '+id);
      }
      assert(get('#audit').textContent.includes('12,580'));
      checks++;
    }
  }
}
assert(get('#spread-chart').innerHTML.includes('<polyline'));
assert(get('#evidence-table').innerHTML.includes('<table>'));
assert(get('#paired-table').innerHTML.includes('<table>'));
if(baseUrl){
  get('#scenario').value='toxic';get('#seed').value='123';get('#spread').value='.06';
  await get('#run').events.click();tabs[0].events.click();
  assert(get('#path-note').textContent.includes('Seed 123.'),'live API result not rendered');
  assert.equal(get('#run').disabled,false);
  checks++;
}
console.log(`Dashboard interaction checks passed: ${checks} view/settings combinations${baseUrl?' including live API':''}.`);
