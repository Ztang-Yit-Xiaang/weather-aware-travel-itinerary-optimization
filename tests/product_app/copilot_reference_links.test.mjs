import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
const source=fs.readFileSync(new URL('../../src/itinerary_system/product_app/static/js/copilot.js',import.meta.url),'utf8');
const start=source.indexOf('  function appendMessageText('),end=source.indexOf('  function appendMessage(',start);
function render(text) {
  assert.ok(start>=0 && end>start,'Message reference renderer missing');
  const paragraph={textContent:'',children:[],append(...items){this.children.push(...items);}};
  const document={createElement:tag=>({tag,textContent:'',href:'',target:'',rel:''})};
  new Function('document',source.slice(start,end)+'\nreturn appendMessageText;')(document)(paragraph,text);
  return paragraph;
}
test('official reference is a usable external anchor with surrounding text',()=>{
  const p=render('See [official FAQ](https://hearstcastle.org/plan/faqs/) before booking.');
  const link=p.children.find(child=>child?.tag==='a');
  assert.equal(link.href,'https://hearstcastle.org/plan/faqs/');
  assert.equal(link.textContent,'official FAQ');
  assert.equal(link.target,'_blank');
  assert.equal(link.rel,'noopener noreferrer');
  assert.equal(p.children[0],'See ');
  assert.equal(p.children.at(-1),' before booking.');
});
for(const text of ['<img src=x onerror=alert(1)>','[click](javascript:alert(1))',
  '[click](data:text/html,boom)','[click](https://user:password@example.com/)',
  '[click](file:///C:/private)','https://example.com/plain']) test(`unsafe or unformatted input stays text: ${text}`,()=>{
  const p=render(text);
  assert.equal(p.textContent,text);
  assert.equal(p.children.length,0);
});
test('HTML in a link label remains literal text',()=>{
  const p=render('[<svg onload=alert(1)>](https://hearstcastle.org/)');
  assert.equal(p.children.find(child=>child?.tag==='a').textContent,'<svg onload=alert(1)>');
  assert.equal(p.children.filter(child=>typeof child==='object').length,1);
});
