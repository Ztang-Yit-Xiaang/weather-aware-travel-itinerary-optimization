import assert from "node:assert/strict";
import path from "node:path";
import {fileURLToPath,pathToFileURL} from "node:url";
import test from "node:test";

const out=path.dirname(fileURLToPath(import.meta.url));
const modulePath=process.argv[2] || path.resolve(out,"../../src/itinerary_system/product_app/static/js/copilot.js");
const {createCopilotController}=await import(pathToFileURL(modulePath));

class Element {
  constructor(tag="div") {
    this.tagName=tag;this.children=[];this.listeners={};this.dataset={};this.style={};this.attributes={};
    this.value="";this.textContent="";this.hidden=false;this.scrollHeight=100;this.scrollTop=0;this.clientHeight=100;
    this.classList={add(){},remove(){}};
  }
  append(...children){this.children.push(...children);}
  replaceChildren(...children){this.children=children;}
  setAttribute(key,value){this.attributes[key]=value;}
  addEventListener(name,action){this.listeners[name]=action;}
  focus(){}
}
const descendants=element=>[element,...element.children.flatMap(descendants)];

async function harness(shape, uncertain=false) {
  const elements=new Map(),element=selector=>{if(!elements.has(selector))elements.set(selector,new Element());return elements.get(selector);};
  globalThis.document={querySelector:element,querySelectorAll:selector=>selector === ".copilot-proposal-actions button"
    ? descendants(element("#conversation")).filter(row=>row.tagName === "button") : [],createElement:tag=>new Element(tag)};
  globalThis.window={addEventListener(){}};
  const operations=shape === "single" ? [{type:"set_stop_role",target:"wharf",parameters:{role:"scenic_stop"}}] : [
    {type:"set_stop_order",target:"kings",parameters:{day:4,sequence_index:0}},
    {type:"set_stop_order",target:"sequoia",parameters:{day:4,sequence_index:1}},
  ];
  const state={session:{session_id:"session_fixture",trip_id:"trip_fixture",run_id:"run_fixture",
    conversation_id:"conversation_fixture",accepted_plan_id:"accepted_parent",revision:0,selected_day:6,selected_stop_id:"wharf",draft:[]},
    registry:{label:"Synthetic trip"},workspace:{interaction:{enabled:true,provider:"deterministic"},timeline:[],
      draft_capabilities:{parent_targets:["wharf","kings","sequoia"],operations:{},executable_combinations:[{operations:["set_stop_order"],cardinality:2}]},
      typed_edit_capabilities:{operations:{set_stop_role:{enabled:true,preview_executable:true},set_stop_order:{enabled:true,preview_executable:true}}}}};
  const conversation={schema_version:"product-conversation-v1",session_id:"session_fixture",trip_id:"trip_fixture",run_id:"run_fixture",
    conversation_id:"conversation_fixture",revision:1,turns:[{client_message_id:"initial",context_revision:0,state:"completed",
      user_message:"Prepare the edit.",assistant_message:"Review the proposed edit.",interpretation:{state:"proposal_ready",
        intents:operations.map(row=>({type:row.type,target_stop_id:row.target,parameters:row.parameters}))}}]};
  let additions=0,chatRequests=0,failChat=false,controller;
  controller=createCopilotController({getState:()=>state,
    api:async()=>({session:structuredClone(state.session),conversation:structuredClone(conversation)}),
    addDraft:async()=>{additions++;if(uncertain){state.mutationRecovery={blocked:true};controller.update();return false;}state.session.revision++;state.session.draft=structuredClone(operations);return true;},
    streamMessage:async(_url,{body})=>{
      chatRequests++;if(failChat)throw new Error("Controlled message failure");
      const turn={client_message_id:body.client_message_id,context_revision:body.expected_revision,state:"completed",
        user_message:body.message,assistant_message:"Current trip explanation.",interpretation:{state:"explanation_ready",intents:[]}};
      conversation.revision++;conversation.turns.push(turn);return {session:structuredClone(state.session),turn};
    },getPreviewState:()=>({status:"PARENT_RENDERED"}),getSelectedStop:()=>null,
    toast(){},renderAll:()=>controller.update(),navigate(){}});
  controller.bind();await controller.restore({announce:false});
  const add=descendants(element("#conversation")).find(row=>row.textContent === (shape === "single" ? "Add to draft" : "Add all to draft"));
  assert.ok(add && !add.disabled);await add.listeners.click({currentTarget:add});
  assert.equal(additions,1);if(!uncertain)assert.match(element("#copilot-lifecycle").textContent,/Draft staged/);
  return {state,controller,element,operations,conversation,additions:()=>additions,chatRequests:()=>chatRequests,
    lifecycle:()=>element("#copilot-lifecycle").textContent,
    send:async failure=>{failChat=failure;element("#copilot-input").value="Review the current trip";
      await element("#copilot-form").listeners.submit({preventDefault(){}});}};
}


for(const shape of ["single","batch"]){
  for(const scenario of ["lost_add_reply","pending_undo"]){
    test(`${shape}: ${scenario} remains accurately blocked across card rerender and restore`,async()=>{
      const h=await harness(shape,scenario === "lost_add_reply");
      h.element("#copilot-input").value="Preserve this unsent message";
      h.state.mutationRecovery={blocked:true};h.controller.update();
      const blocked=()=>{
        assert.match(h.lifecycle(),/Awaiting confirmation/);
        assert.doesNotMatch(h.lifecycle(),/not staged|Failed:/);
        const controls=descendants(h.element("#conversation")).filter(e=>e.tagName === "button");
        assert.ok(controls.length>0);assert.ok(controls.every(e=>e.disabled));
        assert.equal(h.element("#copilot-submit").disabled,true);
        assert.equal(h.element("#copilot-input").value,"Preserve this unsent message");
        assert.equal(h.additions(),1);
      };
      blocked();h.controller.previewStateChanged();blocked();
      await h.controller.restore({announce:false});h.controller.update();blocked();
      h.state.mutationRecovery={blocked:false};
      h.state.session.draft=structuredClone(h.operations);h.state.session.revision++;
      await h.controller.restore({announce:false});h.controller.update();
      assert.doesNotMatch(h.lifecycle(),/Awaiting confirmation/);
      assert.equal(h.element("#copilot-submit").disabled,false);
      assert.ok(descendants(h.element("#conversation")).some(e=>e.tagName === "button" && !e.disabled));
      assert.equal(h.additions(),1);assert.equal(h.state.session.accepted_plan_id,"accepted_parent");
    });
  }
}
