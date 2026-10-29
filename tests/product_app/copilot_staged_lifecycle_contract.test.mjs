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

async function harness(shape) {
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
    addDraft:async()=>{additions++;state.session.revision++;state.session.draft=structuredClone(operations);return true;},
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
  assert.equal(additions,1);assert.match(element("#copilot-lifecycle").textContent,/Draft staged/);
  return {state,controller,element,operations,conversation,additions:()=>additions,chatRequests:()=>chatRequests,
    lifecycle:()=>element("#copilot-lifecycle").textContent,
    send:async failure=>{failChat=failure;element("#copilot-input").value="Review the current trip";
      await element("#copilot-form").listeners.submit({preventDefault(){}});}};
}

for(const shape of ["single","batch"]){
  for(const outcome of ["undo_empty","undo_remaining","selection_only","restore_announcement",
    "new_chat_explanation","new_chat_failure","pending_undo","conversation_switch"]){
    test(`${shape}: staging lifecycle follows ${outcome} without replacing unrelated chat status`,async()=>{
      const h=await harness(shape),staged=h.lifecycle();
      if(outcome === "selection_only"){
        h.state.session.revision++;h.state.session.selected_day=5;h.state.session.selected_stop_id=null;h.controller.update();
        assert.equal(h.lifecycle(),staged);assert.deepEqual(h.state.session.draft,h.operations);
      }else if(outcome === "restore_announcement"){
        await h.controller.restore({announce:true});const announcement=h.lifecycle();assert.match(announcement,/Conversation restored/);
        h.state.session.revision++;h.state.session.draft=[];h.controller.update();assert.equal(h.lifecycle(),announcement);
      }else if(outcome.startsWith("new_chat_")){
        await h.send(outcome === "new_chat_failure");const message=h.lifecycle();
        assert.doesNotMatch(message,/Draft staged/);assert.match(message,outcome === "new_chat_failure" ? /failed|failure/i : /Current trip explanation/);
        h.state.session.revision++;h.state.session.draft=[];h.controller.update();assert.equal(h.lifecycle(),message);
        assert.equal(h.chatRequests(),1);
      }else if(outcome === "conversation_switch"){
        h.state.session.conversation_id="conversation_new";h.state.session.revision++;h.state.session.draft=[];h.controller.update();
        assert.doesNotMatch(h.lifecycle(),/Draft staged/);assert.match(h.lifecycle(),/Previous local conversation cleared/);
      }else{
        if(outcome === "pending_undo"){
          h.state.undoPending=true;h.controller.update();assert.equal(h.lifecycle(),staged);
          assert.equal(h.element("#copilot-submit").disabled,true);
        }
        h.state.session.revision++;
        h.state.session.draft=outcome === "undo_remaining" ? shape === "batch" ? [h.operations[0]] : [{type:"keep_stop",target:"other",parameters:{}}] : [];
        h.controller.update();assert.doesNotMatch(h.lifecycle(),/Draft staged/);
        assert.match(h.lifecycle(),outcome === "undo_remaining" ? /Draft changed.*Preview repair/ : /accepted itinerary.*No evaluated preview/);
        h.state.undoPending=false;h.controller.update();assert.equal(h.element("#copilot-submit").disabled,false);
      }
      assert.equal(h.additions(),1,"Status refresh must not add or retry a draft operation");
      assert.equal(h.state.session.accepted_plan_id,"accepted_parent");
    });
  }
}

for (const shape of ["single", "batch"]) {
  for (const order of ["clear_before_session", "session_before_clear"]) {
    test(`${shape}: recovery lifecycle follows canonical empty draft (${order})`, async () => {
      const h = await harness(shape);
      h.state.mutationRecovery = {blocked:true};
      h.controller.update();
      assert.equal(h.element("#copilot-lifecycle").dataset.state,"awaiting_confirmation");
      if (order === "clear_before_session") {
        h.state.mutationRecovery.blocked=false;
        h.controller.update();
      }
      h.state.session.revision++;
      h.state.session.draft=[];
      h.controller.update();
      h.state.mutationRecovery.blocked=false;
      h.controller.update();
      assert.equal(h.element("#copilot-lifecycle").dataset.state,"idle");
      assert.match(h.lifecycle(),/Showing the accepted itinerary.*No evaluated preview/);
      assert.equal(h.additions(),1);
      assert.equal(h.state.session.accepted_plan_id,"accepted_parent");
    });
  }
}
