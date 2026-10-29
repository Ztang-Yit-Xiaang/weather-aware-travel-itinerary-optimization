"""Passive exact-control input receipts shared by real draft browser journeys."""
from __future__ import annotations

OBSERVE_CONTROL_ACTIONS = """(() => {
  window.__draftControlInputs=[];
  const forms=new WeakMap();let nextForm=0;
  window.__draftControlFormId=form=>{if(!form)return null;if(!forms.has(form))forms.set(form,++nextForm);return forms.get(form);};
  for(const type of ['click','change','submit']) document.addEventListener(type,event=>{
    const target=event.target,form=target.closest?.('.typed-operation-form');
    const button=target.closest?.('button');
    if(!form&&!button) return;
    window.__draftControlInputs.push({type,trusted:event.isTrusted,at:Date.now(),path:location.pathname,
      id:button?.id||target.id||null,name:target.name||null,value:target.type==='checkbox'?target.checked:target.value??null,
      operation:form?.dataset.typedOperation||null,form_id:window.__draftControlFormId(form),
      fields:form?Object.fromEntries([...form.elements].filter(e=>e.name).map(e=>[e.name,e.type==='checkbox'?e.checked:e.value])):null});
  },true);
})();"""


def before_action(page,control):
    return {'event_start':page.evaluate('window.__draftControlInputs.length'),
        'control':control.evaluate("""e=>{const form=e.closest('.typed-operation-form');return {
          id:e.id||null,visible:!!e.getClientRects().length,disabled:e.matches(':disabled'),
          operation:form?.dataset.typedOperation||null,form_id:window.__draftControlFormId(form),path:location.pathname,
          fields:form?Object.fromEntries([...form.elements].filter(x=>x.name).map(x=>[x.name,x.type==='checkbox'?x.checked:x.value])):null};}""")}


def after_action(page,before,*,context,phase,request,response):
    events=page.evaluate('window.__draftControlInputs')
    return {**before,'context':context,'phase':phase,'request':request,'response':response,
        'event_end':len(events),'raw_events':events,'assertions_passed':True}


def select_native_option(control, value):
    """Choose a real enabled option through keyboard input, yielding trusted change events."""
    options = control.locator("option").evaluate_all(
        "elements => elements.map(option => ({value: option.value, disabled: option.disabled}))"
    )
    enabled = [option["value"] for option in options if not option["disabled"]]
    assert value in enabled, f"Option {value!r} is not an enabled native choice"

    def navigate(target):
        control.focus()
        control.press("Home")
        for _ in range(len(options) + 1):
            if control.input_value() == target:
                return
            control.press("ArrowDown")
        raise AssertionError(f"Could not reach enabled option {target!r} with native keyboard input")

    if control.input_value() == value and len(enabled) > 1:
        navigate(next(option for option in enabled if option != value))
    navigate(value)
    assert control.input_value() == value


def fill_native_text(control, value):
    """Edit a text-like input with trusted keystrokes and commit it by moving focus."""
    if control.get_attribute("type") == "time":
        control.fill(str(value))
        control.focus()
        if value:
            # Chromium time inputs do not accept a colon through sequential
            # typing. Nudge and restore the valid value with real key events so
            # the exact final field value owns a trusted change receipt.
            control.press("ArrowUp")
            control.press("ArrowDown")
        control.press("Tab")
        assert control.input_value() == str(value)
        return
    control.focus()
    control.press("Control+A")
    control.press("Backspace")
    if value:
        control.press_sequentially(str(value))
    control.press("Tab")
    assert control.input_value() == str(value)
