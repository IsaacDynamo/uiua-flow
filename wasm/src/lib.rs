use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn uiua_to_dot(msg: String) -> JsValue {

    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();

    let flows = uiua_flow::process(&msg);

    let obj = js_sys::Object::new();
    for (k,v) in flows.iter() {
        let key = JsValue::from(k);
        let value = JsValue::from(uiua_flow::plot(&v));
        js_sys::Reflect::set(&obj, &key, &value).unwrap();
    }
    JsValue::from(obj)
}
