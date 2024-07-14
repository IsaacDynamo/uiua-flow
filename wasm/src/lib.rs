mod utils;

use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn uiua_to_dot(msg: String) -> String {
    let flows = uiua_flow::process(&msg);

    // For now return the first function
    for flow in flows {
        return uiua_flow::plot(&flow.1);
    }

    "".to_string()
}
