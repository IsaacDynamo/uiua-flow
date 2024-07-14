use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::mem::swap;

use linked_hash_map::LinkedHashMap;
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use petgraph::{graph::Graph, graph::NodeIndex};
use uiua::{
    self,
    ast::{Modifier, Word},
    Primitive, Signature, Sp,
};

#[derive(Debug)]
pub enum Op {
    Input(usize),
    Word(Word),
    Output(usize),
}

#[derive(Debug)]
pub struct Var {
    src_slot: usize,
    dst_slot: usize,
}

#[derive(Default)]
struct Dataflow {
    stack: Vec<(NodeIndex, usize)>,
    g: Graph<Op, Var>,
}

impl Dataflow {
    fn pop(&mut self) -> (NodeIndex, usize) {
        self.stack.pop().expect("stack underflow")
    }

    fn peek(&self, index: usize) -> (NodeIndex, usize) {
        self.stack
            .len()
            .checked_sub(index)
            .and_then(|i| i.checked_sub(1))
            .and_then(|i| self.stack.get(i))
            .copied()
            .expect("stack underflow")
    }

    fn add_edge(&mut self, src: (NodeIndex, usize), dst: (NodeIndex, usize)) {
        self.g.add_edge(
            src.0,
            dst.0,
            Var {
                src_slot: src.1,
                dst_slot: dst.1,
            },
        );
    }
}

static ORANGE: &str = "\"#ff8855\"";
static SEA_GREEN: &str = "\"#11cc99\"";
static GREEN: &str = "\"#95d16a\"";
static YELLOW: &str = "\"#f0c36f\"";
static BLUE: &str = "\"#54b0fc\"";

fn signature(words: &[Sp<Word>]) -> Signature {
    if words.len() == 1 {
        match &words[0].value {
            Word::Char(_) | Word::String(_) | Word::Number(_, _) => {
                return Signature {
                    args: 0,
                    outputs: 1,
                }
            }
            Word::Primitive(p) => {
                if let Some(s) = p.signature() {
                    return s;
                }
            }
            Word::Modified(m) => {
                match m.modifier.value {
                    Modifier::Primitive(Primitive::Gap) => {
                        let mut s = signature(&m.operands);
                        s.args += 1;
                        return s;
                    }
                    _ => (),
                }
            }
            _ => (),
        }
    }

    println!("Warning: Guessing signature for {:?}", words);

    Signature {
        args: 1,
        outputs: 1,
    }
}

fn interpret(flow: &mut Dataflow, words: &[Sp<Word>]) {
    for word in words.iter().rev() {
        if word.value.is_code() {
            println!("- {:?}", word.value);
            match &word.value {
                Word::Number(_, _) | Word::Char(_) | Word::String(_) => {
                    let n = flow.g.add_node(Op::Word(word.value.clone()));
                    flow.stack.push((n, 0));
                }
                Word::Strand(_) => {
                    let n = flow.g.add_node(Op::Word(word.value.clone()));
                    flow.stack.push((n, 0));
                }
                Word::Modified(m) => {
                    match m.modifier.value {
                        Modifier::Primitive(Primitive::Fork) => {
                            // Detach stack from flow so we can clone it.
                            let mut stack = Vec::new();
                            swap(&mut flow.stack, &mut stack);
                            let mut stacks = Vec::new();

                            fn inner(
                                flow: &mut Dataflow,
                                stacks: &mut Vec<(Signature, Vec<(NodeIndex, usize)>)>,
                                stack: &Vec<(NodeIndex, usize)>,
                                words: &[Sp<Word>],
                            ) {
                                flow.stack.clone_from(stack);
                                interpret(flow, words);

                                let mut x = Vec::new();
                                swap(&mut flow.stack, &mut x);

                                let s = signature(words);
                                stacks.push((s, x));
                            }

                            // Process function pack or double operands.
                            if m.operands.len() == 1 {
                                if let uiua::ast::Word::Pack(p) = &m.operands[0].value {
                                    for branch in &p.branches {
                                        assert_eq!(
                                            branch.value.lines.len(),
                                            1,
                                            "Expect single-line function pack"
                                        );
                                        let words = &branch.value.lines[0];
                                        inner(flow, &mut stacks, &stack, words);
                                    }
                                } else {
                                    panic!("Expected function pack");
                                }
                            } else if m.operands.len() == 2 {
                                for word in m.operands.chunks(1) {
                                    inner(flow, &mut stacks, &stack, word);
                                }
                            } else {
                                panic!("Expected function pack or two operands");
                            }

                            // Remove used args.
                            let pops = stacks
                                .iter()
                                .map(|(s, _)| s.args)
                                .max()
                                .expect("Expect multiple operands");
                            for _ in 0..pops {
                                stack.pop();
                            }

                            // Push results
                            for (signature, result) in stacks.iter().rev() {
                                for &v in result.iter().rev().take(signature.outputs) {
                                    stack.push(v)
                                }
                            }

                            // Reattach stack to flow
                            flow.stack = stack;
                        }
                        Modifier::Primitive(Primitive::Both) => {
                            assert_eq!(m.operands.len(), 1, "Expect a single operand");

                            // Run once
                            interpret(flow, &m.operands);

                            // Pop result to tmp
                            let s = signature(&m.operands);
                            let mut tmp_stack = Vec::new();
                            for _ in 0..s.outputs {
                                tmp_stack.push(flow.pop())
                            }

                            // Run second time
                            interpret(flow, &m.operands);

                            // Push first results from tmp
                            for v in tmp_stack.into_iter().rev() {
                                flow.stack.push(v)
                            }
                        }
                        Modifier::Primitive(Primitive::Bracket) => {
                            fn inner(
                                flow: &mut Dataflow,
                                words: &[Sp<Word>],
                                tmp_stack: &mut Vec<(NodeIndex, usize)>,
                            ) {
                                interpret(flow, words);
                                let s = signature(words);
                                for _ in 0..s.outputs {
                                    tmp_stack.push(flow.pop())
                                }
                            }

                            let mut tmp_stack = Vec::new();

                            // Process function pack or double operands.
                            if m.operands.len() == 1 {
                                if let uiua::ast::Word::Pack(p) = &m.operands[0].value {
                                    for branch in &p.branches {
                                        assert_eq!(
                                            branch.value.lines.len(),
                                            1,
                                            "Expect single-line function pack"
                                        );
                                        let words = &branch.value.lines[0];
                                        inner(flow, words, &mut tmp_stack);
                                    }
                                } else {
                                    panic!("Expected function pack");
                                }
                            } else if m.operands.len() == 2 {
                                for word in m.operands.chunks(1) {
                                    inner(flow, word, &mut tmp_stack);
                                }
                            } else {
                                panic!("Expected function pack or two operands");
                            }

                            // Push results
                            for v in tmp_stack.into_iter().rev() {
                                flow.stack.push(v);
                            }
                        }
                        Modifier::Primitive(Primitive::Reduce) => {
                            let n = flow.g.add_node(Op::Word(word.value.clone()));
                            let s = Signature {
                                args: 1,
                                outputs: 1,
                            }; // Assume this is true
                            for i in 0..s.args {
                                let src = flow.pop();
                                flow.add_edge(src, (n, i))
                            }
                            for i in 0..s.outputs {
                                flow.stack.push((n, i));
                            }
                        }
                        Modifier::Primitive(Primitive::Gap) => {
                            flow.pop();
                            interpret(flow, &m.operands);
                        }
                        Modifier::Primitive(Primitive::Dip) => {
                            let v = flow.pop();
                            interpret(flow, &m.operands);
                            flow.stack.push(v);
                        }
                        Modifier::Primitive(Primitive::On) => {
                            let v = flow.peek(0);
                            interpret(flow, &m.operands);
                            flow.stack.push(v);
                        }
                        Modifier::Primitive(Primitive::By) => {
                            let s = signature(&m.operands);

                            // Duplicate last argument
                            assert!(s.args > 1);
                            let n = s.args - 1;
                            let mut tmp_stack = Vec::new();
                            for _ in 0..n {
                                tmp_stack.push(flow.pop())
                            }
                            flow.stack.push(flow.peek(0));
                            for v in tmp_stack.into_iter().rev() {
                                flow.stack.push(v);
                            }

                            interpret(flow, &m.operands);
                        }
                        _ => todo!(),
                    }
                }
                Word::Primitive(p) => match *p {
                    Primitive::Dup => {
                        flow.stack.push(flow.peek(0));
                    }
                    Primitive::Identity => (),
                    Primitive::Flip => {
                        let i = flow.pop();
                        let j = flow.pop();
                        flow.stack.push(i);
                        flow.stack.push(j);
                    }
                    Primitive::Over => {
                        flow.stack.push(flow.peek(1));
                    }
                    Primitive::Pop => {
                        flow.pop();
                    }
                    _ => {
                        let n = flow.g.add_node(Op::Word(word.value.clone()));
                        // Pop Args
                        for i in 0..p.args().expect("Expected primitive signature") {
                            let src = flow.pop();
                            flow.add_edge(src, (n, i))
                        }
                        // Push outputs
                        for i in 0..p.outputs().expect("Expected primitive signature") {
                            flow.stack.push((n, i));
                        }
                    }
                },
                Word::Ref(_r) => {
                    let n = flow.g.add_node(Op::Word(word.value.clone()));

                    // TODO: do look up
                    let s = Signature {
                        args: 1,
                        outputs: 1,
                    };

                    // Pop Args
                    for i in 0..s.args {
                        let src = flow.pop();
                        flow.add_edge(src, (n, i))
                    }
                    // Push outputs
                    for i in 0..s.outputs {
                        flow.stack.push((n, i));
                    }
                }
                _ => todo!(),
            }
        }
    }
}

pub fn process(input: &str) -> LinkedHashMap<String, Graph<Op, Var>> {
    let ast = uiua::parse(input, (), &mut uiua::Inputs::default());
    let mut uiua = uiua::Uiua::with_safe_sys();
    uiua.run_str(input).unwrap();
    let funcs = uiua.bound_functions();

    let mut flows = LinkedHashMap::new();

    let mut root = Dataflow::default();

    for item in &ast.0 {
        match item {
            uiua::ast::Item::Words(words) => {
                for line in words {
                    interpret(&mut root, line);
                }
            }
            uiua::ast::Item::Binding(binding) => {
                let name = &binding.name.value;
                let f = funcs.get(name).unwrap();
                let s = f.signature();

                let mut flow = Dataflow::default();

                let n = flow.g.add_node(Op::Input(f.signature().args));
                for i in (0..s.args).rev() {
                    flow.stack.push((n, i));
                }

                interpret(&mut flow, &binding.words);
                assert_eq!(flow.stack.len(), s.outputs, "inconsistent stack");

                let n = flow.g.add_node(Op::Output(flow.stack.len()));
                for (i, &(m, j)) in flow.stack.iter().rev().enumerate() {
                    flow.g.add_edge(
                        m,
                        n,
                        Var {
                            src_slot: j,
                            dst_slot: i,
                        },
                    );
                }

                flows.insert(name.to_string(), flow.g);
            }
            _ => println!("TODO: {:?}", item),
        }
    }

    if !root.stack.is_empty() {
        let n = root.g.add_node(Op::Output(root.stack.len()));
        for (i, &(m, j)) in root.stack.iter().rev().enumerate() {
            root.g.add_edge(
                m,
                n,
                Var {
                    src_slot: j,
                    dst_slot: i,
                },
            );
        }
    }

    if root.g.node_count() > 0 {
        flows.insert("root".to_string(), root.g);
    }

    flows
}

fn literal_value(word: &Word) -> String {
    match word {
        Word::Number(s, _) => {
            fn n(c: &str) -> &str {
                match c {
                    "inf" => "∞",
                    "eta" => "η",
                    "pi"  => "π",
                    "tau" => "τ",
                    _ => c,
                }
            }

            if let Some(rem) = s.strip_prefix(['`', '¯']) {
                format!("¯{}", n(rem))
            } else {
                n(s).to_string()
            }
        },
        Word::Char(c) => {
            format!("@{c}")
        },
        Word::String(s) => {
            format!("\"{s}\"")
        }
        _ => panic!("unexpected literal value")
    }
}

pub fn plot(graph: &Graph<Op, Var>) -> String {
    let mut input_self = HashMap::new();
    let mut output_self = HashMap::new();
    let mut dot = String::new();
    writeln!(dot, "digraph G {{").unwrap();
    writeln!(dot, "graph [fontname = \"DejaVu Sans Mono\"];").unwrap();
    writeln!(dot, "node [fontname = \"DejaVu Sans Mono\"];").unwrap();
    writeln!(dot, "edge [fontname = \"DejaVu Sans Mono\"];").unwrap();

    for (i, op) in graph.node_references() {
        match op {
            Op::Input(n) => {
                let inputs = (0..*n)
                    .map(|a| format!("<o{a}> •"))
                    .reduce(|a, b| format!("{a} | {b}"))
                    .unwrap_or_default();
                writeln!(
                    dot,
                    "n{} [label=\"{{input | {{{}}}}}\" shape=record];",
                    i.index(),
                    inputs
                )
                .unwrap();
            }
            Op::Word(word) => {
                match word {
                    Word::Number(_, _) => {
                        writeln!(
                            dot,
                            "n{} [label=\"<o0> {}\" shape=record style=filled fillcolor={} width=0.2];",
                            i.index(),
                            literal_value(word).replace('"', "\\\""),
                            ORANGE
                        )
                        .unwrap();
                    }
                    Word::Char(_) | Word::String(_) => {
                        writeln!(
                            dot,
                            "n{} [label=\"<o0> {}\" shape=record style=filled fillcolor={} width=0.2];",
                            i.index(),
                            literal_value(word).replace('"', "\\\""),
                            SEA_GREEN
                        )
                        .unwrap();
                    }
                    Word::Strand(words) => {
                        let color = if matches!(words[0].value, Word::Char(_) | Word::String(_)) {
                            SEA_GREEN
                        } else {
                            ORANGE
                        };

                        let s = words.iter()
                            .map(|word| literal_value(&word.value))
                            .reduce(|mut a, b| { write!(a, "_{b}").unwrap(); a })
                            .expect("non empty strand");

                        writeln!(
                            dot,
                            "n{} [label=\"<o0> {}\" shape=record style=filled fillcolor={} width=0.2];",
                            i.index(),
                            s.replace('"', "\\\""),
                            color
                        )
                        .unwrap();
                    }
                    Word::Primitive(p) => {
                        let args = p.args().unwrap();
                        let out = p.outputs().unwrap();
                        let name = p
                            .glyph()
                            .map(|c| c.to_string())
                            .unwrap_or(p.name().to_string());

                        match (args, out) {
                            (1, 1) => {
                                input_self.insert((i, 0), None);
                                output_self.insert((i, 0), None);
                                writeln!(dot, "n{} [label=\"{}\" shape=record style=filled fillcolor={} tooltip=\"{}\" width=0.2];",
                                    i.index(),
                                    name,
                                    GREEN,
                                    p.name()
                                ).unwrap();
                            }
                            (2, 1) => {
                                input_self.insert((i, 0), Some("io0"));
                                output_self.insert((i, 0), Some("io0"));
                                writeln!(dot, "n{} [label=\"<io0> {} | <i1> •\" shape=record style=filled fillcolor={} tooltip=\"{}\"];",
                                    i.index(),
                                    name,
                                    BLUE,
                                    p.name()
                                ).unwrap();
                            }
                            _ => {
                                todo!("support more primitives")
                            }
                        }
                    }
                    Word::Modified(m) => {
                        match m.modifier.value {
                            Modifier::Primitive(p @ Primitive::Reduce) => {
                                input_self.insert((i, 0), None);
                                output_self.insert((i, 0), None);
                                let x = match m.operands[0].value {
                                    Word::Primitive(p) => p.glyph().unwrap(),
                                    _ => todo!("supported more word variants within reduce"),
                                };

                                writeln!(dot, concat!(
                                    "n{} [label=<\n",
                                    "<TABLE BORDER=\"0\" CELLBORDER=\"1\" CELLSPACING=\"0\"><TR>\n",
                                    "<TD ALIGN=\"center\" WIDTH=\"20\" BGCOLOR={}>{}</TD>\n",
                                    "<TD ALIGN=\"center\" WIDTH=\"20\" BGCOLOR={}>{}</TD>\n",
                                    "</TR></TABLE>> shape=record style=filled fillcolor={}];"), i.index(), YELLOW, p.glyph().unwrap(), BLUE, x, GREEN).unwrap();
                            }
                            _ => {
                                writeln!(dot, "n{} [label=\"{:?}\" shape=record];", i.index(), word)
                                    .unwrap();
                            }
                        }
                    }
                    Word::Ref(r) => {
                        // TODO: lookup signature and adapt visualization
                        input_self.insert((i, 0), None);
                        output_self.insert((i, 0), None);
                        let color = GREEN;
                        writeln!(dot, "n{} [label=\"{}\" shape=record style=filled fillcolor={} width=0.2];", i.index(), r.name.value, color)
                            .unwrap();
                    }
                    _ => {
                        todo!("supported more word variants");
                    }
                }
            }
            Op::Output(n) => {
                let outputs = (0..*n)
                    .map(|a| format!("<i{a}> •"))
                    .reduce(|a, b| format!("{a} | {b}"))
                    .unwrap_or_default();
                writeln!(
                    dot,
                    "n{} [label=\"{{{{{}}} | output }}\" shape=record];",
                    i.index(),
                    outputs
                )
                .unwrap();
            }
        }
    }

    for (i, _) in graph.node_references() {
        for edge in graph.edges(i) {
            let var = edge.weight();

            let src = match output_self.get(&(i, var.src_slot)) {
                Some(Some(slot)) => format!("n{}:{}", i.index(), slot),
                Some(None) => format!("n{}", i.index()),
                _ => format!("n{}:o{}", i.index(), var.src_slot),
            };

            let dst = match input_self.get(&(edge.target(), var.dst_slot)) {
                Some(Some(slot)) => format!("n{}:{}", edge.target().index(), slot),
                Some(None) => format!("n{}", edge.target().index()),
                _ => format!("n{}:i{}", edge.target().index(), var.dst_slot),
            };

            writeln!(dot, "{} -> {} [arrowsize=0.5];", src, dst).unwrap();
        }
    }

    writeln!(dot, "}}").unwrap();
    dot
}


#[cfg(test)]
mod test {
    use crate::*;

    #[test]
    fn numbers() {
        plot(process("0 1.1").get("root").unwrap());
        plot(process("eta pi tau inf").get("root").unwrap());
        plot(process("η π τ ∞ ").get("root").unwrap());
        plot(process("`1 `1.1 `pi `π").get("root").unwrap());
        plot(process("¯1 ¯1.1 ¯pi ¯π").get("root").unwrap());
    }

    #[test]
    fn strand() {
        plot(process("0_1.1").get("root").unwrap());
        plot(process("eta_pi_tau_inf").get("root").unwrap());
        plot(process("η_π_τ_∞").get("root").unwrap());
        plot(process("`1_`1.1_`pi_`π").get("root").unwrap());
        plot(process("¯1_¯1.1_¯pi_¯π").get("root").unwrap());
        plot(process("@a_@b").get("root").unwrap());
        plot(process("\"b\"_\"a\"").get("root").unwrap());
    }

}