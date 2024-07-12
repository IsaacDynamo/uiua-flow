use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::{io::Write, mem::swap, path::Path};

use linked_hash_map::LinkedHashMap;
use petgraph::visit::{EdgeRef, IntoNodeReferences};
use petgraph::{dot::Dot, graph::Graph, graph::NodeIndex};
use uiua::{
    self,
    ast::{Modifier, Word},
    Primitive, Signature, Sp,
};

#[derive(Debug)]
enum Op {
    Input(usize),
    Word(Word),
    Output(usize),
}

#[derive(Debug)]
struct Var {
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

fn process(input: &str) -> LinkedHashMap<String, Graph<Op, Var>> {
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

fn main() {
    let input = r#"
    Identity ← ∘
    Kestrel ← ⊙◌
    Kite ← ⋅∘
    Warbler ← ⊂.
    Cardinal ← ⊂:
    Bluebird ← ⊢⇌
    Blackbird ← ⇌⊂
    Starling ← ⊂⟜¯
    VioletStarling ← ≍⇌.
    Dove ← ⊟⊙⇌
    ZebraDove ← ⊟⇌
    Phoenix ← ⊟⊃¯⇌
    Psi ← ⊂∩□
    Dovekie ← ⊟⊓¯⇌
    EasternNicator ← ⊟⟜+
    WesternNicator ← ⊟⊸+
    Eagle ← ⊟⊙+
    GoldenEagle ← ⊟+
    EasternKingbird ← ⊟⊃¯+
    WesternKingbird ← ⊟⊃+¯
    EasternParotia ← ⊟⊓¯+
    WesternParotia ← ⊟⊓+¯
    Pheasant ← ⊟⊃+-
    BaldEagle ← ⊟⊓-+
    "#;

    let _extra = r#"
    #A ← ⊓++⋅∘ 1 2 3 14 15 16
    #B ← ⊓(++)⋅∘ 1 2 3 14 15 16
    #C ← ⊓++ 1 2 3 14 15 16
    #A ← ⊃++⋅∘ 1 2 3 14 15 16
    #B ← ⊃(++)⋅∘ 1 2 3 14 15 16
    #B ← ⊃(++)1 2 3 14 15 16
    #C ← ⊃+1 2 3 14 15 16
    H ← ÷2
    MinAvg ← H/↧+⇌⊃↙↘H⧻.⊏⍏.
    #MinAvg ← ÷2/↧+⇌⊃↙↘÷2⧻.⊏⍏.
    Q ← /↧∊+@A⇡26⌵
    X ← /+
    W←⊃⋅∘∘
    U←⊃(⋅∘|∘)
    "#;

    // let input = "H ← ÷2\nMinAvg ← H⊢+⇌⊃↙↘H⧻.⊏⍏.\nQ ← /↧∊+@A⇡26⌵\nW←⊃⋅∘∘\nU←⊃(⋅∘|∘)";

    let flows = process(input);

    let mut index = HEADER.to_string();

    let output_path = Path::new("output");
    std::fs::create_dir_all(output_path).unwrap();

    for (name, flow) in flows {
        {
            let dot = format!("{:?}", Dot::with_config(&flow, &[]));
            let dot_path = output_path.join(format!("{}_dbg.dot", name));
            let svg_path = output_path.join(format!("{}_dbg.svg", name));

            let mut f = std::fs::File::create(dot_path.clone()).unwrap();
            f.write_all(dot.as_bytes()).unwrap();

            //println!("{}", svg_path.display());

            std::process::Command::new("dot")
                .arg("-Tsvg")
                .arg("-o")
                .arg(svg_path)
                .arg(dot_path)
                .spawn()
                .unwrap();
        }
        {
            let dot = plot(&flow);
            let dot_path = output_path.join(format!("{}.dot", name));
            let svg_path = output_path.join(format!("{}.svg", name));

            let mut f = std::fs::File::create(dot_path.clone()).unwrap();
            f.write_all(dot.as_bytes()).unwrap();

            //println!("{}", svg_path.display());

            std::process::Command::new("dot")
                .arg("-Tsvg:cairo")
                .arg("-o")
                .arg(svg_path)
                .arg(dot_path)
                .spawn()
                .unwrap();
        }

        writeln!(
            index,
            "<li class=\"image-item\">\n<h2>{}</h2>\n<img src=\"{}_dbg.svg\">\n<img src=\"{}.svg\">\n</li>",
            name, name, name
        )
        .unwrap();
    }

    index.push_str(FOOTER);

    let index_path = output_path.join("index.html");
    let mut f = std::fs::File::create(index_path).unwrap();
    f.write_all(index.as_bytes()).unwrap();
}

static HEADER: &str = r###"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Image List with Headers</title>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .image-list {
            list-style-type: none;
            padding: 0;
        }
        .image-item {
            margin-bottom: 20px;
        }
        .image-item img {
            max-width: 100%;
            height: auto;
        }
        .image-item h2 {
            margin: 10px 0;
            font-size: 1.5em;
        }
    </style>
</head>
<body>
    <ul class="image-list">"###;

static FOOTER: &str = r###"</ul>
</body>
</html>
"###;

fn plot(graph: &Graph<Op, Var>) -> String {
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
                    Word::Number(n, _) => {
                        writeln!(
                            dot,
                            "n{} [label=\"<o0> {}\" shape=record style=filled fillcolor={} width=0.2];",
                            i.index(),
                            n,
                            ORANGE
                        )
                        .unwrap();
                    }
                    Word::Char(c) => {
                        writeln!(
                            dot,
                            "n{} [label=\"<o0> {}\" shape=record style=filled fillcolor={} width=0.2];",
                            i.index(),
                            c,
                            SEA_GREEN
                        )
                        .unwrap();
                    }
                    Word::String(s) => {
                        writeln!(
                            dot,
                            "n{} [label=\"<o0> {}\" shape=record style=filled fillcolor={} width=0.2];",
                            i.index(),
                            s,
                            SEA_GREEN
                        )
                        .unwrap();
                    }
                    Word::Primitive(p) => {
                        let args = p.args().unwrap();
                        let out = p.outputs().unwrap();

                        let inputs = (0..args)
                            .map(|a| format!("<i{a}> •"))
                            .reduce(|a, b| format!("{a} | {b}"))
                            .unwrap_or_default();
                        let outputs = (0..out)
                            .map(|a| format!("<o{a}> •"))
                            .reduce(|a, b| format!("{a} | {b}"))
                            .unwrap_or_default();
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
                                writeln!(dot, "n{} [label=\"{{{{{}}} | {} | {{{}}}}}\" shape=record style=filled fillcolor={}];",
                                    i.index(),
                                    inputs,
                                    name,
                                    outputs,
                                    YELLOW // todo
                                ).unwrap();
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
                                    _ => todo!(),
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
                        writeln!(dot, "n{} [label=\"{:?}\" shape=record];", i.index(), word)
                            .unwrap();
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

// uiua::ast::Word::Char(c) => {
//     writeln!(ctx.graph, "    word{} [label=\"{}\"];", word.span.start.byte_pos, c).unwrap();
//     let v = ctx.next_value;
//     ctx.next_value += 1;
//     ctx.stack.push(v);
//     writeln!(ctx.graph, "    v{} [label=\"\" shape=point];", v).unwrap();
//     writeln!(ctx.graph, "    word{} -> v{} [dir=none];", word.span.start.byte_pos, v).unwrap();
// }

// let inputs = inputs.iter().map(|a| format!("<v{a}> v{a}")).reduce(|a, b| format!("{a} | {b}")).unwrap_or_default();
// let outputs = outputs.iter().map(|a| format!("<v{a}> v{a}")).reduce(|a, b| format!("{a} | {b}")).unwrap_or_default();
// let name = p.glyph().map(|c| c.to_string()).unwrap_or(p.name().to_string());

// writeln!(ctx.graph, "    word{} [label=\"{{{{{}}} | {} | {{{}}}}}\" shape=record fillcolor={} style=filled];",
//     word.span.start.byte_pos,
//     inputs,
//     name,
//     outputs,
//     primitive_color(p)
// ).unwrap();

//writeln!(ctx.graph, "    v{} -> word{}:v{};", v, word.span.start.byte_pos, v).unwrap();
//inputs.push(v);
//struct3 [label="hello\nworld |{ b |{c|<here> d|e}| f}| g | h"];

//writeln!(ctx.graph, "    v{} [label=\"\" shape=point];", v).unwrap();
//writeln!(ctx.graph, "    word{}:v{} -> v{} [dir=none];", word.span.start.byte_pos, v, v).unwrap();
