use petgraph::dot::Dot;
use std::fmt::Write as FmtWrite;
use std::{io::Write, path::Path};
use uiua_flow::*;

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
