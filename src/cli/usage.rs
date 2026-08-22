/// The one-line contract: flags first, then exactly one mode. Pinned by
/// tests/cli_contract_test.rs; changing this text updates the pin in the
/// same change.
pub(super) fn print() {
    println!(
        "Usage: llm [--seed <n>] [--model <id-or-path>] [--epochs <n>] [--tiny] [--trace] [--temperature <t>] [--presence <c>] [--repetition <r>] [--top-p <p>] [--fluency <n>] [--eos] [--lr-decay <final_lr>] [--e2e <prompt> | --ask <prompt> | --eval | --train <file.jsonl> | --probe | --models | --demo]"
    );
    println!();

    // One working command per surface.
    println!("Examples:");
    println!("  llm");
    println!("  llm --trace --seed 42");
    println!("  llm --models");
    println!("  llm --model watercycle-latest");
    println!("  llm --e2e \"hello world\"");
    println!("  llm --eval --seed 42");
    println!("  llm --model watercycle-latest --eval --seed 42");
    println!("  llm --model stories-full --ask \"Once upon a time,\"");
    println!("  llm --demo --seed 42");
    println!(
        "  llm --tiny --train models/tinystories/train.jsonl --epochs 2 --model models/ts.bin"
    );
    println!(
        "  llm --tiny --train models/tinystories/demo.jsonl --epochs 6 --lr-decay 5e-5 --model models/ts.bin"
    );
    println!("  llm --tiny --eval --model models/ts.bin --fluency 20");
    println!("  llm --tiny --eval --model models/ts.bin --temperature 0.7 --presence 1.5");
    println!(
        "  llm --tiny --eval --model models/ts.bin --temperature 0.7 --top-p 0.8 --presence 1.5"
    );
    println!("  llm --probe --model stories-full --seed 42");
}
