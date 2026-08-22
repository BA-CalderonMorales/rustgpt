/// The one-line contract: flags first, then exactly one mode. Pinned by
/// tests/cli_contract_test.rs; changing this text updates the pin in the
/// same change. The body follows the temperwright operating-path pattern:
/// an ordered command list where each step prepares the next, then the
/// flags grouped by purpose, then one working example per surface.
pub(super) fn print() {
    println!(
        "Usage: llm [--seed <n>] [--model <id-or-path>] [--epochs <n>] [--tiny] [--trace] [--temperature <t>] [--presence <c>] [--repetition <r>] [--top-p <p>] [--fluency <n>] [--eos] [--lr-decay <final_lr>] [--e2e <prompt> | --ask <prompt> | --eval | --train <file.jsonl> | --probe | --models | --demo]"
    );
    println!();

    // The operating path.
    println!("The operating path: start at the top, move down. Each step prepares");
    println!("the next; every number a step prints carries its seed.");
    println!();
    println!("  1  llm --models                         pick an artifact from the trained catalog");
    println!(
        "  2  llm --model <id> --ask \"<prompt>\"    one answer from it; decode knobs honored"
    );
    println!("  3  llm --model <id>                     chat interactively (/help inside)");
    println!("  4  llm --demo                           watch raw text become a model, end to end");
    println!(
        "  5  llm --tiny --train <corpus.jsonl>    teach your own model (add --eos --lr-decay)"
    );
    println!("  6  llm --tiny --eval --model <path>     score it against held-out data");
    println!("  7  llm --eval --seed 42                 the micro arena oracle: fresh, seeded");
    println!();

    // Decode knobs: every sampling surface shares them.
    println!("Decode knobs (usable with --tiny --eval, --ask, and chat):");
    println!("  --temperature <t>   sampling heat (> 0; 1.0 keeps the greedy pin)");
    println!("  --top-p <p>         nucleus mass cutoff ((0, 1]; 0 = off)");
    println!("  --presence <c>      flat penalty per seen word (>= 0; 0 = off)");
    println!("  --repetition <r>    count-scaled repeat penalty (>= 1; 1 = off)");
    println!();

    // Training levers: the recipe-level knobs.
    println!("Training levers (with --tiny --train):");
    println!("  --eos               append </s> to every row: teach the model to stop");
    println!("  --lr-decay <lr>     linear per-epoch decay down to this final rate");
    println!();

    // Reproducibility is a contract, not a convenience.
    println!("Reproducibility:");
    println!("  --seed <n>          default 42 on machine modes; same seed, same scores");
    println!();
    println!("--e2e and --probe are contract probes: boundaries to test, not quality");
    println!("claims.");
    println!();

    // One working example per path step.
    println!("Examples:");
    println!("  llm --models");
    println!("  llm --model stories-full --ask \"Once upon a time,\"");
    println!(
        "  llm --model stories-full --ask \"Once upon a time,\" --temperature 0.7 --top-p 0.8 --presence 1.5 --repetition 1.1"
    );
    println!("  llm --model watercycle-latest");
    println!("  llm --demo --seed 42");
    println!(
        "  llm --tiny --train my-corpus.jsonl --epochs 6 --eos --lr-decay 5e-5 --model models/ts.bin"
    );
    println!("  llm --tiny --eval --model models/ts.bin --fluency 20");
    println!("  llm --eval --seed 42");
    println!("  llm --probe --model stories-full --seed 42");
}
