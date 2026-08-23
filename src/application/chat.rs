use std::io::Write;

use llm::{DecodeKnobs, LLM};

use super::{Trace, trace_turn};

/// Prompt loop until "exit": the shared chat surface of the training demo
/// and the loaded-model use path. Traced turns capture every decode step;
/// knobbed turns sample through the full stack; the default path keeps the
/// untraced greedy `predict` surface untouched.
pub(crate) fn chat_loop(llm: &mut LLM, trace: &Trace, knobs: &mut DecodeKnobs) {
    println!("\nInteractive mode");
    println!("  type a prompt and press Enter; 'exit' quits; '/help' lists commands");
    let mut input = String::new();
    loop {
        // Read the next prompt.
        input.clear();
        print!("\nyou> ");
        std::io::stdout().flush().unwrap();

        // End of input (closed pipe) ends the session cleanly instead of
        // spinning on an empty buffer.
        if std::io::stdin().read_line(&mut input).unwrap_or(0) == 0 {
            println!();
            println!("Exiting interactive mode.");
            break;
        }

        // The exit word ends the session; a leading slash is a command.
        let trimmed_input = input.trim();
        if trimmed_input.eq_ignore_ascii_case("exit") {
            println!("Exiting interactive mode.");
            break;
        }
        if let Some(command) = trimmed_input.strip_prefix('/') {
            if command.eq_ignore_ascii_case("exit") {
                println!("Exiting interactive mode.");
                break;
            }
            handle_command(command, knobs);
            continue;
        }

        // Traced turns capture every decode step; knobbed turns ride the
        // seeded sampling stack (one fresh rng per turn: same seed, same
        // answer); the default path keeps the pinned greedy stream.
        let formatted_input = format!("User: {trimmed_input}");
        let prediction = if trace.on || !knobs.is_greedy() {
            let mut rng = llm::Xorshift::new(llm::seed());
            let (output, steps) = llm.generate_with_steps(
                &formatted_input,
                knobs.temperature,
                knobs.top_p,
                knobs.presence,
                knobs.repetition,
                &mut rng,
            );
            if trace.on {
                trace_turn(llm, &formatted_input, &steps, trace);
            }
            output
        } else {
            llm.predict(&formatted_input)
        };
        println!("model> {}\n", render_answer(&prediction));
    }
}

/// Render the trailing </s> as a clean end-of-answer: the marker ends the
/// generation, it is not part of the text.
pub(crate) fn render_answer(answer: &str) -> String {
    match answer.strip_suffix("</s>") {
        Some(head) => head.trim_end().to_string(),
        None => answer.to_string(),
    }
}

/// Dispatch one slash command. Unknown names and bad values print a
/// friendly error and leave every knob unchanged.
fn handle_command(command: &str, knobs: &mut DecodeKnobs) {
    let mut parts = command.split_whitespace();
    let name = parts.next().unwrap_or("").to_ascii_lowercase();
    let value = parts.next();

    // Dispatch the command name to its action.
    match name.as_str() {
        "help" => print_help(),
        "config" => print_config(knobs),
        "reset" => {
            knobs.reset();
            println!("Decode knobs reset to greedy.");
            print_config(knobs);
        }
        "temp" | "top-p" | "presence" | "repetition" => set_knob(&name, value, knobs),
        other => println!("Unknown command '/{other}'. Type /help for the list."),
    }
}

/// Apply one knob set; a rejected value leaves the config untouched.
fn set_knob(name: &str, value: Option<&str>, knobs: &mut DecodeKnobs) {
    let Some(raw) = value else {
        println!("error: /{name} needs a number, e.g. /{name} 0.7");
        return;
    };
    let parsed: f32 = match raw.parse() {
        Ok(parsed) => parsed,
        Err(_) => {
            println!("error: '{raw}' is not a number (config unchanged)");
            return;
        }
    };
    let result = match name {
        "temp" => knobs.set_temperature(parsed),
        "top-p" => knobs.set_top_p(parsed),
        "presence" => knobs.set_presence(parsed),
        _ => knobs.set_repetition(parsed),
    };

    // Success echoes the new config; rejection explains itself.
    match result {
        Ok(()) => print_config(knobs),
        Err(message) => println!("error: {message} (config unchanged)"),
    }
}

fn print_help() {
    println!("Commands (anything else is sent to the model):");
    println!(
        "  /temp <t>        sampling temperature (> 0; 1.0 = unscaled -- greedy while no other knob moves)"
    );
    println!("  /top-p <p>       nucleus mass cutoff ((0, 1]; off by default)");
    println!("  /presence <c>    flat penalty per seen word (>= 0; 0 = off)");
    println!("  /repetition <r>  count-scaled repeat penalty (>= 1; 1 = off)");
    println!("  /config          show the current decode knobs");
    println!("  /reset           restore greedy defaults");
    println!("  /exit            quit the session");
}

fn print_config(knobs: &DecodeKnobs) {
    // One knob per line: the mode verdict first, then every value with its
    // neutral reading (off reads as 'off', not as a number that means off).
    let mode = if knobs.is_greedy() {
        "greedy"
    } else {
        "sampling"
    };
    println!("config ({mode})");
    println!("  temperature  {}", knobs.temperature);
    println!("  top-p        {}", neutral_off(knobs.top_p, 0.0));
    println!("  presence     {}", neutral_off(knobs.presence, 0.0));
    println!("  repetition   {}", neutral_off(knobs.repetition, 1.0));
}

/// A knob's display value: the neutral setting reads as 'off'.
fn neutral_off(value: f32, neutral: f32) -> String {
    if value == neutral {
        "off".to_string()
    } else {
        format!("{value}")
    }
}
