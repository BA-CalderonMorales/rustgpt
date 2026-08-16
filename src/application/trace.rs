/// Domain-labeled event trace for the interactive surface: every core event
/// prints "[{domain}] {msg}" on stderr, so a human can see WHICH DOMAIN
/// could be broken when output misbehaves. Domain labels are the src/
/// folder names: the label IS the pointer to the broken code. Emits nothing
/// unless switched on; machine stdout contracts never touch it.
pub(crate) struct Trace {
    pub(crate) on: bool,
}

impl Trace {
    pub(crate) fn new(on: bool) -> Self {
        Self { on }
    }

    pub(crate) fn event(&self, domain: &str, msg: &str) {
        if self.on {
            eprintln!("[{domain}] {msg}");
        }
    }
}
