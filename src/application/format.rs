//! Shared human-facing formatting: the one home for number rendering that
//! several surfaces share, so 385776 never prints two different ways.

/// Thousands separators: 14230655 reads as 14,230,655.
pub(crate) fn thousands(value: u64) -> String {
    let digits = value.to_string();
    let mut out = String::with_capacity(digits.len() + digits.len() / 3);
    for (index, digit) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(digit);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::thousands;

    #[test]
    fn thousands_groups_from_the_right() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(385_776), "385,776");
        assert_eq!(thousands(14_230_655), "14,230,655");
    }
}
