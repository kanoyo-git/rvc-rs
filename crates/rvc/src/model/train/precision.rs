pub(super) struct GradScaler {
    factor: Option<f32>,
    finite_steps: usize,
}

impl GradScaler {
    pub(super) fn new(enabled: bool) -> Self {
        Self {
            factor: enabled.then_some(65_536.0),
            finite_steps: 0,
        }
    }

    pub(super) fn factor(&self) -> Option<f32> {
        self.factor
    }

    pub(super) fn update(&mut self, overflow: bool) {
        let Some(scale) = self.factor.as_mut() else {
            return;
        };
        if overflow {
            *scale *= 0.5;
            self.finite_steps = 0;
        } else {
            self.finite_steps += 1;
            if self.finite_steps == 2000 {
                let grown = *scale * 2.0;
                if grown.is_finite() {
                    *scale = grown;
                }
                self.finite_steps = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::GradScaler;

    #[test]
    fn grows_after_2000_consecutive_steps_and_backs_off_on_overflow() {
        let mut scale = GradScaler::new(true);
        for _ in 0..1999 {
            scale.update(false);
        }
        assert_eq!(scale.factor(), Some(65_536.0));
        scale.update(false);
        assert_eq!(scale.factor(), Some(131_072.0));
        scale.update(true);
        assert_eq!(scale.factor(), Some(65_536.0));
        scale.update(false);
        assert_eq!(scale.factor(), Some(65_536.0));
    }
}
