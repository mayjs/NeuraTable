use std::{
    ops::{AddAssign, DivAssign},
    str::FromStr,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ModelValueMode {
    /// Values are centered on 0 (and have a negative and positive part)
    Symmetric,
    /// Values are not centered on zero and are positive
    Asymmetric,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ModelValueRange {
    value_mode: ModelValueMode,
    max_abs_value: f32,
}

impl ModelValueRange {
    /// Create a new symmetric model value range
    pub fn symmetric(max_abs_value: f32) -> Self {
        Self {
            value_mode: ModelValueMode::Symmetric,
            max_abs_value,
        }
    }

    /// Create a new asymmetric model value range
    pub fn asymmetric(max_abs_value: f32) -> Self {
        Self {
            value_mode: ModelValueMode::Asymmetric,
            max_abs_value,
        }
    }

    /// Transform a single value in the u16 range to a f32 value in the range specified by self
    pub fn pixel_value_to_model(&self, pixel_value: u16) -> f32 {
        let asymmetric_value = ((pixel_value as f32) / (u16::MAX as f32)) * self.max_abs_value;
        match self.value_mode {
            ModelValueMode::Symmetric => (asymmetric_value * 2.0) - self.max_abs_value,
            ModelValueMode::Asymmetric => asymmetric_value,
        }
    }

    /// Transform a value in the value range specified by self into the [0,1] range
    pub fn normalize_model_value<T>(&self, model_value: &mut T)
    where
        T: AddAssign<f32> + DivAssign<f32>,
    {
        if self.value_mode == ModelValueMode::Symmetric {
            *model_value += self.max_abs_value;
            *model_value /= 2.0;
        }

        *model_value /= self.max_abs_value;
    }
}

impl FromStr for ModelValueRange {
    type Err = std::num::ParseFloatError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.starts_with("+-") {
            s[2..].parse().map(|max| ModelValueRange::symmetric(max))
        } else {
            s.parse().map(|max| ModelValueRange::asymmetric(max))
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_parse_symmetric() {
        let parsed = ModelValueRange::from_str("+-123").unwrap();
        assert_eq!(parsed, ModelValueRange::symmetric(123.0));
    }

    #[test]
    fn test_parse_asymmetric() {
        let parsed = ModelValueRange::from_str("1000.00").unwrap();
        assert_eq!(parsed, ModelValueRange::asymmetric(1000.0));
    }
}
