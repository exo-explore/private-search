use std::string::FromUtf8Error;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum PirError {
    #[error("Database error: {0}")]
    Database(String),

    #[error("Embedding error: {0}")]
    Embedding(String),

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Encoding error: {0}")]
    Encoding(String),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("String conversion error: {0}")]
    StringConversion(#[from] FromUtf8Error),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("External command failed: {0}")]
    CommandFailed(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Tensor operation error: {0}")]
    TensorError(String),
}

// Implement From trait for common error conversions
impl From<candle::Error> for PirError {
    fn from(err: candle::Error) -> Self {
        PirError::TensorError(err.to_string())
    }
}

impl From<tokenizers::Error> for PirError {
    fn from(err: tokenizers::Error) -> Self {
        PirError::TokenizerError(err.to_string())
    }
}
