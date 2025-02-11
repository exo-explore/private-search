use anyhow::Result;
use axum::extract::path;
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use serde_json::Value;
use simplepir::*;
use std::{env, path::PathBuf, process::Command};

use crate::{embedding::BertEmbedder, error::PirError, utils::encode_data};

pub trait Database {
    fn new() -> Result<Self>
    where
        Self: Sized;
    fn update(&mut self) -> Result<()>;
    fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>>;
    fn params(&self) -> &SimplePIRParams;
    fn hint(&self) -> &DMatrix<BigInt>;
    fn a(&self) -> &DMatrix<BigInt>;
}

pub struct SimplePirDatabase {
    params: Option<SimplePIRParams>,
    data: DMatrix<BigInt>,
    hint: Option<DMatrix<BigInt>>,
    a: Option<DMatrix<BigInt>>,
}

impl SimplePirDatabase {
    pub fn new(data: DMatrix<BigInt>) -> Self {
        Self {
            data,
            params: None,
            hint: None,
            a: None,
        }
    }

    pub fn update_db(&mut self, data: DMatrix<BigInt>) -> Result<()> {
        self.data = data;

        let params = gen_params(self.data.nrows(), self.data.ncols(), 64);
        let (hint, a) = gen_hint(&params, &self.data);

        self.params = Some(params);
        self.hint = Some(hint);
        self.a = Some(a);

        Ok(())
    }

    pub fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        let params = self
            .params
            .as_ref()
            .ok_or_else(|| PirError::Database("Database not initialized".to_string()))?;
        let answer = process_query(&self.data, query, params.q.clone());
        Ok(answer)
    }

    fn params(&self) -> &SimplePIRParams {
        self.params
            .as_ref()
            .ok_or(PirError::Database("Database not initialized".to_string()))
            .unwrap()
    }

    fn hint(&self) -> &DMatrix<BigInt> {
        self.hint
            .as_ref()
            .ok_or(PirError::Database("Database not initialized".to_string()))
            .unwrap()
    }

    fn a(&self) -> &DMatrix<BigInt> {
        self.a
            .as_ref()
            .ok_or(PirError::Database("Database not initialized".to_string()))
            .unwrap()
    }
}

pub struct EmbeddingDatabase {
    db: SimplePirDatabase,
    embedder: BertEmbedder,
}

impl Database for EmbeddingDatabase {
    fn new() -> Result<Self> {
        Ok(Self {
            db: SimplePirDatabase::new(DMatrix::zeros(1, 1)),
            embedder: BertEmbedder::new().map_err(|e| PirError::Embedding(e.to_string()))?,
        })
    }

    fn update(&mut self) -> Result<()> {
        // If running as a binary, use the current directory otherwise use the manifest directory
        let path = env::var("CARGO_MANIFEST_DIR")
        .map_or_else(
            |_| {
                let mut p = env::current_dir().unwrap();
                p.push("tiptoe-rs/src/python/stocks.py");
                p
            },
            |manifest_dir| {
                let mut p = PathBuf::from(manifest_dir);
                p.push("src/python/stocks.py");
                p
            },
        );

        let stock_json = Command::new("python")
            .arg(path)
            .output()
            .map_err(|e| PirError::CommandFailed(e.to_string()))?;

        if !stock_json.status.success() {
            return Err(PirError::CommandFailed("Failed to update database".to_string()).into());
        }

        let stock_json = String::from_utf8(stock_json.stdout)?;
        let stock_json: Vec<Value> = serde_json::from_str(&stock_json)?;

        let embeddings = self
            .embedder
            .embed_json_array(&stock_json)
            .map_err(|e| PirError::Embedding(e.to_string()))?;

        if embeddings.nrows() != embeddings.ncols() {
            return Err(PirError::Database("Embedding matrix must be square".to_string()).into());
        }

        self.db.update_db(embeddings)?;
        Ok(())
    }

    fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        self.db.respond(query)
    }

    fn params(&self) -> &SimplePIRParams {
        self.db.params()
    }

    fn hint(&self) -> &DMatrix<BigInt> {
        self.db.hint()
    }

    fn a(&self) -> &DMatrix<BigInt> {
        self.db.a()
    }
}

pub struct EncodingDatabase {
    db: SimplePirDatabase,
}

impl Database for EncodingDatabase {
    fn new() -> Result<Self> {
        Ok(Self {
            db: SimplePirDatabase::new(DMatrix::zeros(1, 1)),
        })
    }

    fn update(&mut self) -> Result<()> {
        // If running as a binary, use the current directory otherwise use the manifest directory
        let path = env::var("CARGO_MANIFEST_DIR")
        .map_or_else(
            |_| {
                let mut p = env::current_dir().unwrap();
                p.push("tiptoe-rs/src/python/stocks.py");
                p
            },
            |manifest_dir| {
                let mut p = PathBuf::from(manifest_dir);
                p.push("src/python/stocks.py");
                p
            },
        );

        let stock_json = Command::new("python")
            .arg(path)
            .output()
            .map_err(|e| PirError::CommandFailed(e.to_string()))?;

        if !stock_json.status.success() {
            return Err(PirError::CommandFailed("Failed to update database".to_string()).into());
        }

        let stock_json = String::from_utf8(stock_json.stdout)?;
        let stock_json: Vec<Value> = serde_json::from_str(&stock_json)?;

        let encodings = encode_data(
            &stock_json
                .iter()
                .map(|v| v.to_string())
                .collect::<Vec<String>>(),
        )
        .map_err(|e| PirError::Encoding(e.to_string()))?;

        if encodings.nrows() != encodings.ncols() {
            return Err(PirError::Database("Encoding matrix must be square".to_string()).into());
        }

        self.db.update_db(encodings)?;
        Ok(())
    }

    fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        self.db.respond(query)
    }

    fn params(&self) -> &SimplePIRParams {
        self.db.params()
    }

    fn hint(&self) -> &DMatrix<BigInt> {
        self.db.hint()
    }

    fn a(&self) -> &DMatrix<BigInt> {
        self.db.a()
    }
}
