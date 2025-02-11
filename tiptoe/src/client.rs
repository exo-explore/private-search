use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::One;
use simplepir::{generate_query, recover, SimplePIRParams};
use std::cmp::Ordering;

use crate::{
    embedding::BertEmbedder,
    error::PirError,
    network::{AsyncDatabase, RemoteDatabase},
    server::{Database, EmbeddingDatabase, EncodingDatabase},
};

// Each database can be either local or remote
pub enum DatabaseConnection<T> {
    Local(T),
    Remote(Box<dyn AsyncDatabase>),
}

impl<T: Database> DatabaseConnection<T> {
    #[allow(dead_code)]
    async fn update(&mut self) -> Result<()> {
        match self {
            Self::Local(db) => db
                .update()
                .map_err(|e| PirError::Database(format!("Update failed: {}", e)).into()),
            Self::Remote(_db) => Ok(()),
        }
    }

    async fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        match self {
            Self::Local(db) => db
                .respond(query)
                .map_err(|e| PirError::Database(format!("Response failed: {}", e)).into()),
            Self::Remote(db) => db.respond(query).await,
        }
    }

    async fn params(&self) -> Result<SimplePIRParams> {
        match self {
            Self::Local(db) => Ok(db.params().clone()),
            Self::Remote(db) => db.get_params().await,
        }
    }

    async fn hint(&self) -> Result<DMatrix<BigInt>> {
        match self {
            Self::Local(db) => Ok(db.hint().clone()),
            Self::Remote(db) => db.get_hint().await,
        }
    }

    async fn a(&self) -> Result<DMatrix<BigInt>> {
        match self {
            Self::Local(db) => Ok(db.a().clone()),
            Self::Remote(db) => db.get_a().await,
        }
    }
}

// Unified client that works with both local and remote databases
pub struct Client {
    embedding_db: DatabaseConnection<EmbeddingDatabase>,
    encoding_db: DatabaseConnection<EncodingDatabase>,
    embedder: BertEmbedder,
}

impl Client {
    pub fn new_local() -> Result<Self> {
        Ok(Self {
            embedding_db: DatabaseConnection::Local(EmbeddingDatabase::new()?),
            encoding_db: DatabaseConnection::Local(EncodingDatabase::new()?),
            embedder: BertEmbedder::new()?,
        })
    }

    pub fn new_remote(embedding_url: String, encoding_url: String) -> Result<Self> {
        Ok(Self {
            embedding_db: DatabaseConnection::Remote(Box::new(RemoteDatabase::new(embedding_url))),
            encoding_db: DatabaseConnection::Remote(Box::new(RemoteDatabase::new(encoding_url))),
            embedder: BertEmbedder::new()?,
        })
    }

    #[allow(dead_code)]
    pub(crate) async fn update(&mut self) -> Result<()> {
        self.encoding_db.update().await?;
        self.embedding_db.update().await?;
        Ok(())
    }

    fn adjust_embedding(embedding: DVector<BigInt>, m: usize) -> DVector<BigInt> {
        match embedding.len().cmp(&m) {
            Ordering::Equal => embedding,
            Ordering::Less => {
                let mut new_embedding = DVector::zeros(m);
                new_embedding
                    .rows_mut(0, embedding.len())
                    .copy_from(&embedding);
                new_embedding
            }
            Ordering::Greater => embedding.rows(0, m).into(),
        }
    }

    pub async fn query(&self, query: &str) -> Result<DVector<BigInt>> {
        let embedding = self
            .embedder
            .embed_text(query)
            .map_err(|e| PirError::Embedding(format!("Text embedding failed: {}", e)))?;

        // Query embedding database
        let embedding_params = self.embedding_db.params().await?;
        let adjusted_embedding = Self::adjust_embedding(embedding, embedding_params.m);
        let (s_embedding, query_embedding) = generate_query(
            &embedding_params,
            &adjusted_embedding,
            &self.embedding_db.a().await?,
        );

        let response_embedding = self.embedding_db.respond(&query_embedding).await?;
        let result_embedding = recover(
            &self.embedding_db.hint().await?,
            &s_embedding,
            &response_embedding,
            &embedding_params,
        );

        // Convert to one-hot vector
        let result_vec = {
            let mut vec = DVector::zeros(result_embedding.len());
            let max_idx = result_embedding
                .iter()
                .enumerate()
                .max_by_key(|(_i, val)| (*val).clone())
                .ok_or_else(|| PirError::InvalidInput("Empty embedding result".to_string()))?
                .0;
            vec[max_idx] = BigInt::one();
            vec
        };

        // Query encoding database
        let encoding_params = self.encoding_db.params().await?;
        let adjusted_result = Self::adjust_embedding(result_vec, encoding_params.m);
        let (s, query) = generate_query(
            &encoding_params,
            &adjusted_result,
            &self.encoding_db.a().await?,
        );

        let response = self.encoding_db.respond(&query).await?;
        let result = recover(
            &self.encoding_db.hint().await?,
            &s,
            &response,
            &encoding_params,
        );

        Ok(result)
    }

    pub async fn query_top_k(&self, query: &str, k: usize) -> Result<Vec<DVector<BigInt>>> {
        if k == 0 {
            return Err(PirError::InvalidInput("k must be greater than 0".to_string()).into());
        }

        let embedding = self
            .embedder
            .embed_text(query)
            .map_err(|e| PirError::Embedding(format!("Text embedding failed: {}", e)))?;
        let embedding_params = self.embedding_db.params().await?;
        let encoding_params = self.encoding_db.params().await?;

        let (s_embedding, query_embedding) = generate_query(
            &embedding_params,
            &Self::adjust_embedding(embedding, embedding_params.m),
            &self.embedding_db.a().await?,
        );

        let response_embedding = self.embedding_db.respond(&query_embedding).await?;
        let result_embedding = recover(
            &self.embedding_db.hint().await?,
            &s_embedding,
            &response_embedding,
            &embedding_params,
        );

        let top_indices: Vec<usize> = {
            let mut indexed_values: Vec<(usize, &BigInt)> =
                result_embedding.iter().enumerate().collect();
            indexed_values.sort_by(|(_i1, v1), (_i2, v2)| v2.cmp(v1));
            indexed_values.into_iter().map(|(i, _val)| i).collect()
        };

        if top_indices.is_empty() {
            return Err(PirError::InvalidInput("No results found".to_string()).into());
        }

        let mut results = Vec::with_capacity(k);
        for &idx in top_indices.iter().take(k) {
            let mut vec = DVector::zeros(result_embedding.len());
            vec[idx] = BigInt::one();

            let (s, query) = generate_query(
                &encoding_params,
                &Self::adjust_embedding(vec, encoding_params.m),
                &self.encoding_db.a().await?,
            );

            let response = self.encoding_db.respond(&query).await?;
            let result = recover(
                &self.encoding_db.hint().await?,
                &s,
                &response,
                &encoding_params,
            );

            results.push(result);
        }

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::decode_input;

    use super::*;
    use rand::seq::SliceRandom;
    use serde_json::Value;
    use strsim::jaro_winkler;
    use tokio::test;

    async fn run_test_queries(client: &mut Client) -> Result<()> {
        let names = vec![
            "Bitcoin",
            "Ethereum",
            "SPDR S&P 500",
            "Tesla",
            "NASDAQ Composite",
        ];

        for i in 0..3 {
            println!("\nUpdate iteration {}...", i + 1);
            client.update().await?; // Dummy for remote client
            for name in &names {
                println!("\nQuerying {}...", name);
                let result = client.query(name).await?;
                println!("Raw result: {:?}", result);

                let output = decode_input(&result)?;
                println!("Decoded output: {:?}", output);
            }
        }
        Ok(())
    }

    #[test]
    async fn test_local_client() -> Result<()> {
        let mut client = Client::new_local()?;
        run_test_queries(&mut client).await
    }

    #[ignore]
    #[test]

    async fn test_remote_client() -> Result<()> {
        let mut client = Client::new_remote(
            "http://localhost:3001".to_string(),
            "http://localhost:3000".to_string(),
        )?;
        run_test_queries(&mut client).await
    }

    #[test]
    async fn bench_client_retrieval_accuracy() -> Result<()> {
        fn names_match(name1: &str, name2: &str) -> bool {
            let name1 = name1.trim().to_lowercase();
            let name2 = name2.trim().to_lowercase();

            // Exact match
            if name1 == name2 {
                return true;
            }

            // Check similarity using Jaro-Winkler distance
            let similarity = jaro_winkler(&name1, &name2);
            // Threshold of 0.9 means names need to be 90% similar
            similarity > 0.9
        }

        println!("Testing client query acceptance rate for both single and top-k queries...");
        let mut client = Client::new_local()?;
        let k = 3;

        let symbols = [
            "Agilent",
            "Apple",
            "General Motors Company",
            "Micron Technology",
            "Tesla",
            "Aluminum Futures,Apr-2025",
            "Canadian Dollar Dec 20",
            "E-mini Crude Oil Futures,Mar-20",
            "NASDAQ Composite",
            "EUR/USD",
            "AUD/USD",
            "Dow Jones Transportation Average",
            "HANG SENG INDEX",
            "CBOE Volatility Index",
            "Pacer Data and Digital Revolution",
            "SPDR S&P 500",
            "Washington Mutual Invs Fd Cl A",
            "Vanguard S&P 500 ETF",
            "Xtr.(IE)-Art.Int.+Big Data ETFR",
            "Bitcoin USD",
            "Ethereum USD",
        ];

        let query_templates = [
            "Tell me about {name}",
            "What is the latest price of {name}?",
            "How is {name} performing today?",
            "Give me details on {name}",
            "Fetch data for {name}",
            "What's happening with {name}?",
        ];

        let mut single_success_count = 0;
        let mut single_error_count = 0;
        let mut topk_success_count = 0;
        let mut topk_error_count = 0;
        let mut rng = rand::thread_rng();

        for i in 0..3 {
            println!("\nUpdate iteration {}...", i + 1);
            client.update().await?;

            for name in symbols.iter() {
                let template = query_templates.choose(&mut rng).unwrap();
                let query = template.replace("{name}", name);

                // Test single query
                match client.query(&query).await {
                    Ok(result) => {
                        println!("Single query raw result: {:?}", result);
                        match decode_input(&result) {
                            Ok(output) => {
                                println!("Single query decoded output: {:?}", output);

                                if let Ok(json_output) = serde_json::from_str::<Value>(&output) {
                                    let received_name =
                                        json_output["name"].as_str().unwrap_or("").trim();

                                    if names_match(received_name, name) {
                                        single_success_count += 1;
                                        println!(
                                            "Single query matched: '{}' with '{}'",
                                            received_name, name
                                        );
                                    } else {
                                        single_error_count += 1;
                                        println!(
                                            "Single query data mismatch: Expected ({}), but got ({})",
                                            name, received_name
                                        );
                                    }
                                }
                            }
                            Err(e) => {
                                single_error_count += 1;
                                println!("Single query decoding failed: {:?}", e);
                            }
                        }
                    }
                    Err(e) => {
                        single_error_count += 1;
                        println!("Single query failed: {:?}", e)
                    }
                }

                // Test top-k query
                match client.query_top_k(&query, k).await {
                    Ok(results) => {
                        println!("Top-k query raw results: {:?}", results);
                        let mut found_match = false;
                        let mut match_position = None;

                        for (idx, result) in results.iter().enumerate() {
                            match decode_input(result) {
                                Ok(output) => {
                                    println!("Top-k decoded output {}: {:?}", idx, output);

                                    if let Ok(json_output) = serde_json::from_str::<Value>(&output)
                                    {
                                        let received_name =
                                            json_output["name"].as_str().unwrap_or("").trim();

                                        if names_match(received_name, name) {
                                            found_match = true;
                                            match_position = Some(idx);
                                            println!("Found match at position {}", idx);
                                            println!(
                                                "Matched: '{}' with '{}'",
                                                received_name, name
                                            );
                                            break;
                                        }
                                    }
                                }
                                Err(e) => {
                                    println!("Top-k decoding failed for result {}: {:?}", idx, e);
                                }
                            }
                        }

                        if found_match {
                            topk_success_count += 1;
                            println!("Successfully found match at position {:?}", match_position);
                        } else {
                            topk_error_count += 1;
                            println!("Expected name {} not found in top {} results", name, k);
                            println!(
                                "Top-k results: {:?}",
                                results.iter().map(decode_input).collect::<Vec<_>>()
                            );
                        }
                    }
                    Err(e) => {
                        topk_error_count += 1;
                        println!("Top-k query failed: {:?}", e)
                    }
                }

                // Print current stats
                let single_total = single_success_count + single_error_count;
                let topk_total = topk_success_count + topk_error_count;

                println!("\nCurrent Statistics:");
                println!(
                    "Single Query Acceptance Rate: {:.2}%",
                    if single_total > 0 {
                        (single_success_count as f64 / single_total as f64) * 100.0
                    } else {
                        0.0
                    }
                );
                println!(
                    "Top-k Query Acceptance Rate: {:.2}%",
                    if topk_total > 0 {
                        (topk_success_count as f64 / topk_total as f64) * 100.0
                    } else {
                        0.0
                    }
                );
            }
        }

        println!("\nFinal Statistics:");
        println!("Single Query:");
        println!(
            "  Total Attempts: {}",
            single_success_count + single_error_count
        );
        println!("  Successes: {}", single_success_count);
        println!("  Errors: {}", single_error_count);
        println!(
            "  Final acceptance rate: {:.2}%",
            (single_success_count as f64 / (single_success_count + single_error_count) as f64)
                * 100.0
        );

        println!("\nTop-k Query:");
        println!(
            "  Total Attempts: {}",
            topk_success_count + topk_error_count
        );
        println!("  Successes: {}", topk_success_count);
        println!("  Errors: {}", topk_error_count);
        println!(
            "  Final acceptance rate: {:.2}%",
            (topk_success_count as f64 / (topk_success_count + topk_error_count) as f64) * 100.0
        );

        Ok(())
    }
}
