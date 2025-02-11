use anyhow::Result;
use async_trait::async_trait;
use axum::{
    extract::State,
    Json, Router,
};
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::One;
use reqwest::Client as HttpClient;
use serde::{Deserialize, Serialize};
use simplepir::{gen_params, generate_query, recover, SimplePIRParams};
use std::{str::FromStr, sync::Arc, time::Duration};
use tokio::sync::RwLock;

use crate::{embedding::BertEmbedder, server::Database};

// Shared state for server
pub struct ServerState<T: Database + Send + Sync> {
    db: RwLock<T>,
}

// Request/Response types
#[derive(Serialize, Deserialize)]
pub struct QueryRequest {
    query: Vec<String>, // Serialized BigInt vector
}

#[derive(Serialize, Deserialize)]
pub struct QueryResponse {
    response: Vec<String>,
}

#[derive(Serialize, Deserialize)]
pub struct ParamsData {
    m: usize,
    n: usize,
    q: String,
    p: String,
}

#[derive(Serialize, Deserialize)]
pub struct MatrixResponse {
    rows: usize,
    cols: usize,
    data: Vec<String>,
}

// Helper functions for serialization
fn serialize_vector(vec: &DVector<BigInt>) -> Vec<String> {
    vec.iter().map(|x| x.to_string()).collect()
}

fn deserialize_vector(vec: &[String]) -> DVector<BigInt> {
    let values: Vec<BigInt> = vec.iter().map(|x| x.parse().unwrap()).collect();
    DVector::from_vec(values)
}

fn serialize_matrix(matrix: &DMatrix<BigInt>) -> MatrixResponse {
    MatrixResponse {
        rows: matrix.nrows(),
        cols: matrix.ncols(),
        data: matrix.iter().map(|x| x.to_string()).collect(),
    }
}

fn deserialize_matrix(response: &MatrixResponse) -> DMatrix<BigInt> {
    let data: Vec<BigInt> = response.data.iter().map(|x| x.parse().unwrap()).collect();
    DMatrix::from_vec(response.rows, response.cols, data)
}

fn serialize_params(params: &SimplePIRParams) -> ParamsData {
    ParamsData {
        m: params.m,
        n: params.n,
        q: params.q.to_string(),
        p: params.p.to_string(),
    }
}

fn deserialize_params(data: &ParamsData) -> SimplePIRParams {
    let p = BigInt::from_str(&data.p).unwrap();
    let mod_power = (p.bits() - 1) as u32;
    gen_params(data.m, data.n, mod_power)
}

pub async fn run_server<T: Database + Send + Sync + 'static>(db: T, port: u16) {
    let state = Arc::new(ServerState {
        db: RwLock::new(db),
    });

    let update_state = Arc::clone(&state);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(Duration::from_secs(15));
        loop {
            interval.tick().await;
            println!("Starting database update...");

            let new_db = match tokio::task::spawn_blocking(|| {
                T::new().and_then(|mut instance| {
                    instance.update()?;
                    Ok(instance)
                })
            })
            .await
            {
                Ok(Ok(new_instance)) => new_instance,
                Ok(Err(e)) => {
                    eprintln!("Error building new database: {:?}", e);
                    continue;
                }
                Err(e) => {
                    eprintln!("Blocking task panicked: {:?}", e);
                    continue;
                }
            };

            {
                let mut db_lock = update_state.db.write().await;
                *db_lock = new_db;
            }
            println!("Database update complete!");
        }
    });

    let app = Router::new()
        .route("/query", axum::routing::post(handle_query::<T>))
        .route("/params", axum::routing::get(handle_params::<T>))
        .route("/hint", axum::routing::get(handle_hint::<T>))
        .route("/a", axum::routing::get(handle_a::<T>))
        .with_state(state);

    let addr = format!("0.0.0.0:{}", port).parse().unwrap();
    println!("Starting server on {}", addr);

    axum_server::bind(addr)
        .serve(app.into_make_service())
        .await
        .unwrap();
}

async fn handle_query<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
    Json(request): Json<QueryRequest>,
) -> Json<QueryResponse> {
    let query = deserialize_vector(&request.query);
    let db = state.db.read().await;
    let response = db.respond(&query).unwrap();
    Json(QueryResponse {
        response: serialize_vector(&response),
    })
}

async fn handle_params<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> Json<ParamsData> {
    let db = state.db.read().await;
    Json(serialize_params(db.params()))
}

async fn handle_hint<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> Json<MatrixResponse> {
    let db = state.db.read().await;
    Json(serialize_matrix(db.hint()))
}

async fn handle_a<T: Database + Send + Sync>(
    State(state): State<Arc<ServerState<T>>>,
) -> Json<MatrixResponse> {
    let db = state.db.read().await;
    Json(serialize_matrix(db.a()))
}

// Remote database implementation that connects to server
#[async_trait]
pub trait AsyncDatabase {
    async fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>>;
    async fn get_params(&self) -> Result<SimplePIRParams>;
    async fn get_hint(&self) -> Result<DMatrix<BigInt>>;
    async fn get_a(&self) -> Result<DMatrix<BigInt>>;
}

pub struct RemoteDatabase {
    client: HttpClient,
    base_url: String,
}

impl RemoteDatabase {
    pub fn new(base_url: String) -> Self {
        Self {
            client: HttpClient::builder().build().unwrap(),
            base_url,
        }
    }
}

#[async_trait]
impl AsyncDatabase for RemoteDatabase {
    async fn respond(&self, query: &DVector<BigInt>) -> Result<DVector<BigInt>> {
        let response: QueryResponse = self
            .client
            .post(format!("{}/query", self.base_url))
            .json(&QueryRequest {
                query: serialize_vector(query),
            })
            .send()
            .await?
            .json()
            .await?;

        Ok(deserialize_vector(&response.response))
    }

    async fn get_params(&self) -> Result<SimplePIRParams> {
        let response: ParamsData = self
            .client
            .get(format!("{}/params", self.base_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(deserialize_params(&response))
    }

    async fn get_hint(&self) -> Result<DMatrix<BigInt>> {
        let response: MatrixResponse = self
            .client
            .get(format!("{}/hint", self.base_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(deserialize_matrix(&response))
    }

    async fn get_a(&self) -> Result<DMatrix<BigInt>> {
        let response: MatrixResponse = self
            .client
            .get(format!("{}/a", self.base_url))
            .send()
            .await?
            .json()
            .await?;
        Ok(deserialize_matrix(&response))
    }
}

// Network client implementation
pub struct NetworkClient {
    embedder: BertEmbedder,
    embedding_db: RemoteDatabase,
    encoding_db: RemoteDatabase,
}

impl NetworkClient {
    pub fn new(embedding_url: String, encoding_url: String) -> Result<Self> {
        Ok(Self {
            embedder: BertEmbedder::new()?,
            embedding_db: RemoteDatabase::new(embedding_url),
            encoding_db: RemoteDatabase::new(encoding_url),
        })
    }

    fn adjust_embedding(embedding: DVector<BigInt>, m: usize) -> DVector<BigInt> {
        match embedding.len().cmp(&m) {
            std::cmp::Ordering::Equal => embedding,
            std::cmp::Ordering::Less => {
                let mut new_embedding = DVector::zeros(m);
                new_embedding
                    .rows_mut(0, embedding.len())
                    .copy_from(&embedding);
                new_embedding
            }
            std::cmp::Ordering::Greater => embedding.rows(0, m).into(),
        }
    }

    pub async fn query(&self, query: &str) -> Result<DVector<BigInt>> {
        let embedding = self.embedder.embed_text(query)?;

        let embedding_params = self.embedding_db.get_params().await?;
        let adjusted_embedding = Self::adjust_embedding(embedding, embedding_params.m);
        let (s_embedding, query_embedding) = generate_query(
            &embedding_params,
            &adjusted_embedding,
            &self.embedding_db.get_a().await?,
        );

        let response_embedding = self.embedding_db.respond(&query_embedding).await?;
        let result_embedding = recover(
            &self.embedding_db.get_hint().await?,
            &s_embedding,
            &response_embedding,
            &embedding_params,
        );

        let result_vec = {
            let mut vec = DVector::zeros(result_embedding.len());
            let max_idx = result_embedding
                .iter()
                .enumerate()
                .max_by_key(|(_i, val)| (*val).clone())
                .map(|(i, _val)| i)
                .unwrap();
            vec[max_idx] = BigInt::one();
            vec
        };

        let encoding_params = self.encoding_db.get_params().await?;
        let adjusted_result = Self::adjust_embedding(result_vec, encoding_params.m);
        let (s, query) = generate_query(
            &encoding_params,
            &adjusted_result,
            &self.encoding_db.get_a().await?,
        );

        let response = self.encoding_db.respond(&query).await?;
        let result = recover(
            &self.encoding_db.get_hint().await?,
            &s,
            &response,
            &encoding_params,
        );

        Ok(result)
    }
}
