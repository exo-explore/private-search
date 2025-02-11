use anyhow::Result;
use tiptoe::{
    network::run_server,
    server::{Database, EmbeddingDatabase},
};

#[tokio::main]
async fn main() -> Result<()> {
    let db = EmbeddingDatabase::new()?;
    run_server(db, 3001).await;
    Ok(())
}
