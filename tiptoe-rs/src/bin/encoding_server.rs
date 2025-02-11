use anyhow::Result;
use tiptoe::{
    network::run_server,
    server::{Database, EncodingDatabase},
};

#[tokio::main]
async fn main() -> Result<()> {
    let db = EncodingDatabase::new()?;
    run_server(db, 3000).await;

    Ok(())
}
