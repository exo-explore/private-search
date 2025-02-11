use anyhow::Result;
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::ops::bytes::ToBytes;

use crate::error::PirError;

pub fn encode_input(text: &str) -> Result<DVector<u64>> {
    let bytes = text.as_bytes();
    let tmp = bytes
        .chunks(8)
        .map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            u64::from_le_bytes(buf)
        })
        .collect::<Vec<u64>>();
    Ok(DVector::from_vec(tmp))
}

#[allow(dead_code)]
pub fn decode_input(data: &DVector<BigInt>) -> Result<String> {
    let bytes = data
        .iter()
        .flat_map(|x| x.to_le_bytes())
        .collect::<Vec<u8>>();

    let s = String::from_utf8(bytes)?;
    Ok(s.replace('\0', ""))
}

pub fn encode_data(data: &[String]) -> Result<DMatrix<BigInt>> {
    let max_length = data
        .iter()
        .map(|text| text.len())
        .max()
        .ok_or_else(|| PirError::InvalidInput("Empty data vector".to_string()))?;

    let padded_data = data
        .iter()
        .map(|text| {
            let mut padded = text.clone();
            while padded.len() < max_length {
                padded.push('\0');
            }
            padded
        })
        .collect::<Vec<String>>();

    let embeddings = padded_data
        .iter()
        .map(|text| encode_input(text))
        .collect::<Result<Vec<_>>>()?;

    let num_embeddings = embeddings.len();
    let embedding_size = embeddings[0].len();
    let square_size = std::cmp::max(num_embeddings, embedding_size);

    let mut square_matrix = DMatrix::zeros(square_size, square_size);
    for (i, embedding) in embeddings.iter().enumerate() {
        for (j, &value) in embedding.iter().enumerate() {
            square_matrix[(j, i)] = BigInt::from(value);
        }
    }

    Ok(square_matrix)
}

#[allow(dead_code)]
pub fn decode_data(data: &DMatrix<BigInt>) -> Result<Vec<String>> {
    data.column_iter()
        .map(|row| decode_input(&row.into_owned()).map(|s| s.trim_end_matches('\0').to_string()))
        .filter(|r| r.as_ref().map_or(false, |s| !s.is_empty()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_decode() -> Result<()> {
        let data = vec![
            "Lorem ipsum odor amet, consectetuer adipiscing elit".to_string(),
            "Conubia elementum taciti dapibus vestibulum mattis primis".to_string(),
            "Facilisis fames justo ultricies pharetra rhoncus".to_string(),
            "Nam vel mi aptent turpis purus fusce purus".to_string(),
            "Pretium ultrices torquent vulputate venenatis magnis vitae tempor semper torquent"
                .to_string(),
            "Habitant suspendisse nascetur in quis adipiscing".to_string(),
        ];

        let encoded = encode_data(&data)?;
        let decoded = decode_data(&encoded)?;

        assert_eq!(data, decoded);
        Ok(())
    }
}
