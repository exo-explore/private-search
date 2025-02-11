use anyhow::{Error as E, Result};
use candle::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config, DTYPE};
use hf_hub::{api::sync::Api, Repo, RepoType};
use nalgebra::{DMatrix, DVector};
use num_bigint::BigInt;
use num_traits::One;
use serde_json::Value;
use tokenizers::Tokenizer;

pub struct BertEmbedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
}

impl BertEmbedder {
    pub fn new() -> Result<Self> {
        let device = Device::cuda_if_available(0)?;
        let model_id = "sentence-transformers/all-MiniLM-L6-v2".to_string();
        let revision = "refs/pr/21".to_string();

        let repo = Repo::with_revision(model_id, RepoType::Model, revision);
        let (config_filename, tokenizer_filename, weights_filename) = {
            let api = Api::new()?;
            let api = api.repo(repo);
            let config = api.get("config.json")?;
            let tokenizer = api.get("tokenizer.json")?;
            let weights = api.get("model.safetensors")?;
            (config, tokenizer, weights)
        };

        let config = std::fs::read_to_string(config_filename)?;
        let config: Config = serde_json::from_str(&config)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], DTYPE, &device)? };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn normalize_l2(&self, v: &Tensor) -> Result<Tensor> {
        Ok(v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)?)
    }

    fn embedding_to_bigint(&self, embeddings: &Tensor) -> Result<DVector<BigInt>> {
        let embeddings = embeddings.squeeze(0)?;
        let values = embeddings.to_vec1::<f32>()?;

        let quantized: Vec<BigInt> = values.iter().map(|&x| f32_to_bigint(x)).collect();

        Ok(DVector::from_vec(quantized))
    }

    pub fn embed_json_array(&self, json: &[Value]) -> Result<DMatrix<BigInt>> {
        let embeddings = json
            .iter()
            .map(|v| self.embed_text(&v.to_string()))
            .collect::<Result<Vec<_>>>()?;

        let dim = std::cmp::max(embeddings[0].nrows(), embeddings.len());
        let mut out = DMatrix::zeros(dim, dim);

        for (i, embedding) in embeddings.iter().enumerate() {
            out.row_mut(i).copy_from_slice(embedding.as_slice());
        }

        Ok(out)
    }

    pub fn embed_text(&self, text: &str) -> Result<DVector<BigInt>> {
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();

        let token_ids = Tensor::new(&tokens[..], &self.device)?.unsqueeze(0)?;
        let token_type_ids = token_ids.zeros_like()?;

        let embeddings = self.model.forward(&token_ids, &token_type_ids)?;

        let (_n_sentence, n_tokens, _hidden_size) = embeddings.dims3()?;
        let embeddings = (embeddings.sum(1)? / (n_tokens as f64))?;

        let embeddings = self.normalize_l2(&embeddings)?;

        self.embedding_to_bigint(&embeddings)
    }
}

fn f32_to_bigint(value: f32) -> BigInt {
    if value.is_nan() || value.is_infinite() {
        panic!("Cannot convert NaN or infinite values to BigInt");
    }

    let (mantissa, exponent, sign) = {
        let bits = value.to_bits(); // Get raw IEEE 754 representation
        let sign = if bits >> 31 == 1 { -1 } else { 1 };
        let exponent = ((bits >> 23) & 0xFF) as i32 - 127; // Unbiased exponent
        let mantissa = (bits & 0x7FFFFF) | 0x800000; // Add implicit leading 1
        (mantissa, exponent, sign)
    };

    let mut big_mantissa = BigInt::from(mantissa);

    if exponent >= 0 {
        big_mantissa <<= exponent as usize; // Multiply by 2^exponent
    } else {
        let denominator = BigInt::one() << (-exponent as usize); // Divide by 2^(-exponent)
        big_mantissa /= denominator;
    }

    if sign == -1 {
        big_mantissa = -big_mantissa;
    }

    big_mantissa
}

#[cfg(test)]
mod tests {
    use num_traits::One;
    use simplepir::{gen_hint, gen_params, generate_query, process_query, recover};

    use crate::utils::{decode_input, encode_data};

    use super::*;

    #[test]
    fn test_embedding_shape() -> Result<()> {
        let embedder = BertEmbedder::new()?;
        let embedding = embedder.embed_text("test text")?;

        assert_eq!(embedding.nrows(), 384);
        Ok(())
    }

    #[test]
    fn test_embedding() {
        let expected_idx = 0;

        let text = [
            "Lorem ipsum odor amet, consectetuer adipiscing elit",
            " Conubia elementum taciti dapibus vestibulum mattis primis",
            " Facilisis fames justo ultricies pharetra rhoncus",
            " Nam vel mi aptent turpis purus fusce purus",
            " Pretium ultrices torquent vulputate venenatis magnis vitae tempor semper torquent",
            " Habitant suspendisse nascetur in quis adipiscing",
        ];

        let d = encode_data(text.map(|x| x.to_string()).as_ref())
            .unwrap()
            .transpose();
        let matrix_height = d.nrows();
        let mut v = DVector::zeros(matrix_height);
        v[expected_idx] = BigInt::one();

        // Expected result
        let expected = {
            let mut result = DVector::zeros(matrix_height);
            for i in 0..matrix_height {
                result[i] = d[(i, expected_idx)].clone();
            }
            result
        };

        let params = gen_params(matrix_height, 2048, 64);
        let (hint, a) = gen_hint(&params, &d);
        let (s, query) = generate_query(&params, &v, &a);
        let answer = process_query(&d, &query, params.q.clone());
        let result = recover(&hint, &s, &answer, &params);

        println!("result: {:?}", result);
        println!("expected: {:?}", expected);
        println!("decoded: {:?}", decode_input(&result));
        println!("decoded expected: {:?}", decode_input(&expected));
    }
}
