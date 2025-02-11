# Optimizations

## [SimplePIR](https://github.com/0xWOLAND/simplepir) 
- Posisbly use fixed-size arithmetic instead of `BigInt`
    - I ran into overflow problems when computing the inner product of the embeddings, so this might not be possible
    - Use faster matrix-multiplication algorithms
        - Use `nalgebra`'s built-in sparse finite field matrix multiplication
            - Use `ark-ff` for finite field arithmetic
        - Rayon? -- parallel-computing based matmul
- More efficient matrix packing mechanism
    - Currently this pads the matrix of embeddings and encodings with zeros to be a square
- Batch PIR optimizations
    - Partition the database into `k` hcunks when the client wants to fetch `k` records
    - If records fall in different chunks, only one PIR query per chunk is needed

## Tiptoe
- Use a more efficient encoding algorithm (e.g. LZAQ) to compress the encoding database even more
- Cluster similar documents into clusters of size $~\sqrt{n}$. 
    - Use k-means to generate initial clusters
    - Assign documents to these clusters
        - Vertically partition the matrix $M = (M_1 || \dots || M_W)$ (subset of rows but all columns)
        - $\text{ct} = (\text{ct}_1 || \dots || \text{ct}_W)$
        - Server $i$ computes $a_i = M_i \cdot \text{ct}_i$ and the coordinator computes $a = \sum_{i = 1}^W a_i$