import numpy as np
from scipy import sparse, linalg
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SimplePIRParams:
    n: int
    m: int
    q: int
    p: int
    std_dev: float
    a: np.ndarray

def gen_params(m: int, n: int = 2048, mod_power: int = 17) -> SimplePIRParams:
    q = np.uint64(2**64 - 1)
    p = np.uint64(2**mod_power)
    std_dev = 3.2
    a = np.random.randint(0, q, size=(m, n), dtype=np.uint64)
    
    return SimplePIRParams(n=n, m=m, q=int(q), p=int(p), std_dev=std_dev, a=a)

def gen_secret(q: int, n: int, seed: Optional[int] = None) -> np.ndarray:
    if seed is not None:
        np.random.seed(seed)
    return np.random.randint(0, q, size=n, dtype=np.uint64)

def gen_hint(params: SimplePIRParams, db: np.ndarray) -> np.ndarray:
    q = int(params.q)
    p = int(params.p)
    
    db_offset = db.astype(np.uint64)
    offset = (q * (p // 2)) % q
    db_offset = (db_offset + offset) % q
    
    result = (params.a.T.astype(np.float64) @ db_offset.astype(np.float64)) % q
    
    return result.astype(np.uint64)

def gauss_sample(std_dev: float = 3.2) -> int:
    return int(np.random.normal(0, std_dev)) % 8

def encrypt(secret: np.ndarray, a_matrix: np.ndarray, bits: np.ndarray, q: int, p: int) -> Tuple[np.ndarray, np.ndarray]:
    q = np.uint64(q)
    p = np.uint64(p)
    q_over_p = q // p
    
    error = np.array([gauss_sample() for _ in range(len(bits))], dtype=np.uint64)
    
    b = linalg.blas.dgemv(alpha=1.0,
                         a=a_matrix.astype(np.float64),
                         x=secret.astype(np.float64)) % q
    
    b = (b.astype(np.uint64) + error + bits * q_over_p) % q
    
    return a_matrix, b

def query(index: int, db_size: int, params: SimplePIRParams) -> Tuple[np.ndarray, np.ndarray]:
    query_vector = np.zeros(db_size, dtype=np.uint64)
    query_vector[index] = 1
    
    secret = gen_secret(params.q, params.n)
    _, query_cipher = encrypt(secret, params.a, query_vector, params.q, params.p)
    
    return secret, query_cipher

def answer(query_cipher: np.ndarray, db: np.ndarray, q: int) -> np.ndarray:
    q = int(q)
    
    # Transpose db to work with rows
    db = db.T
    
    result = linalg.blas.dgemv(alpha=1.0,
                              a=db.astype(np.float64),
                              x=query_cipher.astype(np.float64)) % q
    
    return result.astype(np.uint64)

def recover(secret: np.ndarray, hint: np.ndarray, answer_cipher: np.ndarray, 
           query_cipher: np.ndarray, params: SimplePIRParams) -> int:
    ciphertext_mod = int(params.q)
    plaintext_mod = int(params.p)
    q_over_p = ciphertext_mod // plaintext_mod
    
    if answer_cipher.size == 1:
        answer_scalar = int(answer_cipher.item())
    else:
        answer_scalar = int(answer_cipher[0])
    
    ratio = plaintext_mod // 2
    query_sum = linalg.blas.dasum(query_cipher.astype(np.float64))
    hint_term = int(linalg.blas.ddot(secret.astype(np.float64), 
                                    hint.astype(np.float64)) % ciphertext_mod)
    
    noised = (answer_scalar - ratio * int(query_sum) - hint_term) % ciphertext_mod
    denoised = ((noised + q_over_p // 2) // q_over_p) % ciphertext_mod
    
    return (denoised - ratio) % plaintext_mod

def recover_row(secret: np.ndarray, hint: np.ndarray, answer_cipher: np.ndarray, 
               query_cipher: np.ndarray, params: SimplePIRParams) -> np.ndarray:
    records = []
    for i in range(hint.shape[1]):  # Iterate over columns
        record = recover(secret, hint[:, i], np.array([answer_cipher[i]]), 
                       query_cipher, params)
        records.append(record)
    return np.array(records)

if __name__ == "__main__":
    # Test with increasing matrix sizes
    for size in [8, 16, 32, 64]:
        print(f"\nTesting {size}x{size} matrix...")
        
        # Generate random test database with entries between 0 and 255
        test_db = np.random.randint(0, 256, size=(size, size), dtype=np.uint64)
        
        # Generate parameters and hint
        params = gen_params(m=size, mod_power=17)
        hint = gen_hint(params, test_db)
        
        # Test random rows
        num_rows_to_test = min(5, size)  # Test up to 5 random rows
        test_rows = np.random.choice(size, num_rows_to_test, replace=False)
        
        for row_idx in test_rows:
            secret, query_cipher = query(row_idx, size, params)
            ans = answer(query_cipher, test_db, params.q)
            
            # Recover entire row
            recovered_row = recover_row(secret, hint, ans, query_cipher, params)
            true_row = test_db[row_idx]
            
            print(f"\nRow {row_idx}:")
            print(f"Recovered: {recovered_row[:5]}...")  # Show first 5 elements
            print(f"True:      {true_row[:5]}...")
            assert np.array_equal(recovered_row, true_row), f"Mismatch at row {row_idx}"
            
        print(f"All tests passed for {size}x{size} matrix!")
    
    print("\nAll matrix size tests completed successfully!")
