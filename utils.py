import math
import numpy as np
import zlib
from typing import List, Tuple

def string_to_numbers(s: str) -> List[int]:
    """Convert a string into a list of numbers using zlib compression"""
    byte_array = s.encode('utf-8')
    compressed = zlib.compress(byte_array)
    numbers = [int(b) for b in compressed]
    return numbers

def numbers_to_string(numbers: List[int]) -> str:
    """Convert a list of numbers back into a string using zlib decompression"""
    byte_array = bytes(numbers)
    try:
        decompressed = zlib.decompress(byte_array)
        return decompressed.decode('utf-8')
    except zlib.error:
        return ""

def strings_to_matrix(strings: List[str]) -> Tuple[np.ndarray, int]:
    """Convert a list of strings into a square matrix suitable for SimplePIR"""
    number_lists = [string_to_numbers(s) for s in strings]
    max_len = max(len(nums) for nums in number_lists)
    max_width = max_len + 1
    
    total_elements = len(strings) * max_width
    matrix_size = math.ceil(math.sqrt(total_elements))
    
    if matrix_size < max_width:
        matrix_size = max_width
    
    matrix = np.zeros((matrix_size, matrix_size), dtype=np.int64)
    
    for i, numbers in enumerate(number_lists):
        matrix[i, 0] = len(numbers)
        matrix[i, 1:len(numbers)+1] = numbers
    
    return matrix, matrix_size

def matrix_to_strings(matrix: np.ndarray, num_strings: int) -> List[str]:
    """Convert a matrix back into a list of strings"""
    strings = []
    
    for i in range(num_strings):
        length = int(matrix[i, 0])
        if length > 0:
            numbers = matrix[i, 1:length+1].astype(np.int64).tolist()
            try:
                strings.append(numbers_to_string(numbers))
            except:
                strings.append("")
        else:
            strings.append("")
    
    return strings

def test_string_compression():
    """Test the string compression and matrix conversion functions"""
    test_string = "Hello, World!"
    numbers = string_to_numbers(test_string)
    recovered = numbers_to_string(numbers)
    assert test_string == recovered, f"String compression failed: {test_string} != {recovered}"
    
    test_strings = [
        "Hello, World!",
        "This is a test", 
        "PIR is cool",
        "Multiple strings",
        "Some longer string that might need more compression",
        "Short one"
    ]
    
    matrix, size = strings_to_matrix(test_strings)
    recovered_strings = matrix_to_strings(matrix, len(test_strings))
    
    assert test_strings == recovered_strings, f"Matrix conversion failed"
    print("All tests passed successfully!")

if __name__ == "__main__":
    test_string_compression()
