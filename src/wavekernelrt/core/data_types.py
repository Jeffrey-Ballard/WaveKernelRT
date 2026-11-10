import numpy as np
import numpy.typing as npt
from typing import TypeAlias

RealArray: TypeAlias = npt.NDArray[np.float32]
ComplexArray: TypeAlias = npt.NDArray[np.complex64]
NumericArray: TypeAlias = RealArray | ComplexArray

DTYPE_TO_CODE = {np.dtype("uint8"): 1,np.dtype("uint16"): 2,np.dtype("uint32"): 3,np.dtype("uint64"): 4,
    np.dtype("int8"): 5,np.dtype("int16"): 6,np.dtype("int32"): 7,np.dtype("int64"): 8,np.dtype("float16"): 9,
    np.dtype("float32"): 10,np.dtype("float64"): 11,np.dtype("complex64"): 12,np.dtype("complex128"): 13,
    np.dtype("bool"): 14}

DTYPE_TO_BYTES = {np.dtype("uint8"): 1,np.dtype("uint16"): 2,np.dtype("uint32"): 4,np.dtype("uint64"): 8,
    np.dtype("int8"): 1,np.dtype("int16"): 2,np.dtype("int32"): 4,np.dtype("int64"): 8,np.dtype("float16"): 2,
    np.dtype("float32"): 4,np.dtype("float64"): 8,np.dtype("complex64"): 8,np.dtype("complex128"): 16,
    np.dtype("bool"): 1}
CODE_TO_DTYPE = {code: dtype for dtype,code in DTYPE_TO_CODE.items()}

def dtype_to_code(dtype) -> int:
    return DTYPE_TO_CODE[np.dtype(dtype)]

def code_to_dtype(code: int) -> np.dtype:
    return CODE_TO_DTYPE[code]

def dtype_to_bytes(dtype) -> int:
    return DTYPE_TO_BYTES[np.dtype(dtype)]