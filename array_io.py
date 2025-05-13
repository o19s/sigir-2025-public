import re
from typing import Optional, Generator, Tuple

import numpy
import numpy as np

from file_io import Fs

# Tools for storing/loading numpy arrays (of floats)

def save_np_array(fs: Fs, np_array: numpy.ndarray, filename_prefix: str) -> str:
    """save numpy array to binary file, store dtype and shape in filename, e.g. filename_prefix_float32_32000x768.bin"""
    shape_str = "x".join(map(str, np_array.shape))
    filename = f"{filename_prefix}_{np_array.dtype}_{shape_str}.bin"
    fs.write(filename, np_array.tobytes())
    return filename

def load_np_array(fs: Fs, filename: str) -> numpy.ndarray:
    """load numpy array from binary file, reading dtype and shape from filename"""
    fn_match = re.match(r"(.+)_([a-z0-9]+)_([0-9x]+)\.bin", filename)
    if not fn_match:
        # fallback, just try loading anyway using numpy's native load method, only from local file system so far
        return np.load(filename)
    else:
        _, dtype, shape_str = fn_match.groups()
        shape = tuple(map(int, shape_str.split("x")))
        data = fs.read(filename)
        return np.frombuffer(data, dtype=np.dtype(dtype)).reshape(shape)

def batch_embeddings_loader(fs: Fs,
                            file_path_pattern: str,
                            batch_size: int,
                            max_total: Optional[int] = None) -> Generator[Tuple[int, numpy.ndarray], None, None]:
    """Load batched numpy arrays from files following a file{i} pattern (with 0-based increasing i)
       and yield them as (start_index, Tensor) tuples
    :param file_path_pattern - the path and file pattern of files to load, needs to include a {i} placeholder,
                               we'll look for all files _starting_ with the full pattern, e.g. /my/path/file_{i}_ will
                               lead to all files matching /my/path/file_0_* being processed first, then /my/path/file_1_* and so on
    :param batch_size - the (max) number of vectors to load per call (required)
    :param max_total - the maximum number of vectors to load in total (optional)
    """

    total = 0
    batch_start_idx = 0
    for embeddings_file in file_gen(fs, file_path_pattern):
        embeddings = load_np_array(fs, embeddings_file)

        for i in range(0, embeddings.shape[0], batch_size):
            adjusted_batch_size = min(batch_size, len(embeddings) - i)
            if max_total is not None:
                adjusted_batch_size = min(adjusted_batch_size, max_total - total)
            if adjusted_batch_size == 0:
                break
            embeddings_slice = embeddings[i:i+adjusted_batch_size]
            total += adjusted_batch_size
            yield batch_start_idx, embeddings_slice
            batch_start_idx += adjusted_batch_size

        if total == max_total:
            break

def largest_existing_fileidx(fs, file_path_pattern) -> Optional[int]:
    """Returns the largest existing fileidx, i.e. the largest i for which a file pattern like 'file_{i}' references
       an existing file name, returns -1 if none exists"""
    file_idx = 0
    while True:
        file_path = file_path_pattern.replace('{i}', str(file_idx))
        if next(fs.glob(file_path), None) is None:
            return file_idx - 1
        else:
            file_idx += 1

def file_gen(fs, file_path_pattern) -> Generator[str, None, None]:
    file_idx = 0
    while True:
        file_path = file_path_pattern.replace('{i}', str(file_idx))
        files_matched = list(fs.glob(file_path))
        if len(files_matched) == 0:
            break
        for filename in files_matched:
            yield filename
        file_idx += 1
