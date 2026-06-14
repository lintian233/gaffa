import gaffa
import pytest


def test_vector_add_empty() -> None:
    assert gaffa.vector_add([], []) == []


def test_cuda_runtime_version() -> None:
    assert gaffa.cuda_runtime_version() >= 12000

def test_vector_add_cuda_kernel() -> None:
    assert gaffa.vector_add([1.0, 2.5, -3.0], [2.0, 0.5, 3.0]) == [3.0, 3.0, 0.0]
