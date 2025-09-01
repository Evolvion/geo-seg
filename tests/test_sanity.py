def test_imports():
    import torch
    import numpy

    assert torch.__version__ is not None
    assert numpy.__version__ is not None
