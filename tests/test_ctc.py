import torch


def test_ctc_mnist():
    from ctc import mnist_ctc

    ct = mnist_ctc()
    ct.eval()
    x = torch.randn(16, 1, 28, 28)
    assert ct(x)[0].shape == torch.Size([x.shape[0], 2])


def test_cvit_cub():
    from ctc import cub_cvit

    ct = cub_cvit()
    ct.eval()
    x = torch.randn(8, 3, 224, 224)
    assert ct(x)[0].shape == torch.Size([x.shape[0], 200]), "Error CUB has 200 output classes"
