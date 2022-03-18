import torch


def test_mnist():
    from data import explanation_mnist_dataset

    n_train_samples = 100
    trainset, valset, testset = explanation_mnist_dataset(n_train_samples=n_train_samples)

    assert len(trainset) == n_train_samples, "Error in number of training samples"
    assert len(valset) == 5000, "Error in number of training samples"
    assert len(testset) == 10000, "Error in number of training samples"

    for i in range(100):
        image, explanation, _, label = trainset[i]
        parity = torch.nonzero(explanation).item() % 2
        assert label == parity, "Error: explanation parity does not correspond to label"

    # Test DataLoader
    dl = torch.utils.data.DataLoader(trainset, batch_size=32)
    x, ex, _, y = next(iter(dl))

    assert ex.shape == torch.Size([32, 10]), "Explanations are malformed"
    assert y.shape == torch.Size([32]), "Labels are malformed"


def test_tmnist():
    from data import get_cluttered_datasets

    trainset, valset, testset, _ = get_cluttered_datasets()

    assert len(trainset) == 55000, "Error in number of training samples"
    assert len(valset) == 5000, "Error in number of training samples"
    assert len(testset) == 10000, "Error in number of training samples"
