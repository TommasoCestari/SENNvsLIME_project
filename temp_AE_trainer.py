from senn.datasets.dataloaders import load_fashion_mnist
from senn.utils.MNIST_autoencoder import AETrainer

BATCH_SIZE = 200
train_loader, _, _ = load_fashion_mnist("datasets/data/fashion_mnist_data", BATCH_SIZE)

ae_trainer = AETrainer(train_loader, BATCH_SIZE)
ae_trainer.train(epochs=20)
ae_trainer.save_model("senn/utils/FashionMNIST_autoencoder_pretrained.pt")