import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='./saved')
    args = parser.parse_args()
    log_dir = args.save_dir

    log_train_loss = np.load(os.path.join(log_dir, 'log_train_loss.npy'))
    log_train_acc  = np.load(os.path.join(log_dir, 'log_train_acc.npy'))
    log_test_loss  = np.load(os.path.join(log_dir, 'log_test_loss.npy'))
    log_test_acc   = np.load(os.path.join(log_dir, 'log_test_acc.npy'))

    n_epochs = len(log_train_loss)

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(np.arange(1, n_epochs + 1), log_train_loss)
    plt.plot(np.arange(1, n_epochs + 1), log_test_loss)
    plt.title("Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Test'], loc="upper right")

    plt.subplot(1, 2, 2)
    plt.plot(np.arange(1, n_epochs + 1), log_train_acc)
    plt.plot(np.arange(1, n_epochs + 1), log_test_acc)
    plt.title("Accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.xlim([0, n_epochs])
    plt.legend(['Train', 'Test'], loc="lower right")

    os.makedirs(log_dir, exist_ok=True)
    plt.savefig(os.path.join(log_dir, "training_curves.png"), dpi=300, bbox_inches="tight")
    plt.close()
