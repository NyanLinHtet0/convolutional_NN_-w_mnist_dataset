import time
import numpy as np
import matplotlib.pyplot as plt

from mycnn.parallel import CNNMultiCore


def build_confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=int)

    for true_label, pred_label in zip(y_true, y_pred):
        matrix[int(true_label), int(pred_label)] += 1

    return matrix


def compute_per_class_accuracy(conf_matrix):
    per_class_accuracy = []

    for class_idx in range(conf_matrix.shape[0]):
        total_for_class = np.sum(conf_matrix[class_idx, :])

        if total_for_class == 0:
            per_class_accuracy.append(0.0)
        else:
            correct_for_class = conf_matrix[class_idx, class_idx]
            per_class_accuracy.append(correct_for_class / total_for_class)

    return np.array(per_class_accuracy)


def plot_wrong_predictions(x_test, y_test, predictions, class_target=0, max_show=10):
    wrong_target_indices = []

    for i in range(len(x_test)):
        if y_test[i] == class_target and predictions[i] != y_test[i]:
            wrong_target_indices.append(i)

    wrong_to_show = wrong_target_indices[:max_show]

    if len(wrong_to_show) == 0:
        print(f"No wrong predictions found for class {class_target}.")
        return

    rows = int(np.ceil(len(wrong_to_show) / 5))
    cols = min(5, len(wrong_to_show))

    plt.figure(figsize=(2 * cols, 2.5 * rows))

    for j, i in enumerate(wrong_to_show):
        plt.subplot(rows, 5, j + 1)
        plt.imshow(x_test[i], cmap="gray")
        plt.title(f"T:{y_test[i]} P:{predictions[i]}")
        plt.axis("off")

    plt.suptitle(f"Wrong Predictions for Class {class_target}")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(conf_matrix):
    plt.figure(figsize=(6, 5))
    plt.imshow(conf_matrix, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar()

    num_classes = conf_matrix.shape[0]
    plt.xticks(np.arange(num_classes))
    plt.yticks(np.arange(num_classes))

    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(
                j,
                i,
                str(conf_matrix[i, j]),
                ha="center",
                va="center",
                color="black",
            )

    plt.tight_layout()
    plt.show()


def plot_loss_curves(epoch_loss_history, batch_loss_history):
    plt.figure(figsize=(8, 4))
    plt.plot(epoch_loss_history, label="Epoch Loss")
    plt.plot(batch_loss_history, label="Batch Loss", alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    np.random.seed(3)

    output_class = 10
    image_inputsize = (20, 20)
    kernel_shape = (3, 3)
    pool_size = (2, 2)
    stride = (2, 2)

    widths = [32, 64]
    hidden_layerwidths = [128, 64]

    epochs = 10
    learning_rate = 0.005
    mini_batch_size = 64
    class_target = 0

    train_path = (
        f"train_{image_inputsize[0]}x{image_inputsize[1]}_dataset_ysize={output_class}.npz"
    )
    test_path = (
        f"test_{image_inputsize[0]}x{image_inputsize[1]}_dataset_ysize={output_class}.npz"
    )

    train_data = np.load(train_path)
    x_train = train_data["images"]
    y_train = train_data["labels"]

    output_size = (int(y_train.max()) + 1, 1)
    num_classes = output_size[0]
    
    cnn_multicore = CNNMultiCore(
        image_inputsize=image_inputsize,
        output_size=output_size,
        kernel_shape=kernel_shape,
        widths=widths,
        hidden_layerwidths= hidden_layerwidths,
        pool_size=pool_size,
        stride=stride,
    )
    save_path = (
        f"trained_parameters_widths={widths}_fc={hidden_layerwidths}_"
        f"input{image_inputsize}_ysize={output_class}.npz"
    )
    cnn_multicore.load_parameters(save_path)
#------------------------------------------------------------------------------------------------------------------------------------#

    start_time = time.perf_counter()

    epoch_loss_history, batch_loss_history = cnn_multicore.train_batches(
        x_train=x_train,
        y_train=y_train,
        epochs=epochs,
        mini_batch_size=mini_batch_size,
        learning_rate=learning_rate,
    )
    end_time = time.perf_counter()

    print(f"Training Time: {end_time - start_time:.4f} seconds")
    cnn_multicore.save_parameters(save_path)

#-------------------------------------------------------------------------------------------------------------------------------------#

    cnn_multicore.load_parameters(save_path)

    test_data = np.load(test_path)
    x_test = test_data["images"]
    y_test = test_data["labels"]

    predictions = []
    num_correct = 0
    total = len(x_test)

    for i in range(total):
        pred_label = int(cnn_multicore.predict(x_test[i]))
        predictions.append(pred_label)

        if pred_label == int(y_test[i]):
            num_correct += 1

    predictions = np.array(predictions, dtype=int)
    accuracy = num_correct / total

    print(f"\nCorrect Predictions: {num_correct}/{total}")
    print(f"Accuracy: {accuracy:.4%}")

    conf_matrix = build_confusion_matrix(y_test, predictions, num_classes)
    per_class_accuracy = compute_per_class_accuracy(conf_matrix)

    print("\nPer-Class Accuracy:")
    for class_idx, class_acc in enumerate(per_class_accuracy):
        class_total = np.sum(conf_matrix[class_idx, :])
        class_correct = conf_matrix[class_idx, class_idx]
        print(
            f"Class {class_idx}: {class_correct}/{class_total} "
            f"({class_acc:.4%})"
        )

    print("\nConfusion Matrix:")
    print(conf_matrix)

    plot_wrong_predictions(
        x_test=x_test,
        y_test=y_test,
        predictions=predictions,
        class_target=class_target,
        max_show=10,
    )

    plot_confusion_matrix(conf_matrix)
    plot_loss_curves(epoch_loss_history, batch_loss_history)


if __name__ == "__main__":
    main()