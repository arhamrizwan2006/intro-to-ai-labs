# Lab 08 - MNIST digit recognition (CNN)

**Data:** Kaggle Digit Recognizer, 42,000 labelled images (classes balanced, 3,795-4,684 each), 28,000 unlabelled test images.
Split 80/20 stratified (33,600 train / 8,400 validation), pixels scaled to 0-1.

**Model:** Conv32 -> MaxPool -> Conv64 -> MaxPool -> Flatten -> Dense128 -> Dropout 0.3 -> Softmax10, 225,034 parameters, Adam, 10 epochs, batch 64.

## Results (retrained here, seed 42)
| | Train acc | Validation acc |
|---|---|---|
| Epoch 1 | 92.3% | 97.7% |
| Epoch 10 | 99.4% | 98.7% |

- Lowest validation loss at epoch 7 (0.036). After that train accuracy keeps rising while validation loss creeps up (0.036 -> 0.044), a mild sign of overfitting, so 6-8 epochs would be enough.
- Weakest classes: 8 (recall 96.9%) and 9 (precision 97.4%). Most common error: true 4 predicted as 9 (11 images).
- The model you saved earlier (`mnist_cnn_model_submitted.keras`, same architecture) scores 98.93% on the same validation images; the two models agree on 98.8% of them. The small gap is ordinary run-to-run variation.
- On the 28,000 test images, 615 predictions have confidence below 90%.
- Your handwritten 5 (`images/handwritten_5.png`) is predicted as **5** with ~100% confidence after inverting, cropping and centring it MNIST-style. Without that conversion (dark ink on white) the model would be looking at the inverse of what it was trained on.

## Files
`digits_recognition.ipynb` (executed), `mnist_cnn_model.keras` (retrained), `mnist_cnn_model_submitted.keras`, `figures/`, `images/`.
`train.csv` and `test.csv` are not committed (127 MB); place them in `data/` to rerun.
