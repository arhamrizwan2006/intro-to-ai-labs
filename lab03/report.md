# Lab 03 Report - Text and Image Features

**Course:** Introduction to Artificial Intelligence (AI-101L)   **Student:** Arham Rizwan

## Part A - Bag-of-Words (answers to Q1-3)

1. **What does each column represent?** Each column is one unique word (term) from the vocabulary learned across the whole corpus. The number in a cell says how many times that word appears in that row's document. Here there are 11 columns, so 11 features.
2. **What is the machine-understandable format?** The machine-understandable format is the 3 x 11 matrix of integer word counts: every document becomes a fixed-length numeric vector (length 11), and any algorithm that accepts numeric vectors can now work with text. scikit-learn stores it as a sparse matrix.
3. **How is it an oversimplification?** BoW throws away a lot. Word order is gone, so 'dog bites man' and 'man bites dog' get the same vector. Grammar, context and meaning are gone too: 'sun' and 'sunny' become two unrelated columns, and synonyms are not linked. The defaults add more loss: everything is lowercased, so 'The' and 'the' merge, and single-character tokens such as 'a' are dropped. Raw counts also favour long documents, and any word not in the training vocabulary is ignored later.

## Part B - Image features (answers to Q1-4)

1. **Grayscale as feature engineering.** Grayscale collapses the three colour channels (R, G, B) into a single brightness value per pixel (Pillow uses a weighted sum, L = 0.299 R + 0.587 G + 0.114 B). That cuts the number of features to a third, which means less memory, faster training and less risk of overfitting. The price is all colour information, so it is a good choice when shape or intensity matters (digits) and a bad one when colour is the clue (ripe vs unripe fruit).
2. **Flattened length for an H x W image.** H x W. After grayscale each pixel is one number, so a 100 x 100 image gives 10,000 features (the RGB version would have 3 x H x W = 30,000).
3. **Why flatten?** Models such as Logistic Regression or an SVM expect every sample to be one row of numbers, an input of shape (n_samples, n_features). An image is a 2-D grid, so it must be unrolled into a 1-D vector first. The cost is that neighbouring-pixel structure is no longer explicit: the model sees 4,096 independent columns. Convolutional networks exist to keep that spatial structure.
4. **Why resize to one size?** A model has a fixed number of input features, and a matrix X needs equal-length rows. Resizing every image to the same size (for example 64 x 64) guarantees vectors of identical length (4,096), so photos of any original size can be stacked and fed to the same model. Smaller images also cost less memory and compute. The trade-off is lost fine detail, and distortion if the original aspect ratio was not square.

## Exercise results

- Exercise 1: the vocabulary grows from 11 features (unigrams) to 24 with `ngram_range=(1,2)`.
- Exercise 2: the repeated-word document dominates the `sun` column by repetition and length alone.
- Exercise 3: `china.jpg` (640x427) gives a flattened grayscale vector of 273,280 values; resized to 64x64 it gives 4,096 and to 32x32 it gives 1,024.
- Exercise 4: `image_to_features` returns 4,096 values for three images of different original sizes.
- Part A matrix shape: 3 x 11.

## Home assignment

Dataset: 20 synthetic images, class_a = circles (label 0) and class_b = squares (label 1), made by
`make_data.py`. Saved as `dataset.npz` with X of shape (20, 4096) and y of shape (20,).

Twenty samples in 4,096 dimensions is not a sound basis for learning. There are about 0.005 samples per feature, far fewer samples than features, so many different linear boundaries can separate the training images perfectly and the model cannot tell which one generalises (high variance, overfitting). In very high dimensions distances also become less informative (curse of dimensionality). The cross-validation score printed above uses only 4 test images per fold, so it is a rough guide, not a reliable estimate. In the run above, training accuracy is 1.0 while cross-validation accuracy is only 0.5 to 0.75, which is exactly the overfitting this predicts: the model memorises the 20 images but does not generalise. A real task would need far more images, regularisation, dimensionality reduction (for example PCA) or data augmentation, or a convolutional network that shares weights across the image.

## Notes

- `CountVectorizer` defaults used and stated: lowercasing is on, and single-character tokens are dropped.
- The dataset images are generated shapes, not photographs, and the notebook says so.
