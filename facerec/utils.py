import tensorflow as tf


def euclidean_distance(vectors: tf.Tensor) -> tf.Tensor:
    """
    Calculate the Euclidean distance between two vectors.

    Parameters
    ----------
    vectors : tf.Tensor
        A tuple of two tensors (x, y) to calculate the distance.

    Returns
    -------
    tf.Tensor
        The Euclidean distance between the two input tensors.

    Example
    -------
    >>> x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    >>> y = tf.constant([[5.0, 6.0], [7.0, 8.0]])
    >>> distance = euclidean_distance((x, y))
    """
    x, y = vectors
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


def contrastive_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate the contrastive loss for Siamese networks.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth labels (0 for dissimilar, 1 for similar pairs).
    y_pred : tf.Tensor
        Predicted similarity scores.

    Returns
    -------
    tf.Tensor
        Contrastive loss value.

    Example
    -------
    >>> true_labels = tf.constant([1, 0, 1])
    >>> predicted_scores = tf.constant([0.8, 0.2, 0.7])
    >>> loss = contrastive_loss(true_labels, predicted_scores)
    """
    margin = 1

    y_true = tf.cast(y_true, y_pred.dtype)

    square_pred = tf.math.square(y_pred)
    margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
    return tf.math.reduce_mean(
        y_true * square_pred + (1 - y_true) * margin_square
    )


def accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate accuracy.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary labels.
    y_pred : tf.Tensor
        Predicted distances.

    Returns
    -------
    tf.Tensor
        Accuracy score.

    Example
    -------
    >>> true_labels = tf.constant([1, 0, 1])
    >>> predicted_scores = tf.constant([0.8, 0.2, 0.7])
    >>> acc = accuracy(true_labels, predicted_scores)
    """
    y_pred_binary = tf.cast(tf.math.less(y_pred, 0.5), y_pred.dtype)

    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true, y_pred_binary), dtype=tf.float32))

    return accuracy


def specificity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate specificity.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary labels.
    y_pred : tf.Tensor
        Predicted distances.

    Returns
    -------
    tf.Tensor
        Specificity score.

    Example
    -------
    >>> true_labels = tf.constant([1, 0, 1])
    >>> predicted_scores = tf.constant([0.8, 0.2, 0.7])
    >>> spec = specificity(true_labels, predicted_scores)
    """
    y_pred_binary = tf.cast(tf.math.less(y_pred, 0.5), y_pred.dtype)
    # y_pred_binary = tf.cast(y_pred_binary, y_pred.dtype)

    true_negatives = tf.reduce_sum((1 - y_true) * (1 - y_pred_binary))
    false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)

    specificity = true_negatives / (true_negatives + false_positives + tf.keras.backend.epsilon())

    return specificity


def recall(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate recall.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary labels.
    y_pred : tf.Tensor
        Predicted distances.

    Returns
    -------
    tf.Tensor
        Recall score.

    Example
    -------
    >>> true_labels = tf.constant([1, 0, 1])
    >>> predicted_scores = tf.constant([0.8, 0.2, 0.7])
    >>> rec = recall(true_labels, predicted_scores)
    """
    y_pred_binary = tf.cast(tf.math.less(y_pred, 0.5), y_pred.dtype)
    # y_pred_binary = tf.cast(y_pred_binary, y_pred.dtype)

    true_positives = tf.reduce_sum(y_true * y_pred_binary)
    false_negatives = tf.reduce_sum(y_true * (1 - y_pred_binary))

    recall = true_positives / (true_positives + false_negatives + tf.keras.backend.epsilon())

    return recall


def precision(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Calculate precision.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth binary labels.
    y_pred : tf.Tensor
        Predicted distances.

    Returns
    -------
    tf.Tensor
        Precision score.

    Example
    -------::
    >>> true_labels = tf.constant([1, 0, 1])
    >>> predicted_scores = tf.constant([0.8, 0.2, 0.7])
    >>> prec = precision(true_labels, predicted_scores)
    """
    y_pred_binary = tf.cast(tf.math.less(y_pred, 0.5), y_pred.dtype)
    # y_pred_binary = tf.cast(y_pred_binary, y_pred.dtype)

    true_positives = tf.reduce_sum(y_true * y_pred_binary)
    false_positives = tf.reduce_sum((1 - y_true) * y_pred_binary)

    precision = true_positives / (true_positives + false_positives + tf.keras.backend.epsilon())

    return precision
