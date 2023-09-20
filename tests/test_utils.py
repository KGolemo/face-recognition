import pytest
import tensorflow as tf
from facerec.utils import euclidean_distance, contrastive_loss, accuracy, precision, recall, specificity


# Define a small tolerance for floating-point comparisons
TOLERANCE = 1e-6


# Test euclidean_distance function
def test_euclidean_distance():
    x = tf.constant([[1.0, 2.0, 3.0]])
    y = tf.constant([[4.0, 5.0, 6.0]])
    expected_distance = tf.constant([[5.196152]])
    result_distance = euclidean_distance((x, y))
    tf.debugging.assert_near(result_distance, expected_distance, atol=TOLERANCE)


# Test contrastive_loss function
def test_contrastive_loss():
    y_true = tf.constant([0, 1, 0, 1], dtype=tf.float32)
    y_pred = tf.constant([0.1, 0.8, 0.3, 0.9], dtype=tf.float32)
    expected_loss = 0.6875  # Calculate the expected loss manually
    result_loss = contrastive_loss(y_true, y_pred)
    tf.debugging.assert_near(result_loss, expected_loss, atol=TOLERANCE)


# Test Custom_Accuracy class
# def test_custom_accuracy():
#     accuracy_metric = CustomAccuracy()

#     y_true = tf.constant([1, 1, 1, 1], dtype=tf.float32)
#     y_pred = tf.constant([0.9, 0.3, 0.8, 0.1], dtype=tf.float32)

#     # Update the accuracy metric with the test data
#     accuracy_metric.update_state(y_true, y_pred)

#     # Calculate the expected accuracy manually
#     expected_accuracy = 0.5
#     result_accuracy = accuracy_metric.result().numpy()

#     assert result_accuracy == expected_accuracy


# Test accuracy function
def test_accuracy():
    y_true = tf.constant([1, 1, 1, 1], dtype=tf.float32)
    y_pred = tf.constant([0.1, 0.8, 0.3, 0.9], dtype=tf.float32)
    expected_accuracy = 0.5
    result_accuracy = accuracy(y_true, y_pred)
    tf.debugging.assert_near(result_accuracy, expected_accuracy, atol=TOLERANCE)


# Test precision function
def test_precision():
    y_true = tf.constant([1, 1, 1, 1], dtype=tf.float32)
    y_pred = tf.constant([0.1, 0.8, 0.3, 0.9], dtype=tf.float32)
    expected_precision = 1
    result_precision = precision(y_true, y_pred)
    tf.debugging.assert_near(result_precision, expected_precision, atol=TOLERANCE)


# Test recall function
def test_recall():
    y_true = tf.constant([1, 1, 1, 1], dtype=tf.float32)
    y_pred = tf.constant([0.1, 0.8, 0.3, 0.9], dtype=tf.float32)
    expected_recall = 0.5
    result_recall = recall(y_true, y_pred)
    tf.debugging.assert_near(result_recall, expected_recall, atol=TOLERANCE)


# Test specificity function
def test_specificity():
    y_true = tf.constant([0, 0, 0, 0], dtype=tf.float32)
    y_pred = tf.constant([0.1, 0.8, 0.3, 0.9], dtype=tf.float32)
    expected_specificity = 0.5
    result_specificity = specificity(y_true, y_pred)
    tf.debugging.assert_near(result_specificity, expected_specificity, atol=TOLERANCE)


# Run the tests
if __name__ == "__main__":
    pytest.main()
