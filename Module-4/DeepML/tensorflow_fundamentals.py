"""
TensorFlow Fundamentals

This script demonstrates:

1. Tensor Creation
2. Tensor Shape
3. Tensor Rank
4. Tensor Datatypes
5. Tensor Indexing
6. Tensor Slicing
7. Tensor Reshaping
8. Mathematical Operations
9. Matrix Multiplication
10. Reduction Operations
11. Comparison Operations
12. Logical Operations
13. Tensor Statistics
14. Random Tensors
15. Tensor Stacking
16. Tensor Concatenation
17. Tensor Expansion
18. Tensor Squeezing
19. Variables
20. NumPy Conversion
21. GPU Detection

These concepts form the foundation for:

    - Perceptrons
    - Neural Networks
    - Backpropagation
    - TensorFlow Models
    - Deep Learning
"""

import numpy as np
import pandas as pd
import tensorflow as tf

from lib.html import HtmlBuilder
from lib.utility.reports.report_utils import ReportUtils as ru


def main():

    builder = HtmlBuilder()

    # ==========================================================
    # SCALAR
    # ==========================================================

    scalar = tf.constant(10)

    # ==========================================================
    # VECTOR
    # ==========================================================

    vector = tf.constant([
        1, 2, 3, 4
    ])

    # ==========================================================
    # MATRIX
    # ==========================================================

    matrix = tf.constant([
        [1, 2],
        [3, 4]
    ])

    # ==========================================================
    # 3D TENSOR
    # ==========================================================

    tensor_3d = tf.constant([
        [
            [1, 2],
            [3, 4]
        ],
        [
            [5, 6],
            [7, 8]
        ]
    ])

    # ==========================================================
    # RAGGED TENSOR
    # ==========================================================
    #
    # Used when rows have different lengths.
    #
    # Common in:
    #   NLP
    #   Token Sequences
    #   Text Classification
    #

    ragged_tensor = tf.ragged.constant([
        [1, 2],
        [3, 4, 5],
        [6]
    ])

    # ==========================================================
    # NUMPY TO TENSOR
    # ==========================================================

    numpy_array = np.array([
        [1, 2, 3],
        [4, 5, 6]
    ])

    numpy_tensor = tf.convert_to_tensor(
        numpy_array,
        dtype=tf.float32
    )

    # ==========================================================
    # DATAFRAME TO TENSOR
    # ==========================================================

    employee_df = pd.DataFrame({
        "Age": [25, 30, 35],
        "Salary": [50000, 70000, 90000]
    })

    dataframe_tensor = tf.convert_to_tensor(
        employee_df.values,
        dtype=tf.float32
    )

    # ==========================================================
    # DATA TYPES
    # ==========================================================

    int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

    # ==========================================================
    # MATHEMATICAL OPERATIONS
    # ==========================================================

    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])

    # ==========================================================
    # MATRIX MULTIPLICATION
    # ==========================================================

    matrix_a = tf.constant([
        [1, 2],
        [3, 4]
    ])

    matrix_b = tf.constant([
        [5, 6],
        [7, 8]
    ])

    # ==========================================================
    # INDEXING
    # ==========================================================
    #
    # Tensor used throughout this section.
    #
    # Shape: (3, 3)
    #
    sample_tensor = tf.constant([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])

    # First element
    first_element = sample_tensor[0, 0]

    # Last element
    last_element = sample_tensor[-1, -1]

    # First row
    first_row = sample_tensor[0]

    # Second row
    second_row = sample_tensor[1]

    # Third column
    third_column = sample_tensor[:, 2]

    # ==========================================================
    # SLICING
    # ==========================================================
    #
    # Extract a subset of rows/columns.
    #

    # First two rows
    rows_0_2 = sample_tensor[0:2]

    # First two columns
    columns_0_2 = sample_tensor[:, 0:2]

    # Top-left block
    center_block = sample_tensor[0:2, 0:2]

    # Bottom-right block
    bottom_right = sample_tensor[1:, 1:]

    # ==========================================================
    # REDUCTION OPERATIONS
    # ==========================================================
    #
    # Reduce dimensions by applying an operation.
    #

    values = tf.constant([
        10,
        20,
        30,
        40,
        50
    ])

    # Sum of all values
    sum_tensor = tf.reduce_sum(values)

    # Mean value
    mean_tensor = tf.reduce_mean(values)

    # Maximum value
    max_tensor = tf.reduce_max(values)

    # Minimum value
    min_tensor = tf.reduce_min(values)

    # Product of all values
    product_tensor = tf.reduce_prod(values)

    # Variance
    variance_tensor = tf.math.reduce_variance(
        tf.cast(values, tf.float32)
    )

    # Standard Deviation
    std_tensor = tf.math.reduce_std(
        tf.cast(values, tf.float32)
    )

    # ==========================================================
    # ARGMAX / ARGMIN
    # ==========================================================
    #
    # Frequently used in classification models.
    #

    argmax_tensor = tf.argmax(values)

    argmin_tensor = tf.argmin(values)

    # ==========================================================
    # TOP-K
    # ==========================================================
    #
    # Return largest K values and their indices.
    #

    scores = tf.constant([
        0.15,
        0.80,
        0.40,
        0.95,
        0.60
    ])

    top_values, top_indices = tf.math.top_k(
        scores,
        k=3
    )

    # ==========================================================
    # COMPARISON OPERATIONS
    # ==========================================================
    #
    # Returns boolean tensors.
    #

    greater_than_tensor = tf.math.greater(
        values,
        25
    )

    less_than_tensor = tf.math.less(
        values,
        25
    )

    equal_tensor = tf.math.equal(
        values,
        30
    )

    # ==========================================================
    # LOGICAL OPERATIONS
    # ==========================================================
    #
    # Operate on boolean tensors.
    #

    logical_a = tf.constant([
        True,
        False,
        True
    ])

    logical_b = tf.constant([
        True,
        True,
        False
    ])

    logical_and_tensor = tf.logical_and(
        logical_a,
        logical_b
    )

    logical_or_tensor = tf.logical_or(
        logical_a,
        logical_b
    )

    logical_not_tensor = tf.logical_not(
        logical_a
    )

    # ==========================================================
    # RESHAPE
    # ==========================================================

    tensor = tf.constant([
        1, 2, 3, 4, 5, 6
    ])

    reshaped = tf.reshape(
        tensor,
        (2, 3)
    )

    # ==========================================================
    # CONCATENATION
    # ==========================================================

    concat_tensor = tf.concat(
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        axis=0
    )

    # ==========================================================
    # STACKING
    # ==========================================================

    stacked_tensor = tf.stack([
        [1, 2, 3],
        [4, 5, 6]
    ])

    # ==========================================================
    # EXPAND DIMS
    # ==========================================================
    #
    # Common before deep learning input
    #
    expanded = tf.expand_dims(
        vector,
        axis=0
    )

    # ==========================================================
    # SQUEEZE
    # ==========================================================

    squeezed = tf.squeeze(expanded)

    # ==========================================================
    # RANDOM TENSORS
    # ==========================================================
    #
    # Frequently used for:
    #
    # - Weight Initialization
    # - Dropout
    # - Testing
    #
    random_uniform = tf.random.uniform(
        shape=(3, 3)
    )

    random_normal = tf.random.normal(
        shape=(3, 3)
    )

    # ==========================================================
    # REPORT
    # ==========================================================

    html_doc = builder.build_page(
        "TensorFlow Fundamentals Report",
        builder.grid([

            # ==================================================
            # TENSOR OBJECTS
            # ==================================================

            builder.card("Scalar Tensor", builder.render_tensor(scalar)),
            builder.card("Vector Tensor", builder.render_tensor(vector)),
            builder.card("Matrix Tensor", builder.render_tensor(matrix)),
            builder.card("3D Tensor", builder.render_tensor(tensor_3d)),
            builder.card("Ragged Tensor", builder.render_tensor(ragged_tensor)),
            builder.card("Integer Tensor", builder.render_tensor(int_tensor)),
            builder.card("Float Tensor", builder.render_tensor(float_tensor)),

            # ==================================================
            # TENSOR CONVERSION
            # ==================================================

            builder.card("NumPy → Tensor", builder.render_tensor(numpy_tensor)),
            builder.card("DataFrame → Tensor", builder.render_tensor(dataframe_tensor)),

            # ==================================================
            # TENSOR MANIPULATION
            # ==================================================

            builder.card("Reshape", builder.render_tensor(reshaped)),
            builder.card("Concatenation", builder.render_tensor(concat_tensor)),
            builder.card("Stack", builder.render_tensor(stacked_tensor)),
            builder.card("Expand Dims", builder.render_tensor(expanded)),
            builder.card("Squeeze", builder.render_tensor(squeezed)),

            # ==================================================
            # TENSOR OPERATIONS
            # ==================================================

            builder.card("Tensor A", builder.render_tensor(a)),
            builder.card("Tensor B", builder.render_tensor(b)),
            builder.card("Addition", builder.render_tensor(tf.add(a, b))),
            builder.card("Subtraction", builder.render_tensor(tf.subtract(b, a))),
            builder.card("Multiplication", builder.render_tensor(tf.multiply(a, b))),
            builder.card("Division", builder.render_tensor(tf.divide(b, a))),
            builder.card("Matrix Multiplication", builder.render_tensor(tf.matmul(matrix_a, matrix_b))),

            # ==================================================
            # INDEXING
            # ==================================================
            builder.card("Index Tensor", builder.render_tensor(sample_tensor)),
            builder.card("First Element", builder.render_tensor(first_element)),
            builder.card("Last Element", builder.render_tensor(last_element)),
            builder.card("First Row", builder.render_tensor(first_row)),
            builder.card("Second Row", builder.render_tensor(second_row)),
            builder.card("Third Column", builder.render_tensor(third_column)),

            # ==================================================
            # SLICING
            # ==================================================
            builder.card("Rows 0-2", builder.render_tensor(rows_0_2)),
            builder.card("Columns 0-2", builder.render_tensor(columns_0_2)),
            builder.card("Center Block", builder.render_tensor(center_block)),
            builder.card("Bottom Right Block", builder.render_tensor(bottom_right)),

            # ==================================================
            # REDUCTION
            # ==================================================
            builder.card("Values Tensor", builder.render_tensor(values)),
            builder.card("Sum", builder.render_tensor(sum_tensor)),
            builder.card("Mean", builder.render_tensor(mean_tensor)),
            builder.card("Max", builder.render_tensor(max_tensor)),
            builder.card("Min", builder.render_tensor(min_tensor)),
            builder.card("Product", builder.render_tensor(product_tensor)),
            builder.card("Standard Deviation", builder.render_tensor(std_tensor)),
            builder.card("Variance", builder.render_tensor(variance_tensor)),
            # ==================================================
            # ARGMAX / ARGMIN
            # ==================================================
            builder.card("ArgMax", builder.render_tensor(argmax_tensor)),
            builder.card("ArgMin", builder.render_tensor(argmin_tensor)),
            # ==================================================
            # TOP-K
            # ==================================================
            builder.card("Top Values", builder.render_tensor(top_values)),
            builder.card("Top Indices", builder.render_tensor(top_indices)),
            # ==================================================
            # COMPARISON
            # ==================================================
            builder.card("Greater Than 25", builder.render_tensor(greater_than_tensor)),
            builder.card("Less Than 25", builder.render_tensor(less_than_tensor)),
            builder.card("Equal to 30", builder.render_tensor(equal_tensor)),
            # ==================================================
            # LOGICAL
            # ==================================================
            builder.card("Logical A", builder.render_tensor(logical_a)),
            builder.card("Logical B", builder.render_tensor(logical_b)),
            builder.card("Logical OR", builder.render_tensor(logical_or_tensor)),
            builder.card("Logical NOT", builder.render_tensor(logical_not_tensor)),
            builder.card("Logical AND", builder.render_tensor(logical_and_tensor)),
            # ==================================================
            # RANDOM TENSORS
            # ==================================================
            builder.card("Random Uniform", builder.render_tensor(random_uniform)),
            builder.card("Random Normal", builder.render_tensor(random_normal)),
        ]))

    ru.save_html_report(
        __file__,
        "tensorflow_fundamentals_report.html",
        html_doc,
        subfolder="reports",
        open_in_browser=True
    )


if __name__ == "__main__":
    main()
