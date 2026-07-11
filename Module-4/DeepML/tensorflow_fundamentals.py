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
    #
    # Rank = 0
    # Shape = ()
    #
    scalar = tf.constant(10)

    # ==========================================================
    # VECTOR
    # ==========================================================
    #
    # Rank = 1
    # Shape = (4,)
    #
    vector = tf.constant([
        1, 2, 3, 4
    ])

    # ==========================================================
    # MATRIX
    # ==========================================================
    #
    # Rank = 2
    # Shape = (2,2)
    #
    matrix = tf.constant([
        [1, 2],
        [3, 4]
    ])

    # ==========================================================
    # 3D TENSOR
    # ==========================================================
    #
    # Rank = 3
    #
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
    # TENSOR CREATION UTILITIES
    # ==========================================================

    zeros_tensor = tf.zeros((3, 3))

    ones_tensor = tf.ones((3, 3))

    fill_tensor = tf.fill(
        dims=(3, 3),
        value=9
    )

    identity_tensor = tf.eye(3)

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

    ragged_df = pd.DataFrame([
        [
            "Ragged Tensor",
            ragged_tensor.to_list()
        ],
        [
            "Shape",
            str(ragged_tensor.shape)
        ]
    ], columns=[
        "Property",
        "Value"
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

    tensor_objects_df = pd.DataFrame([
        [
            "Scalar",
            scalar.numpy(),
            scalar.shape,
            tf.rank(scalar).numpy(),
            scalar.dtype
        ],
        [
            "Vector",
            vector.numpy().tolist(),
            vector.shape,
            tf.rank(vector).numpy(),
            vector.dtype
        ],
        [
            "Matrix",
            matrix.numpy().tolist(),
            matrix.shape,
            tf.rank(matrix).numpy(),
            matrix.dtype
        ],
        [
            "3D Tensor",
            tensor_3d.numpy().tolist(),
            tensor_3d.shape,
            tf.rank(tensor_3d).numpy(),
            tensor_3d.dtype
        ]
    ], columns=[
        "Tensor",
        "Value",
        "Shape",
        "Rank",
        "Datatype"
    ])
    # ==========================================================
    # SHAPE AND RANK
    # ==========================================================

    shape_df = pd.DataFrame([
        [
            "Scalar",
            scalar.shape,
            tf.rank(scalar).numpy()
        ],
        [
            "Vector",
            vector.shape,
            tf.rank(vector).numpy()
        ],
        [
            "Matrix",
            matrix.shape,
            tf.rank(matrix).numpy()
        ],
        [
            "3D Tensor",
            tensor_3d.shape,
            tf.rank(tensor_3d).numpy()
        ]
    ], columns=[
        "Tensor",
        "Shape",
        "Rank"
    ])

    # ==========================================================
    # DATA TYPES
    # ==========================================================

    int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)

    datatype_df = pd.DataFrame([
        ["Integer Tensor", int_tensor.dtype],
        ["Float Tensor", float_tensor.dtype]
    ], columns=["Tensor", "Datatype"])

    # ==========================================================
    # VARIABLES
    # ==========================================================
    #
    # Neural network weights use tf.Variable
    #
    weight = tf.Variable(
        initial_value=5.0,
        trainable=True
    )

    initial_weight = weight.numpy()

    weight.assign(10.0)

    updated_weight = weight.numpy()

    variable_df = pd.DataFrame([
        ["Initial Weight", initial_weight],
        ["Updated Weight", updated_weight]
    ], columns=["Property", "Value"])

    # ==========================================================
    # MATHEMATICAL OPERATIONS
    # ==========================================================

    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])

    math_df = pd.DataFrame([
        ["Addition", tf.add(a, b).numpy()],
        ["Subtraction", tf.subtract(b, a).numpy()],
        ["Multiplication", tf.multiply(a, b).numpy()],
        ["Division", tf.divide(b, a).numpy()],
        ["Power", tf.pow(a, 2).numpy()],
        ["Square Root", tf.sqrt(tf.cast(b, tf.float32)).numpy()]
    ], columns=["Operation", "Result"])

    abs_tensor = tf.abs(
        tf.constant([-5, -2, 10])
    )

    exp_tensor = tf.exp(
        tf.constant([1.0, 2.0])
    )

    log_tensor = tf.math.log(
        tf.constant([1.0, 2.0, 3.0])
    )

    power_tensor = tf.pow(
        tf.constant([2, 3, 4]),
        2
    )

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

    matrix_result = tf.matmul(
        matrix_a,
        matrix_b
    ).numpy()

    # ==========================================================
    # INDEXING
    # ==========================================================

    sample_tensor = tf.constant([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ])

    indexing_df = pd.DataFrame([
        ["First Element", sample_tensor[0, 0].numpy()],
        ["Last Element", sample_tensor[-1, -1].numpy()],
        ["First Row", sample_tensor[0].numpy()],
        ["Second Row", sample_tensor[1].numpy()],
        ["Third Column", sample_tensor[:, 2].numpy()]
    ], columns=[
        "Operation",
        "Result"
    ])

    # ==========================================================
    # SLICING
    # ==========================================================

    slicing_df = pd.DataFrame([
        [
            "Rows 0:2",
            sample_tensor[0:2].numpy().tolist()
        ],
        [
            "Columns 0:2",
            sample_tensor[:, 0:2].numpy().tolist()
        ],
        [
            "Center Block",
            sample_tensor[0:2, 0:2].numpy().tolist()
        ]
    ], columns=[
        "Operation",
        "Result"
    ])

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
    # REDUCTION OPERATIONS
    # ==========================================================

    values = tf.constant([
        10, 20, 30, 40, 50
    ])

    reduction_df = pd.DataFrame([
        ["Sum", tf.reduce_sum(values).numpy()],
        ["Mean", tf.reduce_mean(values).numpy()],
        ["Maximum", tf.reduce_max(values).numpy()],
        ["Minimum", tf.reduce_min(values).numpy()],
        ["Product", tf.reduce_prod(values).numpy()]
    ], columns=["Metric", "Value"])

    # ==========================================================
    # ARGMAX / ARGMIN
    # ==========================================================
    #
    # Very common in classification models.
    #
    arg_df = pd.DataFrame([
        [
            "Arg Max",
            tf.argmax(values).numpy()
        ],
        [
            "Arg Min",
            tf.argmin(values).numpy()
        ]
    ], columns=[
        "Operation",
        "Index"
    ])

    # ==========================================================
    # COMPARISON OPERATIONS
    # ==========================================================

    comparison_df = pd.DataFrame([
        [
            "Greater Than",
            tf.math.greater(
                values,
                25
            ).numpy().tolist()
        ],
        [
            "Less Than",
            tf.math.less(
                values,
                25
            ).numpy().tolist()
        ]
    ], columns=[
        "Comparison",
        "Result"
    ])

    # ==========================================================
    # LOGICAL OPERATIONS
    # ==========================================================

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

    logical_df = pd.DataFrame([
        [
            "Logical AND",
            tf.logical_and(
                logical_a,
                logical_b
            ).numpy().tolist()
        ],
        [
            "Logical OR",
            tf.logical_or(
                logical_a,
                logical_b
            ).numpy().tolist()
        ]
    ], columns=[
        "Operation",
        "Result"
    ])

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
    # STATISTICS
    # ==========================================================

    statistics_df = pd.DataFrame([
        [
            "Mean",
            tf.reduce_mean(
                values
            ).numpy()
        ],
        [
            "Standard Deviation",
            tf.math.reduce_std(
                tf.cast(
                    values,
                    tf.float32
                )
            ).numpy()
        ]
    ], columns=[
        "Metric",
        "Value"
    ])

    # ==========================================================
    # NUMPY CONVERSION
    # ==========================================================

    numpy_array = vector.numpy()

    numpy_conversion_df = pd.DataFrame([
        [
            "TensorFlow Tensor",
            vector.numpy().tolist()
        ],
        [
            "NumPy Array",
            numpy_array.tolist()
        ]
    ], columns=[
        "Object",
        "Value"
    ])

    # ==========================================================
    # GPU DETECTION
    # ==========================================================

    gpus = tf.config.list_physical_devices(
        "GPU"
    )

    gpu_df = pd.DataFrame([
        ["GPU Count", len(gpus)]
    ], columns=[
        "Property",
        "Value"
    ])

    # ==========================================================
    # REPORT
    # ==========================================================

    html_doc = builder.build_page(
        "TensorFlow Fundamentals Report",
        builder.grid([
            builder.card("Tensor Objects", builder.render_dataframe(tensor_objects_df)),
            builder.card("Ragged Tensor", builder.render_dataframe(ragged_df)),
            builder.card("Tensor Shape & Rank", builder.render_dataframe(shape_df)),
            builder.card("Tensor Datatypes", builder.render_dataframe(datatype_df)),
            builder.card("Tensor Variables", builder.render_dataframe(variable_df)),
            builder.card("Mathematical Operations", builder.render_dataframe(math_df)),
            builder.card("Absolute Value", builder.render_dataframe(pd.DataFrame(abs_tensor.numpy()))),
            builder.card("Exponential", builder.render_dataframe(pd.DataFrame(exp_tensor.numpy()))),
            builder.card("Logarithm", builder.render_dataframe(pd.DataFrame(log_tensor.numpy()))),
            builder.card("Power", builder.render_dataframe(pd.DataFrame(power_tensor.numpy()))),
            builder.card("Indexing Operations", builder.render_dataframe(indexing_df)),
            builder.card("Slicing Operations", builder.render_dataframe(slicing_df)),
            builder.card("Reduction Operations", builder.render_dataframe(reduction_df)),
            builder.card("ArgMax / ArgMin", builder.render_dataframe(arg_df)),
            builder.card("Comparison Operations", builder.render_dataframe(comparison_df)),
            builder.card("Logical Operations", builder.render_dataframe(logical_df)),
            builder.card("Statistics", builder.render_dataframe(statistics_df)),
            builder.card("NumPy Conversion", builder.render_dataframe(numpy_conversion_df)),
            builder.card("GPU Information", builder.render_dataframe(gpu_df)),
            builder.card("Matrix Multiplication", builder.render_dataframe(pd.DataFrame(matrix_result))),
            builder.card("Reshaped Tensor", builder.render_dataframe(pd.DataFrame(reshaped.numpy()))),
            builder.card("Random Uniform Tensor", builder.render_dataframe(pd.DataFrame(random_uniform.numpy()))),
            builder.card("Random Normal Tensor", builder.render_dataframe(pd.DataFrame(random_normal.numpy())))
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
