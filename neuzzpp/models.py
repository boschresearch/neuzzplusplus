# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Module containing neural network architectures."""
from typing import List, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class MLP:
    """
    Basic MLP with one hidden layer as used in the original NEUZZ implementation.

    The output layer is a sigmoid (not softmax), as predictions are coverage bitmaps.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        lr: float,
        ff_dim: int = 4096,
        output_bias: Optional[float] = None,
        fast: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.output_bias = output_bias
        self.ff_dim = ff_dim

        # Create model
        model = tf.keras.models.Sequential()
        model.add(layers.Dense(ff_dim, input_dim=input_dim, activation="relu"))
        model.add(layers.Dense(output_dim, bias_initializer=output_bias, name="logits"))
        model.add(layers.Activation("sigmoid"))

        # Compile
        lr = tf.keras.optimizers.schedules.CosineDecayRestarts(lr, first_decay_steps=1000)
        # lr = CyclicalLearningRate(
        #     initial_learning_rate=1e-6,
        #     maximal_learning_rate=lr,
        #     step_size=2 * (seed_handler.training_size // batch_size + 1),
        #     scale_fn=lambda x: 1 / (2.0 ** (x - 1)),
        #     scale_mode="cycle",
        # )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        metrics = [
            tf.keras.metrics.AUC(
                name="prc", curve="PR", multi_label=True, num_labels=self.output_dim
            )
        ]
        if not fast:
            metrics.extend(
                [
                    tf.keras.metrics.BinaryAccuracy(name="acc"),
                    tf.keras.metrics.AUC(name="auc", multi_label=True, num_labels=self.output_dim),
                ]
            )

        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=metrics,
        )
        self.model = model


def create_logits_model(model):
    """
    Create a model that outputs logits instead of probabilities for mutation gradient computation.

    The assumption is that the input model has a layer called `logits`, which will be used for
    creating the gradient model.

    Args:
        model: Trained model capable of predicting bitmap coverage for the
            target program. Outputs are probabilities.

    Returns:
        A new model of shared input and weights as the original, but that outputs the logits
        before the last activation layer.
    """
    return tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("logits").output)


def predict_coverage(model: tf.keras.models.Model, inputs: List[np.ndarray]) -> np.ndarray:
    """
    Get binary labels from model for non-normalized input data.

    The input data is first normalized and preprocessed to the length required by the model.

    Args:
        model: Keras model predicting coverage bitmap from program input.
        inputs: List or equivalent of non-normalized inputs.
    """
    input_shape = model.input.shape[-1]
    inputs_preproc = tf.keras.preprocessing.sequence.pad_sequences(
        inputs, padding="post", dtype="float32", maxlen=input_shape
    )
    inputs_preproc = inputs_preproc.astype("float32") / 255.0

    preds = model(inputs_preproc).numpy()
    return preds > 0.5
