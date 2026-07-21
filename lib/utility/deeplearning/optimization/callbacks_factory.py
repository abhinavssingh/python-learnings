import tensorflow as tf


class CallbacksFactory:

    @staticmethod
    def create(
        framework: str,
        config=None,
        **kwargs
    ) -> list:

        framework = framework.lower()

        if framework != "tensorflow":
            raise NotImplementedError(
                f"Framework '{framework}' is not supported."
            )

        callbacks = []

        # Early Stopping
        if getattr(config, "early_stopping", False):

            callbacks.append(
                tf.keras.callbacks.EarlyStopping(
                    monitor=getattr(
                        config,
                        "early_stopping_monitor",
                        "val_loss",
                    ),
                    patience=getattr(
                        config,
                        "early_stopping_patience",
                        5,
                    ),
                    restore_best_weights=True,
                )
            )

        # Reduce LR on Plateau
        if getattr(config, "reduce_lr_on_plateau", False):

            callbacks.append(
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor=getattr(
                        config,
                        "reduce_lr_monitor",
                        "val_loss",
                    ),
                    factor=getattr(
                        config,
                        "reduce_lr_factor",
                        0.5,
                    ),
                    patience=getattr(
                        config,
                        "reduce_lr_patience",
                        3,
                    ),
                    verbose=1,
                )
            )

        # Model Checkpoint
        checkpoint_path = getattr(
            config,
            "checkpoint_filepath",
            None,
        )

        if checkpoint_path:

            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=checkpoint_path,
                    monitor=getattr(
                        config,
                        "checkpoint_monitor",
                        "val_loss",
                    ),
                    save_best_only=True,
                    verbose=1,
                )
            )

        # TensorBoard
        log_dir = getattr(
            config,
            "tensorboard_log_dir",
            None,
        )

        if log_dir:

            callbacks.append(
                tf.keras.callbacks.TensorBoard(
                    log_dir=log_dir
                )
            )

        # CSV Logger
        csv_log_file = getattr(
            config,
            "csv_log_file",
            None,
        )

        if csv_log_file:

            callbacks.append(
                tf.keras.callbacks.CSVLogger(
                    csv_log_file
                )
            )

        return callbacks
