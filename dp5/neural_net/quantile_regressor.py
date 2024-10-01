"""runs combinatorial studies on CASCADE data"""

import os
import tensorflow as tf
import numpy as np
import pandas as pd
import tempfile
import zipfile
from functools import partial

import keras
import keras.backend as K
from keras.layers import Input, Lambda, Dense, Add, Layer
from keras.models import Sequential, Model
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay

from scipy.stats import norm, gaussian_kde
from scipy.optimize import curve_fit

from sklearn.base import RegressorMixin, BaseEstimator

from tqdm import tqdm

from dp5.neural_net.CNN_model import build_model, extract_representations, load_NMR_prediction_model

@keras.saving.register_keras_serializable()
def QuantileLoss(perc, delta=1e-4):
    perc = np.array(perc).reshape(-1)
    perc.sort()
    perc = perc.reshape(1, -1)

    @keras.saving.register_keras_serializable()
    def _qloss(y, pred):
        I = tf.cast(y <= pred, tf.float32)
        d = K.abs(y - pred)
        correction = I * (1 - perc) + (1 - I) * perc
        # huber loss
        huber_loss = K.sum(
            correction * tf.where(d <= delta, 0.5 * d**2 / delta, d - 0.5 * delta), -1
        )
        # order loss
        q_order_loss = K.sum(K.maximum(0.0, pred[:, :-1] - pred[:, 1:] + 1e-6), -1)
        return huber_loss + q_order_loss

    return _qloss

class PercentileRegressor(BaseEstimator):
    # create a @classmethod to initialise a model from CASCADE or load an existing one
    def __init__(self, model, quantiles):

        self.quantiles = quantiles
        self.dims = len(self.quantiles)
        self.model = model

    def save(self, archive_path):
        # Use tempfile to create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the NumPy array in the temporary directory
            array_path = os.path.join(temp_dir, "array.npy")
            np.save(array_path, self.quantiles)

            # Save the TensorFlow model in the temporary directory
            model_path = os.path.join(temp_dir, "model.keras")
            self.model.save(model_path)

            # Create a ZIP archive containing both the array and the model
            with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                # Add the NumPy array to the ZIP archive
                zipf.write(array_path, "array.npy")
                zipf.write(model_path, "model.keras")

    @classmethod
    def load(cls, archive_path):
        # Use tempfile to extract the ZIP archive to a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Extract the archive contents into the temporary directory
            with zipfile.ZipFile(archive_path, "r") as zipf:
                zipf.extractall(temp_dir)

            # Load the NumPy array
            array_path = os.path.join(temp_dir, "array.npy")
            arr = np.load(array_path)

            # Load the TensorFlow model
            model_dir = os.path.join(temp_dir, "model.keras")
            model = tf.keras.models.load_model(
                model_dir, custom_objects={"_qloss": QuantileLoss(arr)}
            )

            return cls(model, arr)

    @classmethod
    def from_cascade(
        cls,
        quantiles,
        model_path="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
    ):
        dims = len(quantiles)
        full_model = load_NMR_prediction_model(model_path)
        initial_learning_rate = 5e-4
        lr_schedule = ExponentialDecay(
            initial_learning_rate, decay_steps=100000, decay_rate=0.96, staircase=True
        )
        optimizer = Adam(learning_rate=lr_schedule)
        mean_shift = full_model.get_layer(name="atomwise_shift").weights[0][2].numpy()

        input_rep = Input(shape=(256,), name="input_rep")
        x = full_model.get_layer("loc_1")(input_rep)
        x = full_model.get_layer("loc_2")(x)
        x = full_model.get_layer("loc_3")(x)
        x = Dense(dims, name="loc_reduce")(x)
        output = Dense(dims, name="workaround", trainable=False)(x)

        model = Model(input_rep, output)

        w, b = full_model.get_layer("loc_reduce").weights
        w_c = np.broadcast_to(w, shape=(w.shape[0], dims))
        b_c = np.broadcast_to(b, shape=(dims))
        model.get_layer("loc_reduce").set_weights([w_c, b_c])

        w_w, b_w = np.eye(dims), np.broadcast_to(mean_shift, shape=(dims))
        model.get_layer("workaround").set_weights([w_w, b_w])

        model.compile(optimizer=optimizer, loss=QuantileLoss(quantiles))
        class1 = cls(model, quantiles)
        return class1

    def fit(self, *args, **kwargs):
        early_stopping_monitor = EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=10,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
        return self.model.fit(*args, **kwargs, callbacks=[early_stopping_monitor])

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

regressor = PercentileRegressor.load("/home/jbr46/combinatorial_study_dp5/quantile_regressor.zip")

def generate_quantiles(reps_col, atom_index_col):

    X_test = tf.convert_to_tensor(np.vstack(reps_col.values))
    test_preds = regressor(X_test, batch_size=32)

    # Initialize empty lists to store the mapped predictions
    mapped_quantiles = []
    mapped_mus = []
    mapped_sigmas = []
    
    # Keep track of the current index in the predictions array
    current_index = 0
    
    for indices in atom_index_col.values:
        num_atoms = len(indices)
        # Slice the predictions for this molecule
        mol_predictions = test_preds[current_index:current_index + num_atoms]
        
        # Calculate mu and sigma for each atom
        mus = []
        sigmas = []
        for q in mol_predictions:
            percentiles = regressor.quantiles
            dims = len(percentiles)
            median = q[dims // 2]
            std = (q[dims * 2 // 3] - q[dims // 3]) / 2
            mu, sigma = curve_fit(norm.cdf, q, percentiles, p0=[median, std])[0]
            mus.append(mu)
            sigmas.append(sigma)
        
        # Store the predictions, mus, and sigmas as lists
        mapped_quantiles.append(mol_predictions.tolist())
        mapped_mus.append(mus)
        mapped_sigmas.append(sigmas)
        
        # Update the current index
        current_index += num_atoms
    
    return mapped_quantiles, mapped_mus, mapped_sigmas

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    tf.config.set_visible_devices([], 'GPU')
    df = pd.read_pickle(
        "/home/jbr46/combinatorial_study_dp5/CASCADE_test_df_with_preds.p.gz", compression="gzip"
    )
    df["nCatoms"] = df["atom_index"].apply(len)
    df_by_atom = [df_at for i, df_at in df.groupby("nCatoms")]
    corrects = []
    incorrects = []
    for sub_df in tqdm(df_by_atom, desc="Running Combinatorial Studies"):
        if len(sub_df) < 10:
            continue
        e, d, p = combinatorial_studies_monolith(
            sub_df, nmr_col="Shift", c_preds="shift_arrays", c_inds="atom_index"
        )
        correct, incorrect = corrector(e, d, p)
        corrects.append(correct)
        incorrects.append(incorrect)
    corrects = np.concatenate(corrects, axis=1)
    incorrects = np.concatenate(incorrects, axis=1)
    np.save("correct.npy", corrects)
    np.save("incorrect.npy", incorrects)
    plot_kde(corrects, incorrects, "big_kde_v2.png")
    dp5_vs_mad(corrects, incorrects, "big_madvsdp5_v2.png")


if __name__ == "__main__":
    main()
