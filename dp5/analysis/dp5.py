from pathlib import Path
import logging
import pickle
from abc import abstractmethod

from tqdm import tqdm
import pathos.multiprocessing as mp

from dp5.neural_net.CNN_model import *
from dp5.analysis.utils import scale_nmr, AnalysisData

logger = logging.getLogger(__name__)


class DP5:
    def __init__(self, output_folder: Path, use_dft_shifts: bool):
        logger.info("Setting up DP5 method")
        self.output_folder = output_folder
        self.dft_shifts = use_dft_shifts

        if use_dft_shifts:
            # must load model for error prediction
            self.C_DP5 = ErrorDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Error_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_ERRORrep_Error_decomp.p",
                kde_file="pca_10_kde_ERRORrep_Error_kernel.p",
                dp5_correct_scaling="Error_correct_kde.p",
                dp5_incorrect_scaling="Error_incorrect_kde.p",
            )
        else:
            # must load model for shift preiction
            self.C_DP5 = ExpDP5ProbabilityCalculator(
                atom_type="C",
                model_file="NMRdb-CASCADEset_Exp_mean_model_atom_features256.hdf5",
                batch_size=16,
                transform_file="pca_10_EXP_decomp.p",
                kde_file="pca_10_kde_EXP_kernel.p",
                dp5_correct_scaling=None,
                dp5_incorrect_scaling=None,
            )

        if not self.output_folder.exists():
            self.output_folder.mkdir()

        if not (self.output_folder / "dp5").exists():
            (self.output_folder / "dp5").mkdir()

    def __call__(self, mols):
        # have to generate representations for accepted things
        # must also check is analysis has been done beore!
        data_dic_path = self.output_folder / "dp5" / "data_dic.p"
        dp5_data = DP5Data(mols, data_dic_path)
        if dp5_data.exists:
            logger.info("Found existing DP5 probability file")
            dp5_data.load()
        else:
            logger.info("Calculating DP5 probabilites...")
            (
                dp5_data.Clabels,
                dp5_data.Cshifts,
                dp5_data.Cexp,
                dp5_data.Cerrors,
                dp5_data.Cconf_atom_probs,
                dp5_data.CDP5_atom_probs,
                dp5_data.CDP5_mol_probs,
            ) = self.C_DP5(mols)
            dp5_data.save()
        return dp5_data.output


class DP5ProbabilityCalculator:
    def __init__(
        self,
        atom_type,
        model_file,
        batch_size,
        transform_file,
        kde_file,
        dp5_correct_scaling=None,
        dp5_incorrect_scaling=None,
    ):
        self.atom_type = atom_type
        self.model = build_model(model_file=model_file)
        self.batch_size = batch_size
        with open(Path(__file__).parent / transform_file, "rb") as tf:
            self.transform = pickle.load(tf)
        with open(Path(__file__).parent / kde_file, "rb") as kf:
            self.kde = pickle.load(kf)
        if dp5_correct_scaling is not None:
            with open(Path(__file__).parent / dp5_correct_scaling, "rb") as ckf:
                self.dp5_correct_kde = pickle.load(ckf)
        if dp5_incorrect_scaling is not None:
            with open(Path(__file__).parent / dp5_incorrect_scaling, "rb") as ikf:
                self.dp5_incorrect_kde = pickle.load(ikf)

    @abstractmethod
    def rescale_probabilities(self, mol_probs, errors, error_threshold=2):
        """
        Scales and aggregated atomic probabilities.

        Computes geometric means of atomic probabilities to generate final molecular probabilities.
        """
        total_probs = np.array([np.exp(np.log(arr).mean()) for arr in mol_probs])
        return mol_probs, total_probs

    @abstractmethod
    def kde_probfunction(self, df):
        return NotImplementedError("KDE sampling function not implemented")

    @staticmethod
    def boltzmann_weight(df, col):
        return df.groupby("mol_id")[["conf_population", col]].apply(
            lambda x: (x[col] * x["conf_population"]).sum()
        )

    def __call__(self, mols):
        # must generate representations
        all_labels = []
        rep_df = []
        for mol_id, mol in enumerate(mols):
            calculated, experimental, labels, indices = self.get_shifts_and_labels(mol)
            # drop unassigned !
            has_exp = np.isfinite(experimental)
            new_calcs = calculated[:, has_exp]
            new_exps = experimental[has_exp]
            new_labs = labels[has_exp]
            new_inds = indices[has_exp]

            # generate scaled errors
            scaled = scale_nmr(new_calcs, new_exps)
            corrected_errors = scaled - new_exps[np.newaxis, :]

            all_labels.append(new_labs)

            rep_df.append(
                (
                    mol_id,
                    range(mol.conformers.shape[0]),
                    mol.rdkit_mols,
                    new_inds,
                    mol.populations,
                    new_calcs,
                    new_exps,
                    corrected_errors,
                )
            )

        rep_df = pd.DataFrame(
            rep_df,
            columns=[
                "mol_id",
                "conf_id",
                "Mol",
                "atom_index",
                "conf_population",
                "conf_shifts",
                "exp_shifts",
                "errors",
            ],
        )
        # each row of dataframe represents a geometry
        rep_df = rep_df.explode(
            ["conf_id", "Mol", "conf_shifts", "conf_population", "errors"],
            ignore_index=True,
        )
        logger.info("Extracting atomic representations")
        # now return condensed representations! These are now grouped by conformer
        rep_df["representations"] = extract_representations(
            self.model, rep_df, self.batch_size
        )
        logger.debug("Transforming representations")
        rep_df["representations"] = rep_df["representations"].apply(
            self.transform.transform
        )
        logger.info("Estimating atomic probabilities")
        rep_df["atom_probs"] = self.kde_probfunction(rep_df)
        atom_probs = [np.stack(df) for i, df in rep_df.groupby("mol_id")["atom_probs"]]

        weighted_probs = self.boltzmann_weight(rep_df, "atom_probs")
        weighted_probs = 1 - weighted_probs

        weighted_errors = self.boltzmann_weight(rep_df, "errors")
        cmae = weighted_errors.apply(lambda x: np.mean(np.abs(x)))

        # rescale and aggregate probabilities
        weighted_probs, total_probs = self.rescale_probabilities(weighted_probs, cmae)

        calc_shifts_analysed = self.boltzmann_weight(rep_df, "conf_shifts")
        exp_shifts_analysed = rep_df.groupby("mol_id")["exp_shifts"].first()

        # eventually return atomic probs, weighted atomic probs, DP5 scores
        logger.info("Atomic probabilities estimated")
        return (
            all_labels,
            calc_shifts_analysed,
            exp_shifts_analysed,
            weighted_errors,
            atom_probs,
            weighted_probs,
            total_probs,
        )

    def get_shifts_and_labels(self, mol):
        """
        Arguments:
        - self.atom_type
        - mol: Molecule object
        Returns:
        - calculated conformer shifts
        - assigned experimental shifts
        - 0-based indices of relevat atoms
        """
        at = self.atom_type
        conformer_shifts = getattr(mol, "conformer_%s_pred" % at)
        assigned_shifts = getattr(mol, "%s_exp" % at)
        atom_labels = getattr(mol, "%s_labels" % at)
        atom_indices = np.array([int(label[len(at) :]) - 1 for label in atom_labels])

        return conformer_shifts, assigned_shifts, atom_labels, atom_indices


class ErrorDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def kde_probfunction(self, df):
        # loop through atoms in the test molecule - generate kde for all of them.
        # implement joblib parallel search
        # check if this has been calculated

        min_value = -20
        max_value = 20
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        probs = []
        with mp.ProcessingPool(nodes=mp.cpu_count()) as pool:
            for i, (rep, errors) in tqdm(
                df[["representations", "errors"]].iterrows(),
                total=len(df),
                desc="Computing error KDEs",
                leave=True,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool.map(self.kde, point[:])

                conf_probs = []
                for pdf, error in zip(results, errors):
                    integral = 0
                    if pdf.sum() != 0:
                        max_x = x[np.argmax(pdf)]

                        low_point = max(min_value, max_x - abs(max_x - error))
                        high_point = min(max_value, max_x + abs(max_x - error))

                        low_bound = np.argmin(np.abs(x - low_point))
                        high_bound = np.argmin(np.abs(x - high_point))

                        bound_integral = np.sum(
                            pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
                        )
                        integral = bound_integral / pdf.sum()
                    conf_probs.append(integral)
                probs.append(np.array(conf_probs))

        return probs

    def rescale_probabilities(self, mol_probs, errors, error_threshold=2):
        _, total_probs = super().rescale_probabilities(mol_probs, errors)
        scaled_probs = []
        scaled_total = []
        for prob, error, total in zip(mol_probs, errors, total_probs):
            if error < error_threshold:
                vector = np.concatenate((prob, np.atleast_1d(total)))
                correct = self.dp5_correct_kde(vector)
                incorrect = self.dp5_incorrect_kde(vector)
                scaled = correct / (correct + incorrect)
                scaled_probs.append(scaled[:-1])
                scaled_total.append(scaled[-1])
            else:
                scaled_probs.append(prob)
                scaled_total.append(total)
        return scaled_probs, scaled_total


class ExpDP5ProbabilityCalculator(DP5ProbabilityCalculator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def kde_probfunction(self, df):
        """Since the result is compared to the experimental shifts, weights the representations and runs KDE on those."""
        # loop through atoms in the test molecule - generate kde for all of them.
        total_reps = self.boltzmann_weight(df, "representations")
        exp_data = df.groupby("mol_id")["exp_shifts"].first()
        mol_df = pd.DataFrame({"representations": total_reps, "exp_shifts": exp_data})

        min_value = 0
        max_value = 250
        n_points = 250

        x = np.linspace(min_value, max_value, n_points)

        mol_probs = []
        with mp.ProcessingPool(nodes=mp.cpu_count()) as pool:
            for i, (rep, exp) in tqdm(
                mol_df[["representations", "exp_shifts"]].iterrows(),
                total=len(mol_df),
                desc="Computing experimental KDEs",
                leave=True,
            ):
                num_atoms, num_components = rep.shape
                rep_b = np.broadcast_to(
                    rep[:, :, np.newaxis], shape=(num_atoms, num_components, n_points)
                )
                x_b = np.broadcast_to(
                    x[np.newaxis, np.newaxis, :], shape=(num_atoms, 1, n_points)
                )
                point = np.concatenate((rep_b, x_b), axis=1)

                results = pool.map(self.kde, point[:])

                conf_probs = []
                for pdf, value in zip(results, exp):
                    integral = 0
                    if pdf.sum() != 0:
                        max_x = x[np.argmax(pdf)]

                        low_point = max(min_value, max_x - abs(max_x - value))
                        high_point = min(max_value, max_x + abs(max_x - value))

                        low_bound = np.argmin(np.abs(x - low_point))
                        high_bound = np.argmin(np.abs(x - high_point))

                        bound_integral = np.sum(
                            pdf[min(low_bound, high_bound) : max(low_bound, high_bound)]
                        )
                        integral = bound_integral / pdf.sum()
                    conf_probs.append(integral)
                mol_probs.append(np.array(conf_probs))
        consistency_hack = {i: probs for i, probs in enumerate(mol_probs)}
        consistent_probs = df["mol_id"].map(consistency_hack)
        return consistent_probs

    def rescale_probabilities(self, *args, **kwargs):
        return super().rescale_probabilities(*args, **kwargs)


class DP5Data(AnalysisData):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def output(self):
        """Uncomment when H-DP5 is implemented"""
        output_dict = dict()
        output_dict["C_output"] = []
        # output_dict["H_output"] = []
        output_dict["CDP5_output"] = []
        # output_dict["HDP5_output"] = []
        # output_dict["DP5_output"] = []
        for mol, clab, cshift, cexp, cerr, cpr in zip(
            self.mols,
            self.Clabels,
            self.Cshifts,
            self.Cexp,
            self.Cerrors,
            self.CDP5_atom_probs,
        ):
            output = f"\nAssigned C NMR shift for {mol}:"
            output += self.print_assignment(clab, cshift, cexp, cerr, cpr)
            output_dict["C_output"].append(output)

        # for mol, hlab, hshift, hscal, hexp, herr in zip(
        #    self.mols, self.Hlabels, self.Hshifts, self.Hscaled, self.Hexp, self.Herrors
        # ):
        #    output = f"\nAssigned H NMR shift for {mol}:"
        #    output += self.print_assignment(hlab, hshift, hscal, hexp, herr)
        #    output_dict["H_output"].append(output)

        for mol, cdp5 in zip(self.mols, self.CDP5_mol_probs):
            output_dict["CDP5_output"].append(
                f"Carbon DP5 probability for {mol}: {cdp5}"
            )
        return [
            dict(zip(output_dict.keys(), values))
            for values in zip(*output_dict.values())
        ]

    @staticmethod
    def print_assignment(labels, calculated, exp, error, probs):
        """Prints table for molecule"""

        s = np.argsort(calculated)
        svalues = calculated[s]
        slabels = labels[s]
        sexp = exp[s]
        serror = error[s]
        sprob = probs[s]

        output = f"\nlabel, calc, exp, error, prob"

        for lab, calc, ex, er, p in zip(slabels, svalues, sexp, serror, sprob):
            output += f"\n{lab:6s} {calc:6.2f} {ex:6.2f} {er:6.2f} {p:6.2f}"
        return output
