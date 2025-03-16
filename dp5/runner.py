"""
The main script. Replaces PyDP4.py, orchestrates the high-level workflow.

Molecules is the container class to contain all the data. MM and DFT methods no longer modify it directly to improve maintainability.
"""

import logging
import os
from pathlib import Path

from dp5.run.data_structures import Molecules, Molecules_precalculated
from dp5.nmr_processing import NMRData


logger = logging.getLogger(__name__)


def runner(config):
    logger.info("Starting DP4 workflow")

    # Initialize the Molecules object
    data = Molecules_precalculated(config)
    
    # Conformational search if requested
    if config["workflow"]["conf_search"] and not (
        config["workflow"]["restart_dft"] or config["workflow"]["calculations_complete"]
    ):
        logger.info("Conformational search requested")
        data.get_conformers()
    else:
        logger.info("No conformational search requested")

    # DFT calculations if requested
    if (
        config["workflow"]["dft_energies"]
        or config["workflow"]["dft_nmr"]
        or config["workflow"]["dft_opt"]
    ):
        logger.info("DFT calculations requested")
        data.get_dft_data()
    else:
        logger.info("No DFT calculations requested")

    # Get NMR shifts if not using DFT for NMR
    if not config["workflow"]["dft_nmr"]:
        logger.info("Generating chemical shifts using a neural network")
        data.get_nn_nmr_shifts()

        if config["workflow"]["save_shifts"]:
            logger.info("Saving NMR shifts to SDF file")
            data.save_nmr_shifts_sdf()
    
    # Process NMR data if provided
    if not config.get("nmr_file"):
        logger.info("No NMR file provided, skipping NMR processing and analysis")
        return data
    
    # NMR processing and analysis
    nmr_data = NMRData(
        config["nmr_file"],
        config["solvent"],
        config["output_folder"],
    )
    data.assign_nmr_spectra(nmr_data)

    # Run analysis
    run_analysis(data, config)
    data.print_results()
    
    return data


def get_nmr_shifts(data, config):
    """
    Get NMR shifts either from pre-calculated data or by generating them.
    """
    logger.info("Generating chemical shifts using a neural network")
    data.get_nn_nmr_shifts()

    if config["workflow"]["save_shifts"]:
        logger.info("Saving NMR shifts to SDF file")
        data.save_nmr_shifts_sdf()


def run_analysis(data, config):
    """
    Run DP4 and DP5 analysis if requested.
    """
    if config["workflow"]["dp5"]:
        data.dp5_analysis()
    if config["workflow"]["dp4"]:
        data.dp4_analysis()
