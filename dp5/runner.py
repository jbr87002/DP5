"""
The main script. Replaces PyDP4.py, orchestrates the high-level workflow.

Molecules is the container class to contain all the data. MM and DFT methods no longer modify it directly to improve maintainability.
"""

import logging
import os
from pathlib import Path

from dp5.run.data_structures import Molecules
from dp5.nmr_processing import NMRData


logger = logging.getLogger(__name__)


def runner(config):
    logger.info("Starting DP4 workflow")

    # Check if we should load from a checkpoint
    if config.get("load_checkpoint"):
        checkpoint_path = Path(config["load_checkpoint"])
        if checkpoint_path.exists():
            logger.info(f"Loading from checkpoint: {checkpoint_path}")
            data = Molecules.load(checkpoint_path)
            # Update config in case it has changed
            data.config = config
        else:
            logger.warning(f"Checkpoint file {checkpoint_path} not found. Starting from scratch.")
            data = Molecules(config)
    else:
        data = Molecules(config)

    # Save initial state if checkpoints are enabled
    if config.get("save_checkpoints", False):
        data.save(Path(config["output_folder"]) / "molecules_initial.pkl")

    if config["workflow"]["conf_search"] and not (
        config["workflow"]["restart_dft"] or config["workflow"]["calculations_complete"]
    ):

        logger.info("Conformational search requested")
        data.get_conformers()

    else:
        logger.info("No conformational search requested")

    if (
        config["workflow"]["dft_energies"]
        or config["workflow"]["dft_nmr"]
        or config["workflow"]["dft_opt"]
    ):

        logger.info("DFT calculations requested")
        data.get_dft_data()

    else:
        logger.info("No DFT calculations requested")

    if not config["workflow"]["dft_nmr"]:
        shifts_loaded = False
        try:
            logger.info("Attempting to load NMR shifts from cache")
            data.load_nmr_shifts()
            shifts_loaded = True
        except Exception as e:
            logger.error(f"Error loading NMR shifts from cache: {e}")

        if not shifts_loaded:
            logger.info("Generating chemical shifts using a neural network")
            data.get_nn_nmr_shifts()
    
            # Save newly predicted NMR shifts to JSON file
            logger.info("Saving NMR shifts to JSON file")
            data.save_nmr_shifts()

    # If nmr_file is not provided, skip NMR processing and DP4/DP5 analysis
    if not config.get("nmr_file"):
        logger.info("No NMR file provided, skipping NMR processing and analysis")
        
        # Save final state if checkpoints are enabled
        if config.get("save_checkpoints", False):
            data.save(Path(config["output_folder"]) / "molecules_final.pkl")
            
        # Return the data object for further use
        return data
    
    # heading into legacy code area
    nmr_data = NMRData(
        config["nmr_file"],
        config["solvent"],
        config["output_folder"],
    )
    # process data first!!!!
    data.assign_nmr_spectra(nmr_data)

    # Save after NMR assignment if checkpoints are enabled
    if config.get("save_checkpoints", False):
        data.save(Path(config["output_folder"]) / "molecules_after_nmr_assignment.pkl")

    # now that we have assigned it, time for DP4
    if config["workflow"]["dp5"]:
        data.dp5_analysis()
    if config["workflow"]["dp4"]:
        data.dp4_analysis()

    # Save final state if checkpoints are enabled
    if config.get("save_checkpoints", False):
        data.save(Path(config["output_folder"]) / "molecules_final.pkl")

    data.print_results()
    
    # Return the data object for further use
    return data
