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

    data = Molecules(config)

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
        missing_mols = []
        
        # Try to load shifts from SDF or JSON, depending on configuration
        if config["workflow"].get("shifts_from_cache", False):
            try:
                # First try to load from SDF if available
                logger.info("Attempting to load NMR shifts from SDF cache")
                shifts_loaded, missing_mols = data.load_nmr_shifts_sdf()
                
                # If SDF loading fails, try JSON as fallback
                if not shifts_loaded:
                    logger.info("SDF loading failed, attempting to load from JSON cache")
                    shifts_loaded, missing_mols = data.load_nmr_shifts()
            except Exception as e:
                logger.error(f"Error loading NMR shifts from cache: {e}")
                shifts_loaded = False

        # Generate shifts if loading failed or wasn't requested
        if not shifts_loaded:
            logger.info("Generating chemical shifts using a neural network")
            data.get_nn_nmr_shifts()
    
            # Save newly predicted NMR shifts to both formats
            logger.info("Saving NMR shifts to SDF files")
            data.save_nmr_shifts_sdf()
            
    # If nmr_file is not provided, skip NMR processing and DP4/DP5 analysis
    if not config.get("nmr_file"):
        logger.info("No NMR file provided, skipping NMR processing and analysis")

        return data
    
    # heading into legacy code area
    nmr_data = NMRData(
        config["nmr_file"],
        config["solvent"],
        config["output_folder"],
    )
    # process data first!!!!
    data.assign_nmr_spectra(nmr_data)

    # now that we have assigned it, time for DP4
    if config["workflow"]["dp5"]:
        data.dp5_analysis()
    if config["workflow"]["dp4"]:
        data.dp4_analysis()

    data.print_results()
    
    return data
