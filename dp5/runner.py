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

    # Initialize the Molecules object
    data = Molecules(config)
    
    # Check if we're using pre-calculated data
    if config["workflow"].get("use_precalculated", False):
        logger.info("Using pre-calculated data workflow")
        
        # If a specific pre-calculated SDF file is provided, try to load it
        if config.get("precalculated_sdf"):
            precalc_sdf = Path(config["precalculated_sdf"])
            if precalc_sdf.exists():
                logger.info(f"Loading pre-calculated data from {precalc_sdf}")
                success, missing_mols = data.load_nmr_shifts_sdf(precalc_sdf)
                if not success:
                    logger.warning("Failed to load pre-calculated data from specified SDF file")
                    if missing_mols:
                        logger.warning(f"Missing data for molecules: {', '.join(missing_mols)}")
                    logger.info("Falling back to standard workflow")
                else:
                    logger.info("Successfully loaded pre-calculated data")
                    # Skip to NMR processing if needed
                    if config.get("nmr_file"):
                        logger.info("Proceeding to NMR processing with pre-calculated shifts")
                        # Skip to NMR processing section
                        nmr_data = NMRData(
                            config["nmr_file"],
                            config["solvent"],
                            config["output_folder"],
                        )
                        data.assign_nmr_spectra(nmr_data)

                        # Run DP4/DP5 analysis if requested
                        if config["workflow"]["dp5"]:
                            data.dp5_analysis()
                        if config["workflow"]["dp4"]:
                            data.dp4_analysis()

                        data.print_results()
                        return data
                    else:
                        logger.info("No NMR file provided, skipping NMR processing and analysis")
                        return data
            else:
                logger.warning(f"Pre-calculated SDF file not found: {precalc_sdf}")
                logger.info("Falling back to standard workflow")
        else:
            # Try to load from the default location
            logger.info("Checking for pre-calculated data in default location")
            sdf_file = Path(config["output_folder"]) / "nmr_shifts.sdf"
            if sdf_file.exists():
                logger.info(f"Found pre-calculated data at {sdf_file}")
                success, missing_mols = data.load_nmr_shifts_sdf(sdf_file)
                if not success:
                    logger.warning("Failed to load pre-calculated data from default location")
                    if missing_mols:
                        logger.warning(f"Missing data for molecules: {', '.join(missing_mols)}")
                    logger.info("Falling back to standard workflow")
                else:
                    logger.info("Successfully loaded pre-calculated data")
                    # Skip to NMR processing if needed
                    if config.get("nmr_file"):
                        logger.info("Proceeding to NMR processing with pre-calculated shifts")
                        # Skip to NMR processing section
                        nmr_data = NMRData(
                            config["nmr_file"],
                            config["solvent"],
                            config["output_folder"],
                        )
                        data.assign_nmr_spectra(nmr_data)

                        # Run DP4/DP5 analysis if requested
                        if config["workflow"]["dp5"]:
                            data.dp5_analysis()
                        if config["workflow"]["dp4"]:
                            data.dp4_analysis()

                        data.print_results()
                        return data
                    else:
                        logger.info("No NMR file provided, skipping NMR processing and analysis")
                        return data
            else:
                logger.warning(f"No pre-calculated data found at default location: {sdf_file}")
                logger.info("Falling back to standard workflow")
    
    # If we're using SMILES input and pre-calculated data, try to match by InChI key
    if config["workflow"].get("use_precalculated", False) and config["input_type"] == "smiles":
        logger.info("Checking if SMILES structures match pre-calculated data")
        # This will be handled by the load_nmr_shifts_sdf method which matches by InChI key
        
        # Try to find pre-calculated data in common locations
        possible_locations = [
            Path(config["output_folder"]) / "nmr_shifts.sdf",
            Path.cwd() / "nmr_shifts.sdf",
            Path.cwd() / "data" / "nmr_shifts.sdf"
        ]
        
        for location in possible_locations:
            if location.exists():
                logger.info(f"Found potential pre-calculated data at {location}")
                success, missing_mols = data.load_nmr_shifts_sdf(location)
                if success:
                    logger.info(f"Successfully loaded pre-calculated data from {location}")
                    # Skip to NMR processing if needed
                    if config.get("nmr_file"):
                        logger.info("Proceeding to NMR processing with pre-calculated shifts")
                        # Skip to NMR processing section
                        nmr_data = NMRData(
                            config["nmr_file"],
                            config["solvent"],
                            config["output_folder"],
                        )
                        data.assign_nmr_spectra(nmr_data)

                        # Run DP4/DP5 analysis if requested
                        if config["workflow"]["dp5"]:
                            data.dp5_analysis()
                        if config["workflow"]["dp4"]:
                            data.dp4_analysis()

                        data.print_results()
                        return data
                    else:
                        logger.info("No NMR file provided, skipping NMR processing and analysis")
                        return data
                    break
        else:
            logger.warning("No matching pre-calculated data found for SMILES input")
            logger.info("Falling back to standard workflow")
    
    # Standard workflow if not using pre-calculated data or if loading failed
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
    
            # Save newly predicted NMR shifts to SDF file
            logger.info("Saving NMR shifts to SDF file")
            data.save_nmr_shifts_sdf()
            
    # If nmr_file is not provided, skip NMR processing and DP4/DP5 analysis
    if not config.get("nmr_file"):
        logger.info("No NMR file provided, skipping NMR processing and analysis")
        return data
    
    # NMR processing and analysis
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
