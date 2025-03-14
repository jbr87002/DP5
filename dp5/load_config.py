"""
The goal is to parse configuration. 

First import config file to supersede default, 
and override I/O from command line if required.

Takes one file with line-delineated Smiles or InChIs, or several SDFiles.
Returns the final config for the run
"""

from pathlib import Path
import argparse
import os
import logging
import shutil

import tomli
import json

from dp5.run import prepare_inputs
from .runner import runner
from .logger import setup_logger

LOGLEVEL_CHOICES = tuple(level.lower() for level in logging._nameToLevel.keys())

DEFAULT_BASE_CONFIG_PATH = (
    Path(__file__).parent / "config/default_config.toml"
).resolve()


def run_workflow(
    structure_files, 
    nmr_files=None, 
    output_path=None, 
    input_type="smiles", 
    workflow=None, 
    stereocentres=None, 
    model=None, 
    remove_previous=False, 
    skip_nmr=False,
    shifts_from_cache=False,
    use_precalculated=False,
    precalculated_sdf=None
):
    """
    Run the DP5 workflow.
    
    Args:
        structure_files: List of structure files (SMILES, InChI, or SDF)
        nmr_files: NMR data files (optional)
        output_path: Output directory
        input_type: Type of input files ('smiles', 'inchi', 'sdf')
        workflow: Workflow flags
        stereocentres: List of stereocentres to enumerate
        model: Neural network model to use
        remove_previous: Whether to remove previous output
        skip_nmr: Whether to skip NMR processing
        shifts_from_cache: Whether to load shifts from cache
        use_precalculated: Whether to use pre-calculated data from SDF files
        precalculated_sdf: Path to pre-calculated SDF file (optional)
        
    Returns:
        Molecules object with results
    """

    # load custom configuration
    config_path = DEFAULT_BASE_CONFIG_PATH
    if config_path.suffix == ".toml":
        with open(config_path, "rb") as f:
            config = tomli.load(f)
    elif config_path.suffix == ".json":
        with open(config_path, "rb") as f:
            config = json.load(f)

    logger = setup_logger(
        name=__package__,
        level=config["log_level"].upper(),
        filename="",
        propagate=True,
    )

    logger.info("Preparing configuration")
    
    # Set up precalculated data options
    config["workflow"]["use_precalculated"] = use_precalculated
    if precalculated_sdf:
        config["precalculated_sdf"] = precalculated_sdf
    else:
        config["precalculated_sdf"] = None
    
    # Override workflow flag
    if workflow is not None:
        config["workflow"]["cleanup"] = "c" in workflow
        config["workflow"]["generate"] = "g" in workflow
        config["workflow"]["conf_search"] = "m" in workflow
        config["workflow"]["dft_nmr"] = "n" in workflow
        config["workflow"]["dft_energies"] = "e" in workflow
        config["workflow"]["dft_opt"] = "o" in workflow
        config["workflow"]["dp4"] = "s" in workflow
        config["workflow"]["dp5"] = "w" in workflow
        config["workflow"]["assign_only"] = "a" in workflow
        config["workflow"]["calculate_nmr_shifts"] = "N" in workflow

    # If using precalculated data, we can skip many steps
    if use_precalculated:
        logger.info("Using pre-calculated data, skipping unnecessary workflow steps")
        # If we're using pre-calculated data, we don't need to do these steps
        config["workflow"]["cleanup"] = False
        config["workflow"]["generate"] = False
        config["workflow"]["conf_search"] = False
        config["workflow"]["dft_nmr"] = False
        config["workflow"]["dft_energies"] = False
        config["workflow"]["dft_opt"] = False
        # But we still want to load shifts from cache
        config["workflow"]["shifts_from_cache"] = True
    
    logger.info(f"Structure cleanup required: {config['workflow']['cleanup']}")
    logger.info(f"Diastereomer generation required: {config['workflow']['generate']}")
    logger.info(f"Conformational search required: {config['workflow']['conf_search']}")
    logger.info(f"DFT geometry optimisation required: {config['workflow']['dft_opt']}")
    logger.info(
        f"DFT energy calculation required: {config['workflow']['dft_energies']}"
    )
    logger.info(f"DFT shielding calculation required: {config['workflow']['dft_nmr']}")
    logger.info(f"DP4 analysis required: {config['workflow']['dp4']}")
    logger.info(f"DP5 analysis required: {config['workflow']['dp5']}")
    
    if use_precalculated:
        logger.info(f"Using pre-calculated data: {config['workflow']['use_precalculated']}")
        if precalculated_sdf:
            logger.info(f"Pre-calculated SDF file: {config['precalculated_sdf']}")

    # calculations_complete should set MM and DFT completion flags as True
    if config["workflow"]["calculations_complete"] or use_precalculated:
        config["workflow"]["mm_complete"] = True
        config["dft"]["dft_complete"] = True
        config["workflow"]["dft_complete"] = True

    # reads command line argument if supplied, else reads config
    if structure_files:
        config["structure"] = structure_files
        config["input_type"] = input_type
        config["stereocentres"] = stereocentres
        logger.debug(
            f"Read structures {', '.join(config['structure'])} from command line"
        )
    elif config["structure"]:
        logger.debug(
            f"Read structures {', '.join(config['structure'])} from config file"
        )
    else:
        logger.critical("No structures specified")
        raise ValueError("No structures specified")

    logger.info(f"Structure input files: {', '.join(config['structure'])}")

    if model:
        config["nn_model"]["model"] = model
    
    if shifts_from_cache:
        config["workflow"]["shifts_from_cache"] = shifts_from_cache

    # Handle NMR file - optional input, not needed if just calculating NMR shifts
    if skip_nmr:
        logger.info("Skipping NMR processing as requested")
        config["nmr_file"] = None
    elif nmr_files:
        logger.debug(f"Read NMR File {nmr_files} from command line")
        config["nmr_file"] = nmr_files
    elif config["nmr_file"]:
        logger.debug(f"Read NMR File {config['nmr_file']} from config file")
    else:
        logger.info("No NMR data specified, will only calculate NMR shifts")
        config["nmr_file"] = None
        # Disable DP4 and DP5 analysis if no NMR data
        config["workflow"]["dp4"] = False
        config["workflow"]["dp5"] = False

    # Only set up TMS constants if we're doing DFT NMR calculations
    if config["workflow"]["dft_nmr"] or not config["nmr_file"]:
        # set up TMS constants
        with open((Path(__file__).parent / "dft" / "TMSdata").resolve()) as file:
            _params_found = False
            _solvent = config["dft"]["solvent"] if config["dft"]["solvent"] else "none"
            for line in file:
                line = line.strip()
                if line:
                    functional, basis_set, solvent, tms_c, tms_h = line.split()
                    if (
                        config["dft"]["n_functional"] == functional
                        and config["dft"]["n_basis_set"] == basis_set
                        and _solvent == solvent
                    ):
                        _params_found = True

                        config["dft"]["c13_tms"] = float(tms_c)
                        config["dft"]["h1_tms"] = float(tms_h)

                        break

        if not _params_found:
            logger.warning(
                "No reference shielding found for the conditions, using default values!"
            )
            functional, basis_set, solvent = ("b3lyp", "6-31G**", "none")

        logger.info("Read shielding parameters for: ")
        logger.info(
            "NMR DFT functional: %s, basis set: %s, solvent: %s",
            functional,
            basis_set,
            solvent,
        )

        logger.info(f"13C reference shielding: {config['dft']['c13_tms']:.1f} ppm")
        logger.info(f"1H reference shielding: {config['dft']['h1_tms']:.2f} ppm")

    if output_path:
        config["output_folder"] = output_path
    config["output_folder"] = (Path.cwd() / config["output_folder"]).resolve()

    config["dft"]["solvent"] = config["solvent"]

    # Skip prepare_inputs if using pre-calculated data from SDF
    if use_precalculated and input_type == "sdf":
        logger.info("Using pre-calculated SDF files, skipping input preparation")
        # Just use the structure files as is
    else:
        config["structure"] = prepare_inputs(
            config["structure"],
            config["input_type"],
            config["stereocentres"],
            config["workflow"],
            config["nn_model"],
            ignore_sanitise_error=config["workflow"]["calculate_nmr_shifts"],
            output_folder=config["output_folder"]
        )

    logger.info(f"Final structure input files:{config['structure']}")

    if config["nmr_file"]:
        logger.info(f"NMR input paths:{config['nmr_file']}")

    # Create output directory if it doesn't exist
    config["output_folder"].mkdir(parents=True, exist_ok=True)
    
    with open(config["output_folder"] / "config.json", "w") as f:
        cfg = config.copy()
        cfg["output_folder"] = str(cfg["output_folder"])
        json.dump(cfg, f, indent=4)
    
    if remove_previous:
        for folder in ['dp4', 'dp5']:
            folder_path = config["output_folder"] / folder
            if folder_path.exists():
                logger.info(f"Removing {folder_path}")
                shutil.rmtree(folder_path)

    logger.info("Configuration saved to %s" % str(config["output_folder"]))

    # run the workflow and get the data object
    data = runner(config)

    logger.info("Program terminated normally")
    
    return data