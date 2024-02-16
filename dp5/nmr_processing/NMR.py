# -*- coding: utf-8 -*-
"""
Created on Mon Jan  12 14:42:47 2015

@author: ke291

Takes care of all the NMR description interpretation, equivalent atom
averaging, Boltzmann averaging and DP4 input preparation and running DP4.py. Called by PyDP4.py

FUNCTIONS AFTER REWRITE:
Calculation of NMR shifts based on TMS reference
Equivalent atom averaging
NMR description parsing
NMR raw data interpretation top level organization
"""

import re
import os
import math
import copy
import pickle
import logging

from dp5.nmr_processing import process_carbon, process_proton

from pathlib import Path
import shutil

gasConstant = 8.3145
temperature = 298.15
hartreeEnergy = 2625.499629554010

logger = logging.getLogger(__name__)

# Data structure for loading and keeping all of experimental NMR data in one place.

class NMRData:
    def __init__(self, structure, nmr_file, solvent, output_folder):

        self.cwd = Path(os.getcwd())
        self.structures = structure
        self.InputPath = nmr_file  # Initial structure input file
        self.Type = 'desc'          # desc or fid, depending on whether the description or raw data used
        self.OutputFolder = output_folder
        self.Atoms = []             # Element labels
        self.Cshifts = []           # Experimental C NMR shifts
        self.Clabels = []           # Experimental C NMR labels, if any
        self.Hshifts = []           # Experimental H NMR shifts
        self.Hlabels = []           # Experimental H NMR labels, if any
        self.Equivalents = []       # Atoms assumed to be NMR equivalent in computational data
        self.Omits = []
        self.protondata = {}
        self.carbondata = {}

        logger.info(f'Reading NMR data from{self.InputPath}')

        if len(self.InputPath) == 0:
            logger.critical('No NMR Data Added, quitting...')
            quit()

        else:

            for ind1 , p in enumerate(self.InputPath):

                if p.exists():

                    if p.is_dir():

                        self.Type = 'fid'

                        if p.parts[-1] == "Proton" or p.parts[-1] == "proton":

                            self.ProcessProton(settings,ind1)

                        elif p.parts[-1] == "Carbon" or p.parts[-1] == "carbon":

                            self.ProcessCarbon(settings,ind1)

                    elif p.parts[-1] == "Proton.dx" or p.parts[-1] == "proton.dx":

                        self.Type = 'jcamp'

                        self.ProcessProton(settings,ind1)

                    elif p.parts[-1] == "Carbon.dx" or p.parts[-1] == "carbon.dx":

                        self.Type = 'jcamp'

                        self.ProcessCarbon(settings,ind1)
                    else:

                        self.Type = 'desc'
                        self.ExpNMRFromDesc()

                else:
                    logger.critical('NMR data path does not exist, quitting...')
                    quit()

    def ExpNMRFromDesc(self):

        print('Loading NMR data from ' + str(self.InputPath))

        # Reads the experimental NMR data from the file
        ExpNMR_file = open(self.InputPath[0], 'r')
        Cexp = ExpNMR_file.readline()
        ExpNMR_file.readline()
        Hexp = ExpNMR_file.readline()

        # Check if exp NMR file contains info about equivalent atoms and read it
        # into an array
        # Also reads a list of atoms to omit from analysis

        equivalents = []
        omits = []

        ExpNMR_file.readline()
        for line in ExpNMR_file:
            if not 'OMIT' in line and len(line) > 1:
                equivalents.append(line[:-1].split(','))
            elif 'OMIT' in line:
                omits.extend(line[5:-1].split(','))

        ExpNMR_file.close()

        self.Clabels, self.Cshifts = self.ParseExp(Cexp)
        self.Hlabels, self.Hshifts = self.ParseExp(Hexp)
        self.Equivalents = equivalents
        self.Omits = omits

    def ParseExp(self, exp):

        if len(exp) > 0:

            # Replace all 'or' and 'OR' with ',', remove all spaces and 'any'
            texp = re.sub(r"or|OR", ',', exp, flags=re.DOTALL)
            texp = re.sub(r" ", '', texp, flags=re.DOTALL)

            # Get all assignments, split mulitassignments
            expLabels = re.findall(r"(?<=\().*?(?=\)|;)", texp, flags=re.DOTALL)
            expLabels = [x.replace('any', '') for x in expLabels]
            expLabels = [x.split(',') for x in expLabels]

            # Remove assignments and get shifts

            ShiftData = (re.sub(r"\(.*?\)", "", exp.strip(), flags=re.DOTALL)).split(',')

            print(ShiftData)

            expShifts = [float(x) for x in ShiftData]

        else:

            expLabels = []
            expShifts = []

        return expLabels, expShifts

    def ProcessProton(self, settings,ind):

        pdir = self.OutputFolder / "Pickles"

        gdir = self.OutputFolder /  "Graphs"

        NMR_file = self.InputPath[ind]

        if not Path(gdir).exists():

            os.mkdir(gdir)

            os.mkdir(gdir / self.structures[0])

        else:

            if not Path(gdir / self.structures[0]).exists():

                os.mkdir(gdir / self.structures[0])

        if not pdir.exists():

            os.mkdir(pdir)

            os.mkdir(pdir / self.structures[0])

        else:

            if not Path(pdir / self.structures[0]).exists():

                os.mkdir(pdir / self.structures[0])

        if Path(pdir / self.structures[0] /  "protondata").exists():

            self.protondata = pickle.load(open(pdir / self.structures[0] / "protondata", "rb"))

            self.Hshifts = self.protondata["exppeaks"]


        else:

            protondata = {}

            protondata["exppeaks"], protondata["xdata"], protondata["ydata"], protondata["integrals"], protondata[
                "peakregions"], protondata["centres"], \
            protondata["cummulativevectors"], protondata["integralsum"], protondata["picked_peaks"], protondata[
                "params"], protondata["sim_regions"] \
                = process_proton(NMR_file, settings,self.Type)

            pickle.dump(protondata, Path(pdir / self.structures[0] / "protondata").open(mode =  "wb+"))

            self.Hshifts = protondata["exppeaks"]

            self.protondata = protondata

    def ProcessCarbon(self, settings,ind):

        pdir = self.OutputFolder /  "Pickles"

        gdir = self.OutputFolder / "Graphs"


        NMR_file = self.InputPath[ind]

        if not Path(gdir).exists():

            os.mkdir(gdir)

            os.mkdir(gdir  / self.structures[0])

        else:

            if not Path(gdir  / self.structures[0]).exists():

                os.mkdir(gdir / self.structures[0])

        if not pdir.exists():

            os.mkdir(pdir)

            os.mkdir(pdir / self.structures[0])

        else:

            if not Path(pdir / self.structures[0]).exists():

                os.mkdir(pdir / self.structures[0])

        if Path(pdir / self.structures[0] / "carbondata").exists():

            self.carbondata = pickle.load(open(pdir / self.structures[0] / "carbondata", "rb"))

            self.Cshifts = self.carbondata["exppeaks"]

        else:

            carbondata = {}

            carbondata["ydata"], carbondata["xdata"], carbondata["corrdistance"], carbondata["uc"], \
            carbondata["exppeaks"], carbondata["simulated_ydata"], carbondata["removed"] = process_carbon(
                NMR_file, settings,self.Type)

            pickle.dump(carbondata, Path(pdir / self.structures[0] / "carbondata").open(mode =  "wb+"))

            #pickle.dump(a, Path("/Users/Maidenhair/Desktop/text.txt").open(mode="wb+"))

            self.carbondata = carbondata
            self.Cshifts = carbondata["exppeaks"]




def NMRDataValid(Isomers):

    for isomer in Isomers:
        if (len(isomer.ConformerShieldings) == 0):
            return False

    return True


def CalcNMRShifts(Isomers, settings):

    print('WARNING: NMR shift calculation currently ignores the instruction to exclude atoms from analysis')
    for i, iso in enumerate(Isomers):

        BShieldings = iso.BoltzmannShieldings

        Cvalues = []
        Hvalues = []
        Clabels = []
        Hlabels = []

        for a, atom in enumerate(iso.Atoms):

            if atom == 'C':
                shift = (settings.TMS_SC_C13-BShieldings[a]) / (1-(settings.TMS_SC_C13/10**6))
                Cvalues.append(shift)
                Clabels.append('C' + str(a + 1))

            if atom == 'H':
                shift = (settings.TMS_SC_H1-BShieldings[a]) / (1-(settings.TMS_SC_H1/10**6))
                Hvalues.append(shift)
                Hlabels.append('H' + str(a + 1))

        Isomers[i].Cshifts = Cvalues
        Isomers[i].Hshifts = Hvalues

        Isomers[i].Clabels = Clabels
        Isomers[i].Hlabels = Hlabels

        print('C shifts for isomer ' + str(i) + ": ")
        print(', '.join(['{0:.3f}'.format(x) for x in Isomers[i].Cshifts]))

        print('H shifts for isomer ' + str(i) + ": ")
        print(', '.join(['{0:.3f}'.format(x) for x in Isomers[i].Hshifts]))

        for conf in iso.ConformerShieldings:

            Cconfshifts = []
            Hconfshifts = []

            for a, atom in enumerate(iso.Atoms):

                if atom == 'C':

                    shift = (settings.TMS_SC_C13-conf[a]) / (1-(settings.TMS_SC_C13/10**6))
                    Cconfshifts.append(shift)

                if atom == 'H':
                    shift = (settings.TMS_SC_H1 - conf[a]) / (1 - (settings.TMS_SC_H1 / 10 ** 6))
                    Hconfshifts.append(shift)

            Isomers[i].ConformerCShifts.append(Cconfshifts)
            Isomers[i].ConformerHShifts.append(Hconfshifts)

    return Isomers


def PrintConformationData(AllSigConfs):
    """REDUNDANT"""
    # Make a list of populations and corresponding files for reporting
    # significant conformations
    """from operator import itemgetter
    ConfsPops = [list(x) for x in zip(args, populations)]
    ConfsPops.sort(key=itemgetter(1), reverse=True)
    totpop = 0
    i = 0
    while totpop < 0.8:
        totpop += ConfsPops[i][1]
        i += 1
    SigConfs = ConfsPops[:i]"""
    for Es, pops in zip(RelEs, populations):
        print('\nConformer relative energies (kJ/mol): ' + \
            ', '.join(["{:5.2f}".format(float(x)) for x in Es]))

        print('\nPopulations (%): ' + \
            ', '.join(["{:4.1f}".format(float(x)*100) for x in pops]))

    for i, SigConfs in enumerate(AllSigConfs):
        print("\nNumber of significant conformers for isomer "\
            + str(i+1) + ": " + str(len(SigConfs)) + "\n(pop, filename)")
        for conf in SigConfs:
            print("   " + format(conf[1]*100, "4.2f") + "%   " + conf[0])
        print('----------------')
        print("   " + format(100*sum([x[1] for x in SigConfs]), "4.2f") +\
            "%   in total")


def RemoveEquivalents(Noutp, equivs, OldCval, OldHval, OldClabels, OldHlabels):
    """
    Currently unused, will use older version to divine the original purpose
    """
    Cvalues = list(OldCval)
    Hvalues = list(OldHval)
    Clabels = list(OldClabels)
    Hlabels = list(OldHlabels)
    
    for eqAtoms in equivs:

        eqSums = [0.0]*Noutp
        eqAvgs = [0.0]*Noutp

        if eqAtoms[0][0] == 'H':
            #print eqAtoms, Hlabels
            for atom in eqAtoms:
                eqIndex = Hlabels.index(atom)
                for ds in range(0, Noutp):
                    eqSums[ds] = eqSums[ds] + Hvalues[ds][eqIndex]
            for ds in range(0, Noutp):
                eqAvgs[ds] = eqSums[ds]/len(eqAtoms)

            #Place the new average value in the first atom shifts place
            target_index = Hlabels.index(eqAtoms[0])
            for ds in range(0, Noutp):
                Hvalues[ds][target_index] = eqAvgs[ds]

            #Delete the redundant atoms from the computed list
            #start with second atom - e.g. don't delete the original one
            for atom in range(1, len(eqAtoms)):
                del_index = Hlabels.index(eqAtoms[atom])
                del Hlabels[del_index]
                for ds in range(0, Noutp):
                    del Hvalues[ds][del_index]

        if eqAtoms[0][0] == 'C':
            for atom in eqAtoms:
                eqIndex = Clabels.index(atom)
                for ds in range(0, Noutp):
                    eqSums[ds] = eqSums[ds] + Cvalues[ds][eqIndex]
            for ds in range(0, Noutp):
                eqAvgs[ds] = eqSums[ds]/len(eqAtoms)

            #Place the new average value in the first atom shifts place
            target_index = Clabels.index(eqAtoms[0])
            for ds in range(0, Noutp):
                Cvalues[ds][target_index] = eqAvgs[ds]

            #Delete the redundant atoms from the computed list
            #start with second atom - e.g. don't delete the original one
            for atom in range(1, len(eqAtoms)):
                del_index = Clabels.index(eqAtoms[atom])
                del Clabels[del_index]
                for ds in range(0, Noutp):
                    del Cvalues[ds][del_index]
                    
    return Cvalues, Hvalues, Clabels, Hlabels
    


def PairwiseAssignment(Isomers,NMRData):

    # for each isomer sort the experimental and calculated shifts

    for iso in Isomers:

        sortedCCalc = sorted(iso.Cshifts, reverse=True)
        sortedHCalc = sorted(iso.Hshifts, reverse=True)

        sortedCExp = sorted(NMRData.Cshifts, reverse=True)
        sortedHExp = sorted(NMRData.Hshifts, reverse=True)

        assignedCExp = [''] * len(sortedCCalc)
        assignedHExp = [''] * len(sortedHCalc)

        tempCCalcs = list(iso.Cshifts)
        tempHCalcs = list(iso.Hshifts)

        # do the assignment in order of chemical shift starting with the largest

        # Carbon

        for exp, shift in zip(sortedCExp, sortedCCalc):

            ind = tempCCalcs.index(shift)

            assignedCExp[ind] = exp

            tempCCalcs[ind] = ''

        # Proton

        for exp, shift in zip(sortedHExp, sortedHCalc):

            ind = tempHCalcs.index(shift)

            assignedHExp[ind] = exp

            tempHCalcs[ind] = ''

        # update isomers class

        iso.Cexp = assignedCExp
        iso.Hexp = assignedHExp

    return Isomers






