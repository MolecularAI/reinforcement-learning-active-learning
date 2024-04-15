"""Perform a shape query against a molecule of interest.

Packages: os, sys,
openeye: oechem, oeomega, oequacpac, oeshape,
fileinput,
argparse.

Functions:
get_ROCS_score: minimal requirement takes input smiles and shape_query as sq file and outputs a tanimoto similarity score.

Keyword arguments:
smile -- a smiles string as type (str) - (no default)
shape_query -- input is a shape query, can be produced in vrocs using a reference ligand, type (.sq file) - (no default)
max_stereo -- number of stereocenters to be enumerated in input molecule, type (int) - (default 0)
max_confs -- maximum number of conformers generated of stereoisomers, type (int) - (default 200)
e_window -- energy window, the difference between the lowest and highest energy conformer, type (int) - (default 10)
sim_measure -- measurement of similarity between reference and fit molecule, type (str) - (default 'Tanimoto')
    options: (RefTversky or FitTversky)
colour_factor -- weight applied to score from colour matching in chosen sim_measure, type (int) - (default 0.5)
shape_factor -- weight applied to score from shape matching in chosen sim_measure, type (int) - (default 0.5)
save_overlay -- outputs 3D coordinates of best conformer to SDF file, type (bool) - (default False)
overlay_path -- if save_overlay is True, outputs SDF file to path, type (str) filetype(.sdf) - (default './outmol.sdf')

Example input:

python3 rocs_scoring.py '[O-]c1ccc(C(N)=[NH2+])cc1Oc2nc(c(F)c(c2F)N(C)CC([O-])=O)Oc3cccc(c3)C4=[N+](C)CCN4' 'rl-al-repo/data/CoX2/shape_query/S58.sq' --save_overlay True --overlay_path ./test.sdf

python3 rocs_scoring.py 'NS(=O)(=O)c1ccc(-n2nc(C(F)(F)F)cc2-c2ccc(Br)cc2)cc1' 'rl-al-repo/data/CoX2/shape_query/S58.sq' --save_overlay True --overlay_path ./test.sdf


"""


import os, sys
from openeye import oechem, oeomega, oequacpac, oeshape
import fileinput
import argparse

import os


def recursion(base_dir, input, i=0):
    new_path = os.path.join(base_dir, f"{str(i)}.txt")
    while os.path.exists(new_path):
        i += 1
        new_path = os.path.join(base_dir, f"{str(i)}.txt")
        print(i)
        print(new_path)
    else:
        with open(new_path, "w") as f:
            print("success")
            f.writelines(f"{new_path} is the current text file\n")
            f.writelines(f"this is the smiles string that has been recieved {input}\n")
    return new_path


def get_ROCS_score(
    smile,
    shape_query,
    max_stereo=0,
    max_confs=200,
    e_window=10,
    sim_measure="Tanimoto",
    colour_factor=0.5,
    shape_factor=0.5,
    save_overlay=False,
    overlay_path="./outmol.sdf",
):
    def smiles_to_mol(smile):
        imol = oechem.OEMol()
        outmol = oechem.OEMol()
        oechem.OESmilesToMol(imol, smile)
        # oequacpac.OEGetReasonableProtomer(imol)
        return imol

    def omega_opts(e_window, max_confs):
        omegaOpts = oeomega.OEOmegaOptions()
        omegaOpts.SetStrictStereo(False)
        omegaOpts.SetEnergyWindow(int(e_window))
        omegaOpts.SetMaxConfs(int(max_confs))
        omegaOpts.GetTorDriveOptions().SetUseGPU(False)

        omega = oeomega.OEOmega(omegaOpts)
        return omega

    def flipper(imol, omega, max_stereo):
        enantiomers = list(oeomega.OEFlipper(imol.GetActive(), max_stereo, False, True))
        for k, enantiomer in enumerate(enantiomers):
            enantiomer = oechem.OEMol(enantiomer)
            omega.Build(enantiomer)
            if k == 0:
                imol = oechem.OEMol(enantiomer.SCMol())
                imol.DeleteConfs()
            for x in enantiomer.GetConfs():
                imol.NewConf(x)
        return imol

    def mol_to_multi_conformer(imol, omega):
        ret_code = omega.Build(imol)
        if ret_code == oeomega.OEOmegaReturnCode_Success:
            return imol
        else:
            return 0

    def prep_shape_query(shape_query, imol):
        prep = oeshape.OEOverlapPrep()
        qry = oeshape.OEShapeQuery()
        overlay = oeshape.OEOverlay()
        oeshape.OEReadShapeQuery(shape_query, qry)
        overlay.SetupRef(qry)
        rocs_overlay = overlay
        prep.Prep(imol)
        return rocs_overlay, imol

    def return_scores(imol, sim_measure, rocs_overlay):
        score = oeshape.OEBestOverlayScore()
        if sim_measure == "Tanimoto":
            rocs_overlay.BestOverlay(score, imol, oeshape.OEHighestTanimotoCombo())
            result = (score.GetTanimoto() * float(shape_factor)) + (
                score.GetColorTanimoto() * float(colour_factor)
            )
        elif sim_measure == "RefTversky":
            rocs_overlay.BestOverlay(score, imol, oeshape.OEHighestRefTverskyCombo())
            result = (score.GetRefTversky() * float(shape_factor)) + (
                score.GetRefColorTversky() * float(colour_factor)
            )
        elif sim_measure == "FitTversky":
            rocs_overlay.BestOverlay(score, imol, oeshape.OEHighestFitTverskyCombo())
            result = (score.GetFitTversky() * float(shape_factor)) + (
                score.GetFitColorTversky() * float(colour_factor)
            )
        return score, result


    ## Take a smile string, and convert it to a mol object - OEmol base class.
    imol = smiles_to_mol(smile)

    print("protomer")
    oequacpac.OEGetReasonableProtomer(imol)

    ## Add the options for omega - energy window (gap between lowest and highest energy conformer), max_confs
    ## the maximum number of generated confs
    print("omega")
    omega = omega_opts(e_window, max_confs)

    ## Flipper is a utility which allows for the generation of stereoisomers
    print("flipper")
    flipper(imol, omega, max_stereo)
    ## mol_to_multi_conformer generates multiple conformations for each isomer
    print("imol")
    imol = mol_to_multi_conformer(imol, omega)
    ## ROCS_Overlay is function that optimises for overlap between two molecules, prep_shape_query uses an SQ file to assign
    ## the region that will be overlapped for input molecules against a reference molecule
    print("prep_shape_query")
    try:
        rocs_overlay, imol = prep_shape_query(shape_query, imol)
    except NotImplementedError:
        # new_path = recursion(base_dir=base_dir, input=smile)
        # oechem.OESetSDData(imol, "Smiles", smile)
        # oechem.OESetSDData(imol, "docking_score", str(result))
        print(f"ROCS failed, generating dummy config at {overlay_path}")
        ofs = oechem.oemolostream(overlay_path)
        imol = smiles_to_mol(smile)
        oechem.OESetSDData(imol, "Smiles", smile)
        oechem.OESetSDData(imol, "docking_score", str(0.0))
        oechem.OEWriteMolecule(ofs, imol)

        return 0.0, imol

    ## return_scores searches for the best conformer which overlays the query molecule
    print("return_scores")
    score, result = return_scores(imol, sim_measure, rocs_overlay)
    print(score, result)
    ## outmol is used here to retrieve the 3D coords of the best conformer, and returns a mol object
    print("outmol")
    outmol = oechem.OEGraphMol(imol.GetConf(oechem.OEHasConfIdx(score.GetFitConfIdx())))
    print("overlay")
    if save_overlay:
        ## Exports best conformer as SDF file with score and smiles string
        ofs = oechem.oemolostream(overlay_path)
        oeshape.OERemoveColorAtoms(outmol)
        score.Transform(outmol)
        oechem.OESetSDData(outmol, "Smiles", smile)
        oechem.OESetSDData(outmol, "docking_score", str(result))
        oechem.OEWriteMolecule(ofs, outmol)

    # with open(new_path, "a") as f:
    #     f.write("This is an sdf object")
    #     try:
    #         f.write(f"{outmol}\n")
    #     except:
    #         pass

    return result, outmol


parser = argparse.ArgumentParser()

parser.add_argument("smile", type=str, help="molecule smiles input")
parser.add_argument(
    "shape_query",
    help="Shape queries are 3D colour representations of molecules, vrocs can be used to prepare a shape query and exported as an .sq file",
)
parser.add_argument("--max_stereo", type=int, required=False, default=3)
parser.add_argument("--max_confs", type=int, default=200)
parser.add_argument("--e_window", type=int, default=10)
parser.add_argument("--sim_measure", type=str, default="Tanimoto")
parser.add_argument("--colour_factor", type=int, default=0.5)
parser.add_argument("--shape_factor", type=int, default=0.5)
parser.add_argument("--save_overlay", type=bool, default=True)
parser.add_argument("--overlay_path", type=str, default="./outmol.sdf")

args = parser.parse_args()

result, outmol = get_ROCS_score(
    smile=args.smile,
    shape_query=args.shape_query,
    max_stereo=args.max_stereo,
    max_confs=args.max_confs,
    e_window=args.e_window,
    sim_measure=args.sim_measure,
    colour_factor=args.colour_factor,
    shape_factor=args.shape_factor,
    save_overlay=args.save_overlay,
    overlay_path=args.overlay_path,
)
# print(result)
