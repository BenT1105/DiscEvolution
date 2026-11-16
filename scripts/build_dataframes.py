"""
build_dataframes.py
===================

Read the HDF5 output files of a population synthesis run and save the
dataframes used by notebooks/popsynth_plots.ipynb as pickle files, so the
notebook never has to open the .h5 files itself.

Three dataframes are built, each saved to its own pickle file:

    df_planets_{MODEL}.pkl   one row per planet: final state, implantation and disc parameters
    df_discs_{MODEL}.pkl     one row per disc: the disc parameters from the HDF5 attributes
    df_tracks_{MODEL}.pkl    one row per planet: thinned growth track (position and total mass) and disc parameters

The disc parameters measured from the disc (Mdisk, Mdot, Rd) are rounded back
to the grid values the population synthesis was run with, and Mdot is saved
in Msun/Myr (the HDF5 attribute is in Msun/yr).

Only simulations that have completed are included. A dataframe whose pickle
file already exists is skipped unless its --rebuild flag is given.
"""

import os
import argparse
import h5py
import numpy as np
import pandas as pd

## ----------------
## Default settings
## ----------------

## Name of the population synthesis model
MODEL = "testmodel"

## Directory of .h5 output files and directory the pickle files are saved to
OUTPUT_DIR = "/home/bentobin/simulations/{model}/output"
SAVE_DIR = "/home/bentobin/simulations/{model}/dataframes"

## Maximum number of points kept per growth track
MAX_TRACK_POINTS = 2000

## Set to True to rebuild a dataframe even if it is already saved
REBUILD_PLANETS = False
REBUILD_DISCS = False
REBUILD_TRACKS = False

## Set to True to rebuild every dataframe even if it is already saved
REBUILD = False

## --------------------------
## Methods to read HDF5 files
## --------------------------

## Method to read data from an HDF5 dataset or group

def read_hdf5_array(obj):
    """
    Read data from either an HDF5 dataset or an HDF5 group.
    """

    ## Normal case (the data are stored as one HDF5 dataset)
    if isinstance(obj, h5py.Dataset):
        return np.array(obj)

    ## Ragged case (the data are stored as a group)
    elif isinstance(obj, h5py.Group):
        values = []

        for key in sorted(obj.keys(), key = int):
            values.append(np.array(obj[key]))

        return values

    ## Raise an error if an unexpected HDF5 object is encountered
    else:
        raise TypeError(f"Unexpected HDF5 object type: {type(obj)}")

## Method to load data from an HDF5 file

def load_data(data_dir):
    """
    Load a disc evolution output file (HDF5 only).

    Parameters:
        data_dir: Path to the HDF5 output file.

    Returns:
        data: Dictionary containing the simulation output.
    """

    data = {}

    ## Open the HDF5 file in read-only mode
    with h5py.File(data_dir, "r") as f:

        ## -----------------------
        ## General simulation data
        ## -----------------------

        general_keys = [
            "R",
            "t",
            "disk_Mdot_star",
            "disk_Mass",
            "Tc",
            "Sigc"]

        for key in general_keys:
            ## Only load the quantity if it exists in the file
            if key in f:
                data[key] = np.array(f[key])

        ## -----------------
        ## Scalar attributes
        ## -----------------

        scalar_attributes = [
            "alpha_SS",
            "psi_DW",
            "Mdot",
            "Mdisk",
            "Rd",
            "pla_eff",
            "f_plt",
            "complete",
            "aborted"]

        for key in scalar_attributes:
            ## Only load the attribute if it exists in the file
            if key in f.attrs:
                data[key] = f.attrs[key].item()

        ## Names of the chemical species (only written when chemistry is on)
        if "chem_species" in f.attrs:
            data["chem_species"] = [s.decode() if isinstance(s, bytes) else str(s) for s in f.attrs["chem_species"]]

        ## -------------
        ## Disk profiles
        ## -------------

        disk_keys = [
            "T",
            "time_snap",
            "Sigma_G",
            "Sigma_dust",
            "Sigma_pebbles",
            "Sigma_planetesimals",
            "Vdrift_grains",
            "Vdrift_pebbles",
            "St_grains",
            "St_pebbles",
            "St_planetesimals",
            "e_planetesimals",
            "i_planetesimals",
            "de2_dt",
            "di2_dt",
            "de2_dt_drag",
            "di2_dt_drag",
            "de2_dt_VS_embryo",
            "di2_dt_VS_embryo",
            "de2_dt_VS_pltsml",
            "di2_dt_VS_pltsml",
            "de2_dt_DF",
            "di2_dt_DF",
            "gas_chem",
            "ice_chem",
            "planetesimal_ice_chem"]

        for key in disk_keys:
            ## Only load the quantity if it exists in the file
            if key in f:
                data[key] = read_hdf5_array(f[key])

        ## -----------
        ## Planet data
        ## -----------

        planet_keys = [
            "Mcs",
            "Mes",
            "Rp",
            "disk_Mdot_p",
            "Mdot_planetesimal",
            "Mdot_pebble_core",
            "Mdot_pebble_env",
            "Mdot_migration",
            "Mdot_gas",
            "M_iso_planetesimal",
            "M_iso_pebble",
            "X_cores",
            "X_envs",
            "f_env",
            "t_form"]

        for key in planet_keys:
            ## Only load the quantity if it exists in the file
            if key in f:
                data[key] = read_hdf5_array(f[key])

    return data

## --------------------------------------
## Method to round the disc parameters
## --------------------------------------

GRID_KEYS = ["Mdisk", "Mdot", "Rd", "alpha_SS", "psi_DW", "pla_eff", "f_plt"]

def round_grid_params(params):
    """
    Round the disc parameters that are measured from the disc back to the grid
    values the population synthesis was run with, and convert Mdot to Msun/Myr.
    Mdisk is rounded to 3 decimal places, Mdot to 2 significant figures (in
    Msun/Myr) and Rd to the nearest AU. alpha_SS, pla_eff and f_plt are saved
    exactly as input and psi_DW is solved per disc, so they are left unchanged.

    Parameters:
        params: Dictionary of disc parameters read from the HDF5 attributes (Mdot in Msun/yr).

    Returns:
        params: New dictionary with Mdisk, Mdot [Msun/Myr] and Rd rounded to the grid values.
    """

    params = dict(params)

    if "Mdisk" in params:
        params["Mdisk"] = round(params["Mdisk"], 3)

    if "Mdot" in params:
        params["Mdot"] = float(f"{params['Mdot'] * 1e6:.2g}")

    if "Rd" in params:
        params["Rd"] = round(params["Rd"], 0)

    return params

## -------------------------------
## Methods to build the dataframes
## -------------------------------

## Method to build the planet dataframe

def build_planet_dataframe(output_dir, pkl_file):
    """
    Loop over every completed HDF5 output file, store the final state of each
    planet (plus its initial position, implant time and the simulation metadata)
    as one row of a dataframe, and save the dataframe as a pickle file.

    Parameters:
        output_dir: Directory containing the .h5 output files.
        pkl_file: Path to save the pickled dataframe to.

    Returns:
        df_planets: Dataframe with one row per planet.
    """

    planet_rows = []
    chem_species = None
    n_files = 0

    ## ----------------------
    ## Main loop over files
    ## ----------------------

    for fname in sorted(os.listdir(output_dir)):

        if not fname.endswith(".h5"):
            continue

        fpath = os.path.join(output_dir, fname)

        try:
            ## Load raw simulation output
            data = load_data(fpath)

            ## Only use simulations that have completed
            if not bool(data.get("complete", False)):
                continue

            ## Skip if no planets
            if "Mcs" not in data or len(data["Mcs"]) == 0:
                continue

            n_files += 1
            chem_species = data.get("chem_species", chem_species)

            ## Simulation metadata from the HDF5 attributes (shared by every planet in this file)
            metadata = {
                **round_grid_params({key: data.get(key, np.nan) for key in GRID_KEYS}),
                "last_t": float(data["t"][-1]),
            }

            Mcs = np.array(data["Mcs"])
            Mes = np.array(data["Mes"])
            Rps = np.array(data["Rp"])
            t_form = np.array(data["t_form"])
            X_cores = np.array(data["X_cores"]) if "X_cores" in data else None
            X_envs = np.array(data["X_envs"]) if "X_envs" in data else None

            ## Loop planets
            for ip in range(len(Mcs)):
                planet_rows.append({
                    ## Identifiers
                    "file": fname,
                    "planet_index": ip,

                    ## Final masses and position
                    "Mc_final": Mcs[ip, -1],
                    "Me_final": Mes[ip, -1],
                    "Mp_final": Mcs[ip, -1] + Mes[ip, -1],
                    "Rp_final": Rps[ip, -1],

                    ## Final chemistry
                    "X_core_final": X_cores[ip, :, -1] if X_cores is not None else None,
                    "X_env_final": X_envs[ip, :, -1] if X_envs is not None else None,

                    ## Implantation (the planet is NaN before it is inserted, so its first finite value is its implantation)
                    "Rp_init": Rps[ip][~np.isnan(Rps[ip])][0],
                    "Mp_init": (Mcs[ip] + Mes[ip])[~np.isnan(Rps[ip])][0],
                    "t_implant": float(t_form[ip]),

                    ## Simulation metadata
                    **metadata,
                })

        except Exception as e:
            print(f"Error in file {fname}: {e}")

    ## Convert to dataframe and save to pkl file
    df_planets = pd.DataFrame(planet_rows)
    df_planets.attrs["chem_species"] = chem_species

    print("Processed", n_files, "completed files")
    print("Planets:", df_planets.shape)

    df_planets.to_pickle(pkl_file)

    return df_planets

## Method to build the disc dataframe

def build_disc_dataframe(output_dir, pkl_file, completed_only = True):
    """
    Read the disc parameters stored as attributes in every HDF5 output file,
    and save the dataframe as a pickle file.

    Parameters:
        output_dir: Directory containing the .h5 output files.
        pkl_file: Path to save the pickled dataframe to.
        completed_only: If True, only include simulations that have completed.

    Returns:
        df_discs: Dataframe with one row per simulated disc.
    """

    disc_rows = []

    for fname in sorted(os.listdir(output_dir)):

        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(output_dir, fname), "r") as f:

            if completed_only and not bool(f.attrs.get("complete", False)):
                continue

            ## Disc parameters with Mdisk, Mdot [Msun/Myr] and Rd rounded to the grid values
            disc_rows.append({
                "file": fname,
                **round_grid_params({key: f.attrs[key].item() for key in GRID_KEYS if key in f.attrs}),
            })

    ## Convert to dataframe and save to pkl file
    df_discs = pd.DataFrame(disc_rows)

    print("Discs:", df_discs.shape)

    df_discs.to_pickle(pkl_file)

    return df_discs

## Method to build the growth track dataframe

def build_track_dataframe(output_dir, pkl_file, max_track_points = 2000):
    """
    Read the position and total mass of every planet over time from every
    completed HDF5 output file, thin each growth track out to at most
    max_track_points points, and save the dataframe as a pickle file.

    Parameters:
        output_dir: Directory containing the .h5 output files.
        pkl_file: Path to save the pickled dataframe to.
        max_track_points: Maximum number of points kept per growth track.

    Returns:
        df_tracks: Dataframe with one row per planet. Rp_track and Mp_track hold
            the thinned position [AU] and total mass [Mearth] arrays, from implantation onwards,
            followed by the disc parameters of the planet's simulation.
    """

    track_rows = []

    for fname in sorted(os.listdir(output_dir)):

        if not fname.endswith(".h5"):
            continue

        with h5py.File(os.path.join(output_dir, fname), "r") as f:

            ## Only use simulations that have completed
            if not bool(f.attrs.get("complete", False)):
                continue

            ## Skip if no planets
            if "Mcs" not in f:
                continue

            Mcs = np.array(f["Mcs"])
            Mes = np.array(f["Mes"])
            Rps = np.array(f["Rp"])

            ## Disc parameters with Mdisk, Mdot [Msun/Myr] and Rd rounded to the grid values
            metadata = round_grid_params({key: f.attrs[key].item() if key in f.attrs else np.nan for key in GRID_KEYS})

        for ip in range(len(Mcs)):

            ## The planet is NaN before it is inserted, so only keep the times after implantation
            inserted = ~np.isnan(Rps[ip])
            Mp_track = (Mcs[ip] + Mes[ip])[inserted]
            Rp_track = Rps[ip][inserted]

            if len(Rp_track) == 0:
                continue

            ## Thin the track out to at most max_track_points points (always keeping the first and last point)
            keep = np.unique(np.linspace(0, len(Rp_track) - 1, min(len(Rp_track), max_track_points)).astype(int))

            track_rows.append({
                ## Identifiers (match the planet dataframe)
                "file": fname,
                "planet_index": ip,

                ## Thinned growth track
                "Rp_track": Rp_track[keep],
                "Mp_track": Mp_track[keep],

                ## Simulation metadata
                **metadata,
            })

    ## Convert to dataframe and save to pkl file
    df_tracks = pd.DataFrame(track_rows)

    print("Tracks:", df_tracks.shape)

    df_tracks.to_pickle(pkl_file)

    return df_tracks


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description = __doc__,
        formatter_class = argparse.RawDescriptionHelpFormatter)

    parser.add_argument("--model", type = str, default = MODEL, 
                        help = f"Name of the population synthesis model")
    parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR,
                        help = f"Directory of .h5 output files, {{model}} is filled in with the model name")
    parser.add_argument("--save_dir", type = str, default = SAVE_DIR,
                        help = f"Directory the pickle files are saved to, {{model}} is filled in with the model name")
    parser.add_argument("--max_track_points", type = int, default = MAX_TRACK_POINTS,
                        help = f"Maximum number of points kept per growth track")
    parser.add_argument("--rebuild_planets", action = "store_true", default = REBUILD_PLANETS,
                        help = "Rebuild the planet dataframe even if it is already saved")
    parser.add_argument("--rebuild_discs", action = "store_true", default = REBUILD_DISCS,
                        help = "Rebuild the disc dataframe even if it is already saved")
    parser.add_argument("--rebuild_tracks", action = "store_true", default = REBUILD_TRACKS,
                        help = "Rebuild the growth track dataframe even if it is already saved")
    parser.add_argument("--rebuild", action = "store_true", default = REBUILD,
                        help = "Rebuild every dataframe even if it is already saved")

    args = parser.parse_args()

    output_dir = args.output_dir.format(model = args.model)
    save_dir = args.save_dir.format(model = args.model)

    ## Make the save directory if it does not exist yet
    if not os.path.isdir(save_dir):
        print(f"Creating save directory {save_dir}")
        os.makedirs(save_dir)

    ## Pickle file names (these must match the paths in popsynth_plots.ipynb)
    planets_pkl_file = os.path.join(save_dir, f"df_planets_{args.model}.pkl")
    disc_pkl_file = os.path.join(save_dir, f"df_discs_{args.model}.pkl")
    tracks_pkl_file = os.path.join(save_dir, f"df_tracks_{args.model}.pkl")

    ## Build each dataframe, skipping any that are already saved unless asked to rebuild

    ## Planet dataframe
    if os.path.exists(planets_pkl_file) and not (args.rebuild_planets or args.rebuild):
        print(f"Skipping the planet dataframe, already saved to {planets_pkl_file}")

    else:
        print(f"Building the planet dataframe from {output_dir}")
        build_planet_dataframe(output_dir, planets_pkl_file)
        print(f"Saved to {planets_pkl_file}")

    ## Disc dataframe
    if os.path.exists(disc_pkl_file) and not (args.rebuild_discs or args.rebuild):
        print(f"Skipping the disc dataframe, already saved to {disc_pkl_file}")

    else:
        print(f"Building the disc dataframe from {output_dir}")
        build_disc_dataframe(output_dir, disc_pkl_file)
        print(f"Saved to {disc_pkl_file}")

    ## Growth track dataframe
    if os.path.exists(tracks_pkl_file) and not (args.rebuild_tracks or args.rebuild):
        print(f"Skipping the growth track dataframe, already saved to {tracks_pkl_file}")
        
    else:
        print(f"Building the growth track dataframe from {output_dir}")
        build_track_dataframe(output_dir, tracks_pkl_file, args.max_track_points)
        print(f"Saved to {tracks_pkl_file}")
