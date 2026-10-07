"""
run_model_popsynth_batched.py
=============================

A compact, heavily-commented walkthrough of a single DiscEvolution run,
written for someone seeing this codebase for the first time.

WHAT THIS RUNS
--------------
One physical setup: a viscously- and magnetically-accreting disc, with
two-population dust growth and radial drift, C/O chemistry, planetesimal
formation and planetesimal dynamical stirring, and the Bitsch-model
planet growth/migration with pebble + planetesimal + gas accretion.

UNIT CONVENTIONS (the part that trips everyone up at first)
-----------------------------------------------------------
DiscEvolution works in units where G = 1, length in AU, mass in Msun (see
DiscEvolution/constants.py). Those choices fix the time unit via Kepler's
third law -- it is NOT years. The constant `yr` (= 2*pi) converts: multiply
a duration in real years by `yr` to get code-time `t`; divide a code-time
`t` by `yr` to get real years. That is why `* yr` / `/ yr` appear so often.

    R, Rd, grid.Rc      radius                 AU
    Sigma               surface density        g / cm^2
    M (disc)            mass                   Msun
    M (planet)          core/envelope mass     Mearth
    Mdot                accretion rate         Msun / yr
    T                   temperature            K
    t (this script)     simulation time        code-time (divide by `yr`)

PIPELINE OVERVIEW
-----------------
    1. Load the JSON config (and any --flag overrides).
    2. Build grid + star + time grid.
    3. Solve the initial disc structure (disc_setup.setup_disc).
    4. Attach gas/dust transport and wrap the disc in DustGrowthTwoPop.
    5. Seed the chemistry in equilibrium with the dust.
    6. Place planets and attach the Bitsch2015Model (optional).
    7. Turn on planetesimal formation + dynamics (optional).
    8. Open the HDF5 file and create the in-memory output buffers.
    9. Integrate forward, recording a row every 5 steps in memory and
       appending the buffered rows to the file every FLUSH_INTERVAL steps.
"""

import os
import sys
import json
import time
from array import array

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from DiscEvolution.constants import AU, Msun, yr
from DiscEvolution.grid import Grid
from DiscEvolution.star import SimpleStar
from DiscEvolution.opacity import Tazzari2016
from DiscEvolution.viscous_evolution import ViscousEvolutionFV, HybridWindModel
from DiscEvolution.dust import DustGrowthTwoPop, SingleFluidDrift, PlanetesimalFormation
from DiscEvolution.diffusion import TracerDiffusion
from DiscEvolution.planet_formation import Planets, Bitsch2015Model
from DiscEvolution.chemistry import (SimpleCOChemOberg, EquilibriumCOChemOberg, TimeDepCOChemOberg,
                                     SimpleCOAtomAbund, SimpleCOMolAbund)

from DiscEvolution.disc_setup import setup_disc

GAS_SOLVER = ViscousEvolutionFV   # viscous scheme used when winds are off
CHEM_SPECIES = SimpleCOMolAbund(1).names   # fixed species order used by the CO-chem model
plt.rcParams.update({'font.size': 16})

# ============================================================================
# Step 2: time grid
# ============================================================================

def make_time_grid(sim_params):
    """
    Build the array of snapshot times (code-time units).

    sim_params['t_interval'] may be:
        "power"  -- log-spaced snapshots from t_initial to t_final (years)
        a list   -- explicit snapshot times, in Myr
        a number -- fixed linear spacing, in years
    """

    t_interval = sim_params['t_interval']

    if t_interval == "power":
        if sim_params['t_initial'] == 0:
            num_points = int(np.log10(sim_params['t_final'])) + 1
            years = np.logspace(0, np.log10(sim_params['t_final']), num=num_points)

        else:
            num_points = int(np.log10(sim_params['t_final'] / sim_params['t_initial'])) + 1
            years = np.logspace(np.log10(sim_params['t_initial']), np.log10(sim_params['t_final']), num=num_points)
            
        return years * yr

    elif isinstance(t_interval, list):
        return np.array(t_interval) * 1e6 * yr          # Myr -> code time

    else:
        years = np.arange(sim_params['t_initial'], sim_params['t_final'], t_interval)
        return years * yr

# ============================================================================
# Step 4: transport + dust-growth disc wrapper
# ============================================================================

def build_transport(transport_params, wind_params, disc_params, dust_growth_params, lambda_DW):
    """Build the gas/dust transport operators (any can be switched off)."""

    gas = None
    if transport_params['gas_transport']:
        gas = HybridWindModel(wind_params['psi_DW'], lambda_DW) if wind_params["on"] else GAS_SOLVER()

    diffuse = None
    if transport_params['diffusion']:
        diffuse = TracerDiffusion(Sc=disc_params["Sc"])

    dust = None
    if transport_params['radial_drift']:
        # SingleFluidDrift folds diffusion in internally when handed a
        # `diffusion` object, so pass ours off and stop calling it separately.
        dust = SingleFluidDrift(diffusion=diffuse, settling=dust_growth_params['settling'], van_leer=transport_params['van_leer'])
        diffuse = None

    return gas, dust, diffuse


def build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas):
    """Wrap the bare (grid, star, eos, Sigma) disc in two-population dust growth."""
    return DustGrowthTwoPop(
        grid, star, eos, disc_params['d2g'],
        eps_SI=disc_params.get('d2g_SI', 0.02),   # SI dust fraction (only used by "SI" masses)
        Sigma=Sigma, feedback=dust_growth_params["feedback"], Sc=disc_params["Sc"],
        f_ice=dust_growth_params['f_ice'], thresh=dust_growth_params['thresh'],
        uf_0=dust_growth_params["uf_0"], uf_ice=dust_growth_params["uf_ice"], gas=gas,
        rho_s=dust_growth_params.get('rho_s', 1.0))

# ============================================================================
# Step 5: chemistry
# ============================================================================

_CHEM_MODELS = {
    "Simple":            lambda: SimpleCOChemOberg(),
    "Equilibrium":       lambda: EquilibriumCOChemOberg(a=1e-5),
    "Equilibrium_Fixed": lambda: EquilibriumCOChemOberg(a=1e-5, fix_ratios=True),
    "TimeDep":           lambda: TimeDepCOChemOberg(a=1e-5),
}

def build_chemistry(disc, chemistry_params, d2g_target, N_cell):
    """
    Seed ice/gas abundances in equilibrium with the dust-to-gas ratio.
    Returns (chemistry_model, Nchem). Sets disc.chem and dust_frac.
    """

    if not chemistry_params["on"]:
        disc.chem = None
        return None, 0

    try:
        chemistry = _CHEM_MODELS[chemistry_params["chem_model"]]()

    except KeyError:
        raise ValueError("Valid chemistry model not selected. Choose "
                         "Simple, Equilibrium, Equilibrium_Fixed, or TimeDep")

    X_solar = SimpleCOAtomAbund(N_cell)     # solar atoms-per-H, same in every cell
    X_solar.set_solar_abundances()

    # Ice fraction and dust-to-gas ratio depend on each other, so iterate.
    chem = None
    for _ in range(100):
        if chemistry_params["assert_d2g"]:
            # Force the total dust-to-gas ratio to match disc.d2g exactly.
            M_dust = np.trapezoid(disc.Sigma_D.sum(0), np.pi * disc.grid.Rc ** 2)
            M_gas = np.trapezoid(disc.Sigma_G, np.pi * disc.grid.Rc ** 2)
            disc.dust_frac[:] = disc.dust_frac * (d2g_target / (M_dust / M_gas))

        chem = chemistry.equilibrium_chem(disc.T, disc.midplane_gas_density, disc.dust_frac.sum(0), X_solar)
        disc.initialize_dust_density(chem.ice.total_abund)

    disc.chem = chem
    disc.update_ices(disc.chem.ice)
    return chemistry, disc.chem.ice.data.shape[0]

# ============================================================================
# Step 6: planets
# ============================================================================

def build_planets(disc, planet_params, chemistry_params, wind_params):
    """
    Create the Planets container + Bitsch2015Model and insert the planets.

    Returns (planets, planet_model, pending_SI). Planets whose mass is the
    string "SI" cannot be placed yet -- they need a nonzero planetesimal
    surface density -- so they are returned in `pending_SI` and inserted
    later, inside the time loop, once Sigma_planetesimal > 0 at their radius.
    """

    if not planet_params['include_planets']:
        return None, None, []

    Nchem = disc.chem.ice.data.shape[0] if chemistry_params["on"] else 0
    planets = Planets(Nchem=Nchem)

    planet_model = Bitsch2015Model(
        disc, pb_gas_f=planet_params["pb_gas_f"],
        f_plt=planet_params.get("f_plt", 400),
        migrate=planet_params["migrate"],
        pebble_acc=planet_params["pebble_accretion"],
        gas_acc=planet_params["gas_accretion"],
        planetesimal_acc_migrate=planet_params["planetesimal_accretion_migrate"],
        planetesimal_acc_insitu=planet_params["planetesimal_accretion_insitu"],
        winds=wind_params["on"],
        rho_core=planet_params.get("rho_core", 5.5))
    
    planet_model.set_disc(disc)

    pending_SI = []
    for R_impl, M_impl, t_impl in zip(planet_params['Rp'], planet_params['Mp'], planet_params['implant_time']):
        if str(M_impl).upper() == "SI":
            pending_SI.append((t_impl, R_impl, M_impl))

        else:
            # M may be a number (Mearth) or "PA"/"TR" (birth-mass models);
            # insert_new_planet handles the string cases and sets M_env = 0.
            planet_model.insert_new_planet(t_impl, R_impl, M_impl, planets)

    return planets, planet_model, pending_SI

# ============================================================================
# Step 7: planetesimals
# ============================================================================

def build_planetesimals(disc, planets, planetesimal_params):
    """
    Attach a PlanetesimalFormation object (formation + dynamical stirring).

    `planets` is passed in because embryo viscous stirring (VS_embryo) needs
    to know the protoplanets. The drag / VS / DF switches turn the individual
    eccentricity/inclination terms on and off; set them all False to recover
    the pre-dynamics behaviour.
    """

    disc._planetesimal = None
    if not planetesimal_params['active']:
        return

    disc._planetesimal = PlanetesimalFormation(
        disc, planets,
        d_planetesimal=planetesimal_params['diameter'],
        rho_pltsml=planetesimal_params.get('rho_pltsml', 2.0),
        St_min=planetesimal_params['St_min'],
        St_max=planetesimal_params['St_max'],
        pla_eff=planetesimal_params['pla_eff'],
        drag=planetesimal_params.get('drag', True),
        VS_embryo=planetesimal_params.get('VS_embryo', True),
        VS_pltsml=planetesimal_params.get('VS_pltsml', True),
        DF=planetesimal_params.get('DF', True),
        e_init=planetesimal_params.get('e_init', 'eq'),
        i_init=planetesimal_params.get('i_init', 'eq'))

# ============================================================================
# In-memory output buffers + periodic HDF5 writes
# ============================================================================

# The time series keep their full resolution (one row every 5 steps), but
# rows are collected in memory and only appended to the .h5 file in blocks
# every FLUSH_INTERVAL steps, each disc snapshot, and at the end of the run.

# Rows are held in array('d') buffers (8 bytes per value) between flushes.

SCALAR_NAMES = ["t", "disk_Mdot_star", "disk_Mass", "Tc", "Sigc"]
FLUSH_INTERVAL = 5000   # steps between writes of the buffered rows to disk
ABORT_EXIT_CODE = 3   # exit status for an aborted run (1 = any other failure)


class SimulationAborted(Exception):
    """Raised when the estimated time remaining exceeds abort_timescale."""


def _ei_mechanisms(planetesimal_params):
    """The e/i stirring terms and whether each is switched on."""

    return [
        ("drag",      planetesimal_params.get('drag', True)),
        ("VS_embryo", planetesimal_params.get('VS_embryo', True)),
        ("VS_pltsml", planetesimal_params.get('VS_pltsml', True)),
        ("DF",        planetesimal_params.get('DF', True)),
    ]


def _any_ei(planetesimal_params):
    return any(flag for _, flag in _ei_mechanisms(planetesimal_params))


def output_filename(config):
    """Deterministic output filename for one parameter combination."""

    sim_params = config['simulation']
    disc_params = config['disc']
    wind_params = config['winds']
    run_name = sim_params.get('run_name', 'run')
    return (f"{run_name}_psi{wind_params['psi_DW']}_Mdot{disc_params['Mdot']:.1e}"
            f"_M{disc_params['M']:.1e}_Rd{disc_params['Rd']:.1e}.h5")


def _planet_series_names(planet_params):
    """Names of the per-planet time series tracked for this configuration."""

    names = ["Mcs", "Mes", "Rp", "disk_Mdot_p"]
    if planet_params["planetesimal_accretion_insitu"]:
        names += ["Mdot_planetesimal", "f_env", "M_iso_planetesimal"]

    if planet_params["pebble_accretion"]:
        names += ["Mdot_pebble_core", "Mdot_pebble_env", "M_iso_pebble"]

    if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
        names += ["Mdot_migration"]

    if planet_params["gas_accretion"]:
        names += ["Mdot_gas"]

    return names


def create_buffers(grid, config):
    """
    Create the empty in-memory buffers for every tracked quantity.

    The buffers only ever hold the rows recorded since the last
    flush_buffers() call. Returns a dict with:
        "scalars" : {name: array('d')}                one value per row
        "planets" : {name: [array('d') per planet]}   one value per row
        "X_cores", "X_envs" : [[array('d') per species] per planet]
        "snaps"   : {name: (row_shape, [ndarray per snapshot])}
    """

    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    nR = len(grid.Rc)
    nSpec = len(CHEM_SPECIES)

    buf = {"scalars": {name: array('d') for name in SCALAR_NAMES},
           "planets": {}, "X_cores": [], "X_envs": [], "snaps": {}}

    if planet_params['include_planets']:
        buf["planets"] = {name: [] for name in _planet_series_names(planet_params)}

    # ---- disc-profile snapshots (one row per snapshot time) ----
    snaps = buf["snaps"]
    snaps["time_snap"] = ((), [])
    for name in ["Sigma_G", "Sigma_dust", "Sigma_pebbles", "Vdrift_grains", "Vdrift_pebbles", "St_grains", "St_pebbles", "T"]:
        snaps[name] = ((nR,), [])

    # ---- per-species ice/gas mass-abundance snapshots (grains+pebbles ice, and
    #      gas), needed to reconstruct C/O ratio profiles for the diagnostic plot ----
    if chemistry_params["on"]:
        snaps["gas_chem"] = ((nSpec, nR), [])
        snaps["ice_chem"] = ((nSpec, nR), [])

    if planetesimal_params['active']:
        for name in ["Sigma_planetesimals", "St_planetesimals", "e_planetesimals", "i_planetesimals"]:
            snaps[name] = ((nR,), [])

        if chemistry_params["on"]:
            snaps["planetesimal_ice_chem"] = ((nSpec, nR), [])

        if _any_ei(planetesimal_params):
            snaps["de2_dt"] = ((nR,), [])
            snaps["di2_dt"] = ((nR,), [])

        for name, flag in _ei_mechanisms(planetesimal_params):
            if flag:
                snaps[f"de2_dt_{name}"] = ((nR,), [])
                snaps[f"di2_dt_{name}"] = ((nR,), [])

    return buf


def _backfilled(n_backfill):
    """A new series buffer, backfilled with NaN for the rows already buffered
    (used when an "SI" planet is inserted late). Rows already on disk are
    backfilled by flush_buffers() when the planet's datasets are created."""

    return array('d', [np.nan]) * n_backfill


def add_planet_slot(buf, chemistry_params, Nchem):
    """Add the per-planet series for one newly inserted planet, keeping the
    buffers index-aligned with `planets`."""

    n_backfill = len(buf["scalars"]["t"])

    for series in buf["planets"].values():
        series.append(_backfilled(n_backfill))

    if chemistry_params["on"]:
        buf["X_cores"].append([_backfilled(n_backfill) for _ in range(Nchem)])
        buf["X_envs"].append([_backfilled(n_backfill) for _ in range(Nchem)])


def _snap(buf, name, value):
    """Append one (copied) row to a disc-profile snapshot buffer."""

    buf["snaps"][name][1].append(np.array(value, dtype="f8"))


def record_scalars(buf, t_years, disc, disk_Mdot):
    """Append one row to every scalar time series."""

    scalars = buf["scalars"]
    scalars["t"].append(t_years)
    scalars["disk_Mdot_star"].append(disk_Mdot[0])
    scalars["disk_Mass"].append(disc.Mtot())
    scalars["Tc"].append(disc.T[0])
    scalars["Sigc"].append(disc.Sigma[0])


def record_planet_row(buf, planets, planet_model, grid, disk_Mdot, rates, config):
    """
    Append one row to every per-planet series.

    `rates` is the growth-rate dict for this instant: at t = 0 it comes from
    planet_model._growth_rates(...) (no integrate() has run yet); during the
    loop it is planet_model.rates, the rates that drove the last integrate().
    """

    planet_params = config['planets']
    chemistry_params = config['chemistry']
    series = buf["planets"]

    for ip, planet in enumerate(planets):
        series["Mcs"][ip].append(planet.M_core)
        series["Mes"][ip].append(planet.M_env)
        series["Rp"][ip].append(planet.R)
        series["disk_Mdot_p"][ip].append(np.interp(planet.R, grid.Rc[0:-1], disk_Mdot))

        if planet_params["planetesimal_accretion_insitu"]:
            series["Mdot_planetesimal"][ip].append(rates["Mdot_planetesimal_insitu"][ip] * yr)
            series["f_env"][ip].append(rates["f_env"][ip])
            series["M_iso_planetesimal"][ip].append(planet_model._pla_acc.M_iso_pltsml(planet.R))

        if planet_params["pebble_accretion"]:
            series["Mdot_pebble_core"][ip].append(rates["Mdot_pebble_core"][ip] * yr)
            series["Mdot_pebble_env"][ip].append(rates["Mdot_pebble_env"][ip] * yr)
            series["M_iso_pebble"][ip].append(planet_model._peb_acc.M_iso(planet.R))

        if planet_params["migrate"] and planet_params["planetesimal_accretion_migrate"]:
            series["Mdot_migration"][ip].append(rates["Mdot_planetesimal_migration"][ip] * yr)

        if planet_params["gas_accretion"]:
            series["Mdot_gas"][ip].append(rates["Mdot_gas"][ip] * yr)

        if chemistry_params["on"]:
            for js, x in enumerate(planet.X_core):
                buf["X_cores"][ip][js].append(x)

            for js, x in enumerate(planet.X_env):
                buf["X_envs"][ip][js].append(x)


def record_disc_snapshot(buf, disc, t, planetesimal_params, chemistry_params):
    """Append one row to every disc-profile snapshot buffer."""

    v_drift = disc.v_drift
    stokes = disc.Stokes()

    _snap(buf, "time_snap", t / (1e6 * yr))     # Myr
    _snap(buf, "Sigma_G", disc.Sigma_G)
    _snap(buf, "Sigma_dust", disc.Sigma_D[0])
    _snap(buf, "Sigma_pebbles", disc.Sigma_D[1])
    _snap(buf, "Vdrift_grains", v_drift[0])
    _snap(buf, "Vdrift_pebbles", v_drift[1])
    _snap(buf, "St_grains", stokes[0])
    _snap(buf, "St_pebbles", stokes[1])
    _snap(buf, "T", disc.T)

    if chemistry_params["on"]:
        _snap(buf, "gas_chem", disc.chem.gas.data)
        _snap(buf, "ice_chem", disc.chem.ice.data)

    if not planetesimal_params['active']:
        return

    pl = disc._planetesimal
    _snap(buf, "Sigma_planetesimals", disc.Sigma_D[2])
    _snap(buf, "St_planetesimals", stokes[2])
    _snap(buf, "e_planetesimals", pl.e)
    _snap(buf, "i_planetesimals", pl.i)

    if chemistry_params["on"] and pl.ice_abund is not None:
        _snap(buf, "planetesimal_ice_chem", pl.ice_abund.data)

    e2, i2 = pl._e2, pl._i2
    for name, flag in _ei_mechanisms(planetesimal_params):
        if flag:
            _snap(buf, f"de2_dt_{name}", getattr(pl, f"de2_dt_{name}")(e2, i2) * yr)
            _snap(buf, f"di2_dt_{name}", getattr(pl, f"di2_dt_{name}")(e2, i2) * yr)

    if _any_ei(planetesimal_params):
        _snap(buf, "de2_dt", pl.de2_dt(e2, i2) * yr)
        _snap(buf, "di2_dt", pl.di2_dt(e2, i2) * yr)


def create_output_file(outfile, grid, config, attrs):
    """Create the HDF5 file, its attributes and the per-planet groups.
    The datasets themselves are created by the first flush_buffers() call."""

    planet_params = config['planets']
    chemistry_params = config['chemistry']

    h5f = h5py.File(outfile, "w")

    for key, value in attrs.items():
        h5f.attrs[key] = value

    h5f.attrs["complete"] = False
    h5f.attrs["aborted"] = False

    if chemistry_params["on"]:
        h5f.attrs["chem_species"] = list(CHEM_SPECIES)

    # ---- grid (written once) ----
    h5f.create_dataset("R", data=grid.Rc)

    # ---- per-planet groups: one group per quantity, one dataset per planet ----
    if planet_params['include_planets']:
        for name in _planet_series_names(planet_params):
            h5f.create_group(name)

        if chemistry_params["on"]:
            h5f.create_group("X_cores")
            h5f.create_group("X_envs")

    return h5f


def _append(parent, name, rows, row_shape=(), n_backfill=0):
    """
    Append a block of rows to the growable dataset parent[name], creating it
    first if needed. A newly created dataset starts with `n_backfill` NaN
    rows, so a late-inserted planet stays aligned with the "t" dataset.
    """

    rows = np.array(rows, dtype="f8").reshape((len(rows),) + row_shape)

    if name in parent:
        dset = parent[name]

    else:
        chunks = (1024,) if row_shape == () else True
        dset = parent.create_dataset(name, shape=(n_backfill,) + row_shape, maxshape=(None,) + row_shape, dtype="f8", chunks=chunks)
        if n_backfill:
            dset[:] = np.nan

    if len(rows):
        n = dset.shape[0]
        dset.resize(n + len(rows), axis=0)
        dset[n:] = rows


def flush_buffers(h5f, buf):
    """Append every buffered row to the HDF5 file, then empty the buffers."""

    n_written = h5f["t"].shape[0] if "t" in h5f else 0   # rows already on disk

    # ---- scalar time series ----
    for name, series in buf["scalars"].items():
        _append(h5f, name, series)
        del series[:]

    # ---- per-planet series ----
    for name, per_planet in buf["planets"].items():
        for ip, series in enumerate(per_planet):
            _append(h5f[name], str(ip), series, n_backfill=n_written)
            del series[:]

    for name in ("X_cores", "X_envs"):
        for ip, per_species in enumerate(buf[name]):
            pgrp = h5f[name].require_group(str(ip))
            for js, series in enumerate(per_species):
                _append(pgrp, str(js), series, n_backfill=n_written)
                del series[:]

    # ---- disc-profile snapshots ----
    for name, (row_shape, rows) in buf["snaps"].items():
        _append(h5f, name, rows, row_shape)
        rows.clear()

    h5f.flush()

# ============================================================================
# Main driver
# ============================================================================

def run_model(config, cli_output_dir=None, cli_output_filename=None):
    """Run one disc-evolution simulation and write the result to HDF5.

    Rows are buffered in memory and appended to the file every FLUSH_INTERVAL
    steps (see flush_buffers).

    Returns the path to the output .h5 file. The returned path can be handed
    to plot_diagnostics() to build the post-hoc diagnostic figure.
    """

    grid_params = config['grid']
    sim_params = config['simulation']
    star_params = config['star']
    disc_params = config['disc']
    eos_params = config['eos']
    transport_params = config['transport']
    dust_growth_params = config['dust_growth']
    planet_params = config['planets']
    chemistry_params = config['chemistry']
    planetesimal_params = config['planetesimal']
    wind_params = config['winds']

    # ---- 0. skip immediately if this exact run already finished ----
    # Output directory precedence: --output_dir CLI flag > config.json
    # ('simulation.output_dir') > DISCEVOLUTION_OUTPUT env var >
    # ./output within the current working directory.
    output_dir = (cli_output_dir
                  or sim_params.get('output_dir')
                  or os.environ.get('DISCEVOLUTION_OUTPUT')
                  or os.path.join(os.getcwd(), 'output'))
    os.makedirs(output_dir, exist_ok=True)

    # --output_filename overrides the deterministic name for both the .h5
    # output and (via plot_diagnostics, which derives its name from outfile's
    # basename) the diagnostic PNG.
    if cli_output_filename:
        name = cli_output_filename
        if not name.endswith(".h5"):
            name += ".h5"

    else:
        name = output_filename(config)

    outfile = os.path.join(output_dir, name)

    if os.path.exists(outfile):
        with h5py.File(outfile, "r") as existing:
            if existing.attrs.get("complete", False):
                print(f"Skipping -- output already complete: {outfile}")
                return outfile

            if existing.attrs.get("aborted", False):
                print(f"Skipping -- output was previously aborted: {outfile}")
                sys.exit(ABORT_EXIT_CODE)

        print(f"Output file exists but is incomplete; re-running: {outfile}")

    # ---- 2. grid + star + time grid ----
    grid = Grid(grid_params['rmin'], grid_params['rmax'], grid_params['nr'], spacing=grid_params['spacing'])
    star = SimpleStar(M=star_params["M"], R=star_params["R"], T_eff=star_params['T_eff'])
    times = make_time_grid(sim_params)

    # Opacity: an instance for Tazzari; None lets IrradiatedEOS default to Zhu2012.
    kappa = Tazzari2016() if eos_params["opacity"] == "Tazzari" else None

    # ---- 3. initial disc structure ----
    disc, eos, Sigma, psi, alpha_SS, lambda_DW = setup_disc(grid, star, config, kappa)

    if alpha_SS > 5e-3:
        print(f"Not running model - alpha too high. alpha_SS={alpha_SS:.3e}, "
              f"Rd={disc_params['Rd']}, Mdisk={disc.Mtot()/Msun:.4g} Msun")
        sys.exit(1)

    if psi < 0.0:
        print(f"Not running model - negative wind torque. alpha_SS={alpha_SS:.3e}, "
              f"Rd={disc_params['Rd']}, Mdisk={disc.Mtot()/Msun:.4g} Msun")
        sys.exit(1)

    print(f"Running model. alpha_SS={alpha_SS:.3e}, Rd={disc_params['Rd']}, "
          f"Mdisk={disc.Mtot()/Msun:.4g} Msun")

    # ---- 4. transport + dust growth ----
    gas, dust, diffuse = build_transport(transport_params, wind_params, disc_params, dust_growth_params, lambda_DW)
    disc = build_dust_growth_disc(grid, star, eos, Sigma, disc_params, dust_growth_params, gas)

    # ---- 5. chemistry ----
    chemistry, Nchem = build_chemistry(disc, chemistry_params, disc_params["d2g"], grid_params["nr"])

    # ---- 6. planets, then 7. planetesimals (planets first: VS_embryo needs them) ----
    planets, planet_model, pending_SI = build_planets(disc, planet_params, chemistry_params, wind_params)
    build_planetesimals(disc, planets, planetesimal_params)

    # ---- 8. output file + buffers, 9. integrate ----
    vr_0 = disc._gas.viscous_velocity(disc, disc.Sigma)
    attrs = {
        "alpha_SS": float(alpha_SS),
        "psi_DW": float(wind_params["psi_DW"]),
        "Mdot": float(disc.Mdot(vr_0[0])),
        "Mdisk": float(disc.Mtot() / Msun),
        "Rd": float(disc.RC()),
        "pla_eff": float(planetesimal_params.get("pla_eff", np.nan)),
        "f_plt": float(planet_params.get("f_plt", 400)),
    }

    h5f = create_output_file(outfile, grid, config, attrs)
    buf = create_buffers(grid, config)
    for ip in range(planets.N if planets is not None else 0):
        add_planet_slot(buf, chemistry_params, Nchem)

    aborted = False
    try:
        _integrate(h5f, buf, disc, grid, planets, planet_model, gas, dust, diffuse, chemistry, times, pending_SI, Nchem, config)
        flush_buffers(h5f, buf)

        if planet_params['include_planets']:
            h5f.create_dataset("t_form", data=planets.t_form / yr)   # yr, insertion time per planet

        h5f.attrs["complete"] = True

    except SimulationAborted:
        # Keep the rows recorded so far and mark the file so that a re-launch
        # of the same sweep skips this run.
        flush_buffers(h5f, buf)
        h5f.attrs["aborted"] = True
        aborted = True

    finally:
        h5f.close()

    print(f"Wrote {outfile}")

    if aborted:
        sys.exit(ABORT_EXIT_CODE)

    return outfile


def _disc_star_mdot(disc):
    """Accretion rate onto the star at the current disc state, in Msun/yr."""

    v = disc._gas.viscous_velocity(disc, disc.Sigma)
    return -2 * np.pi * disc._grid.Rc[0:-1] * disc.Sigma[0:-1] * v * (AU * AU) * (yr / Msun)


def _growth_rates_now(planet_model, planets, chemistry_on):
    """Evaluate the planet growth rates at the current planet state (for t = 0,
    before integrate() has populated planet_model.rates)."""

    if chemistry_on:
        M_Z = (planets.X_core * planets.M_core).sum(0) + (planets.X_env * planets.M_env).sum(0)
        M_HHe = planets.M_core + planets.M_env - M_Z

    else:
        M_Z, M_HHe = planets.M_core, planets.M_env

    return planet_model._growth_rates(planets.R, planets.M_core, planets.M_env, M_Z, M_HHe)


def _integrate(h5f, buf, disc, grid, planets, planet_model, gas, dust, diffuse, chemistry, times, pending_SI, Nchem, config):
    """The time-stepping loop, plus the periodic recording into `buf` and
    flushing of `buf` to `h5f`."""

    transport_params = config['transport']
    chemistry_params = config['chemistry']
    planet_params = config['planets']
    planetesimal_params = config['planetesimal']
    have_planets = planet_params['include_planets']

    # ---- t = 0 rows (scalars + per-planet always; disc profile only if
    #      0.0 is not itself a requested snapshot, else the loop records it) ----
    disk_Mdot = _disc_star_mdot(disc)
    record_scalars(buf, 0.0, disc, disk_Mdot)

    if have_planets and planets.N > 0:
        rates0 = _growth_rates_now(planet_model, planets, chemistry_params["on"])
        record_planet_row(buf, planets, planet_model, grid, disk_Mdot, rates0, config)

    if 0.0 not in times:
        record_disc_snapshot(buf, disc, 0.0, planetesimal_params, chemistry_params)

    # Used to estimate the wall-clock time remaining
    loop_start_time = time.time()
    t_end = times[-1]
    abort_timescale = config['simulation'].get('abort_timescale', 10)   # hours

    t, n = 0.0, 0
    for ti in times:
        while t < ti:
            # Physics-limited timestep, capped to land exactly on ti.
            dt = ti - t
            if transport_params['gas_transport']:
                dt = min(dt, disc._gas.max_timestep(disc))

            if transport_params['radial_drift']:
                dt = min(dt, dust.max_timestep(disc))

            dust_frac = getattr(disc, "dust_frac", None)
            gas_chem = disc.chem.gas.data if chemistry_params["on"] else None
            ice_chem = disc.chem.ice.data if chemistry_params["on"] else None

            # --- gas viscous/wind evolution (advects the tracers too) ---
            if transport_params['gas_transport']:
                # exclude the planetesimal band so it doesn't move with the gas
                dust_frac_gas = dust_frac[:-1] if disc._planetesimal else dust_frac
                disc._gas(dt, disc, [dust_frac_gas, gas_chem, ice_chem])

            # --- planetesimal formation + dynamical stirring ---
            if disc._planetesimal:
                disc._planetesimal.update(dt, disc, dust)

            # --- insert any pending "SI" planets now that planetesimals exist ---
            if have_planets and pending_SI and disc._planetesimal:
                still_pending = []
                for t_impl, R_impl, M_impl in pending_SI:
                    if disc.interp(R_impl, disc.Sigma_D[2]) > 0:
                        planet_model.insert_new_planet(t, R_impl, M_impl, planets)
                        add_planet_slot(buf, chemistry_params, Nchem)

                    else:
                        still_pending.append((t_impl, R_impl, M_impl))

                pending_SI = still_pending

            # --- dust radial drift ---
            if transport_params['radial_drift']:
                dust(dt, disc, gas_tracers=gas_chem, dust_tracers=ice_chem)

            # --- turbulent diffusion (only if not already folded into `dust`) ---
            if diffuse is not None:
                if gas_chem is not None:
                    gas_chem[:] += dt * diffuse(disc, gas_chem)

                if ice_chem is not None:
                    ice_chem[:] += dt * diffuse(disc, ice_chem)

                if dust_frac is not None:
                    band = dust_frac[:2] if disc._planetesimal else dust_frac[:]
                    band += dt * diffuse(disc, band)

            # --- enforce physical bounds ---
            disc.Sigma[:] = np.maximum(disc.Sigma, 0)
            disc.dust_frac[:] = np.maximum(disc.dust_frac, 0)
            disc.dust_frac[:] /= np.maximum(disc.dust_frac.sum(0), 1.0)
            if chemistry_params["on"]:
                disc.chem.gas.data[:] = np.maximum(disc.chem.gas.data, 0)
                disc.chem.ice.data[:] = np.maximum(disc.chem.ice.data, 0)

            # --- chemistry adsorption/desorption ---
            if chemistry_params["on"]:
                d2g = disc.dust_frac[:-1].sum(0) if disc._planetesimal else disc.dust_frac.sum(0)
                chemistry.update(dt, disc.T, disc.midplane_gas_density, d2g, disc.chem)
                disc.update_ices(disc.chem.ice)

            # --- planet growth/migration ---
            if have_planets:
                planet_model.integrate(dt, planets)

            disc.update(dt)
            t += dt
            n += 1

            if (n % 1000) == 0:
                print(f"Nstep {n} | t = {t/(1e6*yr):.4g} Myr | dt = {dt/yr:.3g} yr", flush=True)

            # --- estimate time remaining ---
            if (n % 5000) == 0:
                elapsed = time.time() - loop_start_time
                time_remaining = elapsed * (t_end - t) / t   # seconds

                print(f"Estimated time remaining: {time_remaining/3600:.2f} hr "
                      f"({t/(1e6*yr):.3f} / {t_end/(1e6*yr):.3f} Myr done in {elapsed/3600:.2f} hr)", flush=True)

                if time_remaining > abort_timescale * 3600:
                    print(f"Aborting simulation - estimated time remaining exceeds abort_timescale ({abort_timescale} hr).", flush=True)
                    raise SimulationAborted

            # --- record scalar + per-planet series every 5 steps ---
            if (n % 5) == 0:
                disk_Mdot = _disc_star_mdot(disc)
                record_scalars(buf, t / yr, disc, disk_Mdot)          # years
                if have_planets and planets.N > 0:
                    record_planet_row(buf, planets, planet_model, grid, disk_Mdot, planet_model.rates, config)

            # --- append the buffered rows to the file ---
            if (n % FLUSH_INTERVAL) == 0:
                flush_buffers(h5f, buf)

        # --- full disc-profile row once per requested snapshot time ---
        record_disc_snapshot(buf, disc, t, planetesimal_params, chemistry_params)
        flush_buffers(h5f, buf)

# ============================================================================
# Diagnostic plot
# ============================================================================

# top-left     -- surface density profiles (grains/pebbles/gas[/planetesimals])
# top-right    -- C/O ratio profiles (grains+pebbles ice / gas / planetesimals)
# bottom-left  -- planet growth tracks (mass vs. radius)
# bottom-right -- C/O of planets over time

def _atomic_CO_ratio(species_data):
    """C/O number-abundance ratio at every radius/time, from a (Nspec, N) mass-abundance array."""

    mol = SimpleCOMolAbund(species_data.shape[1])
    mol.data[:] = species_data
    atoms = mol.atomic_abundance()
    return np.nan_to_num(atoms.number_abund("C") / atoms.number_abund("O"))


def _first_valid(seq):
    """First non-NaN value in a per-timestep tracking array (skips insertion backfill)."""

    arr = np.asarray(seq, dtype=float)
    valid = arr[~np.isnan(arr)]
    return valid[0] if valid.size else np.nan


def _planet_CO_track(h5f, ip, nspec):
    """C/O ratio over time for planet `ip`, from its stored core+envelope abundances."""

    Mc = h5f["Mcs"][str(ip)][:]
    Me = h5f["Mes"][str(ip)][:]
    M = Mc + Me

    X_core = np.array([h5f["X_cores"][str(ip)][str(js)][:] for js in range(nspec)])
    X_env = np.array([h5f["X_envs"][str(ip)][str(js)][:] for js in range(nspec)])

    with np.errstate(invalid="ignore", divide="ignore"):
        species_data = (X_core * Mc + X_env * Me) / M

    return _atomic_CO_ratio(species_data)


def plot_diagnostics(outfile, fig_dir=None):
    """
    Build the 4-panel diagnostic figure for one completed run_model_popsynth.py
    output file.

    args:
        outfile : path to a completed .h5 output file (as returned by run_model())
        fig_dir : directory to save the PNG in (default: alongside `outfile`)

    returns:
        path to the saved PNG
    """

    with h5py.File(outfile, "r") as h5f:
        R = h5f["R"][:]
        time_snap = h5f["time_snap"][:]
        Sigma_G = h5f["Sigma_G"][:]
        Sigma_dust = h5f["Sigma_dust"][:]
        Sigma_pebbles = h5f["Sigma_pebbles"][:]

        have_pltsml = "Sigma_planetesimals" in h5f
        Sigma_plts = h5f["Sigma_planetesimals"][:] if have_pltsml else None

        have_chem = "gas_chem" in h5f and "ice_chem" in h5f
        gas_chem = h5f["gas_chem"][:] if have_chem else None
        ice_chem = h5f["ice_chem"][:] if have_chem else None

        have_pltsml_chem = have_chem and "planetesimal_ice_chem" in h5f
        pltsml_chem = h5f["planetesimal_ice_chem"][:] if have_pltsml_chem else None

        have_planets = "Rp" in h5f
        planet_R, planet_M, planet_label_R = [], [], []
        planet_R_snap, planet_M_snap = [], []
        if have_planets:
            t_years = h5f["t"][:]
            snap_idx = np.searchsorted(t_years, time_snap * 1e6, side="right") - 1
            snap_idx = np.clip(snap_idx, 0, len(t_years) - 1)

            for key in h5f["Rp"]:
                Rp = h5f["Rp"][key][:]
                Mp = h5f["Mcs"][key][:] + h5f["Mes"][key][:]
                planet_R.append(Rp)
                planet_M.append(Mp)
                planet_label_R.append(_first_valid(Rp))
                planet_R_snap.append(Rp[snap_idx])
                planet_M_snap.append(Mp[snap_idx])

        have_planet_chem = have_planets and "X_cores" in h5f and "X_envs" in h5f
        planet_t, planet_CO = [], []
        if have_planet_chem:
            t_years = h5f["t"][:]
            nspec = len(CHEM_SPECIES)
            for key in h5f["Rp"]:
                planet_t.append(t_years)
                planet_CO.append(_planet_CO_track(h5f, key, nspec))

        alpha_SS = float(h5f.attrs["alpha_SS"])
        Mdot = float(h5f.attrs["Mdot"])
        Mdisk = float(h5f.attrs["Mdisk"])
        Rd = float(h5f.attrs["Rd"])

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    ax_sigma, ax_CO, ax_growth, ax_CO_time = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    n_times = len(time_snap)
    color_gas = plt.cm.Blues(np.linspace(0.4, 1, n_times))[::-1]
    color_grains = plt.cm.Greens(np.linspace(0.4, 1, n_times))[::-1]
    color_pebbles = plt.cm.Greys(np.linspace(0.4, 1, n_times))[::-1]
    color_plts = plt.cm.Reds(np.linspace(0.4, 1, n_times))[::-1]

    # ---- top-left: surface density profiles ----
    time_handles = []
    species_handles = []
    for i, tsnap in enumerate(time_snap):
        l_grain, = ax_sigma.loglog(R, Sigma_dust[i], linestyle="dotted", color=color_grains[i])
        l_pebble, = ax_sigma.loglog(R, Sigma_pebbles[i], linestyle="dashdot", color=color_pebbles[i])
        l_gas, = ax_sigma.loglog(R, Sigma_G[i], linestyle="dashed", color=color_gas[i])

        if have_pltsml:
            l_plts, = ax_sigma.loglog(R, Sigma_plts[i], color=color_plts[i])

        l_grain.set_label(f"t = {tsnap:.3g} Myr")
        time_handles.append(l_grain)

        if i == 0:
            species_handles = [l_grain, l_pebble, l_gas]

            if have_pltsml:
                species_handles.append(l_plts)

    ax_sigma.set_xlabel("Radius [AU]")
    ax_sigma.set_ylabel(r"$\Sigma\ [g/cm^2]$")
    ax_sigma.set_ylim(1e-6, 1e5)
    ax_sigma.set_title("Surface Density Profiles")

    time_legend = ax_sigma.legend(handles=time_handles, loc="lower left", fontsize=6)
    ax_sigma.add_artist(time_legend)

    species_labels = ["Grains", "Pebbles", "Gas"] + (["Planetesimals"] if have_pltsml else [])
    ax_sigma.legend(handles=species_handles, labels=species_labels, loc="upper right", fontsize=8)

    # ---- top-right: C/O ratio profiles ----
    if have_chem:
        CO_handles = []
        for i in range(n_times):
            CO_ice = _atomic_CO_ratio(ice_chem[i])
            CO_gas = _atomic_CO_ratio(gas_chem[i])
            l_ice, = ax_CO.semilogx(R, CO_ice, linestyle="dashdot", color=color_pebbles[i])
            l_gas, = ax_CO.semilogx(R, CO_gas, linestyle="dashed", color=color_gas[i])

            if have_pltsml_chem:
                CO_plts = _atomic_CO_ratio(pltsml_chem[i])
                l_plts, = ax_CO.semilogx(R, CO_plts, color=color_plts[i])

            if i == 0:
                CO_handles = [l_ice, l_gas]

                if have_pltsml_chem:
                    CO_handles.append(l_plts)

        ax_CO.set_ylim(0, 1.2)
        ax_CO.set_ylabel("[C/O]")
        ax_CO.set_xlabel("Radius [AU]")
        ax_CO.set_title("C/O Ratios Throughout the Disk")

        CO_labels = ["Grains+Pebbles", "Gas"] + (["Planetesimals"] if have_pltsml_chem else [])
        ax_CO.legend(handles=CO_handles, labels=CO_labels, loc="upper left", fontsize=8)

    else:
        ax_CO.text(0.5, 0.5, "Chemistry disabled", ha="center", va="center", transform=ax_CO.transAxes)
        ax_CO.set_axis_off()

    # ---- bottom-left: planet growth tracks ----
    if have_planets and planet_R:
        for Rp, M in zip(planet_R, planet_M):
            ax_growth.loglog(Rp, M, color="black")

        for Rp_snap, M_snap in zip(planet_R_snap, planet_M_snap):
            ax_growth.scatter(Rp_snap, M_snap, color="black", s=60, zorder=-1)

        ax_growth.set_xlabel("Radius [AU]")
        ax_growth.set_ylabel("Earth Masses")
        ax_growth.set_title("Planet Growth Tracks")
        ax_growth.set_xlim(1e-1, 500)

    else:
        ax_growth.text(0.5, 0.5, "No planets", ha="center", va="center", transform=ax_growth.transAxes)
        ax_growth.set_axis_off()

    # ---- bottom-right: C/O of planets over time ----
    if have_planet_chem:
        n_planets = len(planet_CO)
        colors = [plt.cm.viridis(i / n_planets) for i in range(n_planets)]

        for i, (t_years, CO) in enumerate(zip(planet_t, planet_CO)):
            ax_CO_time.semilogx(t_years, CO, color=colors[i], label=f"{planet_label_R[i]:.0f} AU")

        ax_CO_time.set_xlabel("Time (yr)")
        ax_CO_time.set_ylabel("[C/O]")
        ax_CO_time.set_title("C/O of planets over time")
        ax_CO_time.legend(loc="lower left", fontsize=6)

    else:
        ax_CO_time.text(0.5, 0.5, "No planets" if not have_planets else "Chemistry disabled", ha="center", va="center", transform=ax_CO_time.transAxes)
        
        ax_CO_time.set_axis_off()

    plt.figtext(0.5, 0, f"Mdot = {Mdot:.3e}, alpha = {alpha_SS:.3e}, Mtot = {Mdisk:.3e}, Rd = {Rd:.3e}", ha="center")
    plt.tight_layout()

    fig_dir = fig_dir or os.path.dirname(outfile) or "."
    os.makedirs(fig_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(outfile))[0]
    fig_path = os.path.join(fig_dir, f"{basename}_diagnostic.png")
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote {fig_path}")
    return fig_path

# ============================================================================
# Command-line entry point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run one full-physics DiscEvolution model with buffered HDF5 output.",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    ## Directory overrides
    parser.add_argument("--config", type=str, required=True, help="Path to configuration JSON file")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--output_filename", type=str, default=None, help="Override output file name")
    parser.add_argument("--plot", action="store_true", help="Produce a diagnostic plot PNG after the run completes")
    parser.add_argument("--figure_dir", type=str, default=None, help="Directory to save the diagnostic plot PNG")

    ## Disk parameter overrides
    parser.add_argument("--M", type=float, default=None, help="Override disc mass [Msun]")
    parser.add_argument("--Mdot", type=float, default=None, help="Override accretion rate [Msun/yr]")
    parser.add_argument("--Rd", type=float, default=None, help="Override characteristic disc radius [AU]")
    parser.add_argument("--psi_DW", type=float, default=None, help="Override disc-wind parameter")
    parser.add_argument("--pla_eff", type=float, default=None, help="Override planetesimal formation efficiency")
    parser.add_argument("--f_plt", type=float, default=None, help="Override embryo birth mass factor")
    parser.add_argument("--alpha", type=float, default=None, help="Override alpha viscosity parameter")

    args = parser.parse_args()

    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file not found: {args.config}", file=sys.stderr)
        sys.exit(1)
    
    with open(args.config, "r") as f:
        config = json.load(f)
    
    print(f"Loaded configuration from: {args.config}")

    overrides = {
        ("disc", "M"): args.M,
        ("disc", "Mdot"): args.Mdot,
        ("disc", "Rd"): args.Rd,
        ("winds", "psi_DW"): args.psi_DW,
        ("planetesimal", "pla_eff"): args.pla_eff,
        ("planets", "f_plt"): args.f_plt,
        ("disc", "alpha"): args.alpha,
    }

    for (section, key), value in overrides.items():
        if value is not None:
            config[section][key] = value
            print(f"Overriding {section}.{key}: {value}")

    start_time = time.time()
    outfile = run_model(config, cli_output_dir=args.output_dir, cli_output_filename=args.output_filename)

    if args.plot:
        # Figure directory precedence: --figure_dir CLI flag > config.json
        # ('simulation.figure_dir') > DISCEVOLUTION_FIGURE_DIR env var >
        # ./figure within the current working directory.
        fig_dir = (args.figure_dir
                   or config['simulation'].get('figure_dir')
                   or os.environ.get('DISCEVOLUTION_FIGURE_DIR')
                   or os.path.join(os.getcwd(), 'figure'))
        plot_diagnostics(outfile, fig_dir=fig_dir)

    print(f"Duration: {time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time))}")
