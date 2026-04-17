# -*- coding: utf-8 -*-
"""AiiDA-abinit output parser."""
import logging
from os import path
import pathlib as pl
import re
from tempfile import TemporaryDirectory

from abipy import abilab
from abipy.dynamics.hist import HistFile
from abipy.abio.inputs import _DATA_PREFIX
from abipy.flowtk import events
from aiida.common.exceptions import NotExistent
from aiida.engine import ExitCode
from aiida.orm import BandsData, Dict, StructureData, TrajectoryData
from aiida.parsers.parser import Parser
import netCDF4 as nc
import numpy as np
from pymatgen.core import units

UNITS_SUFFIX = '_units'
DEFAULT_CHARGE_UNITS = 'e'
DEFAULT_DIPOLE_UNITS = 'Debye'
DEFAULT_ENERGY_UNITS = 'eV'
DEFAULT_FORCE_UNITS = 'eV / Angstrom'
DEFAULT_K_POINTS_UNITS = '1 / Angstrom'
DEFAULT_LENGTH_UNITS = 'Angstrom'
DEFAULT_MAGNETIZATION_UNITS = 'Bohr mag. / cell'
DEFAULT_POLARIZATION_UNITS = 'C / m^2'
DEFAULT_STRESS_UNITS = 'GPa'



def _read_singlefile_text(singlefile):
    with singlefile.open(mode='r') as handle:
        return handle.read()



def _split_nonempty_lines(text):
    return [line.strip() for line in text.splitlines() if line.strip()]



def _resolve_retrieved_path(dirpath, relpath):
    relpath = str(relpath)
    candidate = pl.Path(dirpath) / relpath
    if candidate.exists():
        return candidate

    candidate = pl.Path(dirpath) / pl.Path(relpath).name
    if candidate.exists():
        return candidate

    return None



def _parse_created_output_files(stdout_text):
    created_files = re.findall(r'Creating HDf5 file(?: WITHOUT MPI-IO support)?:\s*(\S+)', stdout_text)
    return sorted(set(created_files))



def _clean_text_value(text):
    return str(text).replace('\x00', '').strip()



def _collapse_char_sequence(array):
    if isinstance(array, np.ma.MaskedArray):
        array = array.filled(b'')

    chartyped = np.asarray(array)
    if chartyped.dtype.kind in {'S', 'U'}:
        try:
            decoded = nc.chartostring(chartyped)
            if isinstance(decoded, np.ndarray):
                return _jsonable_value(decoded.tolist())
            return _clean_text_value(decoded)
        except Exception:
            pass

    values = []
    for item in np.asarray(array).tolist():
        if item is None:
            continue
        if isinstance(item, bytes):
            values.append(item)
        else:
            values.append(str(item).encode('utf-8', errors='ignore'))

    raw = b''.join(values)

    if b'\x00' in raw:
        raw = raw.split(b'\x00', 1)[0]
    elif b'\n' in raw:
        last_newline = raw.rfind(b'\n')
        trailing = raw[last_newline + 1:]
        if trailing and len(set(trailing)) == 1:
            raw = raw[:last_newline + 1]

    return _clean_text_value(raw.decode('utf-8', errors='ignore'))



def _decode_char_array(value):
    array = np.asarray(value)

    if array.ndim == 0:
        item = array.item()
        if isinstance(item, bytes):
            return _clean_text_value(item.decode('utf-8', errors='ignore'))
        return _clean_text_value(item)

    if array.dtype.kind in {'S', 'U'}:
        try:
            decoded = nc.chartostring(array)
            if isinstance(decoded, np.ndarray):
                return _jsonable_value(decoded.tolist())
            return _clean_text_value(decoded)
        except Exception:
            pass

    if array.ndim == 1:
        return _collapse_char_sequence(value)

    return [_decode_char_array(item) for item in array]



def _jsonable_value(value):
    if isinstance(value, np.ma.MaskedArray):
        if value.dtype.kind in {'S', 'U'}:
            fill_value = b'' if value.dtype.kind == 'S' else ''
            return _decode_char_array(value.filled(fill_value))
        return _jsonable_value(value.filled(np.nan))

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {'S', 'U'}:
            return _decode_char_array(value)
        return _jsonable_value(value.tolist())

    if isinstance(value, np.generic):
        return _jsonable_value(value.item())

    if isinstance(value, bytes):
        return _clean_text_value(value.decode('utf-8', errors='ignore'))

    if isinstance(value, str):
        return _clean_text_value(value)

    if isinstance(value, float):
        return value if np.isfinite(value) else None

    if isinstance(value, (int, bool)) or value is None:
        return value

    if isinstance(value, (list, tuple)):
        return [_jsonable_value(item) for item in value]

    if isinstance(value, dict):
        return {str(key): _jsonable_value(val) for key, val in value.items()}

    return str(value)



def _serialize_nc_variable(variable):
    try:
        raw_value = variable[()]
    except Exception:  # pylint: disable=broad-except
        raw_value = variable[:]

    return {
        'dtype': str(variable.dtype),
        'dimensions': list(variable.dimensions),
        'shape': list(variable.shape),
        'attributes': {attr: _jsonable_value(variable.getncattr(attr)) for attr in variable.ncattrs()},
        'value': _jsonable_value(raw_value),
    }



def _serialize_nc_group(group, *, root=False):
    data = {
        'dimensions': {name: len(dim) for name, dim in group.dimensions.items()},
        'variables': {name: _serialize_nc_variable(var) for name, var in group.variables.items()},
    }

    attributes = {attr: _jsonable_value(group.getncattr(attr)) for attr in group.ncattrs()}
    if root:
        data['global_attributes'] = attributes
    else:
        data['attributes'] = attributes

    if group.groups:
        data['groups'] = {name: _serialize_nc_group(subgroup, root=False) for name, subgroup in group.groups.items()}

    return data



def _serialize_nc_file(filepath):
    with nc.Dataset(filepath, 'r') as dataset:  # pylint: disable=no-member
        return _serialize_nc_group(dataset, root=True)


class AbinitParser(Parser):
    """Basic parser for the output of an Abinit calculation."""

    def parse(self, **kwargs):
        """Parse outputs, store results in database."""
        try:
            settings = self.node.inputs.settings.get_dict()
        except NotExistent:
            settings = {}

        # Look for optional settings input node and potential 'parser_options' dictionary within it
        parser_options = settings.get('parser_options', None)
        if parser_options is not None:
            error_on_warning = parser_options.get('error_on_warning', False)
            report_comments = parser_options.get('report_comments', True)
        else:
            error_on_warning = False
            report_comments = True

        try:
            retrieved = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        parameters = self.node.inputs.parameters.get_dict()
        ionmov = parameters.get('ionmov', 0)
        optcell = parameters.get('optcell', 0)
        is_relaxation = ionmov != 0 or optcell != 0

        retrieve_list = self.node.base.attributes.get('retrieve_list')
        output_filename = self.node.base.attributes.get('output_filename')
        
        # Dynamically determine the output prefix
        outdata_prefix = parameters.get('outdata_prefix', _DATA_PREFIX.get('outdata_prefix', 'aiidao'))
        gsr_filename = f'{outdata_prefix}_GSR.nc'
        hist_filename = f'{outdata_prefix}_HIST.nc'

        with TemporaryDirectory() as dirpath:
            retrieved.copy_tree(dirpath)

            if output_filename in retrieve_list:
                stdout_filepath = path.join(dirpath, output_filename)
                exit_code = self._parse_stdout(
                    stdout_filepath, error_on_warning=error_on_warning, report_comments=report_comments
                )
                if exit_code is not None:
                    return exit_code
            else:
                return self.exit_codes.ERROR_OUTPUT_MISSING

            # AiiDA stores plain-string retrieve-list entries using only the basename
            # (os.path.split(remote_path)[1]), so 'outdata/out_GSR.nc' arrives as
            # 'out_GSR.nc' at the root of the retrieved folder.  Always resolve by
            # basename so the lookup works regardless of the outdata_prefix format.
            gsr_basename = pl.Path(gsr_filename).name
            gsr_filepath = pl.Path(dirpath) / gsr_basename
            if gsr_filepath.exists():
                self._parse_gsr(str(gsr_filepath), is_relaxation)
            else:
                # Log a warning instead of killing the job (required for DFPT)
                self.logger.warning(
                    f"{gsr_filename} not found. This is normal for DFPT; creating fallback parameters."
                )
                fallback_data = {
                    'parser_warning': f'No {gsr_filename} found. Fallback parameters only.',
                    'is_scf_run': False
                }
                self.out('output_parameters', Dict(dict=fallback_data))

            # Check for dynamic HIST file
            hist_basename = pl.Path(hist_filename).name
            hist_filepath = pl.Path(dirpath) / hist_basename
            if hist_filepath.exists():
                self._parse_trajectory(str(hist_filepath))
            else:
                if is_relaxation:
                    return self.exit_codes.ERROR_MISSING_HIST_OUTPUT_FILE

        return ExitCode(0)

    def _report_message(self, level, message):
        if not isinstance(level, int):
            level = getattr(logging, level.upper(), None)
        if '\n' in message:
            message_lines = message.strip().split('\n')
            message_lines = [f'\t{line}' for line in message_lines]
            message = '\n' + '\n'.join(message_lines)
        self.logger.log(level, '%s', message)

    def _parse_stdout(self, filepath, error_on_warning=False, report_comments=True):
        """Abinit stdout parser."""
        # Read the output log file for potential errors.
        parser = events.EventsParser()
        try:
            report = parser.parse(filepath)
        except:  # pylint: disable=bare-except
            return self.exit_codes.ERROR_OUTPUT_PARSE

        # Handle `ERROR`s
        if len(report.errors) > 0:
            for error in report.errors:
                self._report_message('ERROR', error.message)
            return self.exit_codes.ERROR_OUTPUT_CONTAINS_ERRORS

        # Handle `WARNING`s
        if len(report.warnings) > 0:
            for warning in report.warnings:
                self._report_message('WARNING', warning.message)
            # Need to figure out how to handle the ordering of errors.
            # In theory, this is restartable, but it can occur alongside out
            # of walltime, in which case it probably _isn't_ restartable.
            # for warning in report.warnings:
            #     if ('nstep' in warning.message and
            #         'was not enough SCF cycles to converge.' in warning.message):
            #         return self.exit_codes.ERROR_SCF_CONVERGENCE_NOT_REACHED
            if error_on_warning:
                # This can be quite harsh; inefficient k-point parallelization can cause
                # a non-zero exit in this case.
                return self.exit_codes.ERROR_OUTPUT_CONTAINS_WARNINGS

        # Handle `COMMENT`s
        if len(report.comments) > 0:
            if report_comments:
                self.logger.setLevel('INFO')
                for comment in report.comments:
                    self._report_message('INFO', comment.message)
            for comment in report.comments:
                if ('Approaching time limit' in comment.message and 'Will exit istep loop' in comment.message):
                    return self.exit_codes.ERROR_OUT_OF_WALLTIME

        # Did the run complete?
        if not report.run_completed:
            return self.exit_codes.ERROR_RUN_NOT_COMPLETED

    def _parse_gsr(self, filepath, is_relaxation):
        """Abinit GSR parser."""
        # abipy.electrons.gsr.GsrFile has a method `from_binary_string`;
        # could try to use this instead of copying the files
        with abilab.abiopen(filepath) as gsr:
            gsr_data = {
                'abinit_version': gsr.abinit_version,
                'nband': gsr.nband,
                'nelect': gsr.nelect,
                'nkpt': gsr.nkpt,
                'nspden': gsr.nspden,
                'nspinor': gsr.nspinor,
                'nsppol': gsr.nsppol,
                'cart_stress_tensor': gsr.cart_stress_tensor.tolist(),
                'cart_stress_tensor' + UNITS_SUFFIX: DEFAULT_STRESS_UNITS,
                'is_scf_run': bool(gsr.is_scf_run),
                'cart_forces': gsr.cart_forces.tolist(),
                'cart_forces' + UNITS_SUFFIX: DEFAULT_FORCE_UNITS,
                # 'forces': gsr.cart_forces.tolist(),  # backwards compatibility
                # 'forces' + UNITS_SUFFIX: DEFAULT_FORCE_UNITS,
                'energy': float(gsr.energy),
                'energy' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_localpsp': float(gsr.energy_terms.e_localpsp),
                'e_localpsp' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_eigenvalues': float(gsr.energy_terms.e_eigenvalues),
                'e_eigenvalues' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_ewald': float(gsr.energy_terms.e_ewald),
                'e_ewald' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_hartree': float(gsr.energy_terms.e_hartree),
                'e_hartree' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_corepsp': float(gsr.energy_terms.e_corepsp),
                'e_corepsp' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_corepspdc': float(gsr.energy_terms.e_corepspdc),
                'e_corepspdc' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_kinetic': float(gsr.energy_terms.e_kinetic),
                'e_kinetic' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_nonlocalpsp': float(gsr.energy_terms.e_nonlocalpsp),
                'e_nonlocalpsp' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_entropy': float(gsr.energy_terms.e_entropy),
                'e_entropy' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'entropy': float(gsr.energy_terms.entropy),
                'entropy' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_xc': float(gsr.energy_terms.e_xc),
                'e_xc' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_xcdc': float(gsr.energy_terms.e_xcdc),
                'e_xcdc' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_paw': float(gsr.energy_terms.e_paw),
                'e_paw' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_pawdc': float(gsr.energy_terms.e_pawdc),
                'e_pawdc' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_elecfield': float(gsr.energy_terms.e_elecfield),
                'e_elecfield' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_magfield': float(gsr.energy_terms.e_magfield),
                'e_magfield' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_fermie': float(gsr.energy_terms.e_fermie),
                'e_fermie' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_sicdc': float(gsr.energy_terms.e_sicdc),
                'e_sicdc' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_exactX': float(gsr.energy_terms.e_exactX),
                'e_exactX' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'h0': float(gsr.energy_terms.h0),
                'h0' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_electronpositron': float(gsr.energy_terms.e_electronpositron),
                'e_electronpositron' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'edc_electronpositron': float(gsr.energy_terms.edc_electronpositron),
                'edc_electronpositron' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e0_electronpositron': float(gsr.energy_terms.e0_electronpositron),
                'e0_electronpositron' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'e_monopole': float(gsr.energy_terms.e_monopole),
                'e_monopole' + UNITS_SUFFIX: DEFAULT_ENERGY_UNITS,
                'pressure': float(gsr.pressure),
                'pressure' + UNITS_SUFFIX: DEFAULT_STRESS_UNITS
            }
            structure = StructureData(pymatgen=gsr.structure)

            try:
                # Will return an integer 0 if non-magnetic calculation is run; convert it to a float
                total_magnetization = float(gsr.ebands.get_collinear_mag())
                gsr_data['total_magnetization'] = total_magnetization
                gsr_data['total_magnetization' + UNITS_SUFFIX] = DEFAULT_MAGNETIZATION_UNITS
            except ValueError as exc:
                # `get_collinear_mag`` will raise ValueError if it doesn't know what to do
                if 'Cannot calculate collinear magnetization' in exc.args[0]:
                    pass
                else:
                    raise exc

            try:
                bands_data = BandsData()
                bands_data.set_kpoints(gsr.ebands.kpoints.get_cart_coords())
                bands_data.set_bands(np.array(gsr.ebands.eigens), units=str(gsr.ebands.eigens.unit))
                self.out('output_bands', bands_data)
            # HACK: refine this exception catch
            except:  # pylint: disable=bare-except
                pass

        self.out('output_parameters', Dict(dict=gsr_data))
        if not is_relaxation:
            self.out('output_structure', structure)

    def _parse_trajectory(self, filepath):
        """Abinit trajectory parser."""

        def _voigt_to_tensor(voigt):
            tensor = np.zeros((3, 3))
            tensor[0, 0] = voigt[0]
            tensor[1, 1] = voigt[1]
            tensor[2, 2] = voigt[2]
            tensor[1, 2] = voigt[3]
            tensor[0, 2] = voigt[4]
            tensor[0, 1] = voigt[5]
            tensor[2, 1] = tensor[1, 2]
            tensor[2, 0] = tensor[0, 2]
            tensor[1, 0] = tensor[0, 1]
            return tensor

        with HistFile(filepath) as hist_file:
            structures = hist_file.structures

        output_structure = StructureData(pymatgen=structures[-1])

        with nc.Dataset(filepath, 'r') as data_set:  # pylint: disable=no-member
            n_steps = data_set.dimensions['time'].size
            energy_ha = data_set.variables['etotal'][:].data  # Ha
            energy_kin_ha = data_set.variables['ekin'][:].data  # Ha
            forces_cart_ha_bohr = data_set.variables['fcart'][:, :, :].data  # Ha/bohr
            positions_cart_bohr = data_set.variables['xcart'][:, :, :].data  # bohr
            stress_voigt = data_set.variables['strten'][:, :].data  # Ha/bohr^3

        stepids = np.arange(n_steps)
        symbols = np.array([specie.symbol for specie in structures[0].species], dtype='<U2')
        cells = np.array([structure.lattice.matrix for structure in structures]).reshape((n_steps, 3, 3))
        energy = energy_ha * units.Ha_to_eV
        energy_kin = energy_kin_ha * units.Ha_to_eV
        forces = forces_cart_ha_bohr * units.Ha_to_eV / units.bohr_to_ang
        positions = positions_cart_bohr * units.bohr_to_ang
        stress = np.array([_voigt_to_tensor(sv) for sv in stress_voigt]) * units.Ha_to_eV / units.bohr_to_ang**3
        total_force = np.array([np.sum(f) for f in forces_cart_ha_bohr]) * units.Ha_to_eV / units.bohr_to_ang

        output_trajectory = TrajectoryData()
        output_trajectory.set_trajectory(stepids=stepids, cells=cells, symbols=symbols, positions=positions)
        output_trajectory.set_array('energy', energy)  # eV
        output_trajectory.set_array('energy_kin', energy_kin)  # eV
        output_trajectory.set_array('forces', forces)  # eV/angstrom
        output_trajectory.set_array('stress', stress)  # eV/angstrom^3
        output_trajectory.set_array('total_force', total_force)  # eV/angstrom

        self.out('output_trajectory', output_trajectory)
        self.out('output_structure', output_structure)


class _AbinitUtilityParser(Parser):
    """Parser for text-based ABINIT helper executables such as mrgddb and anaddb."""

    def _read_stdin_text(self):
        return _read_singlefile_text(self.node.inputs.stdin_file)

    def _parse_stdin_arguments(self, stdin_text):
        return {}

    def _parse_nc_outputs(self, dirpath):
        nc_files = {}
        for filepath in sorted(pl.Path(dirpath).rglob('*.nc')):
            nc_files[str(filepath.relative_to(dirpath))] = _serialize_nc_file(filepath)
        return nc_files

    def parse(self, **kwargs):
        try:
            retrieved = self.retrieved
        except NotExistent:
            return self.exit_codes.ERROR_NO_RETRIEVED_FOLDER

        output_filename = self.node.base.attributes.get('output_filename')
        retrieve_list = self.node.base.attributes.get('retrieve_list', [])
        cmdline_params = list(self.node.base.attributes.get('cmdline_params', []))
        stdin_text = self._read_stdin_text()
        parsed_arguments = self._parse_stdin_arguments(stdin_text)

        with TemporaryDirectory() as dirpath:
            retrieved.copy_tree(dirpath)
            stdout_filepath = _resolve_retrieved_path(dirpath, output_filename)
            if stdout_filepath is None:
                return self.exit_codes.ERROR_OUTPUT_MISSING

            try:
                stdout_text = stdout_filepath.read_text(encoding='utf-8')
            except OSError:
                self.logger.exception('unable to read stdout for CalcJobNode<%s>', self.node.pk)
                return self.exit_codes.ERROR_OUTPUT_READ

            retrieved_files = []
            missing_files = []
            for relpath in retrieve_list:
                candidate = _resolve_retrieved_path(dirpath, relpath)
                if candidate is not None:
                    retrieved_files.append(str(relpath))
                else:
                    missing_files.append(str(relpath))

            nc_files = self._parse_nc_outputs(dirpath)

        self.out('output_parameters', Dict(dict={
            'stdout': stdout_text,
            'cmdline_params': cmdline_params,
            'parsed_arguments': parsed_arguments,
            'retrieved_files': sorted(set(retrieved_files)),
            'missing_retrieved_files': sorted(set(missing_files)),
            'created_files': _parse_created_output_files(stdout_text),
            'nc_files': nc_files,
        }))
        return ExitCode(0)


class MrgddbParser(_AbinitUtilityParser):
    """Parser for `mrgddb` utility jobs."""

    def _parse_stdin_arguments(self, stdin_text):
        lines = _split_nonempty_lines(stdin_text)
        parsed = {}

        if lines:
            parsed['output_ddb'] = lines[0]
        if len(lines) >= 2:
            parsed['description'] = lines[1]
        if len(lines) >= 3:
            try:
                parsed['input_ddb_count'] = int(lines[2])
            except ValueError:
                parsed['input_ddb_count_text'] = lines[2]
        if 'input_ddb_count' in parsed:
            parsed['input_ddbs'] = lines[3:3 + parsed['input_ddb_count']]
        elif len(lines) > 3:
            parsed['input_ddbs'] = lines[3:]

        return parsed


class AnaddbParser(_AbinitUtilityParser):
    """Parser for `anaddb` utility jobs."""

    _ROOT_OUTPUT_SUFFIXES = [
        '_anaddb.nc',
        '_PHBST.nc',
        '_PHBANDS.agr',
        '_PHFRQ',
        '_PHANGMOM',
    ]
    _EXTRA_OUTPUTS = ['PHBST_partial_DOS']

    def _parse_stdin_arguments(self, stdin_text):
        lines = _split_nonempty_lines(stdin_text)
        keys = [
            'input_file',
            'output_file',
            'ddb_filepath',
            'thm_output',
            'gkk_filepath',
            'elphon_output_root',
            'ddk_filenames_filepath',
        ]
        parsed = {key: value for key, value in zip(keys, lines)}

        output_file = parsed.get('output_file')
        if output_file:
            output_root = pl.Path(output_file).stem
            parsed['output_root'] = output_root
            parsed['expected_output_files'] = [output_file]
            parsed['expected_output_files'].extend(f'{output_root}{suffix}' for suffix in self._ROOT_OUTPUT_SUFFIXES)
            parsed['expected_output_files'].extend(self._EXTRA_OUTPUTS)
            thm_output = parsed.get('thm_output')
            if thm_output and not thm_output.endswith('_dummy'):
                parsed['expected_output_files'].append(thm_output)
            parsed['expected_output_files'] = sorted(set(parsed['expected_output_files']))

        return parsed


class OpticParser(_AbinitUtilityParser):
    """Parser for `optic` utility jobs."""

    def _parse_stdin_arguments(self, stdin_text):
        lines = _split_nonempty_lines(stdin_text)
        parsed = {'raw_lines': lines}

        first_line = lines[0].lstrip() if lines else ''
        if first_line.startswith('&'):
            try:
                input_file = pl.Path(self.node.inputs.stdin_file.filename).name
            except Exception:
                input_file = ''
            parsed['input_mode'] = 'direct_namelist'
        elif lines:
            input_file = pl.Path(lines[0]).name
            parsed['input_mode'] = 'control_file'
        else:
            input_file = ''

        if input_file:
            output_root = pl.Path(input_file).stem
            parsed['input_file'] = input_file
            parsed['output_root'] = output_root
            parsed['expected_output_files'] = [f'{output_root}_OPTIC.nc']

        return parsed
