# -*- coding: utf-8 -*-
"""CalcJob class for Abinit."""
import io
import os
import pathlib as pl
import typing as ty

from abipy.abio.inputs import _DATA_PREFIX, AbinitInput
from abipy.core.structure import Structure as AbiStructure
from abipy.data.hgh_pseudos import HGH_TABLE
from aiida import orm
from aiida.common import constants, datastructures, exceptions
from aiida.engine import CalcJob
from aiida_pseudo.data.pseudo import JthXmlData, Psp8Data
from pymatgen.io.abinit.abiobjects import structure_to_abivars

from aiida_abinit.utils import seconds_to_timelimit, uppercase_dict



def _read_singlefile_text(singlefile: orm.SinglefileData) -> str:
    with singlefile.open(mode='r') as handle:
        return handle.read()



def _split_nonempty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]



def _normalize_shift_rows(shift, *, label: str) -> list[list[float]]:
    """Normalize `shiftk`/`shiftq`-style input into a list of 3-component rows."""
    if not isinstance(shift, (list, tuple)):
        raise exceptions.InputValidationError(f'{label} must be a sequence of numeric values.')

    if len(shift) == 0:
        raise exceptions.InputValidationError(f'{label} cannot be empty.')

    if all(not isinstance(value, (list, tuple)) for value in shift):
        try:
            flat = [float(value) for value in shift]
        except (TypeError, ValueError) as exc:
            raise exceptions.InputValidationError(f'{label} must contain only numeric values.') from exc

        if len(flat) % 3 != 0:
            raise exceptions.InputValidationError(
                f'{label} must contain 3 values per shift vector.'
            )

        return [flat[index:index + 3] for index in range(0, len(flat), 3)]

    rows = []
    for row in shift:
        if not isinstance(row, (list, tuple)):
            raise exceptions.InputValidationError(
                f'{label} must be given either as a flat list or a list of 3-component vectors.'
            )
        if len(row) != 3:
            raise exceptions.InputValidationError(f'{label} rows must each contain exactly three values.')
        try:
            rows.append([float(value) for value in row])
        except (TypeError, ValueError) as exc:
            raise exceptions.InputValidationError(f'{label} must contain only numeric values.') from exc

    return rows



def _single_shifts_match(lhs, rhs, *, tol: float = 1.0e-12) -> bool:
    return len(lhs) == len(rhs) == 3 and all(abs(float(a) - float(b)) <= tol for a, b in zip(lhs, rhs))


class AbinitCalculation(CalcJob):
    """AiiDA calculation plugin wrapping the abinit executable."""

    _DEFAULT_PREFIX = 'aiida'
    _DEFAULT_INPUT_EXTENSION = 'in'
    _DEFAULT_OUTPUT_EXTENSION = 'out'
    _PSEUDO_SUBFOLDER = './pseudo/'

    _BLOCKED_KEYWORDS = [
        # Structure-related keywords set automatically from the `StructureData``
        'acell',
        'angdeg',
        'natom',
        'ntypat',
        'rprim',
        'rprimd',
        'brvltt',
        'typat',
        'xcart',
        'xred',
        'znucl',
        'natrd',
        'xyzfile',
        # K-point-related keywords set automatically from the `KpointsData`
        'kpt',
        'ngkpt',
        'nkpath',
        'nkpt',
        'nshiftk',
        'wtk'
    ]

    @classmethod
    def define(cls, spec):
        """Define inputs and outputs of the calculation."""
        # yapf: disable
        super(AbinitCalculation, cls).define(spec)

        spec.input('metadata.options.prefix',
                   valid_type=str,
                   default=cls._DEFAULT_PREFIX)
        spec.input('metadata.options.input_extension',
                   valid_type=str,
                   default=cls._DEFAULT_INPUT_EXTENSION)
        spec.input('metadata.options.output_extension',
                   valid_type=str,
                   default=cls._DEFAULT_OUTPUT_EXTENSION)
        spec.input('metadata.options.withmpi',
                   valid_type=bool,
                   default=True)

        spec.input('structure',
                   valid_type=orm.StructureData,
                   help='The input structure.')
        spec.input('kpoints',
                   valid_type=orm.KpointsData,
                   help='The k-point mesh or path')
        spec.input('parameters',
                   valid_type=orm.Dict,
                   help='The ABINIT input parameters.')
        spec.input('settings',
                   valid_type=orm.Dict,
                   required=False,
                   help='Various special settings.')
        spec.input('parent_calc_folder',
                   valid_type=orm.RemoteData,
                   required=False,
                   help='A remote folder used for restarts.')
        spec.input_namespace('pseudos',
                             valid_type=(Psp8Data, JthXmlData),
                             help='The pseudopotentials.',
                             dynamic=True)
        options = spec.inputs['metadata']['options']
        options['parser_name'].default = 'abinit'
        options['resources'].default = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        options['input_filename'].default = f'{cls._DEFAULT_PREFIX}.{cls._DEFAULT_INPUT_EXTENSION}'
        options['output_filename'].default = f'{cls._DEFAULT_PREFIX}.{cls._DEFAULT_OUTPUT_EXTENSION}'

        # Unrecoverable errors: file missing
        spec.exit_code(100, 'ERROR_MISSING_OUTPUT_FILES',
                       message='Calculation did not produce all expected output files.')
        spec.exit_code(101, 'ERROR_MISSING_GSR_OUTPUT_FILE',
                       message='Calculation did not produce the expected `GSR.nc` output file.')
        spec.exit_code(102, 'ERROR_MISSING_HIST_OUTPUT_FILE',
                       message='Calculation did not produce the expected `HIST.nc` output file.')
        # Unrecoverable errors: resources like the retrieved folder or its expected contents are missing.
        spec.exit_code(200, 'ERROR_NO_RETRIEVED_FOLDER',
                       message='The retrieved folder data node could not be accessed.')
        spec.exit_code(210, 'ERROR_OUTPUT_MISSING',
                       message='The retrieved folder did not contain the `stdout` output file.')
        # Unrecoverable errors: required retrieved files could not be read, parsed or are otherwise incomplete.
        spec.exit_code(301, 'ERROR_OUTPUT_READ',
                       message='The `stdout` output file could not be read.')
        spec.exit_code(302, 'ERROR_OUTPUT_PARSE',
                       message='The `stdout` output file could not be parsed.')
        spec.exit_code(303, 'ERROR_RUN_NOT_COMPLETED',
                       message='The `abipy` `EventsParser` reports that the run was not completed.')
        spec.exit_code(304, 'ERROR_OUTPUT_CONTAINS_ERRORS',
                       message='The output file contains one or more error messages.')
        spec.exit_code(305, 'ERROR_OUTPUT_CONTAINS_WARNINGS',
                       message='The output file contains one or more warning messages.')
        spec.exit_code(312, 'ERROR_STRUCTURE_PARSE',
                       message='The output structure could not be parsed.')
        # Significant errors but calculation can be used to restart
        spec.exit_code(400, 'ERROR_OUT_OF_WALLTIME',
                       message='The calculation stopped prematurely because it ran out of walltime.')
        spec.exit_code(500, 'ERROR_SCF_CONVERGENCE_NOT_REACHED',
                       message='The SCF minimization cycle did not converge.')
        spec.exit_code(501, 'ERROR_GEOMETRY_CONVERGENCE_NOT_REACHED',
                       message='The ionic minimization cycle did not converge.')

        # Outputs
        spec.output('output_parameters',
                    valid_type=orm.Dict,
                    required=True,
                    help='Various output quantities.')
        spec.output('output_structure',
                    valid_type=orm.StructureData,
                    required=False,
                    help='Final structure of the calculation if present.')
        spec.output('output_trajectory',
                    valid_type=orm.TrajectoryData,
                    required=False,
                    help='Trajectory of various output quantities over the calculation if present.')
        spec.output('output_bands',
                    valid_type=orm.BandsData,
                    required=False,
                    help='Final electronic bands if present.')
        spec.default_output_node = 'output_parameters'

    def _validate_parameters(self):
        """Validate the 'parameters' input `Dict` node.

        Check that no blocked keywords are present.
        """
        keyword_intersection = set(self.inputs.parameters.keys()) & set(self._BLOCKED_KEYWORDS)
        if len(keyword_intersection) > 0:
            raise exceptions.InputValidationError(
                f"Some blocked input keywords were provided: {', '.join(list(keyword_intersection))}"
            )

    def _validate_pseudos(self):
        """Validate the 'pseudos' input namespace.

        Check that each 'kind' in the input `StructureData` has a corresponding pseudopotential.
        """
        kinds = [kind.name for kind in self.inputs.structure.kinds]
        if set(kinds) != set(self.inputs.pseudos.keys()):
            pseudos_str = ', '.join(list(self.inputs.pseudos.keys()))
            kinds_str = ', '.join(list(kinds))
            raise exceptions.InputValidationError(
                'Mismatch between the defined pseudos and the list of kinds of the structure.\n'
                f'Pseudos: {pseudos_str};\nKinds:{kinds_str}'
            )

    def _generate_inputdata(self,
                            parameters: orm.Dict,
                            pseudos,
                            structure: orm.StructureData,
                            kpoints: orm.KpointsData) -> ty.Tuple[str, list]:
        """Generate the input file content and list of pseudopotential files to copy.

        :param parameters: input parameters Dict
        :param pseudos: pseudopotential input namespace
        :param structure: input structure
        :param kpoints: input kpoints
        :returns: input file content, pseudopotential copy list
        """
        local_copy_pseudo_list = []

        # `abipy`` has its own subclass of Pymatgen's `Structure`, so we use that
        pmg_structure = structure.get_pymatgen()
        abi_structure = AbiStructure.as_structure(pmg_structure)
        # NOTE: need to refine the `abi_sanitize` parameters
        # Skipping: we do not want to change, at the plugin level, the structure
        # This should be done by the user (or by the workflow) before giving it
        # to us
        #abi_structure = abi_structure.abi_sanitize(symprec=1e-3, angle_tolerance=5,
        #    primitive=False, primitive_standard=False)

        for kind in structure.get_kind_names():
            pseudo = pseudos[kind]
            local_copy_pseudo_list.append((pseudo.uuid, pseudo.filename, f'{self._PSEUDO_SUBFOLDER}{pseudo.filename}'))
        # Pseudopotentials _must_ be listed in the same order as 'znucl' in the input file.
        # So, we need to get 'znucl' as abipy will write it then construct the appropriate 'pseudos' string.
        znucl = structure_to_abivars(abi_structure)['znucl']
        ordered_pseudo_filenames = [pseudos[constants.elements[Z]['symbol']].filename for Z in znucl]
        pseudo_parameters = {
            'pseudos': '"' + ', '.join(ordered_pseudo_filenames) + '"',
            'pp_dirpath': f'"{self._PSEUDO_SUBFOLDER}"'
        }

        input_parameters = parameters.get_dict()

        # Use `abipy`` to write the input file.
        #
        # Multi-dataset ABINIT inputs use suffixed variable names such as
        # `prtden1`, `getwfk3`, `optdriver5`, ... These names are valid for ABINIT but
        # are not recognized by AbiPy's internal variable database during object
        # construction. If we pass them directly via `abi_kwargs`, AbiPy raises an
        # `AbinitInputError` before we even have a chance to disable spell checking.
        #
        # To support both single-dataset and multi-dataset inputs, we therefore:
        #   1. create an empty `AbinitInput`
        #   2. disable spell checking
        #   3. assign variables one-by-one
        #   4. inject the k-mesh information afterwards
        input_parameters = {**input_parameters, **pseudo_parameters}
        kptopt = input_parameters.pop('kptopt', 1)
        explicit_shiftk = input_parameters.pop('shiftk', None)

        mesh = None
        mesh_offset = [0.0, 0.0, 0.0]
        try:
            mesh, mesh_offset = kpoints.get_kpoints_mesh()
        except AttributeError:
            pass

        if mesh is not None:
            mesh = [int(value) for value in mesh]
            mesh_offset = [float(value) for value in mesh_offset]

            if explicit_shiftk is None:
                shiftk = list(mesh_offset)
            else:
                shift_rows = _normalize_shift_rows(explicit_shiftk, label='`shiftk`')
                if len(shift_rows) == 1:
                    if not _single_shifts_match(shift_rows[0], mesh_offset):
                        raise exceptions.InputValidationError(
                            'Explicit `shiftk` does not match the offset stored in `kpoints`. '
                            'Keep them identical or omit `shiftk` and use `kpoints` as the source of truth.'
                        )
                elif not _single_shifts_match(mesh_offset, [0.0, 0.0, 0.0]):
                    raise exceptions.InputValidationError(
                        'Regular-mesh `kpoints` can store only one offset. When specifying multiple `shiftk` vectors, '
                        'set the `kpoints` mesh offset to Gamma `(0, 0, 0)` and pass the complete shift pattern explicitly.'
                    )
                shiftk = shift_rows
        else:
            if explicit_shiftk is not None:
                raise exceptions.InputValidationError(
                    '`shiftk` cannot be specified when `kpoints` does not define a regular mesh.'
                )

        # `AbinitInput` requires a valid pseudo table / list of pseudos, so we give it the `HGH_TABLE`,
        # which should always work. In the end, we do _not_ print these to the input file.
        abi_input = AbinitInput(
            structure=abi_structure,
            pseudos=HGH_TABLE,
            abi_kwargs={}
        )
        abi_input.set_spell_check(False)

        for key, value in input_parameters.items():
            abi_input[key] = value

        if mesh is not None:
            abi_input.set_kmesh(
                ngkpt=mesh,
                shiftk=shiftk,
                kptopt=kptopt
            )
        else:
            abi_input['kptopt'] = kptopt
            abi_input['kptnrm'] = input_parameters.pop('kptnrm', 1)
            abi_input['kpt'] = kpoints.get_kpoints()
            abi_input['nkpt'] = len(abi_input['kpt'])

        return abi_input.to_string(with_pseudos=False), local_copy_pseudo_list

    def _generate_cmdline_params(self, settings: dict) -> ty.List[str]:
        # The input file has to be the first parameter
        cmdline_params = [self.metadata.options.input_filename]

        # If a max wallclock is set in the `options`, we also set the `--timelimit` param
        if 'max_wallclock_seconds' in self.metadata.options:
            max_wallclock_seconds = self.metadata.options.max_wallclock_seconds
            cmdline_params.extend(['--timelimit', seconds_to_timelimit(max_wallclock_seconds)])

        # If a number of OMP threads is set in the options, we set the `--omp-num-threads` param
        if 'num_omp_threads' in self.metadata.options.resources:
            omp_num_threads = self.metadata.options.resources['omp_num_threads']
            cmdline_params.extend(['--omp-num-threads', f'{omp_num_threads:d}'])

        # Enable verbose mode if requested in the settings
        if settings.pop('VERBOSE', False):
            cmdline_params.append('--verbose')

        # Enable a dry run if requested in the settings
        # NOTE: don't pop here, we need to know about dry runs when generating the retrieve list
        if settings.get('DRY_RUN', False):
            cmdline_params.append('--dry-run')

        return cmdline_params

    def _generate_retrieve_list(self, parameters: orm.Dict, settings: dict) -> list:
        """Generate the list of files to retrieve based on the type of calculation requested in the input parameters.

        :param parameters: input parameters
        :returns: list of files to retrieve
        """
        parameters = parameters.get_dict()
        prefix = self.metadata.options.prefix
        outdata_prefix = parameters.get('outdata_prefix', _DATA_PREFIX.get('outdata_prefix', 'aiidao'))
        ndtset = int(parameters.get('ndtset', 1) or 1)

        retrieve_list = [f'{prefix}.{self._DEFAULT_OUTPUT_EXTENSION}']
        retrieve_list += [f'{prefix}.abo']
        retrieve_list += settings.pop('ADDITIONAL_RETRIEVE_LIST', [])

        if not settings.pop('DRY_RUN', False):
            # AiiDA will safely ignore any files in this list that ABINIT didn't actually produce.
            #
            # For multi-dataset jobs (e.g. NLO / rf2-style), ABINIT writes dataset-indexed files such as:
            #   out_DS4_DDB, out_DS5_DDB, out_DS2_GSR.nc, ...
            # The original plugin only retrieved the single-dataset names and therefore silently missed
            # the important DDBs needed for post-processing. Handle both cases here.
            if ndtset <= 1:
                retrieve_list += [
                    # Core DFT / DFPT
                    f'{outdata_prefix}_GSR.nc',
                    f'{outdata_prefix}_OUT.nc',
                    f'{outdata_prefix}_DDB',
                    # Band Structures & DOS
                    f'{outdata_prefix}_BANDS.nc',
                    f'{outdata_prefix}_FATBANDS.nc',
                    f'{outdata_prefix}_DOS.nc',
                    # Optics & Many-Body (GW/BSE)
                    f'{outdata_prefix}_OPT.nc',
                    f'{outdata_prefix}_QPS.nc',
                    f'{outdata_prefix}_MBS.nc',
                    f'{outdata_prefix}_BS.nc',
                    f'{outdata_prefix}_EXC.nc',
                ]
            else:
                retrieve_list += [f'{outdata_prefix}_OUT.nc']
                for idt in range(1, ndtset + 1):
                    ds = f'{outdata_prefix}_DS{idt}'
                    retrieve_list += [
                        f'{ds}_GSR.nc',
                        f'{ds}_DDB',
                        f'{ds}_OPT.nc',
                        f'{ds}_QPS.nc',
                        f'{ds}_MBS.nc',
                        f'{ds}_BS.nc',
                        f'{ds}_EXC.nc',
                    ]

            if parameters.get('ionmov', 0) > 0 or parameters.get('optcell', 0) > 0:
                retrieve_list += [f'{outdata_prefix}_HIST.nc']

        return list(set(retrieve_list))

    def prepare_for_submission(self, folder):
        """Create the input file(s) and execution instructions from the input nodes.

        This method performs the core preparation for the AiiDA engine:
        1. Validates the input parameters and pseudopotentials.
        2. Generates and writes the main ABINIT input file to the local folder.
        3. Stages local pseudopotential files into the appropriate subdirectory.
        4. Sets up the command-line parameters and the list of files to retrieve.

        Additionally, if a `parent_calc_folder` is provided for a restart, it handles
        remote file logistics. It dynamically lists files from both the parent's input 
        and output directories. Output files strictly overwrite input files to ensure 
        fresh data precedence. These files are then safely symlinked or copied to the 
        new calculation's indata directory.

        :param folder: An `aiida.common.folders.Folder` where the plugin will temporarily 
            place all local files needed by the calculation.
        :return: An `aiida.common.datastructures.CalcInfo` instance containing the daemon 
            instructions for execution, retrieval, and remote file linking.
        """
        # Process the `settings`` so that capitalization isn't an issue
        settings = uppercase_dict(self.inputs.settings.get_dict()) if 'settings' in self.inputs else {}

        # Validate the input parameters and pseudopotentials
        self._validate_parameters()
        self._validate_pseudos()

        # Create lists which specify files to copy and symlink
        local_copy_list = []
        remote_copy_list = []
        remote_symlink_list = []

        # Create the subfolder which will contain the pseudopotential files
        folder.get_subfolder(self._PSEUDO_SUBFOLDER, create=True)
        for value in _DATA_PREFIX.values():
            folder.get_subfolder(str(pl.Path(value).parent), create=True)

        # Generate the input file content and list of pseudopotential files to copy
        arguments = [
            self.inputs.parameters,
            self.inputs.pseudos,
            self.inputs.structure,
            self.inputs.kpoints
        ]
        input_filecontent, local_copy_pseudo_list = self._generate_inputdata(*arguments)

        # Merge the pseudopotential copy list with the overall copy list then write the input file
        local_copy_list += local_copy_pseudo_list
        with io.open(folder.get_abs_path(self.metadata.options.input_filename), mode='w', encoding='utf-8') as stream:
            stream.write(input_filecontent)

        # List the files to copy or symlink in the case of a restart.
        # `parent_calc_folder` is the public CalcJob input name but we also accept
        # the legacy internal `parent_folder` key for backwards compatibility.
        #
        # By default, use the parent's `outdata_prefix` as the restart-file prefix.
        # This can be overridden with the plugin setting `PARENT_OUTDATA_PREFIX`, e.g.
        # `aiidao` for older lineages that wrote restart files in the workdir root.
        parent_folder = self.inputs.get('parent_calc_folder', self.inputs.get('parent_folder', None))
        if parent_folder is not None:
            parameters = self.inputs.parameters.get_dict()
            same_computer = self.inputs.code.computer.uuid == parent_folder.computer.uuid
            use_symlink = settings.pop('PARENT_FOLDER_SYMLINK', same_computer)

            # Identify the parent OUTPUT prefix and directory
            default_parent_out_prefix = parameters.get('outdata_prefix', _DATA_PREFIX.get('outdata_prefix', 'aiidao'))
            parent_out_prefix = settings.pop('PARENT_OUTDATA_PREFIX', default_parent_out_prefix)
            parent_out_path = pl.Path(str(parent_out_prefix).strip())
            parent_out_dir = str(parent_out_path.parent) if str(parent_out_path.parent) != '.' else './'
            parent_out_name = parent_out_path.name

            # Identify the parent INPUT prefix and directory
            default_parent_in_prefix = parameters.get('indata_prefix', _DATA_PREFIX.get('indata_prefix', 'aiidai'))
            parent_in_prefix = settings.pop('PARENT_INDATA_PREFIX', default_parent_in_prefix)
            parent_in_path = pl.Path(str(parent_in_prefix).strip())
            parent_in_dir = str(parent_in_path.parent) if str(parent_in_path.parent) != '.' else './'
            parent_in_name = parent_in_path.name

            # Identify the current input prefix (what we are renaming files to)
            current_in_prefix = str(parameters.get('indata_prefix', _DATA_PREFIX.get('indata_prefix', 'aiidai')))
            current_in_path = pl.Path(current_in_prefix)
            current_in_dir = str(current_in_path.parent) if str(current_in_path.parent) != '.' else './'
            current_in_name = current_in_path.name

            # Ensure the destination subdirectory (e.g., 'indata/') exists
            if current_in_dir != './':
                folder.get_subfolder(current_in_dir, create=True)

            # Retrieve parent OUT files
            try:
                list_dir_target_out = parent_out_dir if parent_out_dir != './' else '.'
                existing_out_files = parent_folder.listdir(list_dir_target_out)
            except Exception:
                existing_out_files = []

            # Retrieve parent IN files
            try:
                list_dir_target_in = parent_in_dir if parent_in_dir != './' else '.'
                existing_in_files = parent_folder.listdir(list_dir_target_in)
            except Exception:
                existing_in_files = []

            # Dictionary to track files by their suffix and enforce output precedence
            # Format: { suffix: remote_abs_path }
            files_to_link = {}

            # 1. Process INPUT files first (Lower Precedence)
            for filename in existing_in_files:
                if filename.startswith(parent_in_name):
                    suffix = filename[len(parent_in_name):]
                    remote_abs_path = os.path.join(parent_folder.get_remote_path(), parent_in_dir, filename)
                    files_to_link[suffix] = remote_abs_path

            # 2. Process OUTPUT files second (Higher Precedence, overwrites identical suffixes)
            for filename in existing_out_files:
                if filename.startswith(parent_out_name):
                    suffix = filename[len(parent_out_name):]
                    remote_abs_path = os.path.join(parent_folder.get_remote_path(), parent_out_dir, filename)
                    files_to_link[suffix] = remote_abs_path

            # Map and link all resolved files to the new calculation
            for suffix, remote_abs_path in files_to_link.items():
                new_filename = f"{current_in_name}{suffix}"
                dest_rel_path = os.path.join(current_in_dir, new_filename)

                if use_symlink:
                    remote_symlink_list.append((
                        parent_folder.computer.uuid,
                        remote_abs_path,
                        dest_rel_path,
                    ))
                else:
                    remote_copy_list.append((
                        parent_folder.computer.uuid,
                        remote_abs_path,
                        dest_rel_path,
                    ))

        # Generate the commandline parameters
        cmdline_params = self._generate_cmdline_params(settings)

        # Generate list of files to retrieve from wherever the calculation is run
        retrieve_list = self._generate_retrieve_list(self.inputs.parameters, settings)

        # Set up the `CodeInfo` to pass to `CalcInfo`
        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline_params
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.withmpi = self.inputs.metadata.options.withmpi

        # Set up the `CalcInfo` so AiiDA knows what to do with everything
        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.stdin_name = self.metadata.options.input_filename
        calcinfo.stdout_name = self.metadata.options.output_filename
        calcinfo.retrieve_list = retrieve_list
        calcinfo.remote_symlink_list = remote_symlink_list
        calcinfo.remote_copy_list = remote_copy_list
        calcinfo.local_copy_list = local_copy_list

        return calcinfo


class _AbinitUtilityCalculation(CalcJob):
    """Base CalcJob for small ABINIT companion executables driven by stdin files."""

    _DEFAULT_PREFIX = 'aiida'
    _DEFAULT_INPUT_EXTENSION = 'in'
    _DEFAULT_OUTPUT_EXTENSION = 'out'
    _PARSER_NAME = None
    _DEFAULT_RETRIEVE_LIST = []

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('stdin_file',
                   valid_type=orm.SinglefileData,
                   help='Input file passed to the executable through stdin.')
        spec.input('settings',
                   valid_type=orm.Dict,
                   required=False,
                   help='Special settings such as files to stage and extra files to retrieve.')
        spec.input_namespace('files',
                             valid_type=orm.SinglefileData,
                             required=False,
                             dynamic=True,
                             help='Additional files staged into the working directory.')

        options = spec.inputs['metadata']['options']
        if cls._PARSER_NAME is not None:
            options['parser_name'].default = cls._PARSER_NAME
        options['resources'].default = {'num_machines': 1, 'num_mpiprocs_per_machine': 1}
        options['input_filename'].default = f'{cls._DEFAULT_PREFIX}.{cls._DEFAULT_INPUT_EXTENSION}'
        options['output_filename'].default = f'{cls._DEFAULT_PREFIX}.{cls._DEFAULT_OUTPUT_EXTENSION}'
        options['withmpi'].default = False

        spec.exit_code(200, 'ERROR_NO_RETRIEVED_FOLDER', message='The retrieved folder data node could not be accessed.')
        spec.exit_code(210, 'ERROR_OUTPUT_MISSING', message='The retrieved folder did not contain the stdout output file.')
        spec.exit_code(301, 'ERROR_OUTPUT_READ', message='The stdout output file could not be read.')

        spec.output('output_parameters',
                    valid_type=orm.Dict,
                    required=True,
                    help='Plain-text stdout and retrieved-file bookkeeping.')
        spec.default_output_node = 'output_parameters'

    def _generate_cmdline_params(self, settings: dict) -> ty.List[str]:
        cmdline_params = settings.pop('CMDLINE', [])
        if isinstance(cmdline_params, str):
            cmdline_params = [cmdline_params]
        if not isinstance(cmdline_params, (list, tuple)):
            raise exceptions.InputValidationError('`CMDLINE` setting must be a string, list or tuple.')
        return [str(param) for param in cmdline_params]

    def _infer_retrieve_list(self, stdin_text: str, settings: dict) -> list[str]:
        return []

    def _generate_retrieve_list(self, settings: dict, stdin_text: str) -> list[str]:
        retrieve_list = [self.metadata.options.output_filename]
        retrieve_list += self._infer_retrieve_list(stdin_text, settings)
        retrieve_list += list(self._DEFAULT_RETRIEVE_LIST)
        retrieve_list += settings.pop('ADDITIONAL_RETRIEVE_LIST', [])
        return list(dict.fromkeys(str(path) for path in retrieve_list if path))

    def prepare_for_submission(self, folder):
        settings = uppercase_dict(self.inputs.settings.get_dict()) if 'settings' in self.inputs else {}
        stdin_text = _read_singlefile_text(self.inputs.stdin_file)

        local_copy_list = [
            (
                self.inputs.stdin_file.uuid,
                self.inputs.stdin_file.filename,
                self.metadata.options.input_filename,
            )
        ]

        files_to_copy = settings.pop('FILES_TO_COPY', None)
        if files_to_copy is None:
            files_to_copy = [(label, label) for label in self.inputs.get('files', {}).keys()]

        if not isinstance(files_to_copy, (list, tuple)):
            raise exceptions.InputValidationError('`FILES_TO_COPY` setting must be a list of (input_label, destination) pairs.')

        for item in files_to_copy:
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                raise exceptions.InputValidationError('Invalid `FILES_TO_COPY` entry; expected (input_label, destination).')
            link_name, dest_rel_path = item
            try:
                file_node = self.inputs.files[link_name]
            except (AttributeError, KeyError) as exc:
                raise exceptions.InputValidationError(
                    f'`FILES_TO_COPY` references missing input file namespace key: {link_name}'
                ) from exc
            local_copy_list.append((file_node.uuid, file_node.filename, str(dest_rel_path)))

        cmdline_params = self._generate_cmdline_params(settings)
        retrieve_list = self._generate_retrieve_list(settings, stdin_text)

        codeinfo = datastructures.CodeInfo()
        codeinfo.code_uuid = self.inputs.code.uuid
        codeinfo.cmdline_params = cmdline_params
        codeinfo.stdin_name = self.metadata.options.input_filename
        codeinfo.stdout_name = self.metadata.options.output_filename
        codeinfo.withmpi = self.inputs.metadata.options.withmpi

        calcinfo = datastructures.CalcInfo()
        calcinfo.codes_info = [codeinfo]
        calcinfo.stdin_name = self.metadata.options.input_filename
        calcinfo.stdout_name = self.metadata.options.output_filename
        calcinfo.retrieve_list = retrieve_list
        calcinfo.local_copy_list = local_copy_list

        return calcinfo


class MrgddbCalculation(_AbinitUtilityCalculation):
    """CalcJob for the ABINIT `mrgddb` utility."""

    _PARSER_NAME = 'abinit.mrgddb'

    def _infer_retrieve_list(self, stdin_text: str, settings: dict) -> list[str]:
        lines = _split_nonempty_lines(stdin_text)
        return [lines[0]] if lines else []


class AnaddbCalculation(_AbinitUtilityCalculation):
    """CalcJob for the ABINIT `anaddb` utility."""

    _PARSER_NAME = 'abinit.anaddb'
    _ROOT_OUTPUT_SUFFIXES = [
        '_anaddb.nc',
        '_PHBST.nc',
        '_PHBANDS.agr',
        '_PHFRQ',
        '_PHANGMOM',
    ]
    _EXTRA_OUTPUTS = ['PHBST_partial_DOS']

    def _infer_retrieve_list(self, stdin_text: str, settings: dict) -> list[str]:
        lines = _split_nonempty_lines(stdin_text)
        if len(lines) < 2:
            return []

        retrieve_list = [lines[1]]
        output_root = pl.Path(lines[1]).stem
        if output_root:
            retrieve_list.extend(f'{output_root}{suffix}' for suffix in self._ROOT_OUTPUT_SUFFIXES)
        retrieve_list.extend(self._EXTRA_OUTPUTS)

        if len(lines) >= 4 and not lines[3].endswith('_dummy'):
            retrieve_list.append(lines[3])

        return retrieve_list
