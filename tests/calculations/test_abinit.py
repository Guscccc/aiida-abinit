# -*- coding: utf-8 -*-
"""Tests for the calculation classes."""
import io
import tempfile
from pathlib import Path

from aiida import orm
from aiida.common import datastructures, exceptions
import pytest


def test_abinit_default(fixture_sandbox, generate_calc_job, generate_inputs_abinit, file_regression):
    """Test a default `AbinitCalculation`."""
    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)
    psp8 = inputs['pseudos']['Si']

    cmdline_params = ['aiida.in', '--timelimit', '30:00']
    local_copy_list = [(psp8.uuid, psp8.filename, './pseudo/Si.psp8')]
    retrieve_list = ['aiida.out', 'outdata/out_GSR.nc']

    # Check the attributes of the returned `CalcInfo`
    assert isinstance(calc_info, datastructures.CalcInfo)
    assert isinstance(calc_info.codes_info[0], datastructures.CodeInfo)
    assert calc_info.codes_info[0].cmdline_params == cmdline_params
    assert sorted(calc_info.local_copy_list) == sorted(local_copy_list)
    assert all(ret in calc_info.retrieve_list for ret in retrieve_list)
    assert sorted(calc_info.remote_symlink_list) == sorted([])

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == sorted(['aiida.in', 'pseudo', 'indata', 'outdata', 'tmpdata'])
    file_regression.check(input_written, encoding='utf-8', extension='.in')


# yapf: disable
@pytest.mark.parametrize(
    'ionmov,dry_run,retrieve_list',
    [(0, False, ['aiida.out', 'outdata/out_GSR.nc']),
     (2, False, ['aiida.out', 'outdata/out_GSR.nc', 'outdata/out_HIST.nc']),
     (0, True, ['aiida.out']),
     (2, True, ['aiida.out'])]
)
# yapf: enable
def test_abinit_retrieve(
    fixture_sandbox, generate_calc_job, generate_inputs_abinit, file_regression, ionmov, dry_run, retrieve_list
):  # pylint: disable=too-many-arguments
    """Test an various retrieve list situations for `AbinitCalculation`."""
    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    inputs['parameters']['ionmov'] = ionmov
    inputs['settings'] = orm.Dict(dict={'DRY_RUN': dry_run})
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert all(ret in calc_info.retrieve_list for ret in retrieve_list)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    # Checks on the files written to the sandbox folder as raw input
    assert sorted(fixture_sandbox.get_content_list()) == sorted(['aiida.in', 'pseudo', 'indata', 'outdata', 'tmpdata'])
    file_regression.check(input_written, encoding='utf-8', extension='.in')


# yapf: disable
@pytest.mark.parametrize(
    'settings,cmdline_params',
    [({'DrY_rUn': True, 'verbose': False}, ['aiida.in', '--timelimit', '30:00', '--dry-run']),
     ({'dry_run': True, 'verbose': True}, ['aiida.in', '--timelimit', '30:00', '--verbose', '--dry-run']),
     ({'DRY_RUN': False, 'verbose': True}, ['aiida.in', '--timelimit', '30:00', '--verbose'])]
)
# yapf: enable
def test_abinit_cmdline_params(fixture_sandbox, generate_calc_job, generate_inputs_abinit, settings, cmdline_params):
    """Test various command line parameters for `AbinitCalculation`."""
    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    inputs['settings'] = orm.Dict(dict=settings)
    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert calc_info.codes_info[0].cmdline_params == cmdline_params


def test_abinit_matching_explicit_shiftk_is_accepted(fixture_sandbox, generate_calc_job, generate_inputs_abinit):
    """A single explicit `shiftk` equal to the mesh offset is allowed."""
    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    parameters = inputs['parameters'].get_dict()
    parameters['shiftk'] = [0.0, 0.0, 0.0]
    inputs['parameters'] = orm.Dict(dict=parameters)

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    assert 'ngkpt 2 2 2' in input_written
    assert 'nshiftk 1' in input_written
    assert 'shiftk    0.0    0.0    0.0' in input_written



def test_abinit_mismatched_explicit_shiftk_is_rejected(fixture_sandbox, generate_calc_job, generate_inputs_abinit):
    """A single explicit `shiftk` differing from the mesh offset must fail."""
    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    parameters = inputs['parameters'].get_dict()
    parameters['shiftk'] = [0.5, 0.0, 0.0]
    inputs['parameters'] = orm.Dict(dict=parameters)

    with pytest.raises(exceptions.InputValidationError, match='does not match the offset stored in `kpoints`'):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)



def test_abinit_multishift_explicit_shiftk_is_accepted_for_gamma_centered_mesh(
    fixture_sandbox,
    generate_calc_job,
    generate_inputs_abinit,
):
    """Multiple explicit shifts are allowed when the mesh offset stored in `kpoints` is Gamma."""
    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    parameters = inputs['parameters'].get_dict()
    parameters['shiftk'] = [
        0.5, 0.5, 0.5,
        0.5, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 0.5,
    ]
    inputs['parameters'] = orm.Dict(dict=parameters)

    generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    with fixture_sandbox.open('aiida.in') as handle:
        input_written = handle.read()

    assert 'ngkpt 2 2 2' in input_written
    assert 'nshiftk 4' in input_written
    assert '0.5    0.5    0.5' in input_written
    assert '0.5    0.0    0.0' in input_written
    assert '0.0    0.5    0.0' in input_written
    assert '0.0    0.0    0.5' in input_written



def test_abinit_multishift_explicit_shiftk_with_shifted_kpoints_offset_is_rejected(
    fixture_sandbox,
    generate_calc_job,
    generate_inputs_abinit,
):
    """Multiple explicit shifts are rejected when `kpoints` already encodes a nonzero mesh offset."""
    from aiida.orm import KpointsData

    entry_point_name = 'abinit'

    inputs = generate_inputs_abinit()
    kpoints = KpointsData()
    kpoints.set_kpoints_mesh([2, 2, 2], offset=[0.5, 0.5, 0.5])
    inputs['kpoints'] = kpoints

    parameters = inputs['parameters'].get_dict()
    parameters['shiftk'] = [
        0.5, 0.5, 0.5,
        0.5, 0.0, 0.0,
        0.0, 0.5, 0.0,
        0.0, 0.0, 0.5,
    ]
    inputs['parameters'] = orm.Dict(dict=parameters)

    with pytest.raises(exceptions.InputValidationError, match='set the `kpoints` mesh offset to Gamma'):
        generate_calc_job(fixture_sandbox, entry_point_name, inputs)


@pytest.mark.parametrize(
    'entry_point_name,stdin_content,files_to_copy,files_namespace,expected_retrieve',
    [
        (
            'abinit.mrgddb',
            b'tnlo_3.ddb.out\nlinear + nonlinear response calculation\n2\ntnlo_2o_DS4_DDB\ntnlo_2o_DS5_DDB\n',
            [('ddb_ds4', 'tnlo_2o_DS4_DDB'), ('ddb_ds5', 'tnlo_2o_DS5_DDB')],
            {'ddb_ds4': b'ds4\n', 'ddb_ds5': b'ds5\n'},
            ['aiida.out', 'tnlo_3.ddb.out'],
        ),
        (
            'abinit.anaddb',
            b'tnlo_4.abi\ntnlo_4.abo\ntnlo_3.ddb.out\ntnlo_4_thm_dummy\ntnlo_4_gkk_dummy\ntnlo_4_ep_dummy\ntnlo_4_ddk_dummy\n',
            [('anaddb_input', 'tnlo_4.abi'), ('ddb_merged', 'tnlo_3.ddb.out')],
            {'anaddb_input': b'nlflag 1\n', 'ddb_merged': b'ddb\n'},
            [
                'aiida.out',
                'tnlo_4.abo',
                'tnlo_4_anaddb.nc',
                'tnlo_4_PHBST.nc',
                'tnlo_4_PHBANDS.agr',
                'tnlo_4_PHFRQ',
                'tnlo_4_PHANGMOM',
                'tnlo_4_PHDOS.nc',
                'tnlo_4_PHDOS',
                'tnlo_4_PHDOS_by_atom',
                'tnlo_4_PHDOS_msqd',
                'tnlo_4_MSQD_T',
                'tnlo_4_MSQV_T',
                'tnlo_4_THERMO',
                'PHBST_partial_DOS',
            ],
        ),
    ]
)
def test_abinit_utility_retrieve(
    fixture_sandbox,
    generate_calc_job,
    fixture_code,
    entry_point_name,
    stdin_content,
    files_to_copy,
    files_namespace,
    expected_retrieve,
):
    """Test automatic retrieve list inference and staged files for utility CalcJobs."""
    from aiida.orm import Dict, SinglefileData

    stdin_file = SinglefileData(io.BytesIO(stdin_content), filename='stdin.in')
    files = {
        label: SinglefileData(io.BytesIO(content), filename=f'{label}.dat')
        for label, content in files_namespace.items()
    }

    inputs = {
        'code': fixture_code(entry_point_name),
        'stdin_file': stdin_file,
        'settings': Dict(dict={
            'FILES_TO_COPY': files_to_copy,
        }),
        'files': files,
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert calc_info.codes_info[0].stdin_name == 'aiida.in'
    assert calc_info.codes_info[0].stdout_name == 'aiida.out'
    for retrieved in expected_retrieve:
        assert retrieved in calc_info.retrieve_list
    assert any(item[2] == 'aiida.in' for item in calc_info.local_copy_list)
    for _, destination in files_to_copy:
        assert any(item[2] == destination for item in calc_info.local_copy_list)



def test_anaddb_nc_serializer_reads_template_outputs():
    """Test that template anaddb NetCDF outputs are fully serialized into JSON-safe dictionaries."""
    import netCDF4 as nc
    import numpy as np

    from aiida_abinit.parsers import _serialize_nc_file

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        anaddb_path = tmpdir / 'tnlo_4_anaddb.nc'
        phbst_path = tmpdir / 'tnlo_4_PHBST.nc'

        with nc.Dataset(anaddb_path, 'w') as ds:
            ds.setncattr('abinit_version', '10.6.3')
            ds.createDimension('number_of_cartesian_directions', 3)
            ds.createDimension('number_of_atoms', 2)
            ds.createDimension('number_of_phonon_modes', 6)
            ds.createDimension('six', 6)
            gamma = ds.createVariable('gamma_phonon_modes', 'f8', ('number_of_phonon_modes',))
            gamma[:] = np.arange(6, dtype=float)
            becs = ds.createVariable('becs_cart', 'f8', ('number_of_atoms', 'number_of_cartesian_directions', 'number_of_cartesian_directions'))
            becs[:] = np.arange(18, dtype=float).reshape(2, 3, 3)
            raman = ds.createVariable('raman_sus', 'f8', ('number_of_cartesian_directions', 'number_of_cartesian_directions', 'number_of_phonon_modes'))
            raman[:] = np.arange(54, dtype=float).reshape(3, 3, 6)
            d_tensor = ds.createVariable('d_tensor_relaxed_ion', 'f8', ('number_of_cartesian_directions', 'six'))
            d_tensor[:] = np.arange(18, dtype=float).reshape(3, 6)

        with nc.Dataset(phbst_path, 'w') as ds:
            ds.setncattr('abinit_version', '10.6.3')
            ds.createDimension('number_of_qpoints', 1)
            ds.createDimension('number_of_phonon_modes', 6)
            ds.createDimension('three', 3)
            phfreqs = ds.createVariable('phfreqs', 'f8', ('number_of_qpoints', 'number_of_phonon_modes'))
            phfreqs[:] = np.arange(6, dtype=float).reshape(1, 6)
            phangmom = ds.createVariable('phangmom', 'f8', ('number_of_qpoints', 'number_of_phonon_modes', 'three'))
            phangmom[:] = np.arange(18, dtype=float).reshape(1, 6, 3)

        anaddb_nc = _serialize_nc_file(anaddb_path)
        phbst_nc = _serialize_nc_file(phbst_path)

        assert 'global_attributes' in anaddb_nc
        assert 'dimensions' in anaddb_nc
        assert 'variables' in anaddb_nc
        assert 'gamma_phonon_modes' in anaddb_nc['variables']
        assert anaddb_nc['variables']['gamma_phonon_modes']['shape'] == [6]
        assert 'value' in anaddb_nc['variables']['gamma_phonon_modes']
        assert 'becs_cart' in anaddb_nc['variables']
        assert 'raman_sus' in anaddb_nc['variables']
        assert 'd_tensor_relaxed_ion' in anaddb_nc['variables']

        assert 'global_attributes' in phbst_nc
        assert 'dimensions' in phbst_nc
        assert 'variables' in phbst_nc
        assert 'phfreqs' in phbst_nc['variables']
        assert phbst_nc['variables']['phfreqs']['shape'] == [1, 6]
        assert 'value' in phbst_nc['variables']['phfreqs']
        assert 'phangmom' in phbst_nc['variables']
