# -*- coding: utf-8 -*-
"""Tests for the calculation classes."""
import io

from aiida import orm
from aiida.common import datastructures
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


@pytest.mark.parametrize(
    'entry_point_name,stdout_name,extra_retrieve',
    [
        ('abinit.mrgddb', 'aiida.out', ['tnlo_3.ddb.out']),
        ('abinit.anaddb', 'aiida.out', ['tnlo_4.abo']),
    ]
)
def test_abinit_utility_retrieve(fixture_sandbox, generate_calc_job, fixture_code, entry_point_name, stdout_name, extra_retrieve):
    """Test retrieve list and staged files for small ABINIT utility CalcJobs."""
    from aiida.orm import Dict, SinglefileData

    stdin_file = SinglefileData(io.BytesIO(b'input from stdin\n'), filename='stdin.in')
    staged_file = SinglefileData(io.BytesIO(b'staged payload\n'), filename='payload.dat')

    inputs = {
        'code': fixture_code(entry_point_name),
        'stdin_file': stdin_file,
        'settings': Dict(dict={
            'FILES_TO_COPY': [('ddb_input', 'tnlo_2o_DS4_DDB')],
            'ADDITIONAL_RETRIEVE_LIST': extra_retrieve,
        }),
        'files': {
            'ddb_input': staged_file,
        },
    }

    calc_info = generate_calc_job(fixture_sandbox, entry_point_name, inputs)

    assert isinstance(calc_info, datastructures.CalcInfo)
    assert calc_info.codes_info[0].stdin_name == 'aiida.in'
    assert calc_info.codes_info[0].stdout_name == stdout_name
    assert stdout_name in calc_info.retrieve_list
    for retrieved in extra_retrieve:
        assert retrieved in calc_info.retrieve_list
    assert any(item[2] == 'aiida.in' for item in calc_info.local_copy_list)
    assert any(item[2] == 'tnlo_2o_DS4_DDB' for item in calc_info.local_copy_list)
