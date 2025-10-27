import os
import pytest
import numpy as np
import spc

# Define the path to the test data directory
data_path = os.path.join(os.path.dirname(__file__), "test_data")

# Collect all .spc files in the test_data directory
spc_files = [f for f in os.listdir(data_path) if f.lower().endswith(".spc")]


@pytest.mark.parametrize("spc_file", spc_files)
def test_spc_file(spc_file):
    """
    Test that the data from the .spc file matches the corresponding .txt file.
    """
    spc_path = os.path.join(data_path, spc_file)
    txt_path = os.path.join(data_path, "txt", spc_file + ".txt")

    # Ensure the .txt file exists
    assert os.path.exists(txt_path), f"Reference file missing: {txt_path}"

    # Read the .txt file as a numpy matrix
    expected_data = np.loadtxt(txt_path, delimiter="\t")

    # remove wavelength column
    y_expected = expected_data[:, 1:]
    x_expected = expected_data[:, 0]

    # Load the .spc file
    spectrum = spc.File(spc_path)

    assert len(spectrum.sub) > 0, f"No sub-spectra found in file: {spc_file}"

    # check if all lengths are the same
    # this will prevent creating a nice matrix from the spectra
    # but failing the test because of this seems wrong - skip instead
    # this happens for the m_xyxy.spc test case
    length = [len(s.y) for s in spectrum.sub]
    if not all(l == length[0] for l in length):
        pytest.skip(f"Sub-spectra lengths are not consistent in file: {spc_file}")

    # Extract X and Y data from the spectrum object
    y_data = np.array([s.y for s in spectrum.sub]).T
    if spectrum.dat_fmt == "-xy":
        x_data = spectrum.sub[0].x
    else:
        x_data = spectrum.x

    # Compare the data
    assert y_data.shape == y_expected.shape, (
        f"Shape mismatch {y_data.shape} != {expected_data.shape} in file: {spc_file}"
    )
    assert pytest.approx(y_data, rel=1e-10) == y_expected, (
        f"Data mismatch in file: {spc_file}"
    )
    assert pytest.approx(x_data, rel=1e-10) == x_expected, (
        f"Data mismatch in file: {spc_file}"
    )
