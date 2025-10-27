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

    # Read the .txt file
    expected_data = np.loadtxt(txt_path, delimiter="\t")

    # remove wavelength column
    expected_data = expected_data[:, 1:]

    # Load the .spc file
    spectrum = spc.File(spc_path)
    spectrum_matrix = np.array([s.y for s in spectrum.sub]).T

    # Compare the data
    assert spectrum_matrix.shape == expected_data.shape, f"Shape mismatch {spectrum_matrix.shape} != {expected_data.shape} in file: {spc_file}"
    assert pytest.approx(spectrum_matrix, rel=1e-6) == expected_data, f"Data mismatch in file: {spc_file}"
