import numpy as np
import pickle
import math


def General_zernike_matrix(maxTerm, R, a_i, grid_shape=500):  # highest order term, disc radius of measurement, inner radius in microns of measurement

    maxTerm += 1  # (!) Added this because piston is Z=0

    xs = ys = np.linspace(-R, R, grid_shape)  # disc grid in microns
    X, Y = np.meshgrid(xs, -ys)

    Rs = np.sqrt(X ** 2 + Y ** 2)  # convert to polar coords for Zernike polynomial calculation
    Ts = np.arctan2(Y, X)  #

    js = np.arange(0, maxTerm + 1)  # list of Zernike modes

    ns = [math.ceil((-3 + np.sqrt(9 + 8 * i)) / 2) for i in js]  # incides
    ms = [2 * js[i] - ns[i] * (ns[i] + 2) for i, n in enumerate(js)]  #
    ks = (ns - np.abs(ms)) / 2  #

    z_vals = []
    z_mesh = []

    for i, n in enumerate(js):  # calculate all Zernike polynomials and add to list
        sum_set = np.arange(0, ks[i] + 1)
        R_nm_terms = [(((-1) ** s * math.factorial(ns[i] - s)) / (
                    math.factorial(s) * math.factorial(int(0.5 * (ns[i] + np.abs(ms[i])) - s)) * math.factorial(
                int(0.5 * (ns[i] - np.abs(ms[i])) - s)))) * (Rs / R) ** (ns[i] - 2 * s) for s in sum_set.astype(int)]
        R_nm = sum(R_nm_terms)

        # N_nm = 0
        if ms[i] == 0:
            N_nm = np.sqrt((2 * (ns[i] + 1) / (1 + 1)))
        else:
            N_nm = np.sqrt((2 * (ns[i] + 1) / (1 + 0)))

        if ms[i] >= 0:
            Z_nm = N_nm * R_nm * np.cos(ms[i] * Ts)
        else:
            Z_nm = -N_nm * R_nm * np.sin(ms[i] * Ts)

        test = np.sqrt(X ** 2 + Y ** 2)  #
        inds = np.where((test > R) | (test < a_i))  #
        coords = list(zip(inds[0], inds[1]))  # remove points outside of OD and inside ID
        for j, m in enumerate(coords):  #
            Z_nm[coords[j][0]][coords[j][1]] = np.nan  #

        z_vals.append(Z_nm)  # add to 3D matrix
        z_mesh.append(Z_nm.flatten())  # add to flattened 2D matrix

    Z_3D = np.array(z_vals)  # convert list into array
    Z_matrix = np.array(z_mesh)  # convert list into array

    print('Zernike matrix for image of size ' + str(grid_shape) + ' created for ' + str(maxTerm) + ' terms')

    return Z_matrix.transpose(), Z_3D.transpose(1, 2,
                                                0)  # return 2D array and 3D in tuple form.  Use Z[0] to call the 2D array and Z[1] to call the 3D array. Transpose functions are used to put the matrix in the correct form for matrix arithmetic.


def get_M_and_C(avg_ref, Z):
    #Compute M and C surface height variables that are used for Zernike analysis
    #M is a flattened surface map; C is a list of Zernike coefficients
    M = avg_ref.flatten(), avg_ref
    C = Zernike_decomposition(Z, M, -1)  #Zernike fit
    return M, C


def Zernike_decomposition(Z, M, n):
    Z_processed = Z[0].copy()[:, 0:n]
    Z_processed[np.isnan(Z_processed)] = 0  # replaces NaN's with 0's

    if type(M) == tuple:
        M_processed = M[0].copy()
    else:
        M_processed = M.copy().ravel()
    M_processed[np.isnan(M_processed)] = 0  # replaces NaN's with 0's

    Z_t = Z_processed.transpose()  #

    A = np.dot(Z_t, Z_processed)  #

    A_inv = np.linalg.inv(A)  #

    B = np.dot(A_inv, Z_t)  #

    Zernike_coefficients = np.dot(B, M_processed)  # Solves matrix equation:  Zerninke coefficients = ((Z_t*Z)^-1)*Z_t*M

    Surf = np.dot(Z[1][:, :, 0:n], Zernike_coefficients)  # Calculates best fit surface to Zernike modes

    return Surf.flatten(), Surf, Zernike_coefficients  # returns the vector containing Zernike coefficients and the generated surface in tuple form


def remove_modes(M, C, Z, remove_coef):
    # Remove Zernike modes from input surface map
    removal = M[1] * 0
    for coef in remove_coef:
        term = (Z[1].transpose(2, 0, 1)[coef]) * C[2][coef]
        removal += term

    Surf = M[1] - removal
    return Surf
