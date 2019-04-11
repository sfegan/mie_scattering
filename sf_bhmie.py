#!env python3

# sf_bhmie.py -- Stephen Fegan -- 2019-04-11
#
# Mie scattering function that accepts numpy arrays
#
# Copyright 2019, Stephen Fegan <sfegan@llr.in2p3.fr>
# LLR, Ecole Polytechnique, CNRS/IN2P3
#
# "sf_bhmie.py" is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License version 2 or later, as
# published by the Free Software Foundation.
#
# "sf_bhmie.py" is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.

import numpy

# This function is adapted from bhmie.py, see http://atol.ucsd.edu/scatlib/index.htm

# Which was in turn adapted from the FORTRAN code that Bohren and Huffman published
# in their celebrated book on light scattering [BH1983]

# Absorption and Scattering of Light by Small Particles
# Craig F. Bohren Donald R. Huffman
# Wiley and Sons - 1983
# DOI:10.1002/9783527618156

# Calculation based on Mie scattering theory
# input:
#      x      - size parameter = k*radius = 2 pi / lambda * radius
#                   (lambda is the wavelength in the medium around the scatterers)
#               Can be a scalar or numpy array (dtype=float)
#      m      - refraction index relative to medium around scatterers
#                   (n in complex form for example:  1.5+0.02*i)
#               Can be a complex scalar or numpy array with dtype=complex
# output:
#      Qext   - extinction efficiency
#      Qsca   - scattering efficiency
#      Qbks   - backscatter efficiency
def sf_bhmie(x, m):
    xwasscalar = False
    if(numpy.isscalar(x)):
        x = numpy.asarray([x], float)
        xwasscalar = True

    if(numpy.isscalar(m)):
        m = numpy.zeros_like(x, dtype=complex) + m

    mx = m*x

    # In the forward loop we use the same number of iterations as
    # in Appendix A of BH1983. Note each value of x will have its own
    # stopping point (to avoid instability in calculation of psi).

    nterm = numpy.array(x + 4.0*x**(1.0/3.0) + 2.0, dtype=int)

    # For D however the backward iteration is stable and so we use the
    # same number of iterations for all x-values, given by the maximum
    # of nstop and |mx| with an extra 15 iterations for good luck (see BH1983)

    nD = max(nterm)
    D = numpy.zeros((len(x), nD), dtype=complex)
    # Initialise D iteration with zeros as in Appendix A of BH1983
    Di = numpy.zeros(len(mx), dtype=complex)
    n = max(nD, int(max(numpy.abs(mx)))) + 15
    while(n>0):
        # Equation 4.89 of BH1983
        Di = float(n+1)/mx - 1.0/(Di+float(n+1)/mx)
        n -= 1
        if(n<nD):
            D[:,n] = Di

    # Initialise psi & chi iterations as in Appendix A of BH1983
    psi0 = numpy.cos(x)
    psi1 = numpy.sin(x)
    chi0 = -numpy.sin(x)
    chi1 = numpy.cos(x)
    xi1  = psi1 - chi1*complex(0,1)

    Qext = numpy.zeros_like(x)
    Qsca = numpy.zeros_like(x)
    Qbks = numpy.zeros_like(x, dtype=complex)

    for iterm in range(0,max(nterm)):
        n = iterm+1

        # Only update for elements that have not reached number of required terms
        M = iterm<nterm

        psi    = numpy.zeros_like(psi1)
        chi    = numpy.zeros_like(chi1)
        xi     = numpy.zeros_like(xi1, dtype=complex)

        # Eq 4.11 of BH1983 for psi and chi (and see pg 101)
        psi[M] = (2.0*n-1.0)*psi1[M]/x[M] - psi0[M]
        chi[M] = (2.0*n-1.0)*chi1[M]/x[M] - chi0[M]

        # Eq 4.14 of BH1983 (and see pg 101)
        xi[M]  = psi[M] - chi[M]*complex(0,1)

        # Eq 4.88 of of BH1983
        a = ((D[M,n-1]/m[M] + n/x[M])*psi[M] - psi1[M])/((D[M,n-1]/m[M] + n/x[M])*xi[M] - xi1[M])
        b = ((D[M,n-1]*m[M] + n/x[M])*psi[M] - psi1[M])/((D[M,n-1]*m[M] + n/x[M])*xi[M] - xi1[M])

#         print(n, a, b)

        psi0 = psi1
        chi0 = chi1
        psi1 = psi
        chi1 = chi
        xi1  = xi

        # Eqs 4.61, 4.62, and equation for Qb on Pg 122 of BH1983
        Qsca[M] += (2.0*n+1)*numpy.real(abs(a)**2+abs(b)**2)
        Qext[M] += (2.0*n+1)*numpy.real(a+b)
        Qbks[M] += (2.0*n+1)*(-1.0)**n*(a-b)

    Qext *= 2.0/x**2
    Qsca *= 2.0/x**2
    Qbks = abs(Qbks)**2/x**2

    if xwasscalar:
        Qext = Qext[0]
        Qsca = Qsca[0]
        Qbks = Qbks[0]

    return Qext, Qsca, Qbks

if __name__ == "__main__":
    # Reproduce the values for the "worked example" in Appendix A of B&H 1983

    m = complex(1.55, 0.0)
    wl = 0.6328
    a = 0.525
    x = 2*numpy.pi*a/wl

    print("Refractive index   :",m)
    print("Wavelength         :",wl)
    print("Particle radius    :",a)
    print("Size parameter:    :",x)
    print("")

    Qext, Qsca, Qbks = sf_bhmie(x,m)

    print("Extinction factor  :", Qext)
    print("Scattering factor  :", Qsca)
    print("Backscatter factor :", Qbks)
