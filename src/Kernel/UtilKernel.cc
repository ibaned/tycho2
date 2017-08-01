/*
Copyright (c) 2016, Los Alamos National Security, LLC
All rights reserved.

Copyright 2016. Los Alamos National Security, LLC. This software was produced 
under U.S. Government contract DE-AC52-06NA25396 for Los Alamos National 
Laboratory (LANL), which is operated by Los Alamos National Security, LLC for 
the U.S. Department of Energy. The U.S. Government has rights to use, 
reproduce, and distribute this software.  NEITHER THE GOVERNMENT NOR LOS 
ALAMOS NATIONAL SECURITY, LLC MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR 
ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE.  If software is modified 
to produce derivative works, such modified software should be clearly marked, 
so as not to confuse it with the version available from LANL.

Additionally, redistribution and use in source and binary forms, with or 
without modification, are permitted provided that the following conditions 
are met:
1.      Redistributions of source code must retain the above copyright notice, 
        this list of conditions and the following disclaimer.
2.      Redistributions in binary form must reproduce the above copyright 
        notice, this list of conditions and the following disclaimer in the 
        documentation and/or other materials provided with the distribution.
3.      Neither the name of Los Alamos National Security, LLC, Los Alamos 
        National Laboratory, LANL, the U.S. Government, nor the names of its 
        contributors may be used to endorse or promote products derived from 
        this software without specific prior written permission.
THIS SOFTWARE IS PROVIDED BY LOS ALAMOS NATIONAL SECURITY, LLC AND 
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT 
NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A 
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL LOS ALAMOS NATIONAL 
SECURITY, LLC OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; 
OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, 
WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR 
OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED 
OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#include "UtilKernel.hh"
#include "SweepDataKernel.hh"
#include "GraphTraverserKernel.hh"
#include <math.h>
#include <limits>
#include <Kokkos_Core.hpp>

namespace UtilKernel
{

/*
    psiToPhi
*/
void psiToPhi(PhiData &phi, const PsiData &psi) 
{
    phi.setToValue(0.0);
    
    Kokkos::parallel_for(g_nCells, KOKKOS_LAMBDA(UINT cell) {
    for (UINT angle = 0; angle < g_nAngles; ++angle) {
    for (UINT vertex = 0; vertex < g_nVrtxPerCell; ++vertex) {
    for (UINT group = 0; group < g_nGroups; ++group) {
        phi(group, vertex, cell) +=
            psi(group, vertex, angle, cell) * g_quadrature->getWt(angle);
    }}}});
}


/*
    calcTotalSource
*/
void calcTotalSource(const PsiData &source, const PhiData &phi, 
                     PsiData &totalSource)
{
    Kokkos::parallel_for(g_nCells, KOKKOS_LAMBDA(UINT cell) {
    for (UINT angle = 0; angle < g_nAngles; ++angle) {
    for (UINT vertex = 0; vertex < g_nVrtxPerCell; ++vertex) {
    for (UINT group = 0; group < g_nGroups; ++group) {
        totalSource(group, vertex, angle, cell) = 
            source(group, vertex, angle, cell) + 
            g_sigmaS[cell] / (4.0 * M_PI) *  phi(group, vertex, cell);
    }}}});
}


/*
    sweepLocal

    Solves L_I Psi = L_B Psi_B + Q
*/
void sweepLocal(PsiData &psi, const PsiData &source, PsiBoundData &psiBound)
{
    Mat2<UINT> priorities(g_nCells, g_nAngles);
    SweepDataKernel sweepData(psi, source, psiBound, priorities);
    
    g_graphTraverserKernel->traverse(sweepData);
}


} // End namespace Util
