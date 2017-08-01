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


#ifndef __PSI_DATA_HH__
#define __PSI_DATA_HH__

#include "Assert.hh"
#include "Global.hh"
#include "Quadrature.hh"
#include "TychoMesh.hh"
#include <string>

#include <Kokkos_Core.hpp>

/*
    PsiData

    g = group
    v = vertex
    a = angle
    c = cell
*/
class PsiData {
public:
    using view_type = Kokkos::View<double****, Kokkos::LayoutLeft>;

    // Accessors
    KOKKOS_INLINE_FUNCTION
    /* note that this is const now! */
    double& operator()(size_t g, size_t v, size_t a, size_t c) const
    {
        return c_data(g, v, a, c);
    }

    KOKKOS_INLINE_FUNCTION
    double& operator[](size_t i) { return *(c_data.data() + i); }
    KOKKOS_INLINE_FUNCTION
    double const& operator[](size_t i) const { return *(c_data.data() + i); }
    size_t size() const { return c_data.size(); }

    // Constructor
    PsiData() : c_data("PsiData", g_nGroups, g_nVrtxPerCell, g_nAngles, g_nCells)
    {
    }

    PsiData(double *data) : c_data(data, g_nGroups, g_nVrtxPerCell, g_nAngles, g_nCells)
    {
    }

    // Set constant value
    void setToValue(double value)
    {
        Kokkos::deep_copy(c_data, value);
    }

    // Write to file
    void writeToFile(const std::string &filename);

// Private    
private:
    view_type c_data;
};


/*
    PsiBoundData

    g = group
    v = vertex
    a = angle
    s = side
*/
class PsiBoundData {
public:
    using view_type = Kokkos::View<double****, Kokkos::LayoutLeft>;
    
    // Accessors
    KOKKOS_INLINE_FUNCTION
    double& operator()(size_t g, size_t v, size_t a, size_t s) const
    {
        return c_data(g,v,a,s);
    }

    // Constructor
    PsiBoundData() : c_data("PsiBoundData", g_nGroups, g_nVrtxPerFace, g_nAngles,
        g_tychoMesh->getNSides())
    {
    }

    // Set constant value
    void setToValue(double value)
    {
        Kokkos::deep_copy(c_data, value);
    }

// Private    
private:
    view_type c_data;
};


/*
    PhiData

    g = group
    v = vertex
    c = cell
*/
class PhiData {
public:
    using view_type = Kokkos::View<double***, Kokkos::LayoutLeft>;

    // Accessors
    KOKKOS_INLINE_FUNCTION
    double& operator()(size_t g, size_t v, size_t c) const
    {
        return c_data(g, v, c);
    }

    KOKKOS_INLINE_FUNCTION
    double& operator[](size_t i) { return *(c_data.data() + i); }
    KOKKOS_INLINE_FUNCTION
    double const& operator[](size_t i) const { return *(c_data.data() + i); }
    size_t size() const { return c_data.size(); }

    // Constructor
    PhiData() : c_data("PhiData", g_nGroups, g_nVrtxPerCell, g_nCells)
    {
    }

    PhiData(double *data) : c_data(data, g_nGroups, g_nVrtxPerCell, g_nCells)
    {
    }

    PhiData& operator=(PhiData const& other) {
      Kokkos::deep_copy(c_data, other.c_data);
    }

    // Set to a constant value
    void setToValue(double value)
    {
      Kokkos::deep_copy(c_data, value);
    }

// Private    
private:
    view_type c_data;
};



#endif
