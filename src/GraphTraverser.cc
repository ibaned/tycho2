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

#include "GraphTraverser.hh"
#include "Mat.hh"
#include "Global.hh"
#include "TychoMesh.hh"
#include "Comm.hh"
#include "Timer.hh"
#include "Transport.hh"
#include <vector>
#include <queue>
#include <omp.h>
#include <Kokkos_Core.hpp>

using namespace std;


/*
    Tuple class
*/
namespace {
struct CellAnglePair
{
    UINT cell;
    UINT angle;
};
}


/*
    angleGroupIndex
    
    Gets angle groups for angle index.
    e.g. 20 angles numbered 0...19 with 3 threads.
    Split into 3 angle chunks of size 7,7,6:  0...6  7...13  14...19
    If angle in 0...6,   return 0
    If angle in 7...13,  return 1
    If angle in 14...19, return 2
*/
UINT angleGroupIndex(UINT angle)
{
    UINT numAngles = g_nAngles;
    UINT chunkSize = numAngles / g_nThreads;
    UINT numChunksBigger = numAngles % g_nThreads;
    UINT lowIndex = 0;
    
    
    // Find angleGroup
    for (UINT angleGroup = 0; angleGroup < g_nThreads; angleGroup++) {
        
        UINT nextLowIndex = lowIndex + chunkSize;
        if (angleGroup < numChunksBigger)
            nextLowIndex++;
        
        if (angle < nextLowIndex) 
            return angleGroup;
        
        lowIndex = nextLowIndex;
    }
    
    
    // Should never get here
    Assert(false);
    return 0;
}


/*
    GraphTraverser
*/
GraphTraverser::GraphTraverser(
        PsiData &psi, 
        const PsiData &source, 
        PsiBoundData &psiBound)
: c_initNumDependencies(g_nAngles, g_nCells),
  c_psi(psi), 
  c_psiBound(psiBound), 
  c_source(source)
{
    // Calc num dependencies for each (cell, angle) pair
    for (UINT cell = 0; cell < g_nCells; cell++) {
    for (UINT angle = 0; angle < g_nAngles; angle++) {
        
        c_initNumDependencies(angle, cell) = 0;
        for (UINT face = 0; face < g_nFacePerCell; face++) {
            
            bool incoming = g_tychoMesh->isIncoming(angle, cell, face);
            UINT adjCell = g_tychoMesh->getAdjCell(cell, face);
            
            if (incoming && adjCell != TychoMesh::BOUNDARY_FACE) {
                c_initNumDependencies(angle, cell)++;
            }
        }
    }}
}

using host = Kokkos::DefaultHostExecutionSpace;
using device = Kokkos::DefaultExecutionSpace;
using host_psi_data_t =
  Kokkos::View<double****, host, Kokkos::LayoutLeft>;
using device_psi_data_t =
  Kokkos::View<double****, device, Kokkos::LayoutLeft>;
template <class T>
using host_mat1_t =
  Kokkos::View<T*, host, Kokkos::LayoutLeft>;
template <class T>
using device_mat1_t =
  Kokkos::View<T*, device, Kokkos::LayoutLeft>;
template <class T>
using host_mat2_t =
  Kokkos::View<T**, host, Kokkos::LayoutLeft>;
template <class T>
using device_mat2_t =
  Kokkos::View<T**, device, Kokkos::LayoutLeft>;
template <class T>
using host_mat3_t =
  Kokkos::View<T***, host, Kokkos::LayoutLeft>;
template <class T>
using device_mat3_t =
  Kokkos::View<T***, device, Kokkos::LayoutLeft>;

static 
KOKKOS_INLINE_FUNCTION
void populateLocalPsiBoundKokkos(
    const UINT angle, 
    const UINT cell, 
    device_psi_data_t const& psi, 
    device_psi_data_t const& psiBound,
    double localPsiBound[g_nVrtxPerFace][g_nFacePerCell][g_nMaxGroups],
    device_mat3_t<double> const& omega_dot_n,
    device_mat2_t<UINT> const& adj_cell,
    device_mat3_t<UINT> const& neighbor_vertex,
    device_mat2_t<UINT> const& adj_proc,
    device_mat2_t<UINT> const& sides
    )
{
  for (UINT i = 0; i < g_nVrtxPerFace; i++) {
    for (UINT j = 0; j < g_nFacePerCell; j++) {
      for (UINT k = 0; k < g_nGroups; k++) {
        localPsiBound[i][j][k] = 0.0;
      }
    }
  }

  // Populate if incoming flux
  for (UINT group = 0; group < g_nGroups; group++) {
    for (UINT face = 0; face < g_nFacePerCell; face++) {
      if (omega_dot_n(angle, cell, face) <= 0) {
        UINT neighborCell = adj_cell(cell, face);

        // In local mesh
        if (neighborCell != TychoMesh::BOUNDARY_FACE) {
          for (UINT fvrtx = 0; fvrtx < g_nVrtxPerFace; fvrtx++) {
            UINT neighborVrtx = neighbor_vertex(cell, face, fvrtx);
            localPsiBound[fvrtx][face][group] = 
              psi(group, neighborVrtx, angle, neighborCell);
          }
        }

        // Not in local mesh
        else if (adj_proc(cell, face) != TychoMesh::BAD_RANK) {
          for (UINT fvrtx = 0; fvrtx < g_nVrtxPerFace; fvrtx++) {
            UINT side = sides(cell, face);
            localPsiBound[fvrtx][face][group] = 
              psiBound(group, fvrtx, angle, sides);
          }
        }
      }
    }
  }
}


/*
    traverse
    
    Traverses g_tychoMesh.
*/
void GraphTraverser::traverse()
{
    Timer totalTimer;
    Timer setupTimer;
    

    // Start total timer
    totalTimer.start();
    

    // Kokkos Version
    if ((true)) {
        
        // Start setup timer
        setupTimer.start();


        // Get dependencies
        auto nitems = int(g_nCells * g_nAngles);
        Kokkos::View<int*, host> counts("counts", nitems);
        Kokkos::parallel_for(Kokkos::RangePolicy<int, host>(0,nitems), KOKKOS_LAMBDA(int item) {
            int cell = item % g_nCells;
            int angle = item / g_nCells;
            for (UINT face = 0; face < g_nFacePerCell; ++face) {
                bool is_out = g_tychoMesh->isOutgoing(angle, cell, face);
                if (!is_out) continue;
                auto adjCell = g_tychoMesh->getAdjCell(cell, face);
                if (adjCell == TychoMesh::BOUNDARY_FACE) continue;
                counts(item) += 1;
            }
        }, "count-dependencies");

        
        // Get the row_map
        Kokkos::View<int*, host> row_map;
        Kokkos::get_crs_row_map_from_counts(row_map, counts);
        auto nedges = row_map(row_map.size() - 1);
        Kokkos::View<int*, host> entries("entries", nedges);
        Kokkos::parallel_for(Kokkos::RangePolicy<int, host>(0,nitems), KOKKOS_LAMBDA(int item) {
            int cell = item % g_nCells;
            int angle = item / g_nCells;
            int j = 0;
            for (UINT face = 0; face < g_nFacePerCell; ++face) {
                bool is_out = g_tychoMesh->isOutgoing(angle, cell, face);
                if (!is_out) continue;
                auto adjCell = g_tychoMesh->getAdjCell(cell, face);
                if (adjCell == TychoMesh::BOUNDARY_FACE) continue;
                entries(row_map(item) + j) = adjCell + g_nCells * angle;
                ++j;
            }
            Assert(j + row_map(item) == row_map(item + 1));
        }, "fill-dependencies");


        // Get the policy
        auto device_row_map = Kokkos::create_mirror_view_and_copy(device(), row_map);
        auto device_entries = Kokkos::create_mirror_view_and_copy(device(), entries);
        auto graph = Kokkos::Crs<int,device,void,int>(device_row_map, device_entries);
        auto policy = Kokkos::WorkGraphPolicy<device,int>(graph);

        
        // End setup timer
        setupTimer.stop();


        //copy data to device views
        using host_psi_data_t =
          Kokkos::View<double****, host, Kokkos::LayoutLeft, Kokkos::MemoryUnmanaged>;
        auto host_source =
          host_psi_data_t(c_source.data(),
              g_nGroups,
              g_nVrtxPerCell,
              g_nAngles,
              g_nCells);
        auto host_psi =
          host_psi_data_t(c_psi.data(),
              g_nGroups,
              g_nVrtxPerCell,
              g_nAngles,
              g_nCells);
        auto host_psi_bound =
          host_psi_data_t(c_psiBound.data(),
              g_nGroups,
              g_nVrtxPerFace,
              g_nAngles,
              g_tychoMesh->getNSides());
        auto host_omega_dot_n =
          host_mat3_t<double>(
              g_tychoMesh->c_omegaDotN.data(),
              g_nAngles,
              g_nCells,
              g_nFacePerCell);
        auto host_adj_cell =
          host_mat2_t<UINT>(
              g_tychoMesh->c_adjCell.data(),
              g_nCells,
              g_nFacePerCell);
        auto host_neighbor_vertex =
          host_mat3_t<UINT>(
              g_tychoMesh->c_neighborVrtx.data(),
              g_nCells,
              g_nFacePerCell,
              g_nVrtxPerFace);
        auto host_adj_proc =
          host_mat2_t<UINT>(
              g_tychoMesh->c_adjProc.data(),
              g_nCells,
              g_nFacePerCell);
        auto host_side =
          host_mat2_t<UINT>(
              g_tychoMesh->c_side.data(),
              g_nCells,
              g_nFacePerCell);
        auto host_sigma_t =
          host_mat1_t<double>(
              g_sigmaT.data(),
              g_sigmaT.size());
        auto device_source =
          Kokkos::create_mirror_view_and_copy(
              device(), host_source);
        auto device_psi =
          Kokkos::create_mirror_view_and_copy(
              device(), host_psi);
        auto device_psi_bound =
          Kokkos::create_mirror_view_and_copy(
              device(), host_psi_bound);
        auto device_omega_dot_n =
          Kokkos::create_mirror_view_and_copy(
              device(), host_omega_dot_n);
        auto device_adj_cell =
          Kokkos::create_mirror_view_and_copy(
              device(), host_adj_cell);
        auto device_neighbor_vertex =
          Kokkos::create_mirror_view_and_copy(
              device(), host_neighbor_vertex);
        auto device_adj_proc =
          Kokkos::create_mirror_view_and_copy(
              device(), host_adj_proc);
        auto device_side =
          Kokkos::create_mirror_view_and_copy(
              device(), host_side);
        auto device_sigma_t = 
          Kokkos::create_mirror_view_and_copy(
              device(), host_sigma_t);

        //actually do graph traversal
        auto nCells = g_nCells;
        auto lambda = KOKKOS_LAMBDA(int item) {
          int cell = item % nCells;
          int angle = item / nCells;

          // Update data for this cell-angle pair
        //Transport::update(
        //    cell, angle, device_source, device_psi_bound,
        //    device_psi);
          double localSource[g_nVrtxPerCell][g_nMaxGroups];
          double localPsi[g_nVrtxPerCell][g_nMaxGroups];
          double localPsiBound[g_nVrtxPerFace][g_nFacePerCell][g_nMaxGroups];
    
          // Populate localSource
          for (UINT group = 0; group < g_nGroups; group++) {
            for (UINT vrtx = 0; vrtx < g_nVrtxPerCell; vrtx++) {
              localSource[vrtx][group] =
                device_source(group, vrtx, angle, cell);
            }
          }

          // Populate localPsiBound
          populateLocalPsiBoundKokkos(
              angle, cell, psi, psiBound, 
              localPsiBound, device_omega_dot_n,
              device_adj_cell, device_neighbor_vertex,
              device_adj_proc, device_side);

          // Transport solve
          solve(cell, angle, device_sigma_t(cell),
              localPsiBound, localSource, localPsi);

          // localPsi -> psi
          for (UINT group = 0; group < g_nGroups; group++) {
            for (UINT vrtx = 0; vrtx < g_nVrtxPerCell; vrtx++) {
              device_psi(group, vrtx, angle, cell) = localPsi[vrtx][group];
            }
          }
        };
        Kokkos::parallel_for(policy, lambda, "traverse-dag");
    }

    // Non Kokkos version
    else {
        
        // Variables for non-kokkos version
        vector<queue<CellAnglePair>> canCompute(g_nThreads);
        Mat2<UINT> numDependencies(g_nAngles, g_nCells);

        // Start setup timer
        setupTimer.start();


        // Calc num dependencies for each (cell, angle) pair
        for (UINT cell = 0; cell < g_nCells; cell++) {
        for (UINT angle = 0; angle < g_nAngles; angle++) {
            numDependencies(angle, cell) = c_initNumDependencies(angle, cell);
        }}
        
        
        // Initialize canCompute queue
        for (UINT cell = 0; cell < g_nCells; cell++) {
        for (UINT angle = 0; angle < g_nAngles; angle++) {
            if (numDependencies(angle, cell) == 0) {
                UINT angleGroup = angleGroupIndex(angle);
                CellAnglePair cellAnglePair{cell, angle};
                canCompute[angleGroup].push(cellAnglePair);
            }
        }}


        // End setup timer
        setupTimer.stop();
        
        
        // Do local traversal
        #pragma omp parallel
        {
            UINT angleGroup = omp_get_thread_num();
            while (canCompute[angleGroup].size() > 0)
            {
                // Get cell/angle pair to compute
                CellAnglePair cellAnglePair = canCompute[angleGroup].front();
                canCompute[angleGroup].pop();
                UINT cell = cellAnglePair.cell;
                UINT angle = cellAnglePair.angle;
                
                
                // Update data for this cell-angle pair
                Transport::update(cell, angle, c_source, c_psiBound, c_psi);
                
                
                // Update dependency for children
                for (UINT face = 0; face < g_nFacePerCell; face++) {
                    
                    if (g_tychoMesh->isOutgoing(angle, cell, face)) {

                        UINT adjCell = g_tychoMesh->getAdjCell(cell, face);
                        
                        if (adjCell != TychoMesh::BOUNDARY_FACE) {
                            numDependencies(angle, adjCell)--;
                            if (numDependencies(angle, adjCell) == 0) {
                                CellAnglePair cellAnglePair{adjCell, angle};
                                canCompute[angleGroup].push(cellAnglePair);
                            }
                        }
                    }
                }
            }
        }
    }
    
    
    // Print times
    totalTimer.stop();

    double totalTime = totalTimer.wall_clock();
    Comm::gmax(totalTime);

    double setupTime = setupTimer.wall_clock();
    Comm::gmax(setupTime);

    if (Comm::rank() == 0) {
        printf("      Traverse Timer (setup):   %fs\n", setupTime);
        printf("      Traverse Timer (total):   %fs\n", totalTime);
    }
}

