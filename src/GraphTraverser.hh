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

#ifndef __GRAPH_TRAVERSER_HH__
#define __GRAPH_TRAVERSER_HH__

#include "Global.hh"
#include "Mat.hh"
#include <mpi.h>
#include <vector>
#include <map>

/*
    Boundary Type for faces of a cell.
    They are split into incoming and outgoing wrt sweep direction.
    Interior means adjacent cell is on same proc.
    Interior boundary means adj cell is on different proc.
    Exterior boundary means it is a boundary for the whole mesh.
*/
enum BoundaryType
{
    // Outgoing
    BoundaryType_OutIntBdry,    // Interior boundary
    BoundaryType_OutExtBdry,    // Exterior boundary
    BoundaryType_OutInt,        // Interior
    
    // Incoming
    BoundaryType_InIntBdry,     // Interior boundary
    BoundaryType_InExtBdry,     // Exterior boundary
    BoundaryType_InInt          // Interior
};


enum Direction
{
    Direction_Forward,
    Direction_Backward
};


/*
    TraverseData class
    
    Abstract class defining the methods needed to traverse a graph.
*/
class TraverseData
{
public:
    virtual const char* getData(UINT cell, UINT face, UINT angle) = 0;
    virtual void setSideData(UINT side, UINT angle, const char *data) = 0;
    virtual UINT getPriority(UINT cell, UINT angle) = 0;
    virtual void update(UINT cell, UINT angle, 
                        UINT adjCellsSides[g_nFacePerCell], 
                        BoundaryType bdryType[g_nFacePerCell]) = 0;

protected:
    // Don't allow construction of this base class.
    TraverseData() { }
};



class GraphTraverser
{
public:
    GraphTraverser(Direction direction, bool doComm, UINT dataSizeInBytes);
    ~GraphTraverser();

    void traverse(const UINT maxComputePerStep, TraverseData &traverseData);

private:

    class OneSidedImpl
    {
    public:
        OneSidedImpl(UINT numAdjRanks,
                     std::vector<UINT> offRankLockOffsets,
                     std::vector<UINT> offRankHeaderOffsets,
                     std::vector<UINT> offRankDataOffsets,
                     std::vector<UINT> onRankLockOffsets,
                     std::vector<UINT> onRankHeaderOffsets,
                     std::vector<UINT> onRankDataOffsets,
                     UINT dataSizePerChunk,
                     UINT maxMessages,
                     MPI_Win mpiWin,
                     char *mpiWinMemory);

        
        // Send specific functions
        uint32_t send_getNumWritten(int adjRankIndex);
        void send_setNumWritten(int adjRankIndex, uint32_t numBytesWritten);
        uint32_t send_getNumMessagesSent(int adjRankIndex);
        void send_setNumMessagesSent(int adjRankIndex, uint32_t numMessages);
        void send_switchDataChunk(int adjRankIndex);
        UINT send_getLockOffset(int adjRankIndex);
        UINT send_getNumWrittenOffset(int adjRankIndex);
        UINT send_getIsWrittenOffset(int adjRankIndex);
        UINT send_getDataOffset(int adjRankIndex, uint32_t bytesOffset);
        
        // Recv specific functions
        uint32_t recv_getNumRead(int adjRankIndex);
        void recv_setNumRead(int adjRankIndex, uint32_t numBytesRead);
        void recv_switchDataChunk(int adjRankIndex);
        UINT recv_getBaseLockOffset();
        UINT recv_getBaseHeaderOffset();
        int recv_getLockCount();
        int recv_getHeaderCount();
        uint32_t* recv_getLockPointer();
        uint32_t* recv_getHeaderPointer();
        char* recv_getWinMemory();
        UINT recv_getLockOffset(int adjRankIndex);
        UINT recv_getHeaderOffset(int adjRankIndex);
        UINT recv_getDataOffset(int adjRankIndex, uint32_t bytesOffset);
        uint32_t recv_getLock(int adjRankIndex);
        uint32_t recv_getNumWritten(int adjRankIndex);

        // Other functions
        MPI_Win getWin()
            { return c_mpiWin; }
        UINT getMaxMessages()
            { return c_maxMessages; }
        

    private:
        // Send state variables
        std::vector<uint32_t> c_send_numWrittenVector[2];
        std::vector<uint32_t> c_send_numMessagesSentVector[2];
        std::vector<uint32_t> c_send_currentDataChunkVector;
        std::vector<UINT> c_offRankLockOffsets;
        std::vector<UINT> c_offRankHeaderOffsets;
        std::vector<UINT> c_offRankDataOffsets;
        
        // Recv state variables
        std::vector<uint32_t> c_recv_numReadVector[2];
        std::vector<uint32_t> c_recv_lockDataVector;
        std::vector<uint32_t> c_recv_headerDataVector;
        std::vector<uint32_t> c_recv_currentDataChunkVector;
        std::vector<UINT> c_onRankLockOffsets;
        std::vector<UINT> c_onRankHeaderOffsets;
        std::vector<UINT> c_onRankDataOffsets;

        // Other variables
        UINT c_dataSizePerChunk;
        UINT c_maxMessages;
        MPI_Win c_mpiWin;
        char *c_mpiWinMemory;
    };


    void setupOneSidedMPI();
    void sendData2Sided(const std::vector<std::vector<char>> &sendBuffers) const;
    void recvData2Sided(std::vector<char> &dataPackets) const;
    void sendAndRecvData(const std::vector<std::vector<char>> &sendBuffers,
                         std::vector<char> &dataPackets,
                         std::vector<bool> &commDark,
                         const bool killComm) const;
    void sendData1Sided(const std::vector<std::vector<char>> &sendBuffers) const;
    void recvData1Sided(std::vector<char> &dataPackets) const;
    
    std::vector<UINT> c_adjRankIndexToRank;
    std::map<UINT,UINT> c_adjRankToRankIndex;
    Mat2<UINT> c_initNumDependencies;
    Direction c_direction;
    bool c_doComm;
    UINT c_dataSizeInBytes;

    OneSidedImpl *c_oneSidedImpl;
};

#endif
