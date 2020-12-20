//------------------------------------------------------------------------------
//  GRT Liping Liu Copyright (C) 2003
//
//  File:       grt.h
//
//  License:    GNU GPL (version 2 or later)
//
//  Contents:   Gradient Regression Tree 
//
//  Owner:      liuli@eecs.oregonstate.edu 
//
//  History:    12/13/2010   grt created
//
//------------------------------------------------------------------------------

#ifndef GRT_H
#define GRT_H

#include <R.h>
#include <Rinternals.h>
#include <vector>
#include "buildinfo.h"
#include "tree.h"
#include "dataset.h"
#include "node_factory.h"

class GRT
{
public:

    GRT();
    ~GRT();
    GBMRESULT Initialize(CDataset *pData,
                         unsigned long cDepth,
                         unsigned long cMinObsInNode);
    
    GBMRESULT RegressGradient(double *adTarget, int *aiInBag, double *adReg);
    static GBMRESULT Predict(SEXP rTreeSeq, int nIterKept, int *dIterIndex, CDataset *pDataset, double* dPredicts);

    //treeIndex is the position to be added in the vector
    GBMRESULT TransferTreeToRList(int treeIndex, 
                                  SEXP& rSetOfTrees);

    void SetLambda(double dLambda);

    CDataset *pData;            // the data
    bool fInitialized;          // indicates whether the CGM has been initialized
    CNodeFactory *pNodeFactory;

    // these objects are for the tree growing
    // allocate them once here for all trees to use
    unsigned long *aiNodeAssign;
    CNodeSearch *aNodeSearch;
    PCCARTTree ptreeTemp;
    VEC_P_NODETERMINAL vecpTermNodes;
    double *adFadj;
    double dLambda;
private:
    unsigned long cDepth;
    unsigned long cTrain;
    unsigned long cMinObsInNode;
    unsigned long cSplitCodes;
    bool *afInBag;
};

#endif // CGM_H



