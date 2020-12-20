//------------------------------------------------------------------------------
//  GBM by Greg Ridgeway  Copyright (C) 2003
//
//  File:       dataset.h
//
//  License:    GNU GPL (version 2 or later)
//
//  Contents:   Dataset class 
//        	  
//  Owner:      gregr@rand.org
//
//  History:    3/26/2001   gregr created
//              2/14/2003   gregr: adapted for R implementation
//              Liping Liu has make some revisement: 
//                  Remove Misc
//                  Remove Monotone
//                  Remove Set all weights as 1.0
//                  Remove always no offset
//                  Only data related to adX are left
//------------------------------------------------------------------------------

#ifndef DATASET_H
#define DATASET_H

#include "buildinfo.h"

class CDataset  
{
public:
	CDataset();
	~CDataset();

    GBMRESULT ResetWeights();
    GBMRESULT AllocatePredicts();

    GBMRESULT SetData(double *adX,
                    int *aiXOrder,
                    int cRows,
                    int cCols,
                    int *acVarClasses
                    );

    GBMRESULT Entry(int iRow,
                  int iCol,
                  double &dValue)
    {
        GBMRESULT hr = GBM_OK;

        if((iRow >= cRows) || (iCol >= cCols))
        {
            hr = GBM_INVALIDARG;
            goto Error;
        }

        dValue = adX[iCol*cRows + iRow];

    Cleanup:
        return hr;
    Error:
        goto Cleanup;
    }


    bool fHasOffset;
    double *adX;
    double *adY;
    int *aiXOrder;
    double *adXTemp4Order;

    double *adOffset;
    double *adWeight;
    int *acVarClasses;

    int cRows;
    int cCols;
private:

};

#endif // DATASET_H


