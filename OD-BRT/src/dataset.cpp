//  GBM by Greg Ridgeway  Copyright (C) 2003

#include "dataset.h"

CDataset::CDataset()
{
    adX = NULL;
    aiXOrder = NULL;
    acVarClasses = NULL;

    adWeight = NULL;
        
    cRows = 0;
    cCols = 0;

    adY = NULL;
}


CDataset::~CDataset()
{
    if (adWeight != NULL)
        delete adWeight;
    
    if (adY != NULL)
        delete adY;
}




GBMRESULT CDataset::ResetWeights()
{
    GBMRESULT hr = GBM_OK;
    int i = 0;

    if(adWeight == NULL)
    {
        hr = GBM_INVALIDARG;
        goto Error;
    }

    for(i=0; i<cRows; i++)
    {
        adWeight[i] = 1.0;
    }

Cleanup:
    return hr;
Error:
    goto Cleanup;
}

GBMRESULT CDataset::SetData
(
    double *adX,
    int *aiXOrder,
    int cRows,
    int cCols,
    int *acVarClasses
)
{
    GBMRESULT hr = GBM_OK;

    this->cRows = cRows;
    this->cCols = cCols;

    this->adX = adX;
    this->aiXOrder = aiXOrder;
    this->acVarClasses = acVarClasses;
    
    this->adWeight = new double[cRows];
    
    for (int i = 0; i < cRows; i++)
        this->adWeight[i] = 1.0;

    return hr;
}

GBMRESULT CDataset::AllocatePredicts()
{
    this->adY = new double[cRows];
    if (this->adY != NULL)
        return GBM_OK;
    return GBM_FAIL;
}


void DisplayDataset(CDataset &dataset)
{

    Rprintf("\n------------------------------------------\n");
    for (int j = 0; j < dataset.cRows; j++) 
    {
        for (int k = 0; k < dataset.cCols; k++)
        {
            double value = 0;
            dataset.Entry(j, k, value);
            Rprintf("%f\t", value);
        }
        Rprintf("\n");
    }
}

