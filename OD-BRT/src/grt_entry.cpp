// GBM by Greg Ridgeway  Copyright (C) 2003

#include "grt.h"
#include "dataset.h"

extern "C" {

#include <R.h>
#include <Rinternals.h>

GBMRESULT InitializeDatasets(SEXP rRegData, int cComponents, CDataset aDatasets[]);
GBMRESULT InitializeDataset(SEXP rData, CDataset& dataset);
void InitializeParam(SEXP rParameters, int cComponents, int &cDepth, int &cMinObs, double &dStepLength);

SEXP grt_iterate
(
    SEXP rModel,
    SEXP riT,
    SEXP rParameters,
    SEXP rBagFlag,
    SEXP rRegData,
    SEXP rGradient,
    SEXP rPredicts
)
{
    GBMRESULT hr = GBM_OK;
    /* variables definition block */ 
    SEXP rComponent = NULL;
    int cComponents = LENGTH(rModel);
    int iT = INTEGER(riT)[0];
    
    CDataset *aDatasets = new CDataset[cComponents];
    GRT *aGRT = new GRT[cComponents]; 
   


    SEXP rcDepth = VECTOR_ELT(rParameters, 0);
    SEXP rcMinObsInNode = VECTOR_ELT(rParameters, 1);
    SEXP rdStepLength = VECTOR_ELT(rParameters, 2);
    
    int cDepth = INTEGER(rcDepth)[0];
    int cMinObs = INTEGER(rcMinObsInNode)[0];
    double dStepLength = REAL(rdStepLength)[0];
   

    long iComp = 0;
    int *aiCompBagFlag = NULL;
    double *adTargets = NULL;
    double *adPredicts = NULL; 
   
    SEXP ret;
    PROTECT(ret = allocVector(LGLSXP, 1));
    LOGICAL(ret)[0] = true;
    UNPROTECT(1);

    hr = InitializeDatasets(rRegData, cComponents, aDatasets);
    if (GBM_FAILED(hr))
    {
        goto ERROR;
    }


    for (iComp = 0; iComp < cComponents; iComp++)
    {
        aGRT[iComp].Initialize(&aDatasets[iComp], cDepth, cMinObs);
    }

    for (iComp = 0; iComp < cComponents; iComp++)
    {
        aGRT[iComp].SetLambda(dStepLength);
         
        aiCompBagFlag = LOGICAL(VECTOR_ELT(rBagFlag, iComp));
        adTargets = REAL(VECTOR_ELT(rGradient, iComp));
        adPredicts = REAL(VECTOR_ELT(rPredicts, iComp));
        
        aGRT[iComp].RegressGradient(adTargets, aiCompBagFlag, adPredicts);

        rComponent = VECTOR_ELT(rModel, iComp);
        aGRT[iComp].TransferTreeToRList(iT, rComponent); 
        //grt.ptreeTemp->Print();
    }


CLEAN:
    delete []aDatasets;
    delete []aGRT;
    return ret;

ERROR:
    LOGICAL(ret)[0] = true;
    goto CLEAN;
}

SEXP grt_tree
(
    SEXP rComponent,
    SEXP riT,
    SEXP rParameters,
    SEXP rCompBagFlag,
    SEXP rData,
    SEXP rGradient,
    SEXP rPredicts
)
{
    /* variables definition block */ 
    int iT = INTEGER(riT)[0];
    CDataset dataset;
    
    int cDepth = 0;
    int cMinObs = 0;
    double dStepLength = 1.0;
     
    SEXP radX = VECTOR_ELT(rData, 0);
    SEXP racRows = VECTOR_ELT(rData, 1);
    SEXP racCols = VECTOR_ELT(rData, 2);
    SEXP racVarClasses = VECTOR_ELT(rData, 3);
    SEXP raiXOrder = VECTOR_ELT(rData, 4);

    double *adX = REAL(radX);
    int cRows = INTEGER(racRows)[0];
    int cCols = INTEGER(racCols)[0];
    int *acVarClasses = INTEGER(racVarClasses);
    int *aiXOrder = INTEGER(raiXOrder);
   
    SEXP rcDepth = VECTOR_ELT(rParameters, 0);
    SEXP rcMinObsInNode = VECTOR_ELT(rParameters, 1);
    SEXP rdStepLength = VECTOR_ELT(rParameters, 2);

    cDepth = INTEGER(rcDepth)[0];
    cMinObs = INTEGER(rcMinObsInNode)[0];
    dStepLength = REAL(rdStepLength)[0]; 
    
    SEXP ret;
    PROTECT(ret = allocVector(LGLSXP, 1));
    LOGICAL(ret)[0] = true;
    UNPROTECT(1);


    dataset.SetData(adX, aiXOrder, cRows, cCols, acVarClasses); 

    GRT grt;
    grt.Initialize(&dataset, cDepth, cMinObs);

    grt.SetLambda(1);
    int *aiCompBagFlag = LOGICAL(rCompBagFlag);
    double *adTargets = REAL(rGradient);
    double *adPredicts = REAL(rPredicts);
    
    grt.SetLambda(dStepLength);
    grt.RegressGradient(adTargets, aiCompBagFlag, adPredicts);
    
    //grt.ptreeTemp->Print();
    grt.TransferTreeToRList(iT, rComponent); 
    return ret;
}

SEXP grt_predict
(
    SEXP rTreeSeqs,
    SEXP riT,
    SEXP rRegData
)
{
    int iComp = 0;
    
    int cComponents = LENGTH(rTreeSeqs);
    int *dIterIndex = INTEGER(riT);
    int nIterKept = LENGTH(riT);

    CDataset *aDatasets = new CDataset[cComponents];
    SEXP rTreeSeq; 
    SEXP rPredictions; 
    SEXP rPredicts; 
    InitializeDatasets(rRegData, cComponents, aDatasets);
    
    PROTECT(rPredictions = allocVector(VECSXP, cComponents));
    for (iComp = 0; iComp < cComponents; iComp++)
    {
        PROTECT(rPredicts = allocVector(REALSXP, aDatasets[iComp].cRows * nIterKept));

        rTreeSeq = VECTOR_ELT(rTreeSeqs, iComp);
        GRT::Predict(rTreeSeq, nIterKept, dIterIndex, &aDatasets[iComp], REAL(rPredicts)); 
        SET_VECTOR_ELT(rPredictions, iComp, rPredicts);
    }
    UNPROTECT(cComponents);
    UNPROTECT(1); 
    
    delete []aDatasets;
    return rPredictions;
}

SEXP grt_component_predict
(
    SEXP rTreeSeq,
    SEXP riT,
    SEXP rData
)
{
    /* variables definition block */ 
    int *dIterIndex = INTEGER(riT);
    int nIterKept = LENGTH(riT);
    
    CDataset* pDataset = new CDataset();
    SEXP rPredicts; 
    
    InitializeDataset(rData, *pDataset);
    
    PROTECT(rPredicts = allocVector(REALSXP, pDataset->cRows * nIterKept));
    
    GRT::Predict(rTreeSeq, nIterKept, dIterIndex, pDataset, REAL(rPredicts)); 
    UNPROTECT(1); 
    delete pDataset;
    return rPredicts;
}



void InitializeParam(SEXP rParameters, int cComponents, int &cDepth, int &cMinObs, double &dStepLength)
{
    SEXP rcDepth = VECTOR_ELT(rParameters, 0);
    SEXP rcMinObsInNode = VECTOR_ELT(rParameters, 1);
    SEXP rdStepLength = VECTOR_ELT(rParameters, 2);

    cDepth = INTEGER(rcDepth)[0];
    cMinObs = INTEGER(rcMinObsInNode)[0];
    dStepLength = REAL(rdStepLength)[0]; 
}

GBMRESULT InitializeDatasets(SEXP rRegData, int cComponents, CDataset aDatasets[])
{
    int i = 0;
    bool shared = false;
    SEXP rData;
    SEXP radX;
    SEXP racRows;
    SEXP racCols;
    SEXP raiXOrder;
    SEXP racVarClasses;

    if (LENGTH(rRegData) == 1)
    {
        shared = true; 
    } 
    else if (LENGTH(rRegData) != cComponents)
    {
        Rprintf("Number of matrices is not equal to number of components.\n");
        return GBM_INVALIDARG;
    }

    for (i = 0; i < cComponents; i++)
    {
        if (shared)
            rData = VECTOR_ELT(rRegData, 0);
        else
            rData = VECTOR_ELT(rRegData, i);

        radX = VECTOR_ELT(rData, 0);

        racRows = VECTOR_ELT(rData, 1);
        racCols = VECTOR_ELT(rData, 2);
        racVarClasses = VECTOR_ELT(rData, 3);

        double *adX = REAL(radX);
        int cRows = INTEGER(racRows)[0];
        int cCols = INTEGER(racCols)[0];
        int *acVarClasses = INTEGER(racVarClasses);
        int *aiXOrder = NULL;

        if (LENGTH(rData) > 4 && TYPEOF(VECTOR_ELT(rData, 4)) != NILSXP)
        {
            raiXOrder = VECTOR_ELT(rData, 4);
            aiXOrder = INTEGER(raiXOrder);
        } 
        aDatasets[i].SetData(adX, aiXOrder, cRows, cCols, acVarClasses); 
    }
    return GBM_OK;
}

GBMRESULT InitializeDataset(SEXP rData, CDataset& dataset)
{
    SEXP radX;
    SEXP racRows;
    SEXP racCols;
    SEXP raiXOrder;
    SEXP racVarClasses;

    radX = VECTOR_ELT(rData, 0);

    racRows = VECTOR_ELT(rData, 1);
    racCols = VECTOR_ELT(rData, 2);
    racVarClasses = VECTOR_ELT(rData, 3);

    double *adX = REAL(radX);
    int cRows = INTEGER(racRows)[0];
    int cCols = INTEGER(racCols)[0];
    int *acVarClasses = INTEGER(racVarClasses);
    int *aiXOrder = NULL;

    if (LENGTH(rData) > 4 && TYPEOF(VECTOR_ELT(rData, 4)) != NILSXP)
    {
        raiXOrder = VECTOR_ELT(rData, 4);
        aiXOrder = INTEGER(raiXOrder);
    } 
    dataset.SetData(adX, aiXOrder, cRows, cCols, acVarClasses); 
    return GBM_OK;
}



SEXP grt_plot
(
    SEXP radX,        // vector or matrix of points to make predictions
    SEXP rcRows,      // number of rows in X
    SEXP rcCols,      // number of columns in X
    SEXP raiWhichVar, // length=cCols, index of which var cols of X are
    SEXP rcIterations,// number of iterations to use
    SEXP rTreeSeq,      // tree list object
    SEXP raiVarType   // vector of variable types
)
{
    unsigned long hr = 0;
    int i = 0;
    int iTree = 0;
    int iObs = 0;
    int cRows = INTEGER(rcRows)[0];
    int cCols = INTEGER(rcCols)[0];
    int cIterations = INTEGER(rcIterations)[0];
   

    SEXP rTreeBag = NULL;
    SEXP rThisTree = NULL;
    SEXP rCSplits = NULL;
    SEXP rLambda = NULL;
    double dLambda = 0.0;
    int *aiSplitVar = NULL;
    double *adSplitCode = NULL;
    int *aiLeftNode = NULL;
    int *aiRightNode = NULL;
    int *aiMissingNode = NULL;
    double *adW = NULL;
    int iCurrentNode = 0;
    double dCurrentW = 0.0;
    double dX = 0.0;
    int iCatSplitIndicator = 0;

    SEXP radPredF = NULL;
    int aiNodeStack[40];
    double adWeightStack[40];
    int cStackNodes = 0;
    int iPredVar = 0;

    // allocate the predictions to return
    PROTECT(radPredF = allocVector(REALSXP, cRows));
    if(radPredF == NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }
    for(iObs=0; iObs<cRows; iObs++)
    {
        REAL(radPredF)[iObs] = 0;
    }
    for(iTree=0; iTree<cIterations; iTree++)
    {
        rTreeBag      = VECTOR_ELT(rTreeSeq,iTree);
        rThisTree     = VECTOR_ELT(rTreeBag,0);
        rCSplits      = VECTOR_ELT(rTreeBag, 1);
        rLambda       = VECTOR_ELT(rTreeBag, 2);
        dLambda       = REAL(rLambda)[0];

        aiSplitVar    = INTEGER(VECTOR_ELT(rThisTree,0));
        adSplitCode   = REAL   (VECTOR_ELT(rThisTree,1));
        aiLeftNode    = INTEGER(VECTOR_ELT(rThisTree,2));
        aiRightNode   = INTEGER(VECTOR_ELT(rThisTree,3));
        aiMissingNode = INTEGER(VECTOR_ELT(rThisTree,4));
        adW           = REAL   (VECTOR_ELT(rThisTree,6));
        for(iObs=0; iObs<cRows; iObs++)
        {
            aiNodeStack[0] = 0;
            adWeightStack[0] = 1.0;
            cStackNodes = 1;
            while(cStackNodes > 0)
            {
                cStackNodes--;
                iCurrentNode = aiNodeStack[cStackNodes];

                if(aiSplitVar[iCurrentNode] == -1) // terminal node
                {
                    REAL(radPredF)[iObs] += 
                        adWeightStack[cStackNodes] * adSplitCode[iCurrentNode] * dLambda;
                }
                else // non-terminal node
                {
                    // is this a split variable that interests me?
                    iPredVar = -1;
                    for(i=0; (iPredVar == -1) && (i < cCols); i++)
                    {
                        if(INTEGER(raiWhichVar)[i] == aiSplitVar[iCurrentNode])
                        {
                            iPredVar = i; // split is on one that interests me
                        }
                    }

                    if(iPredVar != -1) // this split is among raiWhichVar
                    {


                        dX = REAL(radX)[iPredVar*cRows + iObs];
                        // missing?
                        if(ISNA(dX))
                        {
                            aiNodeStack[cStackNodes] = aiMissingNode[iCurrentNode];
                            cStackNodes++;                            
                        }
                        // continuous?
                        else if(INTEGER(raiVarType)[aiSplitVar[iCurrentNode]] == 0)
                        {
                            if(dX < adSplitCode[iCurrentNode])
                            {
                                aiNodeStack[cStackNodes] = aiLeftNode[iCurrentNode];
                                cStackNodes++;
                            }
                            else
                            {
                                aiNodeStack[cStackNodes] = aiRightNode[iCurrentNode];
                                cStackNodes++;
                            }
                        }
                        else // categorical
                        {
                            iCatSplitIndicator = INTEGER(
                                VECTOR_ELT(rCSplits,
                                           (int)adSplitCode[iCurrentNode]))[(int)dX];
                            if(iCatSplitIndicator==-1)
                            {
                                aiNodeStack[cStackNodes] = aiLeftNode[iCurrentNode];
                                cStackNodes++;
                            }
                            else if(iCatSplitIndicator==1)
                            {
                                aiNodeStack[cStackNodes] = aiRightNode[iCurrentNode];
                                cStackNodes++;
                            }
                            else // handle unused level
                            {
                                iCurrentNode = aiMissingNode[iCurrentNode];
                            }
                        }
                    } // iPredVar != -1
                    else // not interested in this split, average left and right 
                    {
                        aiNodeStack[cStackNodes] = aiRightNode[iCurrentNode];
                        dCurrentW = adWeightStack[cStackNodes];
                        adWeightStack[cStackNodes] = dCurrentW *
                            adW[aiRightNode[iCurrentNode]]/
                            (adW[aiLeftNode[iCurrentNode]]+
                             adW[aiRightNode[iCurrentNode]]);
                        cStackNodes++;
                        aiNodeStack[cStackNodes] = aiLeftNode[iCurrentNode];
                        adWeightStack[cStackNodes] = 
                            dCurrentW-adWeightStack[cStackNodes-1];
                        cStackNodes++;
                    }
                } // non-terminal node
            } // while(cStackNodes > 0)
        } // iObs
    } // iTree

Cleanup:
    UNPROTECT(1); // radPredF
    return radPredF;
Error:
    goto Cleanup;
} // grt_plot

}
