//  CGM by Greg Ridgeway  Copyright (C) 2003

#include "grt.h"

GRT::GRT()
{
    adFadj = NULL;
    aiNodeAssign = NULL;
    aNodeSearch = NULL;
    
    cDepth = 0;
    cMinObsInNode = 0;
    fInitialized = false;

    pData = NULL;
    pNodeFactory = NULL;
    ptreeTemp = NULL;
    afInBag = NULL;
    cTrain = 0;
    dLambda = 1.0;
    cSplitCodes = 0;
}


GRT::~GRT()
{
    if(adFadj != NULL)
    {
        delete [] adFadj;
        adFadj = NULL;
    }
    if(aiNodeAssign != NULL)
    {
        delete [] aiNodeAssign;
        aiNodeAssign = NULL;
    }
    if(aNodeSearch != NULL)
    {
        delete [] aNodeSearch;
        aNodeSearch = NULL;
    }
    if(ptreeTemp != NULL)
    {
        delete ptreeTemp;
        ptreeTemp = NULL;
    }
    // must delete the node factory last!!! at least after deleting trees
    if(pNodeFactory != NULL)
    {
        delete pNodeFactory;
        pNodeFactory = NULL;
    }
 
    if(afInBag != NULL)
    {
        delete afInBag;
        afInBag = NULL;
    }
    
}


GBMRESULT GRT::Initialize
(
    CDataset *pData,
    unsigned long cDepth,
    unsigned long cMinObsInNode
)
{
    GBMRESULT hr = GBM_OK;
    unsigned long i=0;

    if(pData == NULL)
    {
        hr = GBM_INVALIDARG;
        goto Error;
    }
    this->pData = pData;
    this->cDepth = cDepth;
    this->cMinObsInNode = cMinObsInNode;
    this->cTrain = pData->cRows;
    // allocate the tree structure
    ptreeTemp = new CCARTTree;
    if(ptreeTemp == NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }

    adFadj = new double[pData->cRows];
    if(adFadj == NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }

    afInBag = new bool[cTrain];
    if(afInBag==NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }
    
    pNodeFactory = new CNodeFactory();
    if(pNodeFactory == NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }
    hr = pNodeFactory->Initialize(cDepth);
    if(GBM_FAILED(hr))
    {
        goto Error;
    }
    ptreeTemp->Initialize(pNodeFactory);

    // aiNodeAssign tracks to which node each training obs belongs
    aiNodeAssign = new ULONG[pData->cRows];
    if(aiNodeAssign==NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }
    // NodeSearch objects help decide which nodes to split
    aNodeSearch = new CNodeSearch[2*cDepth+1];
    if(aNodeSearch==NULL)
    {
        hr = GBM_OUTOFMEMORY;
        goto Error;
    }
    for(i=0; i<2*cDepth+1; i++)
    {
        aNodeSearch[i].Initialize(cMinObsInNode);
    }
    vecpTermNodes.resize(2*cDepth+1,NULL);

    dLambda = 1.0;
    fInitialized = true;

Cleanup:
    return hr;
Error:
    goto Cleanup;
}


void GRT::SetLambda(double dLambda)
{
    this->dLambda = dLambda;
}

GBMRESULT GRT::RegressGradient(double *adTarget, int *aiInBag, double *adReg)
{
    GBMRESULT hr = GBM_OK;
    unsigned long i = 0;
    unsigned long cTotalInBag = 0;
    double *adZ = adTarget;

    if(!fInitialized)
    {
        hr = GBM_FAIL;
        goto Error;
    }

    vecpTermNodes.assign(2*cDepth+1,NULL);

    for(i = 0; i < cTrain; i++)
    {
        if(aiInBag[i] != 0)
        {
            afInBag[i] = true;
            cTotalInBag++;
        }
        else
        {
            afInBag[i] = false;
        }
    }

    hr = ptreeTemp->Reset();

    hr = ptreeTemp->grow(adZ,pData,pData->adWeight,adFadj,
                         cTrain,cTotalInBag,dLambda,cDepth,
                         cMinObsInNode,
                         afInBag,
                         aiNodeAssign,aNodeSearch,vecpTermNodes);
    if(GBM_FAILED(hr))
    {
        goto Error;
    }

    //The function below have done nothing, since the distribution is gaussian
    //hr = pDist->FitBestConstant(pData->adY,
    //                            pData->adMisc,
    //                            pData->adOffset,
    //                            pData->adWeight,
    //                            adF,
    //                            adZ,
    //                            aiNodeAssign,
    //                            cTrain,
    //                            vecpTermNodes,
    //                            (2*cNodes+1)/3, // number of terminal nodes
    //                            cMinObsInNode,
    //                            afInBag,
    //                            adFadj);

    // update training predictions
    // fill in missing nodes where N < cMinObsInNode
    hr = ptreeTemp->Adjust(aiNodeAssign,adFadj,cTrain,
                           vecpTermNodes,cMinObsInNode);
    if(GBM_FAILED(hr))
    {
        goto Error;
    }
    ptreeTemp->SetShrinkage(dLambda);

    // update the training predictions
    for(i=0; i < cTrain; i++)
    {
        adReg[i] = adFadj[i];
    }

Cleanup:
    return hr;
Error:
    goto Cleanup;
}


GBMRESULT GRT::TransferTreeToRList(int treeIndex, SEXP& rComponent)
{
    unsigned long hr = 0;
    int cNodes = 0;
    SEXP rTreeBag = NULL; 
    
    const int cTreeComponents = 8;
    // riNodeID,riSplitVar,rdSplitPoint,riLeftNode,
    // riRightNode,riMissingNode,rdErrorReduction,rdWeight
    SEXP rNewTree = NULL;
    
    unsigned int i = 0;
    SEXP rSplitCodes;
    SEXP rSplitCode;

    ptreeTemp->GetNodeCount(cNodes);
    
    PROTECT(rTreeBag = allocVector(VECSXP, 3));
    SET_VECTOR_ELT(rComponent, treeIndex, rTreeBag);
    UNPROTECT(1); //TreeBag
    
    PROTECT(rNewTree = allocVector(VECSXP, cTreeComponents));

    SEXP riSplitVar = NULL;
    SEXP rdSplitPoint = NULL;
    SEXP riLeftNode = NULL;
    SEXP riRightNode = NULL;
    SEXP riMissingNode = NULL;
    SEXP rdErrorReduction = NULL;
    SEXP rdWeight = NULL;
    SEXP rdPred = NULL;

    PROTECT(riSplitVar = allocVector(INTSXP, cNodes));
    PROTECT(rdSplitPoint = allocVector(REALSXP, cNodes));
    PROTECT(riLeftNode = allocVector(INTSXP, cNodes));
    PROTECT(riRightNode = allocVector(INTSXP, cNodes));
    PROTECT(riMissingNode = allocVector(INTSXP, cNodes));
    PROTECT(rdErrorReduction = allocVector(REALSXP, cNodes));
    PROTECT(rdWeight = allocVector(REALSXP, cNodes));
    PROTECT(rdPred = allocVector(REALSXP, cNodes));
    SET_VECTOR_ELT(rNewTree,0,riSplitVar);
    SET_VECTOR_ELT(rNewTree,1,rdSplitPoint);
    SET_VECTOR_ELT(rNewTree,2,riLeftNode);
    SET_VECTOR_ELT(rNewTree,3,riRightNode);
    SET_VECTOR_ELT(rNewTree,4,riMissingNode);
    SET_VECTOR_ELT(rNewTree,5,rdErrorReduction);
    SET_VECTOR_ELT(rNewTree,6,rdWeight);
    SET_VECTOR_ELT(rNewTree,7,rdPred);
    UNPROTECT(cTreeComponents); 
    
    SET_VECTOR_ELT(rTreeBag, 0, rNewTree);
    UNPROTECT(1); // rNewTree

    VEC_VEC_CATEGORIES vecSplitCodes;
    hr = ptreeTemp->TransferTreeToRList(pData,
                                        INTEGER(riSplitVar),
                                        REAL(rdSplitPoint),
                                        INTEGER(riLeftNode),
                                        INTEGER(riRightNode),
                                        INTEGER(riMissingNode),
                                        REAL(rdErrorReduction),
                                        REAL(rdWeight),
                                        REAL(rdPred),
                                        vecSplitCodes,
                                        0,
                                        dLambda);

    PROTECT(rSplitCodes = allocVector(VECSXP, vecSplitCodes.size()));
    for(i = 0; i < vecSplitCodes.size(); i++)
    {
        PROTECT(rSplitCode = 
                    allocVector(INTSXP, vecSplitCodes[i].size()));
        SET_VECTOR_ELT(rSplitCodes,i,rSplitCode);
        UNPROTECT(1); // rSplitCode

        for(unsigned int j = 0; j < vecSplitCodes[i].size(); j++)
        {
            INTEGER(rSplitCode)[j] = vecSplitCodes[i][j];
        }
    
    }
 
    SET_VECTOR_ELT(rTreeBag, 1, rSplitCodes);
   
    
    UNPROTECT(1); //SplitCodes
    return hr;
}


GBMRESULT GRT::Predict(SEXP rTreeSeq, int nIterKept, int *dIterIndex, CDataset *pDataset, double* dPredicts)
{
    int iObs = 0;
    int cRows = pDataset->cRows;

    SEXP rTreeBag = NULL;
    SEXP rThisTree = NULL;
    SEXP rCSplits = NULL;
    SEXP rLambda = NULL;

    int *aiSplitVar = NULL;
    double *adSplitCode = NULL;
    int *aiLeftNode = NULL;
    int *aiRightNode = NULL;
    int *aiMissingNode = NULL;
    int iCurrentNode = 0;
    double dX = 0.0;
    int iCatSplitIndicator = 0;
    double dLambda = 0.0;
    
    //memset(dPredicts, 0, cRows * sizeof(double));


    int iTree = 0;
    int iIterKept = 0;

    for(iTree = 0; ; iTree++)
    {
        rTreeBag      = VECTOR_ELT(rTreeSeq, iTree);
        rThisTree     = VECTOR_ELT(rTreeBag, 0);
        rCSplits      = VECTOR_ELT(rTreeBag, 1);
        rLambda       = VECTOR_ELT(rTreeBag, 2);
        dLambda       = REAL(rLambda)[0];
        
        // these relate to columns returned by pretty.gbm.tree()
        aiSplitVar    = INTEGER(VECTOR_ELT(rThisTree,0));
        adSplitCode   = REAL   (VECTOR_ELT(rThisTree,1));
        aiLeftNode    = INTEGER(VECTOR_ELT(rThisTree,2));
        aiRightNode   = INTEGER(VECTOR_ELT(rThisTree,3));
        aiMissingNode = INTEGER(VECTOR_ELT(rThisTree,4));

    
        for(iObs=0; iObs<cRows; iObs++)
        {
            iCurrentNode = 0;
            while(aiSplitVar[iCurrentNode] != -1)
            {
                dX = pDataset->adX[aiSplitVar[iCurrentNode]*cRows + iObs];
                // missing?
                if(ISNA(dX))
                {
                    iCurrentNode = aiMissingNode[iCurrentNode];
                }
                // continuous?
                else if(pDataset->acVarClasses[aiSplitVar[iCurrentNode]] == 0)
                {
                    if(dX < adSplitCode[iCurrentNode])
                    {
                        iCurrentNode = aiLeftNode[iCurrentNode];
                    }
                    else
                    {
                        iCurrentNode = aiRightNode[iCurrentNode];
                    }
                }
                else // categorical
                {
                    iCatSplitIndicator = INTEGER(VECTOR_ELT(rCSplits,
                                                 (int)adSplitCode[iCurrentNode]))[(int)dX];
                    if(iCatSplitIndicator==-1)
                    {
                        iCurrentNode = aiLeftNode[iCurrentNode];
                    }
                    else if(iCatSplitIndicator==1)
                    {
                        iCurrentNode = aiRightNode[iCurrentNode];
                    }
                    else // categorical level not present in training
                    {
                        iCurrentNode = aiMissingNode[iCurrentNode];
                    }
                }
            }
            
            if (iTree == 0)
                dPredicts[iObs] = adSplitCode[iCurrentNode] * dLambda; // add the prediction
            else
                dPredicts[cRows * iIterKept + iObs] += adSplitCode[iCurrentNode] * dLambda; // add the prediction
        } // iObs

        if (iTree == dIterIndex[iIterKept])
        {
            iIterKept++;
            if (iIterKept >= nIterKept)
                break;
            else
            {
                for(iObs=0; iObs<cRows; iObs++)
                {
                    dPredicts[cRows * iIterKept + iObs] = dPredicts[cRows * (iIterKept  - 1) + iObs];
                } 
            }
        }

    } // iTree
    
    return GBM_OK;
}
