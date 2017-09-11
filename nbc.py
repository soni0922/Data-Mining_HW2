# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import re
import random
import numpy
import matplotlib.pyplot as grp
import math

class feature:
    def __init__(self, word, freq):
                self.word = word
                self.freq = freq
    def __str__(self):
        return "%s %d" % (self.word, self.freq)


def read_file(file):
    dictionaryForm = {}
    with open(file) as f:           
        for row in f:
            token=row.split('\t')       
            reviewToken=re.sub(r'[^a-zA-Z0-9\s]','',token[2]).strip().split()  
            docId=token[0]
            label=token[1]
            for i in range(len(reviewToken)):
                reviewToken[i]=reviewToken[i].strip().lower()
                reviewToken[i]=re.sub(r'\W+','',reviewToken[i])     
            dictionaryForm[docId]=(label,reviewToken)
#            print(reviewToken)
#            exit()
    
    return dictionaryForm

if(len(sys.argv) == 3):   
    #print(type(sys.argv[1]))       
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]
    
    featureWords=[]
    featureFreq=[]
    featureList=[]
    vectorTupleTrain={}
    vectorTupleTest={}
    pX0C0List=[]
    pX0C1List=[]
    pX1C0List=[]
    pX1C1List=[]
    
    
    
    dictTrainFile = read_file(trainFileName)
    dictTestFile = read_file(testFileName)
    
    ##count unique no of words in each review
    for key in dictTrainFile:
        #print(dictTrainFile[key][1])
        eachReviewSet = set(dictTrainFile[key][1])       #this is a review text
        for setValue in eachReviewSet:                  #setValue : each word in review
            if (setValue in featureWords):
                ##increment freq
                wordIndex=featureWords.index(setValue)
                featureFreq[wordIndex]= featureFreq[wordIndex] + 1
            else:
                featureWords.append(setValue)
                wordIndex=featureWords.index(setValue)
                featureFreq.insert(wordIndex,1)
#    print(featureWords)
#    print(featureFreq)
    for i in range(len(featureWords)):
        featureList.append(feature(featureWords[i],featureFreq[i]))
    featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
#    for i in range(len(featureList)):
#        print (featureList[i])          # features formed
    #print(len(featureList))
#    for i in range(101):
#        print (featureList[i])
#    print(len(featureList[0:101]))
    featureListNew = featureList[100:]
    #print(len(featureListNew)) 
    
    ##print top ten words                        
    print("WORD1 ",featureListNew[0].word)
    print("WORD2 ",featureListNew[1].word)
    print("WORD3 ",featureListNew[2].word)
    print("WORD4 ",featureListNew[3].word)
    print("WORD5 ",featureListNew[4].word)
    print("WORD6 ",featureListNew[5].word)
    print("WORD7 ",featureListNew[6].word)
    print("WORD8 ",featureListNew[7].word)
    print("WORD9 ",featureListNew[8].word)
    print("WORD10 ",featureListNew[9].word)


    ##consider features from 100 to 500
    featureListNew = featureListNew[0:500]
    #print(len(featureListNew))
    
    ##now,construct 500-dimensional vector for each review!
    for key in dictTrainFile:
        eachVectorListTrain=[]
        eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
        #print(eachReviewSet)
        for i in range(len(featureListNew)):
            #print(featureListNew[i].word)
            if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                eachVectorListTrain.append(1)
            else:
                eachVectorListTrain.append(0)
            #print(len(eachVectorListTrain))
        #print(eachVectorListTrain)
        vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
        #print("My tuple: ",vectorTupleTrain[key])
       
        
    ##next,forming vector tuples for test set as well
    for key in dictTestFile:
        eachVectorListTest=[]
        eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
        #print(eachReviewSet)
        for i in range(len(featureListNew)):
            #print(featureListNew[i].word)
            if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                eachVectorListTest.append(1)
            else:
                eachVectorListTest.append(0)
        #print(eachVectorListTest)
        vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
        #print("My tuple: ",vectorTupleTest[key])
    
    ##learn NBC model
    for ele in range(len(featureListNew)):
        countC0 = 0
        countC1 = 0
        countX0C0 = 0
        countX1C0 = 0
        countX0C1 = 0
        countX1C1 = 0
        for key in vectorTupleTrain:
            classLabel=vectorTupleTrain[key][0]
            #print("tuple for review id : "+str(key)+" is :",vectorTupleTrain[key])
            if(classLabel=='0'):
                countC0 = countC0 + 1
                pVectorEle = vectorTupleTrain[key][1][ele]    
                if(pVectorEle==0):
                    countX0C0 = countX0C0 + 1
                elif(pVectorEle==1):
                    countX1C0 = countX1C0 + 1
                
            elif(classLabel=='1'):
                countC1 = countC1 + 1
                pVectorEle = vectorTupleTrain[key][1][ele]
                if(pVectorEle==0):
                    countX0C1 = countX0C1 + 1
                elif(pVectorEle==1):
                    countX1C1 = countX1C1 + 1
               
        totalReviews = len(dictTrainFile)
        probC0 = countC0/totalReviews
        probC1 = countC1/totalReviews
               
#        print("countC0: ",countC0)
#        print("countC1: ",countC1)
#        print("countX0C0: ",countX0C0)
#        print("countX1C0: ",countX1C0)
#        print("countX0C1: ",countX0C1)
#        print("countX1C1: ",countX1C1)
      
        ##Laplace smoothing
        probX0C0 = (countX0C0 + 1)/(countC0 + 2)
        probX1C0 = (countX1C0 + 1)/(countC0 + 2)
        probX0C1 = (countX0C1 + 1)/(countC1 + 2)
        probX1C1 = (countX1C1 + 1)/(countC1 + 2)
        
        ##append each prob for each feature
        pX0C0List.append(probX0C0)
        pX1C0List.append(probX1C0)
        pX0C1List.append(probX0C1)
        pX1C1List.append(probX1C1)
      
#    print(probC0)    
#    print(probC1)    
#    print(pX0C0List[0])
#    print(pX0C1List[0])
#    print(pX1C0List[0])
#    print(pX1C1List[0])
    
    ##apply the learned model to test data
    misclassify=0
    totalClassify=len(vectorTupleTest)
    for key in vectorTupleTest:
        probTestDataC0=1
        probTestDataC1=1
        classLabel=vectorTupleTest[key][0]
        
        for ele in range(len(vectorTupleTest[key][1])):
            if(vectorTupleTest[key][1][ele]==0):
                probTestDataC0=probTestDataC0*pX0C0List[ele]
                probTestDataC1=probTestDataC1*pX0C1List[ele]
            elif(vectorTupleTest[key][1][ele]==1):
                probTestDataC0=probTestDataC0*pX1C0List[ele]
                probTestDataC1=probTestDataC1*pX1C1List[ele]
        probTestDataC0=probTestDataC0*probC0
        probTestDataC1=probTestDataC1*probC1
            
        if(probTestDataC0 > probTestDataC1):
            predClassLabel = '0'
        elif(probTestDataC1 >= probTestDataC0):
            predClassLabel = '1'
        
        if(classLabel!=predClassLabel):
            misclassify=misclassify+1
    
    zeroOneLoss=misclassify/totalClassify
    print("ZERO-ONE-LOSS ",zeroOneLoss)
        
        
    nextQues=0
    if(nextQues==1):
        
        vectorTupleTrain={}
        vectorTupleTest={}
        pX0C0List=[]
        pX0C1List=[]
        pX1C0List=[]
        pX1C1List=[]
        
        dictTrainFile = read_file('yelp_data.csv')
        total=len(dictTrainFile)
        per=[1, 5, 10, 20, 50, 90]
        avgZeroOneLossList=[]
        stdZeroOneLossList=[]
        baselineAvgZeroOneLossList=[]
        baselineStdZeroOneLossList=[]
        for perc in per:
            zeroOneLossList=[]
            baselineZeroOneLossList=[]
            trainSize=math.ceil(perc*0.01*total)
            dictTrainFileNew={}
            dictTestFileNew={}
            for i in range(10):
                keys=list(dictTrainFile.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=dictTrainFile[keys[k]]
                for k in range(trainSize+1,total):
                    dictTestFileNew[keys[k]]=dictTrainFile[keys[k]]
                
                featureWords=[]
                featureFreq=[]
                featureList=[]
                vectorTupleTrain={}
                vectorTupleTest={}
                pX0C0List=[]
                pX0C1List=[]
                pX1C0List=[]
                pX1C1List=[]
                
                ##count unique no of words in each review
                for key in dictTrainFileNew:
                    #print(dictTrainFile[key][1])
                    eachReviewSet = set(dictTrainFileNew[key][1])       #this is a review text
                    for setValue in eachReviewSet:                  #setValue : each word in review
                        if (setValue in featureWords):
                            #increment freq
                            wordIndex=featureWords.index(setValue)
                            featureFreq[wordIndex]= featureFreq[wordIndex] + 1
                        else:
                            featureWords.append(setValue)
                            wordIndex=featureWords.index(setValue)
                            featureFreq.insert(wordIndex,1)
           
                for i in range(len(featureWords)):
                    featureList.append(feature(featureWords[i],featureFreq[i]))
                featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
          
                featureListNew = featureList[100:]
                #print(len(featureListNew)) 
                       
                ##consider features from 100 to 500
                featureListNew = featureListNew[0:500]
                #print(len(featureListNew))             
                
                for key in dictTrainFileNew:
#                    key=str(key)
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFileNew[key][1])          #->unique words in each review
                    #print(eachReviewSet)
                    for i in range(len(featureListNew)):
                        #print(featureListNew[i].word)
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                        #print(len(eachVectorListTrain))
                    #print(eachVectorListTrain)
                    vectorTupleTrain[key]=(dictTrainFileNew[key][0],eachVectorListTrain)
          
                for key in dictTestFileNew:
#                    key=str(key)
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFileNew[key][1])          #->unique words in each review
                    #print(eachReviewSet)
                    for i in range(len(featureListNew)):
                        #print(featureListNew[i].word)
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    #print(eachVectorListTest)
                    vectorTupleTest[key]=(dictTestFileNew[key][0],eachVectorListTest)
                    
                ##learn NBC model
                for ele in range(len(featureListNew)):
                    countC0 = 0
                    countC1 = 0
                    countX0C0 = 0
                    countX1C0 = 0
                    countX0C1 = 0
                    countX1C1 = 0
                    for key in vectorTupleTrain:
                        classLabel=vectorTupleTrain[key][0]
                        #print("tuple for review id : "+str(key)+" is :",vectorTupleTrain[key])
                        if(classLabel=='0'):
                            countC0 = countC0 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]    
                            if(pVectorEle==0):
                                countX0C0 = countX0C0 + 1
                            elif(pVectorEle==1):
                                countX1C0 = countX1C0 + 1
                            
                        elif(classLabel=='1'):
                            countC1 = countC1 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]
                            if(pVectorEle==0):
                                countX0C1 = countX0C1 + 1
                            elif(pVectorEle==1):
                                countX1C1 = countX1C1 + 1
                           
                    totalReviews = len(dictTrainFile)
                    probC0 = countC0/totalReviews
                    probC1 = countC1/totalReviews
                    
                    ##Laplace smoothing
                    probX0C0 = (countX0C0 + 1)/(countC0 + 2)
                    probX1C0 = (countX1C0 + 1)/(countC0 + 2)
                    probX0C1 = (countX0C1 + 1)/(countC1 + 2)
                    probX1C1 = (countX1C1 + 1)/(countC1 + 2)
                    
                    ##append each prob for each feature
                    pX0C0List.append(probX0C0)
                    pX1C0List.append(probX1C0)
                    pX0C1List.append(probX0C1)
                    pX1C1List.append(probX1C1)
                    
                ##apply the learned model to test data
                if(probC0 > probC1):
                    baselineLabel = '0'
                else:
                    baselineLabel = '1'
                
                misclassify=0
                baselineMisclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    probTestDataC0=1
                    probTestDataC1=1
                    classLabel=vectorTupleTest[key][0]
                    
                    for ele in range(len(vectorTupleTest[key][1])):
                        if(vectorTupleTest[key][1][ele]==0):
                            probTestDataC0=probTestDataC0*pX0C0List[ele]
                            probTestDataC1=probTestDataC1*pX0C1List[ele]
                        elif(vectorTupleTest[key][1][ele]==1):
                            probTestDataC0=probTestDataC0*pX1C0List[ele]
                            probTestDataC1=probTestDataC1*pX1C1List[ele]
                    probTestDataC0=probTestDataC0*probC0
                    probTestDataC1=probTestDataC1*probC1
                        
                    if(probTestDataC0 > probTestDataC1):
                        predClassLabel = '0'
                    elif(probTestDataC1 >= probTestDataC0):
                        predClassLabel = '1'
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                    
                    if(baselineLabel!=predClassLabel):
                        baselineMisclassify = baselineMisclassify + 1
                
                zeroOneLoss=misclassify/totalClassify
                baselineZeroOneLoss=baselineMisclassify/totalClassify
                
                zeroOneLossList.append(zeroOneLoss)
                baselineZeroOneLossList.append(baselineZeroOneLoss)
                
            avgZeroOneLoss=numpy.average(zeroOneLossList)
            stdZeroOneLoss=numpy.std(zeroOneLossList)
            baselineAvgZeroOneLoss=numpy.average(baselineZeroOneLossList)
            baselineStdZeroOneLoss=numpy.std(baselineZeroOneLossList)
            
            avgZeroOneLossList.append(avgZeroOneLoss)
            stdZeroOneLossList.append(stdZeroOneLoss)
            baselineAvgZeroOneLossList.append(baselineAvgZeroOneLoss)
            baselineStdZeroOneLossList.append(baselineStdZeroOneLoss)
        #end of perc list
       
#        print("avgZeroOneLossList: ",avgZeroOneLossList)
#        print("stdZeroOneLossList: ", stdZeroOneLossList)
#        print("baselineAvgZeroOneLossList: ",baselineAvgZeroOneLossList)
#        print("baselineStdZeroOneLossList: ",baselineStdZeroOneLossList)
        
        grp.figure(1)
        grp.errorbar(per, avgZeroOneLossList, stdZeroOneLossList,  marker='^',  label = "NBC 0-1 loss")
        grp.errorbar(per, baselineAvgZeroOneLossList, baselineStdZeroOneLossList,  marker='^',  label = "baseline 0-1 loss")
        grp.xlabel('Training set size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('training_size_loss_q3.png')
        
        
#        #question4
        vectorTupleTrain={}
        vectorTupleTest={}
        pX0C0List=[]
        pX0C1List=[]
        pX1C0List=[]
        pX1C1List=[]
        
        dictTrainFile = read_file('yelp_data.csv')
        total=len(dictTrainFile)
        W=[10, 50, 250, 500, 1000, 4000]
        avgZeroOneLossList=[]
        stdZeroOneLossList=[]
        baselineAvgZeroOneLossList=[]
        baselineStdZeroOneLossList=[]
        for w in W:
            zeroOneLossList=[]
            baselineZeroOneLossList=[]
            trainSize=math.ceil(50*0.01*total)
            dictTrainFileNew={}
            dictTestFileNew={}
            for i in range(10):
                keys=list(dictTrainFile.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=dictTrainFile[keys[k]]
                for k in range(trainSize+1,total):
                    dictTestFileNew[keys[k]]=dictTrainFile[keys[k]]
                
                featureWords=[]
                featureFreq=[]
                featureList=[]
                vectorTupleTrain={}
                vectorTupleTest={}
                pX0C0List=[]
                pX0C1List=[]
                pX1C0List=[]
                pX1C1List=[]
                
                ##count unique no of words in each review
                for key in dictTrainFileNew:
                    #print(dictTrainFile[key][1])
                    eachReviewSet = set(dictTrainFileNew[key][1])       #this is a review text
                    for setValue in eachReviewSet:                  #setValue : each word in review
                        if (setValue in featureWords):
                            #increment freq
                            wordIndex=featureWords.index(setValue)
                            featureFreq[wordIndex]= featureFreq[wordIndex] + 1
                        else:
                            featureWords.append(setValue)
                            wordIndex=featureWords.index(setValue)
                            featureFreq.insert(wordIndex,1)
           
                for i in range(len(featureWords)):
                    featureList.append(feature(featureWords[i],featureFreq[i]))
                featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
          
                featureListNew = featureList[100:]
                #print(len(featureListNew)) 
                       
                ##consider features from 100 to 500
                featureListNew = featureListNew[0:w]
                #print(len(featureListNew))             
                
                for key in dictTrainFileNew:
#                    key=str(key)
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFileNew[key][1])          #->unique words in each review
                    #print(eachReviewSet)
                    for i in range(len(featureListNew)):
                        #print(featureListNew[i].word)
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                        #print(len(eachVectorListTrain))
                    #print(eachVectorListTrain)
                    vectorTupleTrain[key]=(dictTrainFileNew[key][0],eachVectorListTrain)
          
                for key in dictTestFileNew:
#                    key=str(key)
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFileNew[key][1])          #->unique words in each review
                    #print(eachReviewSet)
                    for i in range(len(featureListNew)):
                        #print(featureListNew[i].word)
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    #print(eachVectorListTest)
                    vectorTupleTest[key]=(dictTestFileNew[key][0],eachVectorListTest)
                    
                ##learn NBC model
                for ele in range(len(featureListNew)):
                    countC0 = 0
                    countC1 = 0
                    countX0C0 = 0
                    countX1C0 = 0
                    countX0C1 = 0
                    countX1C1 = 0
                    for key in vectorTupleTrain:
                        classLabel=vectorTupleTrain[key][0]
                        #print("tuple for review id : "+str(key)+" is :",vectorTupleTrain[key])
                        if(classLabel=='0'):
                            countC0 = countC0 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]    
                            if(pVectorEle==0):
                                countX0C0 = countX0C0 + 1
                            elif(pVectorEle==1):
                                countX1C0 = countX1C0 + 1
                            
                        elif(classLabel=='1'):
                            countC1 = countC1 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]
                            if(pVectorEle==0):
                                countX0C1 = countX0C1 + 1
                            elif(pVectorEle==1):
                                countX1C1 = countX1C1 + 1
                           
                    totalReviews = len(dictTrainFile)
                    probC0 = countC0/totalReviews
                    probC1 = countC1/totalReviews
                    
                    ##Laplace smoothing
                    probX0C0 = (countX0C0 + 1)/(countC0 + 2)
                    probX1C0 = (countX1C0 + 1)/(countC0 + 2)
                    probX0C1 = (countX0C1 + 1)/(countC1 + 2)
                    probX1C1 = (countX1C1 + 1)/(countC1 + 2)
                    
                    ##append each prob for each feature
                    pX0C0List.append(probX0C0)
                    pX1C0List.append(probX1C0)
                    pX0C1List.append(probX0C1)
                    pX1C1List.append(probX1C1)
                    
                ##apply the learned model to test data
                if(probC0 > probC1):
                    baselineLabel = '0'
                else:
                    baselineLabel = '1'
                
                misclassify=0
                baselineMisclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    probTestDataC0=1
                    probTestDataC1=1
                    classLabel=vectorTupleTest[key][0]
                    
                    for ele in range(len(vectorTupleTest[key][1])):
                        if(vectorTupleTest[key][1][ele]==0):
                            probTestDataC0=probTestDataC0*pX0C0List[ele]
                            probTestDataC1=probTestDataC1*pX0C1List[ele]
                        elif(vectorTupleTest[key][1][ele]==1):
                            probTestDataC0=probTestDataC0*pX1C0List[ele]
                            probTestDataC1=probTestDataC1*pX1C1List[ele]
                    probTestDataC0=probTestDataC0*probC0
                    probTestDataC1=probTestDataC1*probC1
                        
                    if(probTestDataC0 > probTestDataC1):
                        predClassLabel = '0'
                    elif(probTestDataC1 >= probTestDataC0):
                        predClassLabel = '1'
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                    
                    if(baselineLabel!=predClassLabel):
                        baselineMisclassify = baselineMisclassify + 1
                
                zeroOneLoss=misclassify/totalClassify
                baselineZeroOneLoss=baselineMisclassify/totalClassify
                
                zeroOneLossList.append(zeroOneLoss)
                baselineZeroOneLossList.append(baselineZeroOneLoss)
                
            avgZeroOneLoss=numpy.average(zeroOneLossList)
            stdZeroOneLoss=numpy.std(zeroOneLossList)
            baselineAvgZeroOneLoss=numpy.average(baselineZeroOneLossList)
            baselineStdZeroOneLoss=numpy.std(baselineZeroOneLossList)
            
            avgZeroOneLossList.append(avgZeroOneLoss)
            stdZeroOneLossList.append(stdZeroOneLoss)
            baselineAvgZeroOneLossList.append(baselineAvgZeroOneLoss)
            baselineStdZeroOneLossList.append(baselineStdZeroOneLoss)
        #end of perc list
#        
#        print("avgZeroOneLossList: ",avgZeroOneLossList)
#        print("stdZeroOneLossList: ",stdZeroOneLossList)
#        print("baselineAvgZeroOneLossList: ",baselineAvgZeroOneLossList)
#        print("baselineStdZeroOneLossList: ",baselineStdZeroOneLossList)
        
        grp.figure(2)
        grp.errorbar(W, avgZeroOneLossList, stdZeroOneLossList,  marker='^',  label = "NBC 0-1 loss")
        grp.errorbar(W, baselineAvgZeroOneLossList, baselineStdZeroOneLossList,  marker='^',  label = "baseline 0-1 loss")
        grp.xlabel('feature size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('feature_size_loss_q4.png')
        
    
else:
    print("Number of arguments is not equal to three. Hence invalid input!!")
    exit()
    
                            
