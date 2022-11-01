import os
import sys
import time
import random as rand
import math
import pandas as pd # for printing pretty tables
from fastinference.implementations.neuralnet.c.MSP430.memorysettings import NNLayer
from fastinference.implementations.neuralnet.c.MSP430.memorysettings import MSPSettings, NNSettings,BITLENGTH # used to dynamically from the module
# from memorysettings import MSPSettings, NNSettings

# TODO change to a relative / fast inference path
# todo use getattr for flexibility

ABSOLUTEPROJECTPATH = "I:\\Bibliotheken\\Programmieren\\BaechlorArbeitTools\\CompatibilityFastInference\\fastinference\\tests\\msp430Models\\"

TMPDIRNAME = "temporary"

# remove the old files of the 



if (not os.path.isdir(ABSOLUTEPROJECTPATH)):
    os.mkdir(ABSOLUTEPROJECTPATH)

# create the directory in the Path
if (not os.path.isdir(ABSOLUTEPROJECTPATH + TMPDIRNAME)):
    os.mkdir(ABSOLUTEPROJECTPATH + TMPDIRNAME)


MAINFILENAME = ABSOLUTEPROJECTPATH + "main.c"
FILENAMECPYCOMMANDS = ABSOLUTEPROJECTPATH +  TMPDIRNAME + "\\" + "memCpyCommands.txt"
INCLUDEDIRECTORY = ABSOLUTEPROJECTPATH + "include" + "\\"

FILENAMEFRAMWEIGHT = ABSOLUTEPROJECTPATH + TMPDIRNAME + "\\" + "framWeightPositions.txt"
FILENAMEFRAMWEIGHTPRETTY = ABSOLUTEPROJECTPATH + TMPDIRNAME + "\\" + "prettyframWeightPositions.txt"
FILENAMERAMWEIGHTPRETTY = ABSOLUTEPROJECTPATH + TMPDIRNAME + "\\" + "prettyramWeightPositions.txt"
FILENAMEWEIGHTS = "currentWeights"
FILENAMEWEIGHTSFULL = ABSOLUTEPROJECTPATH + FILENAMEWEIGHTS
USERNAME = "Jonathan L"

# MSPMODES = ["CPUFULLMEMORY","CPUMEMORY", "LEA", "CPULEA"] # TODO old mode idea, maybe use later
# MSPMODE = "CPUFULLMEMORY"

TRANSFORMWEIGHTSTOFIXEDPOINT = True
ONLYMEMCPYENABLED = True # global decided to use memcpy or dma (with memcpy for smaller areas) TODO make this an argument
LABELTEST = True # TODO move this to arguements/start
TESTLOOPVARNAME = "testCases"
CORRECTLABELCOUNTERNAME = "labelCounter"

class FillSection:
    # def __init__(self,partName,currentSectionName,currentSectionStart,currentSectionLength,currentSectionEnd, neededElements = [], neededElementsLength = [], dataBitLength = 32):
    def __init__(self,kind,partName,currentSectionName,currentSectionStart,currentSectionLength,currentSectionEnd,layerInd,copyFromMemory = False,dataBitLength = 32,neededElementLength = None,concattsOutput=False):
        self.partName = partName
        self.currentSectionName = currentSectionName
        self.currentSectionStart = currentSectionStart
        self.currentSectionLength = currentSectionLength # the needed values of every section
        self.currentSectionEnd = currentSectionEnd
        self.neededElementLength = neededElementLength
        self.dataBitLength = dataBitLength
        self.layerInd = layerInd # the layer index where the section came up
        self.kind = kind #f.e. "forward"
        self.copyFromMemory = copyFromMemory # 
        self.isReused = False
        self.isEmpty = False
        self.memArrayName = ""
        self.ramArrayName = ""

        self.extendedInput = 0
        self.concattsOutput = concattsOutput

        # self.neededElements = neededElements

    def hasNeededElements(self): # for the first layer to indicate that we dont need 32 * 32 values for the input copied, but 1 32 value 
        return self.neededElementLength != None

    def byteLength(self):
        # print(self.kind, self.layerInd, self.partName)
        # print("result",(self.currentSectionLength * self.dataBitLength) / 8 )
        # print("values:",self.currentSectionLength , "  ,bitlength:  " ,self.dataBitLength)
        if (self.neededElementLength == None):
            return (self.currentSectionLength * self.dataBitLength) / 8
        else:
            print("the neededElementLength != None", self.neededElementLength)
            return (self.neededElementLength * self.dataBitLength) / 8 # the length for layers that get a concatted input

    def neededElementByteLength(self):
        return (self.neededElementLength * self.dataBitLength) / 8


    def newReuseSection(self,nameAppend,curLayer,extendedInput=0):
        section = None
        if (self.concattsOutput):
            newLength = int(math.ceil(self.currentSectionLength / self.dataBitLength))
            print("the new length", newLength)
            section = FillSection(self.kind,self.partName + str(nameAppend),self.currentSectionName,self.currentSectionStart,newLength,self.currentSectionEnd,layerInd=curLayer,dataBitLength = self.dataBitLength,neededElementLength = None,concattsOutput=self.concattsOutput)
            print(self.partName + str(nameAppend))
            section.setExtendedInput(extendedInput)
            # todo edit the needed elementLength to be the newLength???
        else:
            section = FillSection(self.kind,self.partName + str(nameAppend),self.currentSectionName,self.currentSectionStart,self.currentSectionLength,self.currentSectionEnd,layerInd=curLayer,dataBitLength = self.dataBitLength,neededElementLength = None)
            section.setExtendedInput(extendedInput)
        
        # CTODO also update the memend in the case of concattenation
        section.isReused = True
        return section

    def newEmptySection(self,curLayer):
        section = FillSection(self.kind,self.partName + "empty",self.currentSectionName,self.currentSectionStart,0,self.currentSectionEnd,layerInd=curLayer,dataBitLength = self.dataBitLength,neededElementLength = None)
        section.isEmpty = True
        return section
    
    def asPandasDataframeString(self):
        return str(pd.DataFrame([[   self.partName,self.currentSectionName,self.currentSectionStart,self.currentSectionLength,self.currentSectionEnd,
                                    self.neededElementLength,self.dataBitLength,self.layerInd,self.kind,
                                    self.copyFromMemory,self.isReused,self.isEmpty,self.memArrayName,self.ramArrayName,self.concattsOutput]]
                        ,columns=["partName","currentSectionName","currentSectionStart","currentSectionLength","currentSectionEnd","neededElementLength","dataBitLength","layerInd","kind","copyFromMemory","isReused","isEmpty","memArrayName","ramArrayName","reducesOutput"]))

    def setExtendedInput(self, val): # TODO this probably doesnt need a setter
        self.extendedInput = val

def sectionListToPandaDataFrameString(sectionList : list[FillSection]):
    dataFrame = pd.DataFrame([[el.partName,el.currentSectionName,el.currentSectionStart,el.currentSectionLength,el.currentSectionEnd,
                                    el.neededElementLength,el.dataBitLength,el.layerInd,el.kind,
                                    el.copyFromMemory,el.isReused,el.isEmpty,el.memArrayName,el.ramArrayName] for el in sectionList], 
                                    columns=["partName","currentSectionName","currentSectionStart","currentSectionLength","currentSectionEnd","neededElementLength","dataBitLength","layerInd","kind","copyFromMemory","isReused","isEmpty","memArrayName","ramArrayName",]) 
    return str(dataFrame)
        
class MemSectionGenerator:
    def __init__(self,mode,maxMemory,memorySections,kind="RAM"):#,dataBitLength): # memorySections [dct("name" = name,"startInt" = Int,"lengthInt" = lengthInt,"length" = length)]
        # TODO check that length is divisble
        self.mode = mode
        self.maxMemory = maxMemory # TODO use this
        self.memorySections = memorySections
        self.curMemoryAdress = 0 # is like array indices,every section starts from 0
        self.kind = kind # or FRAM, used for testing against the need for alignment

        # find the first useable memory section
        self.curMemorySectionInd = None
        for ind,memSec in enumerate(memorySections):
            if (memSec["willUseSection"]):
                self.curMemorySectionInd = ind
                break
        
        self.fillSections = []

        self.curMemoryName = self.memorySections[self.curMemorySectionInd]["name"]

    
        self.interruptVectorString = ""

    #TODO problem if the layer doesnt fit into memory -> only 1 section overflows an we are in the last section -> we will not get the full memory copied idea check wheteher the full layer fits 
    # if the full layer doesnt fit reset the memory adress copying the values to the beginning to prevent fragmentation and make use easier
    # TODO overwrite section that will get used by the function call, to hold new data afterwards
    # TODO let BATCHNORM/STEP layer be splitable

    # curLayer = Index of the layer f.e. 0
    def fillSectionWithLayers(self,kind,names,layerValueCounts,curLayer,copyFromMemory=[],neededInputElements=None,concattsOutput=False,extendedInput=0):
        prevMemoryInd = self.curMemorySectionInd
        prevMemoryAdress = self.curMemoryAdress
        
        lastSectionConcattenation = False # in the last section indicates concatennation, if it was given as param

        for (indOuter, (name,valueCount)) in enumerate(zip(names, layerValueCounts)):
            curNeededElements = None
            if (neededInputElements != None):
                try:
                    curNeededElements = neededInputElements[indOuter]
                except IndexError:
                    curNeededElements = None

            if (curNeededElements != None): # only copy as much memory as is needed from memory
                valueCount = curNeededElements


            memCopy = False
            memValueCount = self.memorySections[self.curMemorySectionInd]["length"] / (self.memorySections[self.curMemorySectionInd]["arrayBitLength"]/8)
            startMemoryAdress = self.curMemoryAdress
            
            

            # TODO STORE for every section whether it has to be used with LEA

            if (self.curMemoryAdress + valueCount > memValueCount): # went over the in the region fittable elements

                # find the next MemorySectionInd based on being a useable region
                # if we are already on the last index ->
                if (self.curMemorySectionInd == len(self.memorySections)):
                    self.curMemorySectionInd = 0
                for ind in range(self.curMemorySectionInd,len(self.memorySections)):
                    if (self.memorySections[ind]["willUseSection"]):
                        #self.curMemorySectionInd = (self.curMemorySectionInd + 1) % len(self.memorySections) # set memory section to the next section or the first depending on how filled it is
                        self.curMemorySectionInd = ind # set memory section to the next section or the first depending on how filled it is
                        break
                self.curMemoryName = self.memorySections[self.curMemorySectionInd]["name"]
                self.curMemoryAdress = valueCount
                startMemoryAdress = 0 # reset the start memory adress
                if (self.curMemorySectionInd == prevMemoryInd and self.curMemoryAdress > prevMemoryAdress):
                    print("the memory doesn't fit")
                    sys.exit()
            else:
                self.curMemoryAdress = self.curMemoryAdress + valueCount
            try:
                memCopy = copyFromMemory[indOuter]
            except IndexError:
                memCopy = False
            
            if (curNeededElements != None):
                curNeededElements = int(curNeededElements)

            alignmentValue = 0
            if (((self.curMemoryAdress * 16) / 32) % 1 != 0.0 and self.kind == "RAM" and self.memorySections[self.curMemorySectionInd]["arrayBitLength"] == 16):
                # determine wheter the adress is in a 32bit block by divding by 32?
                alignmentValue = 1
                print("before change", self.curMemoryAdress)
                print("found  a not fitting value count in", curLayer,kind,names,valueCount,self.curMemoryAdress)

            # CTODO update memoryAdress safely, pack in function
            self.curMemoryAdress = self.curMemoryAdress + alignmentValue

            if (alignmentValue != 0):
                print("after change", self.curMemoryAdress)

                print(alignmentValue, valueCount, self.curMemoryAdress)

            # print("can i find the threshold?", ind, len(names), name, kind, startMemoryAdress)
            # if (ind == (len(names) - 1)): # this if was wrong for forward layer concattenation!!!
            #     print("is the last in the step layer?", ind, len(names), name, kind, startMemoryAdress)
            lastSectionConcattenation = concattsOutput


            newSection = FillSection(kind, name,self.curMemoryName,startMemoryAdress,valueCount,self.curMemoryAdress,curLayer
            ,dataBitLength =self.memorySections[self.curMemorySectionInd]["arrayBitLength"], copyFromMemory=memCopy
                    ,neededElementLength=curNeededElements,concattsOutput=lastSectionConcattenation)
            if (indOuter == 0): # TODO dont just use the ind = 0
                newSection.setExtendedInput(extendedInput)
            self.fillSections.append(newSection)

        # update the curMemoryAdress


    # TODO add name assignment
    def fillEmptySection(self,name): # section to stay aligned to other memory
        prevSection : FillSection = self.fillSections[len(self.fillSections) - 1].newEmptySection()
        self.fillSections.append(prevSection)


    def reusePrevSectionNNLayer(self,newName,sectionOffset=0,extendedInput=0):
        section : FillSection = self.fillSections[len(self.fillSections) - 1 - sectionOffset]
        reusedSection = section.newReuseSection("RE", section.layerInd,extendedInput) # TODO is the layerid Correct? or has it to be the next id?
        self.fillSections.append(reusedSection) # TODO skip sections that are too small


    def __generateMemCpy(self,goalName,goalOffset,sourceName,sourceOffset,dataByteLength): # TODO on the lea memory use a the msp copy command
        if (dataByteLength <= 100 or ONLYMEMCPYENABLED): # for less or equal 100 byte use the memcpy , if memcpy is enabled always use it
            # HACK if the layer is smaller than 1 byte long what to do here -> round up
            dataByteLength = math.ceil(dataByteLength) # round if there is only 1 value
            resStr = "memcpy(" + goalName + " + " + str(int(goalOffset)) + "," + sourceName + " + "  \
                    + str(int(sourceOffset)) + "," + str(int(dataByteLength)) + ");"
        else: # enable interrups and block copy mode furthermore add the interrupt vector
            needed16BitBlocks = str(int(int(dataByteLength) * 8 / 16)) # the DMA always copies in 16 bit blocks
            resStr = "dmaCopying = 1;\n"+\
            "__enable_interrupt();\n" +\
            "DMA0SZ = " + needed16BitBlocks + "; //"+ str(int(dataByteLength)) + "bytes\n" +\
            "__bis_SR_register(GIE);// allow interrupts\n"+\
            "__data20_write_long((uintptr_t) &DMA0SA, (uintptr_t)(" + sourceName + " + "+ str(int(sourceOffset)) + "));// set the source to the memarray\n"+\
            "__data20_write_long((uintptr_t) &DMA0DA, (uintptr_t)(" + goalName + " + " + str(int(goalOffset))+ "));// set the destination to the ram array\n"+\
            "DMA0CTL = DMADT_5 | DMASRCINCR_3 | DMADSTINCR_3;\n" +\
            "DMA0CTL |= DMAEN | DMAIE__ENABLE; // set the enable bit and enable interrupt bit\n"+\
            "while(dmaCopying == 1)\n"+\
            "{\n\
                DMA0CTL |= DMAREQ_1;                  // Trigger block transfer, just once for Block-Transfer\n\
            }\n" + \
            "__disable_interrupt();\n" # TODO probably useless to enable and disable interrupts 
            self.interruptVectorString = \
            "#pragma vector= DMA_VECTOR\n" +\
            "__interrupt void dmaInterrupt(void){\n" + \
            "dmaCopying = 0;\n" +\
            "DMA0CTL = DMAIE__DISABLE;// stop triggering the interrupt again\n" +\
            "}\n"
        return resStr # TODO mark globals better

    def generateCommandsNNLayer(self,ind,otherCont ,mspSettingsObj : MSPSettings):
        fillSection : FillSection = self.fillSections[ind]
        if (fillSection.layerInd != 0 and fillSection.partName == "input"): # dont copy input layers, HACK fix only dont copy input layers if we didnt move to memstart
            return ""
        if (fillSection.isEmpty):
            strResult = "#" + str(fillSection.partName) + " using the Area (Activation/OutputSection)" \
            + "of Ram in: " + str((fillSection.memArrayName)) + "," + str(int(fillSection.currentSectionStart)) + \
            " to "  + str(int(fillSection.currentSectionEnd))
            return strResult
        strResult = ""

        print("current ind",ind, fillSection.partName)
        if (fillSection.copyFromMemory): # use the memory, for example inputs will not be copied from memory # FIX the copy from memory!
            memSection : FillSection = otherCont.fillSections[ind]
            bytesToCopy = memSection.byteLength() # only copy the bytes that are really needed f.e. input is in a longer part
            
            newSectionStart = 0

            if (ind == 0):# TODO move this somewhere better f.e the fillsection creation, refactor
                #print("the searched length", mspSettingsObj.inputSize)
                # if (LABELTEST): # TODO use the labelTest to determine how to copy # make compatible with DMA Copy
                memSection.currentSectionName = mspSettingsObj.inpArrName + " + " + TESTLOOPVARNAME + " * " + str(mspSettingsObj.inputSize)
                newSectionStart = 0
                bytesToCopy = int(mspSettingsObj.neededBytesInputs / mspSettingsObj.wantedInputCount)
            else:
                newSectionStart = memSection.currentSectionStart

            strResult = self.__generateMemCpy(fillSection.currentSectionName,fillSection.currentSectionStart,memSection.currentSectionName,
                                                                    newSectionStart,bytesToCopy) + "#" + str(fillSection.partName) 
        else: # use the last section
            print("fill section wasnt copied")
            # bytesToCopy = self.currentSectionLengths[ind]
            bytesToCopy = fillSection.byteLength()
            if (fillSection.neededElementLength != None): # 
                bytesToCopy = memSection.neededElementByteLength() # only copy the bytes that are really needed f.e. input is in a longer part
            # this make no sense to call
            #strResult = self.__generateMemCpy(fillSection.currentSectionName,fillSection.currentSectionStart , fillSection.currentSectionName
                                                                    #,fillSection.currentSectionStart,bytesToCopy) + "#" + str(fillSection.partName)
        return strResult




    def generateFuncCallFillSections(self,ind,mspSettingsObj : MSPSettings):#,lastSection=False):
        resultString = ""
        curSection : FillSection= self.fillSections[ind]
        if (ind == len(self.fillSections) - 1):
            lastSection = True
        kind = curSection.kind
        
        if (kind == "forward"):
            # use the last 3 sections
            # TODO the fill Section should be directly grouped in layers
            prevSection : FillSection= self.fillSections[ind - 1]
            prevprevSection : FillSection= self.fillSections[ind - 2]
            
            
            extendedInput = prevprevSection.extendedInput # every input section safes whether it has an extended input
            
            # print(sectionListToPandaDataFrameString([prevprevSection,prevSection,curSection]))
            inputSectionStart,inputSectionName = prevprevSection.currentSectionStart, prevprevSection.currentSectionName # inputSection 
            weightSectionStart,weightSectionName = prevSection.currentSectionStart, prevSection.currentSectionName
            outputSectionStart,outputSectionName = curSection.currentSectionStart, curSection.currentSectionName
            # if (self.neededElements[ind-2] != 0):
            # if (prevprevSection.neededElementLength != 0):
            #     containedDataBitValues = prevprevSection.neededElementLength * prevprevSection.dataBitLength
            # else:
                # containedDataBitValues = int(self.currentSectionLengths[ind-2] / (self.dataBitSectionLengths[ind] / 8))

            # CTODO fix this error here!
            inputVectorCount = int(prevprevSection.currentSectionLength) # the input values will be expressed as concatted and therefore should be divided by the databitlength
                                                                                                        #   unless we are in the first layer
            #    unless we are in the first layer
            # if (self.neededElements[ind-1] != 0): # equals the output length
            #     weightVectorsCount = int(self.neededElementsLength[ind])
            # else:
            #     print("section length", self.currentSectionLengths[ind])
            #     print("dataBitSectionLengths", self.dataBitSectionLengths[ind])
            #     #weightVectorsCount = int(self.currentSectionLengths[ind] / (self.dataBitSectionLengths[ind] / 8)) 
            #     # this has to be in values! we convert the bytes to length with * 8

            # weightVectorsCount = 0
            
            weightVectorsCount = int(curSection.currentSectionLength)

            if (extendedInput == 0):
                if (curSection.dataBitLength == 32 or curSection.dataBitLength == 16): # TODO look the bitLength of the sections up
                    # print("length of input + databits", prevprevSection.currentSectionLength, prevprevSection.dataBitLength)
                    # just use XorFix = 0
                    resultString = resultString + "linearLayerForward" + str(curSection.dataBitLength) + "bitFlatPointersXorFix" + "(" + "0,"  + str(inputVectorCount) + "," + str(weightVectorsCount) + "," + inputSectionName + " + " + str(int(inputSectionStart))
                    resultString = resultString + "," + weightSectionName + " + " + str(int(weightSectionStart)) + "," + outputSectionName + " + " + str(int(outputSectionStart)) + ");"
                else: # other bit lengths are not supported
                    print("the bit length is not supported")
                    sys.exit()
            else: 
                if (curSection.dataBitLength == 32 or curSection.dataBitLength == 16): # TODO look the bitLength of the sections up
                    resultString = resultString + "linearLayerForward" + str(curSection.dataBitLength) + "bitFlatPointersXorFix" + "(" + str(extendedInput) + "," + str(inputVectorCount) + "," + str(weightVectorsCount) + "," + inputSectionName + " + " + str(int(inputSectionStart))
                    resultString = resultString + "," + weightSectionName + " + " + str(int(weightSectionStart)) + "," + outputSectionName + " + " + str(int(outputSectionStart)) + ");"
                else:
                    print("the bit length is not supported")
                    sys.exit()
            # if (prevprevSection.neededElementLength != 0):
            #     containedDataBitValues = prevprevSection.neededElementLength * prevprevSection.dataBitLength
            # else:
            #     # containedDataBitValues = int(self.currentSectionLengths[ind-2] / (self.dataBitSectionLengths[ind] / 8))
            #     containedDataBitValues = int(prevprevSection.currentSectionLength / (prevprevSection.dataBitSectionLength / 8))
            #  # TODO calculate this values, for example 2 databit values with dataBitLength of 32 -> 64 bit, so we have 2 contained values
            # # HACK fix the weightVectorsCount
            # if (self.neededElements[ind-1] != 0): # equals the output length #TODO what is this branch for?
            #     weightVectorsCount = int(self.neededElementsLength[ind])
            # else:
            #     print("section length", self.currentSectionLengths[ind])
            #     print("dataBitSectionLengths", self.dataBitSectionLengths[ind])
            #     #weightVectorsCount = int(self.currentSectionLengths[ind] / (self.dataBitSectionLengths[ind] / 8)) 
            #     # this has to be in values! we convert the bytes to length with * 8
            #     weightVectorsCount = int(self.currentSectionLengths[ind] * 8)
            #     # TODO is this correct?
            # if (extendedInput == 0):
            #     if (self.dataBitSectionLengths[ind] == 32): # TODO look the bitLength of the sections up
            #         resultString = resultString + "linearLayerForward32bitFlatPointers" + "(" + str(containedDataBitValues) + "," + str(weightVectorsCount) + "," + inputSectionName + " + " + str(int(inputSectionStart))
            #         resultString = resultString + "," + weightSectionName + " + " + str(int(weightSectionStart)) + "," + "tmp," + outputSectionName + " + " + str(int(outputSectionStart)) + ");"
            #     elif (self.dataBitSectionLengths[ind] == 16):
            #         resultString = resultString + "linearLayerForward16bitFlatPointers" + "(" + str(containedDataBitValues) + "," + str(weightVectorsCount) + "," + inputSectionName + " + " + str(int(inputSectionStart))
            #         resultString = resultString + "," + weightSectionName + " + " + str(int(weightSectionStart)) + "," + "tmp," + outputSectionName + " + " + str(int(outputSectionStart)) + ");"
            #     else: # other bit lengths are not supported
            #         print("the bit length is not supported")
            #         sys.exit()
            # else: 
            #     if (self.dataBitSectionLengths[ind] == 32): # TODO look the bitLength of the sections up
            #         print("fixing the input length")
            #         resultString = resultString + "linearLayerForward32bitFlatPointersXorFix" + "(" + str(extendedInput) + "," + str(containedDataBitValues) + "," + str(weightVectorsCount) + "," + inputSectionName + " + " + str(int(inputSectionStart))
            #         resultString = resultString + "," + weightSectionName + " + " + str(int(weightSectionStart)) + "," + "tmp," + outputSectionName + " + " + str(int(outputSectionStart)) + ");"
        elif (kind == "batchnormUnOp"): # a better BN alternative would be a merged BN with only one parameter
            # if this follows directly on a forward (probably in most cases reuse the output of that)
            prevSection : FillSection= self.fillSections[ind - 1]
            prevprevSection : FillSection= self.fillSections[ind - 2]
            
            extendedInput = prevprevSection.extendedInput

            inputSectionStart,inputSectionName = prevprevSection.currentSectionStart, prevprevSection.currentSectionName # inputSection 
            activationSectionStart,activationSectionName = prevSection.currentSectionStart, prevSection.currentSectionName
            biasOutSectionStart,biasOutSectionName = curSection.currentSectionStart, curSection.currentSectionName
            # inputSectionStart,inputSectionName = self.currentSectionStarts[ind-2][1], self.currentSectionNames[ind-2]
            # activationSectionStart,activationSectionName = self.currentSectionStarts[ind-1][1], self.currentSectionNames[ind-1]
            # biasSectionStart,biasSectionName = self.currentSectionStarts[ind][1], self.currentSectionNames[ind]
            batchNormLength = int(activationSectionStart - inputSectionStart)
            if (curSection.dataBitLength == 32 or curSection.dataBitLength == 16):
                # syntax: bNF32(input, bias, activation,result)
                lastLayerCompareString = ""
                leftSide = "" # initalize variable that hold the amount of correctly classified inputs
                if (lastSection):
                    lastLayerCompareString = "LastLayerF(" + mspSettingsObj.outArrName + " , " + str(TESTLOOPVARNAME) + " , "  # use the floating point last layer and compare there with the given labels from the label array
                    leftSide = str(CORRECTLABELCOUNTERNAME) + "[0] = " + str(CORRECTLABELCOUNTERNAME) + "[0] + " 
                    
                    # testloop var name gets resolved to a number
                else:
                    lastLayerCompareString = "("
                resultString = resultString + leftSide + "batchNorm" + str(curSection.dataBitLength) + "bit" + lastLayerCompareString + inputSectionName + " + " + str(int(inputSectionStart)) + ","
                resultString = resultString + activationSectionName + " + " + str(int(activationSectionStart)) + ","
                resultString = resultString + biasOutSectionName + " + " + str(int(biasOutSectionStart)) + "," 
                resultString = resultString + biasOutSectionName + " + " + str(int(biasOutSectionStart)) + "," + str(batchNormLength) + ");"
                #resultString = resultString + inputSectionName + " + " + str(int(inputSectionStart)) + "," + str(batchNormLength) + ");"
                # probably better to -> write the result in the activation
        elif (kind == "step"):
            
            # if this follows directly on a forward (probably in most cases reuse the output of that) : but this doesnt always follow on a forward!
            # HACK for first Layer
            if (ind - 1 < 0):
                ind = 1  #should also use the next section?
            prevSection : FillSection = self.fillSections[ind - 1] # signArray section
            prevprevSection : FillSection = self.fillSections[ind - 2] # input section
            # TODO use extended input var insetad of first layer calculation

            inputSectionStart,inputSectionName = prevprevSection.currentSectionStart, prevprevSection.currentSectionName
            signArraySectionStart,signArraySectionName = prevSection.currentSectionStart, prevSection.currentSectionName
            thresholdSectionStart,thresholdSectionName = curSection.currentSectionStart, curSection.currentSectionName
            # as only the first layer should be with a value different than a databitlength dividable this should work
            stepLength = curSection.currentSectionLength #int((thresholdSectionStart - inputSectionStart))  # the values will get concatted
            
            signIndexStart = prevprevSection.currentSectionLength % prevprevSection.dataBitLength
            if (signIndexStart == 0):
                signIndexStart = prevprevSection.dataBitLength
            
            signArrayLength = math.ceil(stepLength/16)

            # problem layers that concat have a result start -> and we dont want to waste ram with overlapping memory -> editing the step method so it return in this smaller indices -1
            # use the steplength for that
            belowDataBitLength = 0 # for every value below databitlength fills with a -1

            # onlx in first layer
            if (curSection.layerInd == 0):
                divi = stepLength / prevSection.dataBitLength
                rem = stepLength % prevSection.dataBitLength
                if (rem != 0):
                    stepLength = math.ceil(divi) * prevSection.dataBitLength
                    belowDataBitLength = int((1 - (divi)) * prevSection.dataBitLength) # the amount of values that will get filled with -1


            if (curSection.dataBitLength == 32 or curSection.dataBitLength == 16):
                # syntax: bNF32(input, bias, activation,result)
                lastLayerString = ""
                signFix = "SignFix" # string for the sign fix method call
                if (ind == curSection.layerInd): # TODO there will probably be no step layer on the last layer
                    lastLayerString = "LastLayer"
                # if (belowDataBitLength == 0): # just use the "bitbelowLength call instead"
                #     resultString = resultString + "step" + str(curSection.dataBitLength) + "bit" + lastLayerString + "(" + inputSectionName + " + " + str(int(inputSectionStart)) + ","
                #     resultString = resultString + thresholdSectionName + " + " + str(int(thresholdSectionStart)) + ","
                #     resultString = resultString + thresholdSectionName + " + " + str(int(thresholdSectionStart)) + "," + str(stepLength) + ");"
                # else:
                # TODO use arglist or something to simplify this string concattenation
                resultString = resultString + "step" + str(curSection.dataBitLength) + "bitBelowLength" + signFix + lastLayerString + "(" + inputSectionName + " + " + str(int(inputSectionStart)) + ","
                resultString = resultString + thresholdSectionName + " + " + str(int(thresholdSectionStart)) + ","
                resultString = resultString + thresholdSectionName + " + " + str(int(thresholdSectionStart)) + "," + str(stepLength) + "," + str(int(belowDataBitLength)) 
                resultString = resultString + "," + signArraySectionName + " + " + str(int(signArraySectionStart)) + "," 
                resultString = resultString + str(int(signArrayLength)) + "," + str(int(signIndexStart)-1) + ");"
                 # TODO calculate this layer names and refactor!
                #resultString = resultString + inputSectionName + " + " + str(int(inputSectionStart)) + "," + str(batchNormLength) + ");"
                # probably better to -> write the result in the activation

            
            else:
                print(str(curSection.dataBitLength) + " bit step not yet implemented")
                sys.exit()
        elif (kind == "logsoftmax"): # often used in pytorch for classification # FIX # not useful, we will just use output
            inoutSectionStart,inoutSectionName = curSection.currentSectionStart, curSection.currentSectionName
            if (curSection.dataBitLength == 32): # TODO use section databit length
                # syntax: logSoftMax32bit(length,inputValues[])
                lastLayerString = ""
                resultString = resultString + "logSoftMax" + str(curSection.dataBitLength) + "bit" + lastLayerString + "(" 
                resultString = resultString + str(int(curSection.currentSectionLength / (curSection.dataBitLength / 8))) + "," # TODO calculate his layer names and refactor!
                resultString = resultString + inoutSectionName + " + " + str(int(inoutSectionStart)) + ");"
                #resultString = resultString + inputSectionName + " + " + str(int(inputSectionStart)) + "," + str(batchNormLength) + ");"
                # probably better to -> write the result in the activation
        else:
            pass
        return resultString

    def willFitContinuousMemory(self, bytesList): # the bytes should be a list to see if all bytesParts fit
        # calculate complete remaining memory length, that is left in not filled sections
        if (len(self.currentSectionNames) == 0):
            currentSection = 0
        else:
            currentSection = self.currentSectionNames[len(self.currentSectionNames) - 1]
        startSectionInd = 0
        memoryAdress = self.curMemoryAdress # simulate filling the memory
        byteIndex = 0
        for ind,memSections in enumerate(self.memorySections): # assume "sorted" sections , find the last section in the list of already used sections
            if memSections["name"] == currentSection:
                startSectionInd = ind
        while (byteIndex <= (len(bytesList) - 1) * len(self.memorySections)):
            curBytes = bytesList[byteIndex]
            # TODO use the right self.dataBitSectionLengths[ind] but how to calculate it?
#            if (i == startSectionInd): # dont use the section itself unless?
            if (memoryAdress + curBytes / (self.dataBitLength/8) > (self.memorySections[startSectionInd]["length"]) / (self.dataBitLength/8)): # TODO only calculate bytes count if necessary
                # use the next section
                if (startSectionInd == len(self.memorySections) - 1):
                    return False # the last section Ind was reached so we cant put everything in the memory
                startSectionInd += 1
                memoryAdress = 0
            else:
                memoryAdress = memoryAdress + curBytes / (self.dataBitLength / 8)
                byteIndex = byteIndex + 1
            if (byteIndex > len(bytesList) - 1): # all bytes from the list got used
                return True
        return True

    def willFitContinuousMemorySections(self ,bytesList):
                # calculate complete remaining memory length, that is left in not filled sections
        if (len(self.currentSectionNames) == 0):
            currentSection = 0
        else:
            currentSection = self.currentSectionNames[len(self.currentSectionNames) - 1]
        startSectionInd = 0
        memoryAdress = self.curMemoryAdress # simulate filling the memory
        byteIndex = 0
        for ind,memSections in enumerate(self.memorySections): # assume "sorted" sections , find the last section in the list of already used sections
            if memSections["name"] == currentSection:
                startSectionInd = ind
        while (byteIndex <= (len(bytesList) - 1) * len(self.memorySections)):
            curBytes = bytesList[byteIndex]
            # TODO use the right self.dataBitSectionLengths[ind] but how to calculate it?
#            if (i == startSectionInd): # dont use the section itself unless?
            if (memoryAdress + curBytes / (self.dataBitLength/8) > (self.memorySections[startSectionInd]["length"]) / (self.dataBitLength/8)): # TODO only calculate bytes count if necessary
                # use the next section
                if (startSectionInd == len(self.memorySections) - 1):
                    return False # the last section Ind was reached so we cant put everything in the memory
                startSectionInd += 1
                memoryAdress = 0
            else:
                memoryAdress = memoryAdress + curBytes / (self.dataBitLength / 8)
                byteIndex = byteIndex + 1
            if (byteIndex > len(bytesList) - 1): # all bytes from the list got used
                return True
        return True
    
    # calculate whether we can fit all needed weights, inputs, outputs, in the memory Sections
    def willFitMemory(self, bytesList,dataBitLength = 32):
           # calculate complete remaining memory length, that is left in not filled sections
        # if (len(self.currentSectionNames) == 0):
        #     currentSection = 0
        # else:
        #     currentSection = self.currentSectionNames[len(self.currentSectionNames) - 1]
        currentSection = self.curMemorySectionInd
        startSectionInd = 0
        memoryAdress = self.curMemoryAdress # simulate filling the memory
        byteIndex = 0
        for ind,memSections in enumerate(self.memorySections): # assume "sorted" sections , find the last section in the list of already used sections
            if memSections["name"] == currentSection:
                startSectionInd = ind
        while (byteIndex <= (len(bytesList) - 1) * len(self.memorySections)):
            curBytes = bytesList[byteIndex]
            # TODO use the right self.dataBitSectionLengths[ind] but how to calculate it?
#            if (i == startSectionInd): # dont use the section itself unless?
            if (memoryAdress + curBytes / (dataBitLength/8) > (self.memorySections[startSectionInd]["length"]) / (self.memorySections[startSectionInd]["arrayBitLength"]/8)): # TODO only calculate bytes count if necessary
                # use the next section
                if (startSectionInd == len(self.memorySections) - 1):
                    return False # the last section Ind was reached so we cant put everything in the memory
                startSectionInd += 1
                memoryAdress = 0
            else:
                memoryAdress = memoryAdress + curBytes / (dataBitLength / 8)
                byteIndex = byteIndex + 1
            if (byteIndex > len(bytesList) - 1): # all bytes from the list got used
                return True
        return True

    def willFitMemorySections(self,bytesList,dataBitLength = 32):
        currentSection = self.curMemorySectionInd
        startSectionInd = 0
        memoryAdress = self.curMemoryAdress # simulate filling the memory
        byteIndex = 0
        for ind,memSections in enumerate(self.memorySections): # assume (by adress)sorted sections , find the last section in the list of already used sections
            if memSections["name"] == currentSection:
                startSectionInd = ind
        while (byteIndex <= (len(bytesList) - 1) * len(self.memorySections)):
            curBytes = bytesList[byteIndex]
            if (memoryAdress + curBytes / (dataBitLength/8) > (self.memorySections[startSectionInd]["length"]) / (self.memorySections[startSectionInd]["arrayBitLength"]/8)): # TODO only calculate bytes count if necessary
                # use the next section
                if (startSectionInd == len(self.memorySections) - 1):
                    return False # the last section Ind was reached so we cant put everything in the memory # propbably TODO check whether there is space for a for example smaller outputseciton
                startSectionInd += 1
                memoryAdress = 0
            else:
                memoryAdress = memoryAdress + curBytes / (dataBitLength / 8)
                byteIndex = byteIndex + 1
            if (byteIndex > len(bytesList) - 1): # all bytes from the list got used
                return True
        return True


    
    def toMemStart(self,sectionIndGoal = 0):
        section : FillSection = self.fillSections[len(self.fillSections) - 1]
        if (len(section.currentSectionLength) == 0):
            return "" # in case no section was created we shouldnt have to move memory
        lastSecMemory = section.currentSectionLength # the memory the last added section uses
        lastSecStart = section.currentSectionStart
        lastSecName = section.currentSectionName
        if (0 + lastSecMemory > self.memorySections[sectionIndGoal]["length"] / (self.memorySections[sectionIndGoal]["arrayBitLength"] / 8)):
            return "" # dont put it in the section if the section has too few memory
        self.curMemoryAdress = 0
        self.curMemorySectionInd = sectionIndGoal
        # TODO alternatively just generate a section that holds the same information as the last? does this make sense?
        stringResult = self.__generateMemCpy(self.memorySections[sectionIndGoal]["name"],0,lastSecName,lastSecStart,lastSecMemory) # transform to the right format
        self.curMemoryAdress = self.curMemoryAdress + lastSecMemory
        # store the section length of the moved section?
        # self.dataBitSectionLengths.append()
        return stringResult

    def makeFit(self,nnLayer : NNLayer):
        forwardNeededBytes = [nnLayer.forwardBytes()]
        dataBitLength = nnLayer.dataBitLength
        willFit = self.willFitMemory(forwardNeededBytes,dataBitLength) # TODO wrap this in a function or even better inside the controller
        # print("will it fit? FW ",willFit)
        gotMovedToMemStart = False
        if (not willFit):
            if (len(self.fillSections) == 0):
                print("the section was too big in layer", nnLayer.layerName)
                print("the amount of bytes was", forwardNeededBytes)
                sys.exit()
            memCpy = self.toMemStart()
            gotMovedToMemStart = True
             
            if (not self.willFitContinuousMemory(forwardNeededBytes)):
                print("cant fit the memory", "in layer",nnLayer.layerName)
                print("the amount of bytes was", forwardNeededBytes)
                sys.exit()
            if memCpy == "":
                return (gotMovedToMemStart,None)
            else:
                return (gotMovedToMemStart,memCpy)
        else:
            return (gotMovedToMemStart,None)
    
    def makeFitSection(self,nnLayer : NNLayer):
        forwardNeededByteSections : list[int] = nnLayer.forwardByteSection()
        dataBitLength = nnLayer.dataBitLength
        willFit = self.willFitMemorySections(forwardNeededByteSections,dataBitLength)
        print("will it fit? FW ",willFit)
        gotMovedToMemStart = False
        if (not willFit):
            if (len(self.fillSections) == 0):
                print("the section was too big in layer", nnLayer.layerName)
                print("the amount of bytes was", forwardNeededByteSections, "summed up", sum(forwardNeededByteSections))
                sys.exit()
            memCpy = self.toMemStart()
            gotMovedToMemStart = True
             
            if (not self.willFitContinuousMemorySections(forwardNeededByteSections)):
                print("cant fit the memory", "in layer",nnLayer.layerName)
                print("the amount of bytes was", forwardNeededByteSections)
                sys.exit()
            if memCpy == "":
                return (gotMovedToMemStart,None)
            else:
                return (gotMovedToMemStart,memCpy)
        else:
            return (gotMovedToMemStart,None)


    def getCombinedMemorySize(self): # returns the amount of indices that can be stored
        # combinedMemoryBytes = 0
        combinedMemoryValues = 0
        for ind,memorySection in enumerate(self.memorySections):
            combinedMemoryValues = combinedMemoryValues + int(memorySection["length"] / (memorySection["arrayBitLength"] / 8)) # TODO use indices to control the databitlength of each section
        return combinedMemoryValues

    def getSection(self,ind):
        return self.fillSections[ind]
    
    def getSections(self):
            return self.fillSections

    def stringSections(self):
        # name, layerInd, memSection,memStartIndex,memLength(inBytes),memEnd,arrayName
        return str([(sec.partName, sec.layerInd, sec.currentSectionStart,sec.currentSectionLength,sec.currentSectionEnd,sec.currentSectionName) for sec in self.fillSections])

    def stringSection(self,ind): # TODO fix notation
        #name,layerInd,memSection,memStartIndex,memLength,memEnd,arrayName,bitLength
        sec : FillSection = self.fillSections[ind]
        return (sec.partName, sec.layerInd, self.curMemorySectionInd,sec.currentSectionStart,sec.currentSectionLength,sec.currentSectionEnd,sec.currentSectionName,sec.dataBitLength)

    def sortedSectionContents(self): # sorted by sectionStart
        def getMemStart(a):
            return int(a[3])
        tplList = [(sec.partName, sec.layerInd, self.curMemorySectionInd,sec.currentSectionStart,sec.currentSectionLength,sec.currentSectionEnd,sec.currentSectionName,sec.dataBitLength)
                    for sec in self.fillSections]
        tplList = [tpl for tpl in tplList if tpl[0] != "empty"]
        tplList.sort(key=getMemStart)
        return tplList

    def pandasStringTable(self):
        #name,layerInd,memSection,memStartIndex,memLength,memEnd,arrayName,bitLength
        return str(pd.DataFrame([(sec.partName, sec.layerInd, self.curMemorySectionInd,sec.currentSectionStart,sec.currentSectionLength,sec.currentSectionEnd,sec.currentSectionName,sec.dataBitLength) for sec in self.fillSections],
                            columns=["name","layerInd","memSection","memStartIndex","memLength","memEnd","arrayName","bitLength"]))


    def sectionsLength(self):
        return len(self.fillSections)

    # def sectionsLength(self):
    #     return len(self.partNames)

    def getLayerIdOfSection(self,ind):
        return self.fillSections[ind].layerInd
        return self.partNames[ind][1]
        

    def getInterruptVectorString(self):
        return self.interruptVectorString

def startGeneration(mspSettingsObj : MSPSettings,nnSettingObj : NNSettings):
    # if os.path.exists(FILENAMECPYCOMMANDS):
    #     os.remove(FILENAMECPYCOMMANDS)
    # else:
    #     print("the file" , FILENAMECPYCOMMANDS , "ist not created")
    
    # remove the old files
    print("removing the old files")
    if os.path.isfile(ABSOLUTEPROJECTPATH + "include\\teste.h"):
        os.remove(ABSOLUTEPROJECTPATH + "include\\teste.h")
    if os.path.isfile(ABSOLUTEPROJECTPATH + "temporary\\framWeightPositions.txt"):
        os.remove(ABSOLUTEPROJECTPATH + "temporary\\framWeightPositions.txt")
    if os.path.isfile(ABSOLUTEPROJECTPATH + "temporary\\memCpyCommands.txt"):
        os.remove(ABSOLUTEPROJECTPATH + "temporary\\memCpyCommands.txt")
    if os.path.isfile(MAINFILENAME):
        os.remove(MAINFILENAME)
    if os.path.isfile(ABSOLUTEPROJECTPATH + FILENAMEWEIGHTS + ".c"):
        os.remove(ABSOLUTEPROJECTPATH + FILENAMEWEIGHTS + ".c")



    # print("inputlength in bits", layerInputLengths, "in bytes/kB",layerInputLengthsBytes)
    # print("outputlength in bits", layerOutputLengths, "in bytes/kB", layerOutputLengthsBytes)

    # print("count of needed weights",layerNecessaryWeights)
    # print("weights in bits",layerWeightLenghts, "in bytes/kB", layerWeightLenghtsBytes)

    # print("real adress", realRamAdress1)
    # # one layer is made up of either forward or batch norm forward
    # print("available ram(in bit):", ramSize1+ramSize2, "available memory(in bit):", memorySize)
    # assumes the goalName, sourceName are variable Names
    # assumes the dataByteLength, goalOffset and sourceOffset are Int's
    memStartCopy = [] # shows when the memcpy command gets generated

    ramController = MemSectionGenerator("MSPMODE",0,mspSettingsObj.ramSections,kind="RAM")
    framController = MemSectionGenerator("MSPMODE",0,mspSettingsObj.framSections,kind="FRAM")

    gotMovedToMemStart = False # if it got moved to the start of the memory:the memcpy to the start will be generated and the last section will not be reused, if not reuse the last memSection as in/output
    # fit the memory sections one by one on to RAM, start from the start if the end got reached


    for layerInd in range(nnSettingObj.layerCount):
        # check for the outputLength if it is one or less than the input length in Forward/Batchnorm/Step Layer we
        layer = nnSettingObj.getLayer(layerInd)
        layerType = layer.getType()
        if (layerType == "forward"):
            gotMovedToMemStart, fitResult = ramController.makeFitSection(layer)
            memStartCopy.append(fitResult)
            # print("the input length", layer.inputLength)
            # print("the weight LENGTH", layer.getWeightLength(), layer.weightValues)
            if (layerInd == 0):
                ramController.fillSectionWithLayers(layerType,["input","weights","output"],[layer.inputLength,layer.getWeightLength(),layer.outputLength],layerInd,copyFromMemory=[True,True,False],
                    neededInputElements=[layer.inputLength,None,None],extendedInput=layer.extendedInput)
                framController.fillSectionWithLayers(layerType,["inputMemory","weightsMemory","empty"],[layer.inputLength,layer.getWeightLength(),0],layerInd,neededInputElements=[layer.inputLength,None,None],extendedInput=layer.extendedInput)# the -1 to fix what?,extendedInput=layer.extendedInput?
                # print("creating the first Layer from sections")
                # print(layerType, layer.inputLength, layer.neededInputLength,"\n")
            elif (gotMovedToMemStart): #or layerInd==0):
                ramController.fillSectionWithLayers(layerType,["input","weights","output"],[layer.inputLength,layer.getWeightLength(),layer.outputLength],layerInd,copyFromMemory=[True,True,False],extendedInput=layer.extendedInput)
                framController.fillSectionWithLayers(layerType,["inputMemory","weightsMemory","empty"],[layer.inputLength,layer.getWeightLength(),0],layerInd,extendedInput=layer.extendedInput)# the -1 to fix what?,extendedInput=layer.extendedInput?
            else:
                ramController.reusePrevSectionNNLayer("inputRE",extendedInput=layer.extendedInput)
                ramController.fillSectionWithLayers(layerType,["weights","output"],[layer.getWeightLength(),layer.outputLength],layerInd,copyFromMemory=[True,False],extendedInput=layer.extendedInput)
                framController.fillSectionWithLayers(layerType,["empty","weightsMemory","empty"],[0,layer.getWeightLength(),0],layerInd,extendedInput=layer.extendedInput)
                gotMovedToMemStart = False
            # now create the command using the 3 sections
        elif (layerType == "batchnormUnOp"): # assumes that a forward layer was in front of this
            gotMovedToMemStart, fitResult = ramController.makeFitSection(layer)
            memStartCopy.append(fitResult)
            if (not gotMovedToMemStart):
                ramController.fillSectionWithLayers(layerType,["activation","biasOut"],[int(layer.getWeightLength() / 2),int(layer.getWeightLength() / 2)],layerInd,copyFromMemory=[True,True]) # HACK for weight Length that includes bias and ac,extendedInput=layer.extendedInputt
                framController.fillSectionWithLayers(layerType,["activationMemory","biasOutMemory"],[int(layer.getWeightLength() / 2),int(layer.getWeightLength() / 2)],layerInd,extendedInput=layer.extendedInput)
            else:
                ramController.reusePrevSectionNNLayer("inputBNRE",extendedInput=layer.extendedInput)
                ramController.fillSectionWithLayers(layerType,["activation","biasOut"],[int(layer.getWeightLength() / 2),int(layer.getWeightLength() / 2)],layerInd,copyFromMemory=[True,True],extendedInput=layer.extendedInput)
                framController.fillSectionWithLayers(layerType,["empty","activationMemory","biasOutMemory"],[0,int(layer.getWeightLength() / 2),int(layer.getWeightLength() / 2)],layerInd,extendedInput=layer.extendedInput)
                gotMovedToMemStart = False
        elif (layerType == "step"): # assumes that a forward layer was in front of this step activation, that combined BN and Activation
            gotMovedToMemStart, fitResult = ramController.makeFitSection(layer)
            memStartCopy.append(fitResult)
            print("does the step layer fit?", gotMovedToMemStart)
            if (not gotMovedToMemStart):
                if (layerInd == 0):
                    ramController.fillSectionWithLayers(layerType,["inputStep","signsThreshold", "threshold"],[layer.inputLength,layer.combinedSignValues(),layer.inputLength],layerInd,copyFromMemory=[True,True,True],concattsOutput=True,extendedInput=layer.extendedInput)
                    framController.fillSectionWithLayers(layerType,["inputStepMemory","signsThresholdMemory","thresholdMemory"],[0,layer.combinedSignValues(),layer.inputLength],layerInd,extendedInput=layer.extendedInput) # input length is set 0 to cpy from inp array
                else:
                    ramController.fillSectionWithLayers(layerType,["signsThreshold","threshold"],[layer.combinedSignValues(),layer.inputLength],layerInd,copyFromMemory=[True,True],concattsOutput=True,extendedInput=layer.extendedInput)
                    framController.fillSectionWithLayers(layerType,["signsThresholdMemory","thresholdMemory"],[layer.combinedSignValues(),layer.inputLength],layerInd,extendedInput=layer.extendedInput)
            else:
                ramController.reusePrevSectionNNLayer("inpuStepRE",extendedInput=layer.extendedInput)
                ramController.fillSectionWithLayers(layerType,["signsThreshold","threshold"],[layer.combinedSignValues(),layer.inputLength],layerInd,copyFromMemory=[False,True,True],concattsOutput=True,extendedInput=layer.extendedInput)
                framController.fillSectionWithLayers(layerType,["empty","signsThresholdMemory","thresholdMemory"],[0,layer.combinedSignValues(),layer.inputLength],layerInd,extendedInput=layer.extendedInput)
                gotMovedToMemStart = False
        elif (layerType == "logsoftmax"): # will probably not be used as logits max is the same here
            # neededInputBytes = nnSettingObj.getLayer(layerInd).inputBytes() # operate on the output of the last forward
            # ramController.reusePrevSection(("inputLOGSOFTMAX",str(layerInd)))
            # framController.fillSection(0,("empty",str(layerInd)))
            pass
        else:
            pass

# TODO align some things on LEA -> add a flag for 4KB fitting memory , or alternatively align everything only on lea

    print("the combined RAM memory size (in values): ", ramController.getCombinedMemorySize())
    print("the combined FRAM memory size (in values): ", framController.getCombinedMemorySize())

    print("the command solution:\n")

    commandCreateIndices = [] # store the indices where the func call has to be
    startIndex = -1 # for the amount of steps that each layer has, memcpy gets safed as step TODO make this inside the object
    # for fillSection in ramController.fillSections: #.nnLayers:
    for nnLayer in nnSettingObj.nnLayers: #.nnLayers:
        layer = nnLayer.layerName
        if (layer == "forward"):
            startIndex += 3             
        elif (layer == "batchnormUnOp"):
            startIndex += 2
        elif (layer == "step"):
            # if (startIndex == -1): # the step layer is in the first layer, so we need an input section
            #     print("using custome start")
            #     startIndex += 2 # input + output
            # else:
            #     startIndex += 1 # only the output
            if (startIndex == -1): # the step layer is in the first layer, so we need an input section
                print("using custome start")
                startIndex += 3 # input + signs + output
            else:
                startIndex += 2 # only the output and signs
        elif (layer == "logsoftmax"): # dosnt need memcpy or outputs
            startIndex += 0 # only the output + the command? 
        commandCreateIndices.append(startIndex)

    funcCallInd = 0
    commandoString = ""
    commandoString = commandoString + "\n"


    print("the indices where commands will be created", commandCreateIndices,"\n")


    for i in range(ramController.sectionsLength()):
        # layerInd = ramController.getLayerIdOfSection(i)
        # print(layerInd)
        # print("memCopyArr", memStartCopy, "already")
        commandoString = commandoString + ramController.generateCommandsNNLayer(i,framController,mspSettingsObj) + "\n"

        if (i == commandCreateIndices[funcCallInd]):

            # old version of extended input calculation
            # # determine the nnLayer of the fillSeciton  to extract the extendedInput of the layer
            # curSection : FillSection  = ramController.fillSections[i]
            # sectionLayerId = curSection.layerInd
            # curNNLayer : NNLayer = nnSettingObj.nnLayers[sectionLayerId]
            # extendedInput = curNNLayer.extendedInput

            # lastSection = (ramController.sectionsLength() - 1) == i 

            # commandoString = commandoString + ramController.generateFuncCallFillSections(i,extendedInput,lastSection=lastSection) + "\n" # TODO add the extended input to the fillSections
            commandoString = commandoString + ramController.generateFuncCallFillSections(i,mspSettingsObj) + "\n" # TODO add the extended input to the fillSections 

            funcCallInd = funcCallInd + 1
        else:
            pass


    # write the remaining content after the last loop
    with open(FILENAMECPYCOMMANDS,"a") as f:
        f.write(commandoString)
        f.write("\n")

    framPositionString = ""
    # print("the fram sections")
    for framSectionInd in range(framController.sectionsLength()):
        framPositionString = framPositionString + str(framController.stringSection(framSectionInd)) + "\n"




    with open(FILENAMEFRAMWEIGHT,"a") as f: # (gets deleted beforehand)
        f.write(str(["LayerName", "SectionStart","SectionLength", "SectionEnd","SectionName","SectionBitLength"]) + "\n")
        f.write(framPositionString)



    framPrettyPositionString = framController.pandasStringTable()
    with open(FILENAMEFRAMWEIGHTPRETTY,"w") as f:
        f.write(framPrettyPositionString)

    ramPrettyPositionString = ramController.pandasStringTable()
    with open(FILENAMERAMWEIGHTPRETTY,"w") as f:
            f.write(ramPrettyPositionString)



    # generate the array strings for the c program

#        "_iq31 " + x["name"] + "[" + str(int(x["length"] / (x["arrayBitLength"] / 8))) + \

    
    arrayRamSectionStrings = "".join([ 
        "#pragma DATA_SECTION(" + x["name"] + ",\"" + x["sectionName"] + "\")\n" + \
        "#pragma DATA_ALIGN(" + x["name"] + ",4)\n \
        " + x["usedDatatype"] + " " + x["name"] + "[" + str(int(x["length"] / (x["arrayBitLength"] / 8))) + \
        "] = " \
        + str([0 for _ in range(int(x["length"] / (x["arrayBitLength"] / 8)))]).replace("[","{").replace("]","}") \
        + "; \n\
        // used for the first RAM section \n"\
        for x in mspSettingsObj.ramSections])
    # this aligned the data and puts it in the lea section: DSPLIB_DATA(" + x["name"] + ",4) \n\ 
    # instead now uses the data align pragma

    # editedCommandoString = [for line in commandoString.replace("#", "//")]

    #TODO add way to remove Benchmark 



    # this blocks contain
    benchmarkTestCodeStart = ""
    benchmarkTestCodeEnd = ""
    # replace 
    if (LABELTEST):
        
        benchmarkTestCodeStart = "\n"\
            + "\t" + "uint16_t " + CORRECTLABELCOUNTERNAME +"[1]" + "= {0}; // use an array to prevent compiler optimization\n"\
            + "\t" + "uint16_t " + TESTLOOPVARNAME + " = " + str(mspSettingsObj.wantedInputCount) + ";\n"\
            + "\t" + "while(" + TESTLOOPVARNAME + "--){\n\t\
                \
            \
        "
        benchmarkTestCodeEnd = "\n\
            };\n"
        

    # generate the main file 
    wdtstr2 = "WDTCTL = WDTPW | WDTHOLD;	// stop watchdog timer \n" * 0 # disable the second wdtstr

    # _iq31 tmp = 0; // can be removed from the parameters \n" + \

    mainFileString = " \
    #include <msp430.h> \n\
    #include \"DSPLib.h\" \n\
    #include <stdio.h> \n\
    #include \"" + FILENAMEWEIGHTS + ".h\" \n\
    #include \"neuronalNetOperations.h\" \n"\
    + arrayRamSectionStrings +\
    "/* Benchmark cycle counts */ // \n\
    volatile uint32_t cycleCount = 0; \n\n\
    int dmaCopying = 0; //marks whether block transfer is ongoing\n\
    /** \n\
    * main.c \n\
    */ \n\
    int main(void){ \n \
        /* Disable WDT. */ \n\
        WDTCTL = WDTPW + WDTHOLD; \n\
        // disable the unused input ports to reduce the current consumption\n\
        P1DIR = 0xFF;\n\
        P1OUT = 0x00;\n\
        P2DIR = 0xFF;\n\
        P2OUT = 0x00;\n\
        P3DIR = 0xFF;\n\
        P3OUT = 0x00;\n\
        P4DIR = 0xFF;\n\
        P4OUT = 0x00;\n\
        P5DIR = 0xFF;\n\
        P5OUT = 0x00;\n\
        P6DIR = 0xFF;\n\
        P6OUT = 0x00;\n\
        P7DIR = 0xFF;\n\
        P7OUT = 0x00;\n\
        P8DIR = 0xFF;\n\
        P8OUT = 0x00;\n\
        P9DIR = 0xFF;\n\
        P9OUT = 0x00;\n\
        PADIR = 0xFF;\n\
        PAOUT = 0x00;\n\
        PBDIR = 0xFF;\n\
        PBOUT = 0x00;\n\
        PCDIR = 0xFF;\n\
        PCOUT = 0x00;\n\
        PDDIR = 0xFF;\n\
        PDOUT = 0x00;\n\
        PEDIR = 0xFF;\n\
        PEOUT = 0x00;\n\
        " + wdtstr2 +"\n\
        msp_status status; \n\
        #ifdef __MSP430_HAS_PMM__ \n\
            /* Disable GPIO power-on default high-impedance mode for FRAM devices */ \n\
            PM5CTL0 &= ~LOCKLPM5; \n\
        #endif \n"\
    + benchmarkTestCodeStart\
    + commandoString.replace("#", "//").replace("\n", "\n    ") \
    + benchmarkTestCodeEnd\
    +   "__low_power_mode_4(); // enter low energy consuming l4 \n"\
    +   "    __no_operation(); // loop forever \n"\
    + "}\n"\
    + ramController.getInterruptVectorString() + "\n"

    with open(MAINFILENAME,"w") as f:
        f.write(mainFileString)




    # sectionList = [] # list that contains: name, layerInd, memSection,memStartIndex,memLength(inBytes),memEnd,arrayName 
    # if (os.path.exists(FILENAMEFRAMWEIGHT)): 
    #     with open(FILENAMEFRAMWEIGHT,"r") as f: # this file got generated from the section creator TODO just use the objects
    #         for id,line in enumerate(f):
    #             if (id != 0): # TODO remove just for debugging    
    #                 #TODO use format
    #                 splitLine = line.replace("(","").replace(")","").replace("'","").replace("\n","").split(",")
    #                 fixedStrip = []
    #                 for el in splitLine:
    #                     fixedStrip.append(el.strip())
    #                 sectionList.append(fixedStrip)
    # # # remove the empty elements
    # # # sectionList = [x for x in sectionList if x[0] != "empty"]
    # # # def getMemStart(a):
    # # #     return int(a[3])
    # # # sectionList.sort(key=getMemStart)


    arrayNames = [x["name"] for x in mspSettingsObj.framSections]



    sectionIndices = {} # stores how far the index went
    arrNameToWeightsDict = {} # stores the resultValues TODO what are the result values?? # stores the weights in the correct array Section
    curFRAMIndex = 0
    curList = []
    useRandomWeights = False
    lastSectionIndDict = {}
    for arrName in arrayNames:
        lastSectionIndDict[arrName] = 0 # for every seciton stores how far the index already went, 

    

    # use the given weights to 

    print("the arr Names", arrayNames)

    for arrName in arrayNames: # parse through the fram Section names, that each have corresponding ram sections # TODO maybe sort by array name
        for name,layerInd,memSection,memStartIndex,memLength,memEnd,arrayName,bitLength in framController.sortedSectionContents():
            if (arrName == arrayName):
                try:
                    curFRAMIndex = sectionIndices[arrName]
                except KeyError:
                    sectionIndices[arrName] = 0
                    curFRAMIndex = 0
                # find the list on the framSection
                try:
                    curList = arrNameToWeightsDict[arrayName] # create a list reference
                except KeyError:
                    arrNameToWeightsDict[arrayName] = []
                    curList = arrNameToWeightsDict[arrayName]

                # only append weights from new sections, as there are multiple sections in a layer
                if (lastSectionIndDict[arrName] == int(layerInd)):
                    if (useRandomWeights):
                        for ind in range(int(float(memEnd)) - curFRAMIndex): # TODO fix this for neededElement layers?
                            curList.append(rand.random())
                    else:
                        # curList.extend(nnSettingObj.weights[curFRAMIndex:int(float(memEnd))])#TODO assume this are in case of a binary forward concatted values,
                        # use the previously safed weights and load them into the curValue List
                        curNNLayer : NNLayer = nnSettingObj.getLayer(int(layerInd))
                        curList.extend(curNNLayer.weightValues)  # the layerInds have to be only visited once, as the nnLayer hold all the weights already
                    lastSectionIndDict[arrName] = lastSectionIndDict[arrName] + 1
                else:
                    pass
                    # TODO add possibility to tranform values f.e. 32 bit values to 16 bit?
                sectionIndices[arrName] = int(float(curFRAMIndex)) + int(float(memLength))


# FI Absolute path # TODO make relative
    inputWeightPath = "I:\\Bibliotheken\\Programmieren\\BaechlorArbeitTools\\CompatibilityFastInference\\fastinference\\tests\\generatedModels\\testWeights\\neuralWeightsNormalizedPredicted" + str(mspSettingsObj.inputBitSize) + "bit.txt"

    firstInput = []
    inputValues = []
    lastValues = None
    labelValue = [] # placeholder for the first layer
    wantedInputs = mspSettingsObj.wantedInputCount
    with open(inputWeightPath,"r") as f:
        for row in f: # expects one value as label
            if (wantedInputs > 0):
                splitRow = row.split(",")
                inputValues.extend(splitRow[0:len(splitRow) - 1]) 
                labelValue.extend(splitRow[len(splitRow)-1:len(splitRow)]) # expect the label to be one value
                wantedInputs = wantedInputs - 1
            else: 
                # old: in the last run safe the values for the first input layer 
                # splitRow = row.split(",")
                # lastValues = splitRow[0:len(splitRow) - 1]
                # lastLabel = splitRow[len(splitRow)-1:len(splitRow)][0] # expect the label to be one value in a list
                # labelValue[0] = lastLabel
                # print("the zeroth label", labelValue[0])
                # firstInput.extend(lastValues)
                break
    try:
        curList = arrNameToWeightsDict[mspSettingsObj.outArrName] # create a list reference
    except KeyError:
        arrNameToWeightsDict[mspSettingsObj.outArrName] = []
        curList = arrNameToWeightsDict[mspSettingsObj.outArrName]
    curList.extend(labelValue)

    try:
        curList = arrNameToWeightsDict[mspSettingsObj.inpArrName] # create a list reference
    except KeyError:
        arrNameToWeightsDict[mspSettingsObj.inpArrName] = []
        curList = arrNameToWeightsDict[mspSettingsObj.inpArrName]
    curList.extend(inputValues)


    print(arrNameToWeightsDict)

    #generate the header file
    arrString = ""
    for key in mspSettingsObj.framSectionDict:
        arrString = arrString + "       extern const " + mspSettingsObj.framSectionDict[key]["usedDatatype"]  + " " + mspSettingsObj.framSectionDict[key]["name"] + "[" + \
                    str(int(mspSettingsObj.framSectionDict[key]["length"] / (mspSettingsObj.framSectionDict[key]["arrayBitLength"] / 8))) + "];\n"
                    # arrString = arrString + "       extern const" + _iq31 " + mspSettingsObj.framSectionDict[key]["name"] + "[" + \
                    # str(int(mspSettingsObj.framSectionDict[key]["length"] / (mspSettingsObj.framSectionDict[key]["arrayBitLength"] / 8))) + "];\n"


    headerString = "/* \
        * " + FILENAMEWEIGHTS + ".h \n \
        *\n \
        *  Created by a script\n \
        *  Created on: " + time.strftime("%d.%m.%Y at %H:%M:%S", time.localtime()) + "\n" + "\
        *  Author:" + USERNAME + "\n" + " \
        */\n \
        #include \"DSPLib.h\" \n \
        #include <stdint.h> \n \
        #ifndef WEIGHTS_H_ \n \
        #define WEIGHTS_H_ \n" + \
        arrString + " \
        #endif /* WEIGHTS_H_ */ \n"

    ##pragma DATA_SECTION(givenWeights, \".weightTest\") \
    #extern _iq31 givenWeights[2]; is there a way to not use const here? \
    ##pragma DATA_SECTION(givenWeights, \".weightTest\") \

    if (not os.path.isdir(INCLUDEDIRECTORY)):
        os.mkdir(INCLUDEDIRECTORY)

    with open(INCLUDEDIRECTORY + FILENAMEWEIGHTS + ".h","w") as f:
        f.write(headerString)
    
    #replace the values of the first zeroed input with the input values
    #arrNameToWeightsDict

    with open(FILENAMEWEIGHTSFULL + ".c","w") as f:
        f.write("#include \"DSPLib.h\"\n"
                "#include \"" + FILENAMEWEIGHTS + ".h\"\n")
        for key in arrNameToWeightsDict:
            # TODO file setup + custome pragmas + datatype + databitlength + float support
            f.write("#pragma DATA_SECTION(" + key + ", \"" + "." + mspSettingsObj.framSectionDict[key]["sectionName"] + "\")\n" )
            # TODO calculate the datatype depending on the section
            dataType = mspSettingsObj.framSectionDict[key]["usedDatatype"]
            f.write("const " + dataType + " " + key + "[" + str(int(mspSettingsObj.framSectionDict[key]["length"] / (mspSettingsObj.framSectionDict[key]["arrayBitLength"] / 8))) + "] = " + "{") # the compiler fills the reamining values with 0es
            list = arrNameToWeightsDict[key]
            passedCharacters = 0
            for ind,el in enumerate(list):
                # if (TRANSFORMWEIGHTSTOFIXEDPOINT):
                #     f.write("_IQ31(" + str(el) + "),") # use explicit type conversion with saturation
                # else:
                stringElement = str(el)
                passedCharacters += len(stringElement) + 1 # commata also count as characters
                # only append commata before the last position
                if (ind < len(list) - 1):
                    f.write(stringElement + ",")
                else: # dont append a commata
                    f.write(stringElement)
                if (passedCharacters >= 140):
                    passedCharacters = 0
                    f.write("\n")
            f.write("};\n")


# from memorysettings import MSPSettings, NNSettings
# mspSettings = MSPSettings()
# nnSettings = NNSettings()

# nnSettings.addNNLayer("forward",2, 5)
# nnSettings.addNNLayer("batchnormUnOp",5, 5)
# nnSettings.addNNLayer("forward",5, 5)

# startGeneration(mspSettings,nnSettings)