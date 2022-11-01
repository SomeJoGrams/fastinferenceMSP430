#defines constants and the memory architecture + useful functions

#MSP430 has combined 8KB of normal RAM



import sys
import pandas as pd # used for easy display of created objects
import math


def kiloByteToBit(kB):
    return kB * 1024 * 8

def bitsToKiloBytes(bits):
    return (bits/8,(bits / 8 ) / 1024)



BITLENGTH = 16
inputBitSize = BITLENGTH # size that shows how the input values are stored
sectionBitLengths = [16,16,16,16] 
#sectionBitLengths = [16,16,16,16] # NOTE would be better to wrap all of this in arguments
willUseRamSections = [False,True] # first secion is the RAM Section, second the lea section
willUseFramSection = [True,True]

class MSPSettings:
    def __init__(self):
        self.reservedMemoryRam1 = 10
        self.reservedMemoryRam2 = 10
        self.reservedMemoryStorage = kiloByteToBit(30) # assumes the code and constants in the program are 30KB large
        self.ramSize1 = kiloByteToBit(4) - self.reservedMemoryRam1 # the ram

        self.ramSize2 = kiloByteToBit(4) - self.reservedMemoryRam2 # the lea ram

        self.memorySize = kiloByteToBit(256) - self.reservedMemoryStorage# 256 is the overall memory

        # TODO add section append
        # self.ramInputStart1 = "inputStart1" # the used c adress name 
        self.realRamAdress1 = 0x1C00 # the adress converted as int, equals the amount of bits (bc of the conversion)
        self.realRamLength1 = 0x1000 # = 4096 byte = 4KB = 4096 * 8 bit = 32768 bit = 32768[bit] / 32[bit / data] = 1024 data => every adress adresses as many values as indices are skipped
        self.realRamLength1 = 0x1000 / (1.2)
         # TODO name this differently as this is the fixed value #dividing by 4 to get the 4 bit seperated addess end
        self.realRamStop1 = (self.realRamAdress1 + self.realRamLength1) / 4

        # self.ramInputStart2 = "inputStart2"
        self.realRamAdress2 = 0x2C00
        self.realRamLength2 = (0x1000 - 0x138)  #  lea maxlength - lea stack size
        self.realRamStop2 = (self.realRamAdress2 + self.realRamLength2) / 4

        # self.memoryInputStart1 = "framInputStart1"
        self.realMemoryAdress1 = 0x4000
        self.realMemoryLength1 = 0xBF80 # amount of byte 
        self.realMemoryLength1 = 0xBF80 / 2 # amount of byte # TODO calculate the available size on the chip considering the generated code/ dont randomly divide the length
        self.realMemoryStop1 = (self.realMemoryAdress1 + self.realMemoryLength1) / 4

        # self.memoryInputStart2 = "framInputStart2"
        self.realMemoryAdress2 = 0x10000
        self.realMemoryLength2 = 0x33FF8
        self.realMemoryLength2 = 0x33FF8 / 4 # TODO calculate the available size on the chip considering the generated code/ dont randomly divide the length
        self.realMemoryStop2 = self.realMemoryAdress2 + self.realMemoryLength2 / 4

        # use the hardware information to set the start values
        #try to fit as many layers and operations into the available ram as possible, maybe use a bagpack algorithm?

        # TODO create Arrays after the sections!, that allows to choose the size of those arrays!
        self.ramArrayNames = ["arr1","arr2"]
        self.ramSections =[ {"name": self.ramArrayNames[0],"area" : "RAM","sectionName" : ".myRam",
                            "start" : self.realRamAdress1,"length":self.realRamLength1, "arrayBitLength" : sectionBitLengths[0], "willUseSection" : willUseRamSections[0]},
                            {"name": self.ramArrayNames[1],"area" : "LEARAM","sectionName" : ".leaRAM",
                            "start" : self.realRamAdress2,"length":self.realRamLength2, "arrayBitLength" : sectionBitLengths[1], "willUseSection" : willUseRamSections[1]}]

      
        self.framArrayNames = ["memArr1","memArr2"]
        self.framSections =[    {"name": self.framArrayNames[0],"area" : "FRAM", "start" : self.realMemoryAdress1,
                                "sectionName" : "myFRAM","length":self.realMemoryLength1, "arrayBitLength" : sectionBitLengths[2],  "willUseSection" : willUseFramSection[0]},
                                {"name": self.framArrayNames[1],"area" : "FRAM2", "start" : self.realMemoryAdress2,
                                "sectionName": "myFRAM2", "length":self.realMemoryLength2, "arrayBitLength" : sectionBitLengths[3],  "willUseSection" : willUseFramSection[1]}]
        
        for sectionDict in self.ramSections:
            if (sectionDict["arrayBitLength"] == 32):
                sectionDict["usedDatatype"] = "_iq31"
            elif (sectionDict["arrayBitLength"] == 16):
                sectionDict["usedDatatype"] = "_q15"
            else:
                print("the sectionBitLength is not supported")
                sys.exit()
        for sectionDict in self.framSections:
            if (sectionDict["arrayBitLength"] == 32):
                sectionDict["usedDatatype"] = "_iq31"
            elif (sectionDict["arrayBitLength"] == 16):
                sectionDict["usedDatatype"] = "_q15"
            else:
                print("the sectionBitLength is not supported")
                sys.exit()
        
        self.framSectionDict = {}
        for i in self.framSections:
            self.framSectionDict[i["name"]] = i # restructure the framSection list for ease of use
        #print("fram Section Dictionary", framSectionDict)

        self.inputBitSize = inputBitSize
    
    # dynamically create output sections for every element in the last layer, 
    # should always be used for a batchnorm layer as temp storage
    # also fram section with correct labels should be created
    # that section will use 32 bit precision for batchnorm, 
    # CTODO
    # the eeg data uses an inputSize of 14 values
    def addOutputSection(self,labelCount=2,inputSize=14,inputCount=10): 
        self.outArrName = "resultLabel" # contains the indices of the correct classification, TODO use logits and compare logit precision
        self.inpArrName = "inputTestValues" # contains the test values inputs, those get copied to the start of the memArray
        self.wantedInputCount = inputCount
        self.inputSize = inputSize
        self.labelCount = labelCount
        # use full32 bit for the input and 15 bit for  the labels
        # add the arrays that will be used to store the labels + inputs
        self.framArrayNames.append(self.outArrName)
        self.framArrayNames.append(self.inpArrName) 
        neededBytesLabels = (inputCount * 16) / 8
        neededBytesInputs = (inputCount * inputSize * self.inputBitSize) / 8
        self.neededBytesInputs = neededBytesInputs
        overallNeededBytes = neededBytesLabels + neededBytesInputs
        # update the space of the older sections
        self.framSections[len(self.framSections) - 1]["length"] = self.framSections[len(self.framSections) - 1]["length"] - overallNeededBytes
        # use the space of the framSection second memArray # TODO use FRAM 1 if available
        self.framSections.append({"name": self.outArrName,"area" : "FRAM2", "start" : self.framSections[len(self.framSections) - 1]["start"] + self.framSections[len(self.framSections) - 1]["length"],
                                "sectionName" : "myFRAM2","length": neededBytesLabels, "arrayBitLength" : 16,  "willUseSection" : False,"usedDatatype": "_q15"})
        inputDataType = ""
        if (BITLENGTH == 32):
            inputDataType = "_iq31"
        elif (BITLENGTH == 16):
            inputDataType = "_q15"
        else:
            print("unsupported input datatype")
            sys.exit()
        
        self.framSections.append({"name": self.inpArrName,"area" : "FRAM2", "start" : self.framSections[len(self.framSections) - 1]["start"] + self.framSections[len(self.framSections) - 1]["length"] + neededBytesLabels,
                                "sectionName" : "myFRAM2","length": neededBytesInputs, "arrayBitLength" : BITLENGTH,  "willUseSection" : False, "usedDatatype" : inputDataType}) # TODO inputshould also be able to be read as 16 bit
        #update the FRAM section dictionary 
        self.framSectionDict = {}
        for i in self.framSections:
            self.framSectionDict[i["name"]] = i # restructure the framSectio list for ease of use


    def asPandasDataframe(self):
        combinedSections = self.ramSections + self.framSections
        # elements = [ (section["name"],section["area"],section["start"],section["length"],section["sectionName"],section["arrayBitLenth"]
        #                     ,section["willUseSection"],section["usedDataType"]) for ind,section in enumerate(combinedSections)]
        # elements = [val for key,val in combinedSections.items()]
        # return pd.DataFrame(elements,columns=["name","area","start","length","sectionName","arrayBitLenth","willUseSection","usedDatatype"])
        return pd.DataFrame(combinedSections,columns=[key for key,val in combinedSections[0].items()])

    def showSettings(self):
        print(self.asPandasDataframe())



class NNLayer():
    def __init__(self,layerName,inputLength,outputLength,dataBitLength=32,extendedInput = 0,weights=[],firstLayerInputSize=0,neededInputLength=None,signLength=0): # neededInputLength to be used for memcpy
        self.layerName = layerName
        self.inputLength = inputLength # the amount of input values
        self.neededInputLength = neededInputLength
        self.outputLength = outputLength # the amount of output values
        self.dataBitLength = dataBitLength 
        self.weightValues = weights
        self.weightNames = []
        self.extendedInput = extendedInput
        self.firstLayerInputSize = firstLayerInputSize

        #self.isSplitable = None #TODO implement splitable operations

        self.signLength = signLength #the amount of sign values(with already filled to datalength inputs) based on the outputlength in a setp layer!, 

    def valueList(self):
        return [(self.layerName + "Input",self.inputLength), (self.layerName + "Weight",len(self.weightValues)),(self.layerName + "Output", self.outputLength)]

    def getWeightLength(self): # assuming length from zero index # TODO maybe rename this
        return len(self.weightValues) - self.firstLayerInputSize # the weights will also contain the input, that shouldnt be counted as value # TODO fix -> input size maybe be more than one

    def bytes(self):
        return (self.inputBits() + self.weightBits() + self.outputBits) / 8

    def inputBits(self):
        return self.inputLength * self.dataBitLength
    
    def weightBits(self):
        return len(self.weightValues) * self.dataBitLength

    def outputBits(self):
        return self.outputLength * self.dataBitLength
    
    def inputBytes(self):
        return self.inputBits() / 8

    def weightBytes(self):
        return self.weightBits() / 8

    def outputBytes(self):
        return self.outputBits() / 8

    def signBytes(self): # only not zero in the step layer
        return (self.signLength/self.dataBitLength) * 8

    def combinedSignValues(self): # the sign values combined in databitLength pairs
        return self.signLength/self.dataBitLength

    def forwardBytes(self): # the bytes that have to be reserved to calculate a layer
        if (self.layerName == "forward"):
            return self.outputBytes() * 2 + self.weightBytes() # the input is stored in an output length
        elif (self.layerName == "batchNormUnOp"):
            return self.outputBytes() * 2 + self.weightBytes()
        elif (self.layerName == "step"):
            return self.signBytes() + self.inputBytes()
        else:
            pass
        return 0
    
    def forwardByteSection(self): # TODO FIX
        if (self.layerName == "forward"):
            return [self.outputBytes(),self.weightBytes(),self.outputBytes()] # the input is stored in an output length
        elif (self.layerName == "batchNormUnOp"):
            return [self.outputBytes() * 2,self.weightBytes()]
        elif (self.layerName == "step"):
            return [self.signBytes(),self.inputBytes()]
        else:
            pass
        return []
    
    def hasWeights(self):
        return not (len(self.weightValues) == 0)

    def getType(self):
        return self.layerName

    # def listSettings():


class NNSettings: # optional todo replace with FI model obj
    def __init__(self):

        self.nnLayers = []
        self.layerCount = 0

  

    def addNNLayer(self,layerName,inputLength,outputLength,dataBitLength=32,extendedInput=0,weights=[],neededInputLength=None): # TODO check if inputLength and outputLength are set correctly
        # TODO check supported, make sure the attributes do not get changed from the outside   
        firstLayer = False
        if (self.layerCount == 0):
            firstLayer = True
        else:
            firstLayer = False

        # determine if input and output are concatted , with that information create the nnLayers, TODO maybe in Fastinference we will need a special layer for Linear and Concatted Linear or how can that 
        # information get passed?
        signLength = 0
        if (layerName == "step"):
            # output gets guranteed concatted
            outputLength = int(math.ceil(outputLength / dataBitLength))
            # if (self.layerName == "step"):
            # for every input 1 bit,but filled up to the bit length!
            inputSizeCeil = math.ceil(inputLength / dataBitLength)
            signLength = int(inputSizeCeil * dataBitLength)

        elif (layerName == "batchnormUnOp"): # also determine here concattenation condition
            pass # TODO
        elif (layerName == "forward"):
            try:
                prevNNLayer = self.nnLayers[len(self.nnLayers) - 1]
            except IndexError:
                prevNNLayer = None
            if (prevNNLayer != None and prevNNLayer.layerName == "step"):
                inputLength = int(math.ceil(inputLength / dataBitLength))



        nnObj = NNLayer(layerName,inputLength,outputLength,dataBitLength,extendedInput,weights,firstLayerInputSize=int(firstLayer),neededInputLength=neededInputLength,signLength=signLength)
        self.nnLayers.append(nnObj)
        self.layerCount = self.layerCount + 1

    def labelCount(self): # return the amount of labels in the last layer
        lastLayer : NNLayer = self.nnLayers[len(self.nnLayers)-1]
        return lastLayer.outputLength

    def getLayer(self,ind) -> NNLayer:
        return self.nnLayers[ind]

    def layerBytes(self,ind):
        return self.nnLayers[ind].neededBytes()

    def showSettings(self):
        print("layerCount:", self.layerCount)
        print([x.getType() for x in self.nnLayers])
        print(self.nnLayerToPandasDataFrameString())

    def nnLayerToPandasDataFrameString(self):
        dataframe = pd.DataFrame([[el.layerName,el.inputLength,el.neededInputLength,el.outputLength,el.dataBitLength,el.weightValues,el.weightNames,el.extendedInput,el.firstLayerInputSize] for el in self.nnLayers],
            columns=["layerName","inputLength","neededInputLength","outputLength","dataBitLength","weightValues","weightNames","extendedInput","firstLayerInputSize"])
        return str(dataframe)

