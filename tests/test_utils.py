#!/usr/bin/env python3

import copy
from decimal import DecimalTuple
from genericpath import exists
import itertools
import sys
import os
import subprocess
import json
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier



from sklearn.tree import DecisionTreeClassifier

# help to find the local (not with pip installed) fastinference package
import pathlib
curPathParent = pathlib.Path(__file__).parent.parent
print(curPathParent)
sys.path.append(str(curPathParent)) # add the upper directory to find the simplePackage , es wurde nicht in nen STRING umgewandelt
import fastinference.Loader

from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

import tempfile
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler,MinMaxScaler,RobustScaler
from sklearn.model_selection import train_test_split
from scipy.io.arff import loadarff
import urllib.request

from fastinference.models.Ensemble import Ensemble

def download(url, filename, tmpdir = None):
    """Download the file under the given url and store it in the given tmpdir udner the given filename. If tmpdir is None, then `tempfile.gettmpdir()` will be used which is most likely /tmp on Linux systems.

    Args:
        url (str): The URL to the file which should be downloaded.
        filename (str): The name under which the downlaoded while should be stored.
        tmpdir (Str, optional): The directory in which the file should be stored. Defaults to None.

    Returns:
        str: Returns the full path under which the file is stored. 
    """
    if tmpdir is None:
        tmpdir = os.path.join(tempfile.gettempdir(), "data")

    os.makedirs(tmpdir, exist_ok=True)

    if not os.path.exists(os.path.join(tmpdir,filename)):
        print("{} not found. Downloading.".format(os.path.join(tmpdir,filename)))
        urllib.request.urlretrieve(url, os.path.join(tmpdir,filename))
    return os.path.join(tmpdir,filename)

def read_arff(path, class_name):
    """Loads the ARFF file under the given path and transforms it into a pandas dataframe. Each column which does not match class_name is copied into the pandas frame without changes. The column with the name `class_name` is renamed to `label` in the DataFrame. The behaviour of this method is undefined if the ARFF file already contains a `label` column and `class_name != 'label'`. 

    Args:
        path (str): The path to the ARFF file.
        class_name (str): The label column in the ARFF file

    Returns:
        pandas.DataFrame : A pandas dataframe containing the data from the ARFF file and an additional `label` column.
    """
    data, meta = loadarff(path)
    Xdict = {}
    for cname, ctype in zip(meta.names(), meta.types()):
        # Get the label attribute for the specific dataset:
        #   eeg: eyeDetection
        #   elec: class
        #   nomao: Class
        #   polish-bankruptcy: class
        if cname == class_name:
        #if cname in ["eyeDetection", "class",  "Class"]:
            enc = LabelEncoder()
            Xdict["label"] = enc.fit_transform(data[cname])
        else:
            Xdict[cname] = data[cname]
    return pd.DataFrame(Xdict)

def get_dataset(dataset, tmpdir = None, split = 0.3):
    """Returns XTrain, YTrain, XTest, YTest of the given dataset by name. If the dataset does not exist it will be automatically downloaded.

    Args:
        dataset (str): The name of the dataset to be returned (and downloaded if required.). Currently supports {magic, mnist, fashion, eeg}
        tmpdir (str, optional): The temporary folder to which the dataset is downloaded if it does not exist. If None then uses tempfile.gettempdir() to query for an appropriate temp folder. Defaults to None.
        split (float, optional): The applied train/test split. If the data-set comes with a pre-defined split (e.g. mnist) this value is ignored. Defaults to 0.3

    Raises:
        ValueError: Raises a ValueError if an unsupported dataset is passed as an argument

    Returns:
        XTrain, YTrain, XTest, YTest (2d np.array, np.array, 2d np.array, np.array): Returns the (N, d) train/test data and the (N, ) train/test labels where N is the number of data points and d is the number of features. 
    """

    if dataset == "magic":
        magic_path = download("http://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data", "magic.csv", tmpdir)
        df = pd.read_csv(magic_path)
        X = df.values[:,:-1].astype(np.float64)
        Y = df.values[:,-1]
        Y = np.array([0 if y == 'g' else 1 for y in Y])
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=split, random_state=42)
    elif dataset == "fashion" or dataset == "mnist":
        def load_mnist(path, kind='train',downscaleSize=(14,14)):
            # Taken from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
            import os
            import gzip
            import numpy as np

            """Load MNIST data from `path`"""
            # imagesPath = os.path.join(path,"downscaledImages%s%s.npy"%downscaleSize[0] %downscaleSize[1])
            

            labels_path = os.path.join(path,'%s-labels-idx1-ubyte.gz'% kind)
            # if (os.path.isfile(imagesPath)):
            #     memArray = np.lib.format.open_memmap(imagesPath) # load images as numpy memmap
            # else:
            images_path = os.path.join(path,'%s-images-idx3-ubyte.gz'% kind)

            with gzip.open(labels_path, 'rb') as lbpath:
                labels = np.frombuffer(lbpath.read(), dtype=np.uint8,offset=8)

            with gzip.open(images_path, 'rb') as imgpath:
                images = np.frombuffer(imgpath.read(), dtype=np.uint8, offset=16).reshape(len(labels), 784)
            


            # # TODO remove after Testing
            # # TODO only use if needed
            # print("original", (images[0:1]))
            # print("original shape", np.shape(images)) #  (600000, with 784 values
            # print("the length", len(images))
            # import torchvision.transforms.functional as torchTransforms
            # import torch
            # print(images[0])
            # print(len(images))
            # print(len(images[0]))
            # print(images.size)
            # print(images[0].size)

            scaledImages = np.array([])
            # ind = 2
            # for id,image in enumerate(images): # 60000 images in train set
            #     reshaped = torch.reshape(torch.from_numpy(image.copy()),(1,28,28)) # copy the image to supress warning
            #     resized = torchTransforms.resize(reshaped,(14,14)) # scale the image down
            #     scaled = torch.reshape(resized,(1,14*14)) # the down scaled image is flattened
                # if (id+1 % 10000 == 0):
                #     break
                # trying to find the right format
                # print("the image size", (scaled.size()))
                # print("the scaled image", scaled[0])
                # print("the scaled image2", scaled)
                # print("the scaled image3", scaled[0].numpy())
                # scaledImages = np.append(scaledImages,scaled[0].numpy(),axis=0)
                # ind = ind - 1
                # if (ind == 0):
                #     break
            # print("the scaled images")
            # print(scaledImages)
            # print("exiting for now")
            # sys.exit()
            # print("the final array", np.asarray(scaledImages))
            # print(scaledImages[0])
            # print("second")
            # print(scaledImages[1])
            # images = scaledImages
            return images, labels

        if dataset == "fashion":
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "fashion")
            else:
                out_path = os.path.join(tmpdir, "data", "fashion")

            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)
        else:
            if tmpdir is None:
                out_path = os.path.join(tempfile.gettempdir(), "data", "mnist")
            else:
                out_path = os.path.join(tmpdir, "data", "mnist")

            train_path = download("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", "train-images-idx3-ubyte.gz", out_path)
            train_path = download("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", "train-labels-idx1-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", "t10k-images-idx3-ubyte.gz", out_path)
            test_path = download("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", "t10k-labels-idx1-ubyte.gz", out_path)

        XTrain, YTrain = load_mnist(out_path, kind='train')
        XTest, YTest = load_mnist(out_path, kind='t10k')
    elif dataset == "eeg":
        eeg_path = download("https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff", "eeg.arff", tmpdir)
        

        df = read_arff(eeg_path, "eyeDetection")
        df = pd.get_dummies(df)
        df.dropna(axis=1, inplace=True)
        Y = df["label"].values.astype(np.int32)
        df = df.drop("label", axis=1)

        X = df.values.astype(np.float64)
        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=split, random_state=23)

        # edit the Train Sets to have no comma
        # XTrain = np.multiply(XTrain, 100)
        # XTrain = np.floor(XTrain)
        # YTrain = np.multiply(YTrain, 100)
        # YTrain = np.floor(YTrain)
        
        scaler1 = StandardScaler()
        #scaler1 = MinMaxScaler((-1,1))
        #scaler1 = RobustScaler()
        
        # scalerTrain = scaler1.fit(XTrain)
        # scalerTest = scaler1.fit(XTest)

        XTrain = scaler1.fit_transform(XTrain)
        XTest = scaler1.fit_transform(XTest)
        
        print("xvalues,yvalues", XTest,YTest)
        testWeightsPath = ""
        if tmpdir is None:
            testWeightsPath = os.path.join(tempfile.gettempdir(), "testWeights")
        else:
            testWeightsPath = os.path.join(tmpdir, "testWeights")
        # save test inputs for later use
        with open(os.path.join(testWeightsPath,"testMyNeuralNetWeights.txt"),"w") as f:
            for ind,valList in enumerate(XTest):
                f.write(str(valList) + ":" + str(YTest[ind]) + "\n")


        for bitCount in [16,32]:
            with open(os.path.join(testWeightsPath,"neuralWeightsNormalized" + str(bitCount) + "bit.txt"),"w") as f:
                stringVals = ""
                bitCast = ""
                if (bitCount == 16):
                    bitCast = "_Q15"
                elif (bitCount == 32):
                    bitCast = "_IQ31"
                for ind,batch in enumerate(XTest):
                    for val in batch:
                        stringVals = stringVals + bitCast + "(" + str(val) + "),"
                    stringVals = stringVals + str(YTest[ind]) + "\n"

                f.write(stringVals + "\n")
        
        # X_val = scaler.transform(X_val)
        # X_test = scaler.transform(X_test)
        # X_train, y_train = np.array(X_train), np.array(y_train)
        # X_val, y_val = np.array(X_val), np.array(y_val)
        # X_test, y_test = np.array(X_test), np.array(y_test)
    else:
        raise ValueError("Unsupported dataset provided to get_dataset in test_utils.py: {}. Currently supported are {mnist, fashion eeg, magic}".format(dataset))
        # return None, None

    return XTrain, YTrain, XTest, YTest

def make_hash(o):
    """Generates a positive hash from the given object. Does also work for tuples / dicts and lists

    Args:
        o (The object to be hashed): A positive hash value
    """
    def freeze(o):
        # if isinstance(o, tuple):
        #     return frozenset( freeze(oi) for oi in o)

        if isinstance(o,dict):
            return frozenset({ k:freeze(v) for k,v in o.items()}.items())
        elif isinstance(o,(list,tuple,set)):
            return tuple([freeze(v) for v in o])
        else: 
            return hash(str(o))   
        # return o
    
    return str(hash(freeze(o)) + sys.maxsize + 1) 

def cfg_to_str(d):
    """A simple helper functions that formats a dictionary or lists of dictionaries into  readable string by removing large numpy arrays from them. 

    Args:
        d (dict or list of dict): The dictionary or list of dictionaries to be converted into a string

    Returns:
        str: The string
    """
    _d = copy.deepcopy(d)

    if isinstance(_d, list):
        return str([cfg_to_str(di) for di in _d])
    else:
        for k in list(_d.keys()):
            v = _d[k]
            if isinstance(v, np.ndarray) and (len(v.shape) > 2 or len(v) > 5):
                del _d[k]
                #_d[k] = "np.array"
        return str(_d)

def prepare_onnxmodel(onnx_path, out_path, name, benchmark_file, implementation_type, implementation_args = {}, base_optimizer = [], base_optimizer_args = [], ensemble_optimizer = [], ensemble_optimizer_args = []):
    # print("Loading testing data")
    # df = pd.read_csv(benchmark_file)
    # y_test = df["label"].to_numpy()
    # x_test = df.drop(columns=["label"]).to_numpy()
    # print("")
    # accuracy = accuracy_score(y_test, model.predict(x_test))*100.0

    fi_model = fastinference.Loader.NeuralNet(onnx_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print("Exporting {} with {} to {}".format(
        {"implementation_type":implementation_type, **implementation_args},{n:a for n, a in zip(base_optimizer, base_optimizer_args)}, name,out_path)
    )
    if len(base_optimizer) > 0 and base_optimizer[0] is not None:
        fi_model.optimize(base_optimizer, base_optimizer_args)
    fi_model.implement(out_path, "model", "cpp.{}".format(implementation_type), **implementation_args)

    prepare_and_compile = """
    cp ./main.cpp {outpath} && 
    cp {test_file} {outpath}/ && 
    cp ./CMakeLists.txt {outpath}
    """.replace("{outpath}", out_path).replace("{test_file}", benchmark_file)
    
    subprocess.call(prepare_and_compile, shell=True)

def run_experiment(out_path, name, feature_type, label_type, benchmark_file, n_repeat = 5):
    """Compiles and executes the cpp code in the given filename using the supplied benchmark file.

    Note 1: This code requires cmake for the compilation.
    Note 2: This call will likely only work on Linux / MAC as it utilizes cp to move some files around
    
    TODO: Make it platform independent. 

    Args:
        out_path (str): Folder in which all the cpp files are located.
        name (str): The name of the function that should be tested. In most cases this is the model name
        benchmark_file (str): A *.csv file that contains the test data
        n_repeat (int, optional): The number of repetitions the experiment is repeated to get a more accurate estimation of the latency. Defaults to 5.

    Returns:
        dict: A dictionary that contains the output of the binary. It has the following fields: "accuracy", "diff accuracy", "latency [ms]", "size [Bytes]"
    """    
    
    prepare_and_compile = """
    cd {outpath} &&
    cmake . -DMODELNAME={name} -DLABEL_TYPE={label_type} -DFEATURE_TYPE={feature_type} &&
    make""".replace("{outpath}", out_path).replace("{name}", name).replace("{feature_type}", feature_type).replace("{label_type}", label_type).replace("{test_file}", benchmark_file)
    
    # TODO add compiler arguments + compiler of code composer studio + linker args

    print("Calling {}".format(prepare_and_compile))
    subprocess.call(prepare_and_compile, shell=True)

    print("Running {} {} {}".format(os.path.join(out_path, "testCode"), benchmark_file, str(n_repeat)))
    output = subprocess.check_output([
        os.path.join(out_path, "testCode"),
        benchmark_file,
        str(n_repeat)
    ]).decode(sys.stdout.encoding).strip()
    
    accuracy = output.split("\n")[-1].split(",")[0]
    diff = output.split("\n")[-1].split(",")[2]
    latency = output.split("\n")[-1].split(",")[3]
    
    return {
        "accuracy": accuracy,
        "diff accuracy": diff,
        "latency [ms]": latency,
        "size [Bytes]": os.path.getsize(os.path.join(out_path, "testCode"))
    }

def prepare_fastinference(model_path, out_path, implementation_type, implementation_args = {}, base_optimizer = [], base_optimizer_args = [], ensemble_optimizer = [], ensemble_optimizer_args = []):
    """Prepares all files for the given model and optimizations / implementations for the cpp backend.

    Note: This call will likely only work on Linux / MAC as it utilizes cp to move some files around

    TODO: Make it platform independent. 

    Args:
        model_path ([type]): The model to be generated
        out_path ([type]): The path in which all cpp files should be stored
        implementation_type (str): The cpp implementation. 
        implementation_args (dict, optional): A dictionaries of additional parameters used during implementation. Defaults to {}.
        base_optimizer (list of string, optional): A list of optimizations that are applied before implementing the model. Defaults to [].
        base_optimizer_args (list of dict, optional): A list of parameters for each optimizer. Defaults to [].
        ensemble_optimizer (list of string, optional): A list of optimizations that are applied to the ensemble. Defaults to [].
        ensemble_optimizer_args (list of dict, optional): A list of parameters for each ensemble optimizer. Defaults to [].
    """    
    fi_model = fastinference.Loader.model_from_file(model_path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if isinstance(fi_model, (Ensemble)):
        print("Exporting {} using {}:{} with {}:{} and {}:{} to {}".format(
            fi_model.name, implementation_type, cfg_to_str(implementation_args), ensemble_optimizer, cfg_to_str(ensemble_optimizer_args), base_optimizer, cfg_to_str(base_optimizer_args), out_path
        ))
        if len(ensemble_optimizer) > 0 and ensemble_optimizer[0] is not None:
            fi_model.optimize(ensemble_optimizer, ensemble_optimizer_args, base_optimizer, base_optimizer_args)
        fi_model.implement(out_path, "model", "cpp", "cpp.{}".format(implementation_type), **implementation_args)
    else:
        print("Exporting {} using {}:{} with {}:{} to {}".format(
            fi_model.name, implementation_type, cfg_to_str(implementation_args), base_optimizer, cfg_to_str(base_optimizer_args), out_path
        ))
        if len(base_optimizer) > 0 and base_optimizer[0] is not None:
            fi_model.optimize(base_optimizer, base_optimizer_args)

        # TODO add the c path and the implementation type as arguements
        # currentProgrammingLanguage = "cpp"
        currentProgrammingLanguage = "c" 
        implementation_type = implementation_type
        fi_model.implement(out_path, "model", "{}.{}".format(currentProgrammingLanguage,implementation_type), **implementation_args)
    
    # TODO use the for the programming languange fitting compiler + settings
    prepare_and_compile = """
    cp ./main.cpp {outpath} && 
    cp ./CMakeLists.txt {outpath}
    """.replace("{outpath}", out_path).replace("{name}", fi_model.name).replace("{feature_type}", "double")
    
    print("Calling {}".format(prepare_and_compile))
    subprocess.call(prepare_and_compile, shell=True)

def test_implementations(model, dataset, split, implementations, base_optimizers = [([None], [{}])], ensemble_optimizers = [([None], [{}])], out_path = ".", model_name="Model", n_repeat=5,trainedModelPath=None):

    XTrain, YTrain, XTest, YTest = get_dataset(dataset,out_path,split)
    path_to_model = None
    if (trainedModelPath == None):
        print("Loading {}".format(dataset))

        print("Fitting model")
        model.fit(XTrain, YTrain)

        print("Storing model")
        acc = 0
        if (dataset == "eeg"):
            from sklearn.metrics import zero_one_loss
            modelPrediction = model.predict(XTest).detach()
            # torch.set_printoptions(profile="full")
            # print("the result labels", YTest)
            # transformedLabels = []
            # for inp in YTest:
            #     if (inp == 0):
            #         transformedLabels.append((0,1)) # class 0 
            #     else: # inp == 1
            #         transformedLabels.append((1,0)) # class 1
            # transformedLabels = torch.tensor(transformedLabels)

            # print("predic", modelPrediction)

            zeroOneLoss = zero_one_loss(modelPrediction,YTest)
            # print("the zero one loss", zeroOneLoss)
            # torch.set_printoptions(profile="full")
            acc = (1-zeroOneLoss)*100.0 # detach to allow with this
            print("the accuracy", acc)
            djson = {
                "accuracy":acc,
                "name":model_name
            }
            
            with open(os.path.join(out_path, model_name + ".json"), "w") as outfile:  
                json.dump(djson, outfile) #, cls=NumpyEncoder

            print("running on some extra cases")
            XExtra, YExtra = XTest[0:int(len(YTest)/2)], YTest[0:int(len(YTest)/2)]
            

            # also printing the first 4 extra cases for easier debugging
            extraElementLimit = 0
            if (len(XExtra) >= 10):
                extraElementLimit = 10
            else:
                extraElementLimit = len(XExtra)
            XExtraCut, YExtraCut = XExtra[0:extraElementLimit], YExtra[0:extraElementLimit]
            print("the 10 elements")
            print("the inputs vals" ,XExtraCut)
            predicition0 = model.predictLogits(XExtraCut,printForward=True)
            print("the logits",predicition0)
            print("the labels",YExtraCut)
            print("\n\n")

            extraPrediction = model.predictLogits(XExtra).detach()
            # convert the extraPrediction to an argMax list
            extraPredictionArgMax = []
            for (a,b) in extraPrediction:
                if (a > b):
                    extraPredictionArgMax.append(0)
                else:
                    extraPredictionArgMax.append(1)
            for bits in [16,32]:
                with open(os.path.join(out_path,"testWeights","neuralWeightsNormalizedPredicted" + str(bits) + "bit.txt"), "w") as logitTests:
                    for indX,vals in enumerate(XExtra): # 
                        stringVals = ""
                        dataType = ""
                        if (bits == 32):
                            dataType = "_IQ31("
                        elif (bits == 16):
                            dataType = "_Q15("
                        else:
                            print("the sectionBitLength is not supported")
                            sys.exit()
                        for ind,val in enumerate(vals):
                            stringVals = stringVals + dataType + str(val) + "),"
#                        stringVals = stringVals + (str(YExtra[indX]) + "\n")
                        stringVals = stringVals + (str(extraPredictionArgMax[indX]) + "\n")# use this to compare that pytorch the same label
                        logitTests.write(stringVals)

        else:
            acc = accuracy_score(model.predict(XTest), YTest)*100.0
        if isinstance(model, (DecisionTreeClassifier, RidgeClassifier, QuadraticDiscriminantAnalysis, RandomForestClassifier)):
            fimodel = fastinference.Loader.model_from_sklearn(model, name = model_name, accuracy = acc)
            path_to_model = fastinference.Loader.model_to_json(fimodel, os.path.join(out_path), file_name=model_name)
            print("SK ACC:", acc)
            print("MY ACC:", accuracy_score(fimodel.predict(XTest), YTest)*100.0)
        else:
            path_to_model = model.store(out_path, acc, model_name)
    else: # model is already trained so we dont need to calc an accuracy
        path_to_model = trainedModelPath + ".onnx"
        print("the stored model accuracy is:")
        with open(trainedModelPath + ".json", "r") as outfile:
                print(json.load(outfile)) #, cls=NumpyEncoder



    print("Storing test data")
    dfTest = pd.concat([pd.DataFrame(XTest, columns=["f{}".format(i) for i in range(len(XTrain[0]))]), pd.DataFrame(YTest,columns=["label"])], axis=1)
    path_to_testfile = os.path.join(out_path, "testing.csv")
    dfTest.to_csv(path_to_testfile, header=True, index=False)
    performance = []
    print(implementations)
    print(base_optimizers)
    print(ensemble_optimizers)
    for impl, bopt, eopt in itertools.product(implementations, base_optimizers, ensemble_optimizers):
        impl_path = os.path.join(out_path, model_name + "_" + make_hash(impl) + "_" + make_hash(bopt) + "_" + make_hash(eopt))

        prepare_fastinference(path_to_model, impl_path, implementation_type = impl[0], implementation_args = impl[1], base_optimizer = bopt[0], base_optimizer_args = bopt[1], ensemble_optimizer = eopt[0], ensemble_optimizer_args = eopt[1])

        feature_type = impl[1].get("feature_type", "double")
        label_type = impl[1].get("label_type", "double")

        performance.append(
            {
                "impl":impl[0],
                #"implt_args":cfg_to_str(impl[1]),
                "base_opt":bopt[0],
                #"base_opt_args":cfg_to_str(bopt[1]),
                "opt":eopt[0],
                #"opt_args":cfg_to_str(eopt[1]),
                **run_experiment(impl_path, model_name, feature_type, label_type, path_to_testfile, n_repeat)
            }
        )

        if len(bopt[0]) == 1 and bopt[0][0] is None and len(eopt[0]) == 1 and eopt[0][0] is None and abs(float(performance[-1]["diff accuracy"])) > 1e-5:
            print("FAILED: Diff accuracy did not match in un-optimized implementation. Difference is {}".format(performance[-1]["diff accuracy"]))
            sys.exit(1)
    
    return performance