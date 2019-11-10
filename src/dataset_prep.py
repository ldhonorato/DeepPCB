import glob
import numpy as np
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
#from sklearn.preprocessing import LabelEncoder

def splitTrainingValidationDataset(datasetPath, trainingExamples=1500):
    badFiles = [] #files of pcb with defects
    goodFiles = [] #files of pcb without defects
  
    trainval_txtFile = datasetPath + 'trainval.txt'
    
    #Each line of trainval file has the structure "path/imageName.jpg path/notes.txt"
    #But, there is no imageName.jpg, its just the beginer file name for the templete image and the test image
    #There are two images: imageName_temp.jpg, imageName_test.jpg - template and test image respectively
    #The txt file has the locations and classifications of defects for the test image. 
    #We are not doing defect classification, 
    #Its enough to know that every "test" image is a defect example and every "temp" image is a good example
    f = open(trainval_txtFile, "r")
    lines = f.readlines()
    for l in lines:
        paths = l.split()
        basePath = paths[0] #takes only the "path/imageName.jpg"
        
        pointIndex = basePath.find('.') #takes only the "path/anotations.txt"
        badFiles.append(datasetPath + basePath[:pointIndex] + "_test.jpg")
        goodFiles.append(datasetPath + basePath[:pointIndex] + "_temp.jpg")
    f.close()

    assert (len(badFiles) + len(goodFiles)) > trainingExamples

    good_train = np.random.choice(goodFiles, size=(int)(trainingExamples/2), replace=False)
    bad_train = np.random.choice(badFiles, size=(int)(trainingExamples/2), replace=False)
    training_files = np.concatenate([good_train, bad_train])

    goodFiles = list(set(goodFiles) - set(good_train))
    badFiles = list(set(badFiles) - set(bad_train))
    
    #what remains is for validation
    good_val = np.array(goodFiles)
    bad_val = np.array(badFiles)
    validation_files = np.concatenate([good_val, bad_val])

    return (training_files, validation_files)


def getTestImageFiles(datasetPath='DeepPCB/PCBData/', test_dir='PCB_test_data'):
    #same process for training
    badFiles_test = [] #files of pcb with defects
    goodFiles_test = [] #files of pcb without defects

    test_txtFile = datasetPath + 'test.txt'

    f = open(test_txtFile, "r")
    lines = f.readlines()
    for l in lines:
        basePath = l.split()[0] #takes only the "path/imageName.jpg"
        pointIndex = basePath.find('.')
        badFiles_test.append(datasetPath + basePath[:pointIndex] + "_test.jpg")
        goodFiles_test.append(datasetPath + basePath[:pointIndex] + "_temp.jpg")
    f.close()

    test_files = np.concatenate([goodFiles_test, badFiles_test])

    if test_dir != '':
        saveFilesToPath(test_files, test_dir)

    return test_files

def getAnotationPath(testFilePath):
    splited_path = testFilePath.split('/')
    anotations_path = ''
    for i in range(0, len(splited_path)-2):
        anotations_path += splited_path[i] + '/'
    
    anotations_fileName = splited_path[-1]
    removeIndex = anotations_fileName.find('_test')
    anotations_fileName = anotations_fileName[0:removeIndex] + '.txt'
    anotations_path += splited_path[-2] + '_not/' + anotations_fileName
    
    return anotations_path

def getLabelsFromAnotation(anotationFilePath):
    file = open(anotationFilePath, "r")
    contents = file.readlines()
    labels = [0,0,0,0,0,0]
    for line in contents:
        line = line.strip("\n")
        idx = int(line[-1])
        labels[idx-1] = 1
    file.close()
    
    return labels

def saveFilesToPath(files_list, dest_path):
    if os.path.exists(dest_path) and os.path.isdir(dest_path):
       shutil.rmtree(dest_path)
    
    os.mkdir(dest_path)

    for fn in files_list:
        labels = [0,0,0,0,0,0]
        if 'test' in fn:
            anotations_path = getAnotationPath(fn)
            labels = getLabelsFromAnotation(anotations_path)
        
        imageName = fn.split('/')[-1]
        imageName = imageName[0:-4] + ''.join(map(str, labels)) + imageName[-4:]
        shutil.copy(fn, dest_path + '/' + imageName)

def loadImagesFromFolder(folderPath, imgDim=(224,224), multi_label=False):
    files = glob.glob(folderPath + '/*')
    #images = [img_to_array(load_img(img, target_size=imgDim))[:,:,0:1] for img in files]
    images = [img_to_array(load_img(img, target_size=imgDim)) for img in files]
    images = np.array(images)
    
    if multi_label:
        labels = []
        for fn in files:
            labels.append(list(map(int, list(fn[-10:-4]))))
    else:
        labels = [1 if "test" in fn.split('/')[-1] else 0 for fn in files]
    return (images, labels)

def number2label(numberClass):
    return np.array(numberClass).astype(bool)

def class2number(labelClass):
    return np.array(labelClass)*1 #labelClass must be boolean array

def run_dataset_preparation(datasetPath='DeepPCB/PCBData/',train_dir='PCB_training_data', val_dir = 'PCB_validation_data', IMG_DIM=(224,224), multi_label = False):
    shutil.rmtree(train_dir, ignore_errors=True)
    shutil.rmtree(val_dir, ignore_errors=True)

    (training_files, validation_files) = splitTrainingValidationDataset(datasetPath)

    saveFilesToPath(training_files, train_dir)
    saveFilesToPath(validation_files, val_dir)

    (train_imgs, train_labels) = loadImagesFromFolder(train_dir, IMG_DIM, multi_label)
    (validation_imgs, validation_labels) = loadImagesFromFolder(val_dir, IMG_DIM, multi_label)

#    train_imgs_scaled = train_imgs / 255.
#    validation_imgs_scaled = validation_imgs / 255.

    #le = LabelEncoder()
    #le.fit(train_labels)
    #train_labels_enc = le.transform(train_labels)
    #validation_labels_enc = le.transform(validation_labels)
    
#    if not multi_label:
#        train_labels_enc = class2number(train_labels)
#        validation_labels_enc = class2number(validation_labels)

    print('Trainining files = ', len(training_files))
    print('Validation files = ', len(validation_files))
    #return (train_imgs, train_imgs_scaled, train_labels, train_labels_enc), (validation_imgs, validation_imgs_scaled, validation_labels, validation_labels_enc)
    return (train_imgs, train_labels), (validation_imgs, validation_labels)



