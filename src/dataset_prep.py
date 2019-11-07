import glob
import numpy as np
import os
import shutil
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array, array_to_img
from sklearn.preprocessing import LabelEncoder

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
        basePath = l.split()[0] #takes only the "path/imageName.jpg"
        pointIndex = basePath.find('.')
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


def getTestImageFiles(datasetPath='DeepPCB/PCBData/'):
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

    return test_files

def saveFilesToPath(files_list, dest_path):
    if os.path.exists(dest_path) and os.path.isdir(dest_path):
       shutil.rmtree(dest_path)
    
    os.mkdir(dest_path)

    for fn in files_list:
        shutil.copy(fn, dest_path)

def loadImagesFromFolder(folderPath, imgDim):
    files = glob.glob(folderPath + '/*')
    #images = [img_to_array(load_img(img, target_size=imgDim))[:,:,0:1] for img in files]
    images = [img_to_array(load_img(img, target_size=imgDim)) for img in files]
    images = np.array(images)
    labels = ["test" in fn for fn in files]
    return (images, labels)

def run_dataset_preparation(datasetPath='DeepPCB/PCBData/',train_dir='PCB_training_data', val_dir = 'PCB_validation_data', IMG_DIM=(224,224)):
	shutil.rmtree(train_dir, ignore_errors=True)
	shutil.rmtree(val_dir, ignore_errors=True)

	(training_files, validation_files) = splitTrainingValidationDataset(datasetPath)

	print('Trainining files = ', len(training_files))
	print('Validation files = ', len(validation_files))
	
	saveFilesToPath(training_files, train_dir)
	saveFilesToPath(validation_files, val_dir)

	(train_imgs, train_labels) = loadImagesFromFolder(train_dir, IMG_DIM)
	(validation_imgs, validation_labels) = loadImagesFromFolder(val_dir, IMG_DIM)
	
	train_imgs_scaled = train_imgs / 255.
	validation_imgs_scaled = validation_imgs / 255.
	
	le = LabelEncoder()
	le.fit(train_labels)
	train_labels_enc = le.transform(train_labels)
	validation_labels_enc = le.transform(validation_labels)
	
	return (train_imgs, train_imgs_scaled, train_labels, train_labels_enc), (validation_imgs, validation_imgs_scaled, validation_labels, validation_labels_enc)



