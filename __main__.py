from Classifier import ClassifierDesign
from PreprocessingEngine import PreprocessingEngine
from final_model import Discriminator
import os
import glob
from scipy import misc
import numpy as np
import re
np.random.seed(2018)
PATH="C:\\Users\\Dell inspiron\\Desktop\\FinalProject"
class Main(object):
    def __init__(self):
        self;
    def forged_data(self):
        re="forged"+"*"+".jpg"
        os.chdir(PATH+"\\dataset")
        return glob.glob(re)
    def genuine_data(self):
        re="sample"+"*"+".jpg"
        os.chdir(PATH+"\\dataset")
        return glob.glob(re)
    def test_data(self):
        pass
    def generate_batch(self,id):
        os.chdir(PATH+"\\ready")
        regular_expression1="sample"+str(id)+"*.png"
        regular_expression2="forged"+str(id)+"*.png"
        a=glob.glob(regular_expression1)
        b=glob.glob(regular_expression2)
        return a+b
    def preprocess(self,filename):
        os.chdir(PATH+"\\dataset")
        img=misc.imread(filename)
        p=PreprocessingEngine(img)
        img=p.getGrayImage();
        img=p.ScaleImage();
        img=p.reduceNoise();
        img=p.eliminateBackground();
        return img
    def save_ready(self,FILENAME,img):
        os.chdir(PATH+'\\ready')
        img=img.astype(np.float32)
        misc.imsave(FILENAME.replace('.jpg','.png'), img)
    def extractImage(self,filename,folder):
        os.chdir(PATH+folder)
        img=misc.imread(filename)
        return img
    def startTraining(self,folder):
        result=list()
        os.chdir(PATH+folder)
        forged_data=glob.glob("forged"+"*"+".png")
        genuine_data=glob.glob("sample"+"*"+".png")
        complete_dataset=forged_data+genuine_data
        label=list()
        for filename in complete_dataset:
            m=re.search('\d', filename)
            label.append(filename[m.start()])
        image_list=list()
        for filename in complete_dataset:
            image_list.append(self.extractImage(filename,folder))
        x=len(image_list)
        for i in range(x):
            img=image_list[i]
            conv=ClassifierDesign(img, label[i])
            conv.train()
            result.append(conv.main_conv())
        return result

def main():
    obj=Main()
    GENUINE=obj.genuine_data()
    FORGED=obj.forged_data();
    COMPLETE_DATASET=GENUINE+FORGED
    for filename in COMPLETE_DATASET:
        a=obj.preprocess(filename)
        obj.save_ready(filename, a)
    res=obj.startTraining('\\ready')
    res=np.array(res)
    test=testing()
    model=Discriminator(res,test)
    res=model.predict()
    os.chdir(PATH+'\\test_set')
    forged_data=glob.glob("forged"+"*"+".png")
    genuine_data=glob.glob("sample"+"*"+".png")
    complete_dataset=forged_data+genuine_data
    b=np.ravel(res)
    print('Name\t\tstatus\t\tprediction')
    for i in range(len(complete_dataset)):
        status='forged'
        if 'forged' in complete_dataset[i]:
            status='forged'
        else:
            status='genuine'
        pred='forged'
        if b[i]>=0.5:
            pred='genuine'
        print(str(complete_dataset[i])+'\t'+status+'\t\t'+pred)
    

    
def testing():
    folder='\\test_set'
    test_obj=Main()
    primitive=test_obj.startTraining(folder)
    primitive=np.array(primitive)
    return primitive    
    
if __name__=="__main__":
    main();

