from django.shortcuts import render
from django.http import HttpResponse
from django.templatetags.static import static
import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
#import xgboost as xgb
import pickle

workpath = os.path.dirname(os.path.abspath(__file__))
vehiclefile= os.path.join(workpath, 'vehiclesFinal.csv')
scalerfile= os.path.join(workpath, 'StandardScaler.sav')
modelfile= os.path.join(workpath, 'XGBoostDeploy.sav')


df=pd.read_csv(vehiclefile)
cat_cols=['manufacturer','condition','cylinders','fuel','transmission','drive','size','type','paint_color']
	
temp={}
for i in cat_cols:
	temp[i]=df[i].unique().flatten() 
# Create your views here.
def home(request):
	return render(request,'home.html',{'data':temp});

def predict(request):
        
        standardscaler= pickle.load(open(scalerfile, 'rb'))
        mymodel = pickle.load(open(modelfile, 'rb'))
        
        year=int(request.POST['year'])
        odometer=int(request.POST['odometer'])
        year_odometer=pd.DataFrame(data=[[year,odometer]],columns=['year','odometer'])
        x=standardscaler.transform(year_odometer[['year','odometer']]).flatten()
        
        testcols=['year','manufacturer','condition','cylinders','fuel','odometer','transmission','drive','size','type','paint_color']
        testdata=[int(request.POST['year']),int(request.POST['manufacturer']),int(request.POST['condition']),int(request.POST['cylinders'])
        ,int(request.POST['fuel']),int(request.POST['odometer']),int(request.POST['transmission']),int(request.POST['drive']),
        int(request.POST['size']),int(request.POST['type']),int(request.POST['paint_color'])]
        
        test=pd.DataFrame(data=[testdata],columns=testcols,dtype=None)
        pred=mymodel.predict(test)
        price=np.exp(pred[0])   
        temp['price']=price
        
        return render(request,'home.html',{'data':temp});
