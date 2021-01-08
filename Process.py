import pandas as pd
import pickle
def getPrediction(CurrStat,workshop,FavGame,DesInOne,DesInOnee,Inclined,DesYou,CoWorkerPrec,YourFocus,jobRole,InspiredBy,SatisfyBy):
    lst=[[CurrStat,workshop,FavGame,DesInOne,DesInOnee,Inclined,DesYou,CoWorkerPrec,YourFocus,jobRole,InspiredBy,SatisfyBy]]
    df=pd.DataFrame(lst,columns=['CurrStat','workshop','FavGame','DesInOne','DesInOnee','Inclined','DesYou','CoWorkerPrec','YourFocus','jobRole','InspiredBy','SatisfyBy'])
    with open('stand_scalar', 'rb') as f:
        sc=pickle.load(f)
    with open('model', 'rb') as f:
        ppn = pickle.load(f)
    dataf=sc.transform(df)
    pred=ppn.predict(dataf)
    return str(pred[0])

