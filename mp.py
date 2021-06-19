#Applying Libraries
from flask import Flask, render_template,request
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


#Reading csv file
df=pd.read_csv('movie_dataset.csv')

df.drop_duplicates(subset="movie_title",keep=False,inplace=True)
df.columns

df['Sno']=np.arange(len(df))


characteristics=['Sno','movie_title','director_name','genres','actor_1_name','actor_2_name','actor_3_name']

#Data_cleaning

for a in characteristics:
    df[a]=df[a].fillna('')
    

df['movie_title']=df['movie_title'].apply(lambda x:x.replace(u'\xa0',u''))
df['movie_title']=df['movie_title'].apply(lambda x:x.strip())
df[characteristics]



def mergeCharacteristics(row):
    return row['director_name']+" "+row["genres"]+" "+row['actor_1_name']+" "+row['actor_2_name']+" "+row['actor_3_name']
df["merged"]=df.apply(mergeCharacteristics,axis=1)




cv= CountVectorizer()
count_matrix=cv.fit_transform(df['merged'])


cos_sim=cosine_similarity(count_matrix)

def get_title(index):
    return df[df.Sno== index]["movie_title"].values[0]

def get_index(title):
    return df[df.movie_title == title]["Sno"].values[0]

def recommender(movie_by_user):
    movie_index=get_index(movie_by_user)
    similar_movies=list(enumerate(cos_sim[movie_index]))
    sorted_similar_movies=sorted(similar_movies,key=lambda x:x[1],reverse=True)
    i=0
    l=[]
    for movie in sorted_similar_movies:
        x=get_title(movie[0])
        l.append(x)
        i=i+1
        print(x)
        if i>10:
            break
    return l



#print( recommender('Star Wars: Episode VII - The Force Awakens'))



    #Flask 

app = Flask(__name__)
@app.route('/')

def home():
   return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method=='GET':
        user_liking=request.args.get('movie')
    else:
        user_liking=request.form['movie']
    o=recommender(user_liking)  
    return render_template('show.html',output=o)
if __name__ == '__main__':
   app.run(host="0.0.0.0",debug=True)
