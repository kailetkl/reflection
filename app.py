#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask


# In[2]:


app = Flask(__name__)


# In[3]:


from flask import request, render_template
import joblib

@app.route("/", methods = ["GET", "POST"])
def index():
    if request.method == "POST":
        # getting results 
        q1 = request.form.get("q1")
        q2 = request.form.get("q2")
        q3 = request.form.get("q3")
        
        # cleaning results 
        from nltk.stem import PorterStemmer
        from nltk.corpus import stopwords 
        import re
        ps = PorterStemmer() 
        all_stopwords = stopwords.words('english')
        
        q1 = re.sub('[^a-zA-Z]', ' ', q1)
        q1 = q1.lower()
        q1 = q1.split()
        q1 = [ps.stem(word) for word in q1 if not word in set(all_stopwords)]
        q1 = ' '.join(q1)
        
        q2 = re.sub('[^a-zA-Z]', ' ', q2)
        q2 = q2.lower()
        q2 = q2.split()
        q2 = [ps.stem(word) for word in q2 if not word in set(all_stopwords)]
        q2 = ' '.join(q2)
        
        q3 = re.sub('[^a-zA-Z]', ' ', q3)
        q3 = q3.lower()
        q3 = q3.split()
        q3 = [ps.stem(word) for word in q3 if not word in set(all_stopwords)]
        q3 = ' '.join(q3)
        
        # converting to list (iterable) for cv.fit_transform
        q1 = [q1]
        q2 = [q2]
        q3 = [q3]
        
        # apply cv.fit_transform
        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
        cv = CountVectorizer(max_features = 400, max_df = 1) 
        
        q1 = cv.fit_transform(q1).toarray()
        q2 = cv.fit_transform(q2).toarray()
        q3 = cv.fit_transform(q3).toarray()
        
        # load models and predict
        model1 = joblib.load("q1")
        pred1 = model1.predict(q1)
        print(pred1)
        pred1 = pred1[0]
        
        model2 = joblib.load("q2")
        pred2 = model2.predict(q2)
        print(pred2)
        pred2 = pred2[0]
        
        model3 = joblib.load("q3")
        pred3 = model3.predict(q3)
        print(pred3)
        pred3 = pred3[0]
        
        s = "The predicted category codes are " + str(pred1) + ", " + str(pred2) + " and " + str(pred3)
        return(render_template("index.html", result = s))
    else:
        return(render_template("index.html", result = "2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




