from django.shortcuts import render
from django.http import JsonResponse, HttpResponse
from google.oauth2 import service_account
import dialogflow_v2 as df
from keras.models import model_from_json
import numpy as np
import pandas as pd

# Create your views here.

def home(request):
    return render(request, 'home/index.html')

def chatbot(request):
    if request.method == "POST":
        credentials = service_account.Credentials.from_service_account_file('prasad-eaaf07e3eb38.json')
        session_client = df.SessionsClient(credentials=credentials)
        session = session_client.session_path("prasad-b784b","1234567")
        print("=" * 40)
        inputText = request.POST.get('msg')
        text_input = df.types.TextInput(text = inputText,language_code = "en") 
        query_input = df.types.QueryInput(text = text_input)
        response = session_client.detect_intent(session=session, query_input=query_input)
        print(response.query_result)
        print(response.query_result.intent.display_name)
        try:
            response_text = response.query_result.fulfillment_messages[0].simple_responses.simple_responses[0].text_to_speech
            intent = response.query_result.intent.display_name
        except:
            response_text = response.query_result.fulfillment_text
            intent = response.query_result.intent.display_name
        print(response_text)
    return (JsonResponse({"Response":response_text,
                          "intent":intent}))
    
def predict(request):
    if request.method == "POST":
        maritalStatus = request.POST.get('maritalStatus')
        age = request.POST.get('age')
        race = request.POST.get('race')
        primarySite = request.POST.get('primarySite')
        laterality = request.POST.get('laterality')
        surgery = request.POST.get('surgery')
        
        # Load the Model from Json File
        json_file = open('./model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        
        # Load the weights
        loaded_model.load_weights("./model.h5")
        
        # Compiling the ANN
        loaded_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        #result = loaded_model.predict(np.array([[2,1,2,5,3,1,2,3,12]]))
        result = loaded_model.predict(np.array([[maritalStatus,age,primarySite,laterality,surgery]]))
        print(result)
        np.argmax(result)
        
        dataset = pd.read_csv('seer_preprocessed_final.csv')
        X = dataset.iloc[:, [1,2,3,6,7,8,9,10]].values
        y = dataset.iloc[:, 11].values
        
        # Splitting the dataset into the Training set and Test set
        from sklearn.cross_validation import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
        
        # Fitting Multiple Linear Regression to the Training set
        from sklearn.linear_model import LinearRegression
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        
        # Predicting the Test set results
        y_pred = regressor.predict(X_test)
        
        a = regressor.predict(np.array([[maritalStatus,age,primarySite,laterality,surgery]]))
        
        print(a)
        
    return JsonResponse({"cs":str(np.argmax(result)),
                         "sm":str(int(a[0]/6))})
