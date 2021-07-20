# EmotionsAPI
This is will take in URL and Image File as inputs/post requests and return the emotion of the human as the output. 

[![MIT License](https://img.shields.io/github/license/vivekboss99/EmotionsAPI)](https://github.com/vivekboss99/EmotionsAPI/blob/master/LICENSE)

To Run:
```
python manage.py makemigrations
python manage.py migrate
python manage.py runserver 7000
```
Files Summary:

- emotion_detector : Main Application Directory
  - urls.py :  Contains all the application URLs.
  - views.py : Contains all the functions which run the application.
  - cascades:
    * haarcascade_frontalface_default.xml : This is the Haar Cascade Classifier which helps in detecting the faces.
    * model.h5 : The trained neural network, saved along with its weights and biases.
- EmotionsAPI : Main Project Directory.
  - settings.py :  Contains all the registered applications and settings of the web application.
  - urls.py : Contains the URLs which link to the Main Application.

