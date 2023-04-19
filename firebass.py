import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

def fire():
    cred = credentials.Certificate("./btech-project-5247e-firebase-adminsdk-1vanj-261c641fa6.json")
    firebase_admin.initialize_app(cred, {
        'storageBucket' : 'btech-project-5247e.appspot.com'
    })
    bucket = storage.bucket()   
    blob = bucket.blob('new_ann_model.h5')
    return blob

