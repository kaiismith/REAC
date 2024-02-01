import tensorflow as tf
import numpy as np

class GP038:
    def __init__(self):
        try:
            self.__model = tf.keras.models.load_model('GP038.keras')
            print("Model loaded")
            self.__model.summary()
        except:
            print("Load error")
    
    def predCheat(self, data):
        data = tf.convert_to_tensor(np.array(data), dtype=tf.float32)
        prediction = self.__model.predict(data)
        
        return prediction
    
class GP046:
    def __init__(self):
        try:
            self.__model = tf.keras.models.load_model('GP046.keras')
            print("Model loaded")
            self.__model.summary()
        except:
            print("Load error")
    
    def predCheat(self, data):
        data = tf.convert_to_tensor(np.array(data), dtype=tf.float32)
        prediction = self.__model.predict(data)
        
        return prediction