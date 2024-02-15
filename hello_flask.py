# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, jsonify, json
import anomaly

import pandas as pd

# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
 
# The route() function of the Flask class is a decorator, 
# which tells the application which URL should call 
# the associated function.
@app.route('/country/<name>')
# ‘/’ URL is bound with hello_world() function.
def hello_world(name):
    print(f'Country name incoming: {name}')
    similar_countries = get_countries(name)

    return jsonify({'similar_countries': similar_countries})

def get_countries(name):
    return anomaly.get_countries (name) 
 
# main driver function
if __name__ == '__main__':
 
    # run() method of Flask class runs the application 
    # on the local development server.
    app.run()