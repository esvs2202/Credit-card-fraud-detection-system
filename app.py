import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

model = joblib.load('randomforest_model.pkl')  # loading the saved random forest classifier model.


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    For rendering results on HTML GUI

    """
    if request.method == "POST":
        f_list = [request.form.get('txn_amt'), request.form.get('txn_time'), request.form.get('merch_catg')]  # inputs

        final_features = np.array(f_list).reshape(-1, 3)

        df = pd.DataFrame(final_features,
                          columns=['amt', 'trans_hour', 'category'])
        # transforming the columns
        df['category'] = df['category'].map({'Travel': 0,
                                             'Grocery-Online': 1,
                                             'Gas-Transport': 2,
                                             'Kids-Pets': 3,
                                             'Health & Fitness': 4,
                                             'Personal care': 5,
                                             'Food & Dining': 6,
                                             'Miscellaneous-POS': 7,
                                             'Home': 8,
                                             'Grocery-POS': 9,
                                             'Entertainment': 10,
                                             'Miscellaneous-Online': 11,
                                             'Shopping-POS': 12,
                                             'Shopping-Online': 13})

        df['trans_hour'] = df['trans_hour'].astype('int')

        df['amt'] = df['amt'].astype('float')
        df['amt'] = df['amt'].apply(lambda x: np.log(x + 1))

        print(df)
        prediction = model.predict(df)
        result_dict = {0: 'Non-fraudulent', 1: 'Fraudulent'}
        result = result_dict.get(prediction[0])

        return render_template('index.html',
                               prediction_text=f"Result: Initiated transaction of ${f_list[0]} at {f_list[1]}:00 "
                                               f"hours for "
                                               f"the "
                                               f"merchant category '{f_list[2]}' = {result}")


if __name__ == "__main__":
    app.run(debug=True)
