from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField
from wtforms.validators import DataRequired
from naive_bayes_classifier import NB
import pickle

# load the model.pickle  file 
# model = pickle.load(open('model.pickle','rb'))
p_in = open("model.pkl","rb")  # it needs class def, so dont' forget -> from naive_bayes_classifier import NB
classifier = pickle.load(p_in)

app = Flask(__name__)
app.config['SECRET_KEY'] = '1234252333' # some secret key




@app.route('/', methods=['GET', 'POST'])
def index():
    pred = None
    form = reviewForm()
    if form.validate_on_submit():
        review = form.review.data
        pred = classifier.test(review) #use the classifier
    return render_template('home.html', form=form, pred=pred)

# for handling errors -------------------------------
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404
@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500
# ----------------------------------------------------

class reviewForm(FlaskForm):
    # text = TextAreaField('Text', render_kw={"rows": 70, "cols": 11})
    review = TextAreaField('Review',validators=[DataRequired()])
    submit = SubmitField('Analyze')

# ----------------------------------------------------



if __name__ == '__main__':
    app.run(debug=1)