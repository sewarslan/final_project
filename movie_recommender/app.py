from flask import Flask,render_template,request
import knn_large_ds


app = Flask(__name__, template_folder='templates')


@app.route("/")
def main():
    return render_template("main_page.html")

@app.route("/", methods=['GET','POST'])
def results():
    if request.method == 'POST':
        my_favorite = request.form.get('my_favorite')
        raw_recommends, reverse_mapper = recommender(my_favorite)
        return render_template("recommends.html", raw_recommends = raw_recommends,reverse_mapper=reverse_mapper, my_favorite = my_favorite)
    else:
        return render_template("recommends.html", hata = 'Bir sorun oluştu!')
    

if __name__ == "__main__":
    app.run(debug=True)

def recommender(my_favorite):
    return knn_large_ds.make_recommendation(my_favorite)