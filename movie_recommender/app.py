from flask import Flask,render_template,request
import knn_large_ds
import nn_predictor
import pickle

with open('./pickles/movieLinkLarge.pickle', 'rb') as f:
    # Load the dictionary from the file using pickle
    movieLink = pickle.load(f)


with open('./pickles/idxtomovieid_dict.pickle', 'rb') as f:
    # Load the dictionary from the file using pickle
    idxToMovieId = pickle.load(f)

app = Flask(__name__, template_folder='templates')


@app.route("/")
def main():
    return render_template("main_page.html")

@app.route("/", methods=['GET','POST'])
def results():
    if request.method == 'POST':
        my_favorite = request.form.get('my_favorite')
        raw_recommends, reverse_mapper, movie_name = knn_recommender(my_favorite)
        nn_id_list, nn_title_list = nn_recommender(my_favorite)
        return render_template("recommends.html", raw_recommends = raw_recommends,reverse_mapper=reverse_mapper, my_favorite = my_favorite, movieLink = movieLink, idxToMovieId = idxToMovieId, nn_id_list = nn_id_list, nn_title_list = nn_title_list, movie_name = movie_name)
    else:
        return render_template("recommends.html", hata = 'Bir sorun oluştu!')

if __name__ == "__main__":
    app.run(debug=True)

def knn_recommender(my_favorite):
    return knn_large_ds.make_recommendation(my_favorite)

def nn_recommender(my_favorite):
    return nn_predictor.make_pred(my_favorite)