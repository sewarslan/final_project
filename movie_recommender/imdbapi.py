import requests

url = "https://imdb8.p.rapidapi.com/title/get-videos"

querystring = {"tconst":"tt0113845","limit":"25","region":"US"}

headers = {
	"X-RapidAPI-Key": "039ac7d1d2mshf957d0522af31ccp14af92jsn63ce186d655c",
	"X-RapidAPI-Host": "imdb8.p.rapidapi.com"
}

response = requests.get(url, headers=headers, params=querystring)

print(response.json())