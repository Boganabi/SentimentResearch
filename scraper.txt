import requests # pip install requests
from bs4 import BeautifulSoup # pip install bs4

# need to first search google for your user agent

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36"

URL = "https://www.amazon.com/Hydro-Flask-Around-Tumbler-Trillium/dp/B0C353845H?ref_=Oct_DLandingS_D_1cd6023e_0&th=1"

HEADERS = ({'User-Agent': USER_AGENT,
            'Accept-Language': 'en-US, en;q=0.5'})

# grabs data from url
def getData(url):
    r = requests.get(url, headers=HEADERS)
    return r.text

# pass the url into getdata
def search_URL(url):
    htmldata = getData(url)
    soup = BeautifulSoup(htmldata, 'html.parser')
    return (soup)

# driver code

# grabs the entire HTML page
soup = search_URL(URL)

# grab all the reviews
reviewContainer = soup.find('div', id="cm-cr-dp-review-list")

for review in reviewContainer:
    # grab the star rating, title, and review text
    rating = review.find('span', class_="a-icon-alt").get_text().split(" ")[0]
    title = review.find_all('span')[3].get_text()
    text = review.find('div', class_="a-expander-content reviewText review-text-content a-expander-partial-collapse-content").span.get_text()

    print(rating, title, text)
    print()