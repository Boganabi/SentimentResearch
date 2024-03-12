# NOTE: 
# this program does not run all the way through, but already gathers well over 40000 reviews
# i may edit later if we want more

from selenium import webdriver # pip install selenium
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.common import exceptions
from webdriver_manager.chrome import ChromeDriverManager # pip install webdriver-manager
import time
import csv

options = webdriver.ChromeOptions()
options.add_experimental_option('excludeSwitches', ['enable-logging'])
# options.add_experimental_option('detach', True)

driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

# got test data from https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis?resource=download

# URL = "https://www.amazon.com/gp/bestsellers/?ref_=nav_cs_bestsellers"
URL = "https://www.amazon.com/deals?ref_=nav_cs_gb"
# URL = "https://www.amazon.com/gp/bestsellers/automotive/ref=zg_bs_automotive_sm"
# URL = "https://www.amazon.com/INSE-Cordless-Rechargeable-Powerful-Lightweight/dp/B0BVMGBXQN/"

driver.get(URL)

# sometimes it asks for the human test so this waits for you to complete it
driver.implicitly_wait(10)

# store scraped data into this list
rows = []

fields = ["Rating", "Title", "Text"]

FILENAME = "reviews.csv"

# sidebar_num = 0
# BS_num = 0
count = 0
item_count = 0
inner_item_count = 0

SINGLE_PROD = False
DEAL_PAGE = True
accepted_ratings = ["1.0", "2.0"]

# write data to csv file
with open(FILENAME, 'a', encoding="utf-8", newline="") as csvfile: # change 'w' to 'a' to append to file instead of overwriting it
    csvwriter = csv.writer(csvfile)

    # writing the fields
    # csvwriter.writerow(fields) # remove this if appending

    # unfortunately, the review limit for each product is 100. beyond that we need a possibly charge-incurring account to get more out of each product. 
    # so i will instead look through 1000 products in a loop.

    if(SINGLE_PROD):
        # scrape a single product
        # log the url to track which ones are used
        curr_url = driver.current_url
        with open("urls.txt", "a") as f:
            f.write(curr_url + "\n\n")

        # first we get the full review page
        driver.find_element(By.XPATH, "//*[@id=\"reviews-medley-footer\"]/div[2]/a").click()
        # driver.find_element(By.XPATH, "//*[@id=\"reviews-medley-footer\"]/div[2]/a").click()

        # we scrape until we get to amazons review limit
        while(True):

            time.sleep(0.5) # if we move too fast we get a stale element exception

            # gets review container
            reviewContainer = driver.find_element(By.XPATH, "//*[@id=\"cm_cr-review_list\"]")

            # creates list of all reviews on page
            reviews = reviewContainer.find_elements(By.CLASS_NAME, "a-section.review.aok-relative")

            for review in reviews:
                # grab rating, title, and text from review
                try:
                    # since everything is in a span
                    textReview = review.find_elements(By.TAG_NAME, "span")
                    
                    # the rating scrape is different because the text here isnt actually visible
                    rating = textReview[1].get_attribute("innerText").split(" ")[0]
                    title = textReview[3].text
                    text = textReview[6].text

                    if rating in accepted_ratings:
                        rows.append([rating, title, text])

                    # print(rating, title, text)
                except exceptions.StaleElementReferenceException as e:
                    print("skipping this review")

            try:
                next_page = driver.find_element(By.XPATH, "//*[@id=\"cm_cr-pagination_bar\"]/ul/li[2]/a")
                print(next_page)
                next_page.click()
            except:
                print("found end of reviews!")

                count += 1

                # writing the data for this product
                csvwriter.writerows(rows)
                break

    else:

        # i will use one of amazons pages to search through all the products on it in a loop

        # this will loop through all the items on the first page, as the products are one step deeper
        while(True):
            if(not DEAL_PAGE):
                # grab the container for all the divs
                box = driver.find_element(By.XPATH, "//*[@id=\"grid-main-container\"]/div[3]/div")
                items = box.find_elements(By.TAG_NAME, "div")
                if(item_count > len(items) - 1):
                    # find next button and click it, then reset item count
                    next_button = driver.find_element(By.XPATH, "//*[@id=\"dealsGridLinkAnchor\"]/div/div[3]/div/ul/li[7]/a")
                    try:
                        next_button.click()
                        item_count = 0
                    except:
                        print("next button not clickable. exiting...")
                        break
                else:
                    items[item_count].find_element(By.TAG_NAME, "a").click()

                # save url to come back here later
                page_url = driver.current_url

            # loop through all products here
            prod_count = 0
            while(True):

                # save current url to come back to after getting reviews
                prod_url = driver.current_url

                if(not DEAL_PAGE):
                    # get container for all divs
                    con = driver.find_element(By.XPATH, "//*[@id=\"grid-main-container\"]/div[3]/div") 
                    prods = con.find_elements(By.TAG_NAME, "li")
                    if(prod_count > len(prods) - 1):
                        # click the next page button
                        next_prod = driver.find_element(By.XPATH, "//*[@id=\"octopus-dlp-pagination\"]/div/ul/li[5]/a")
                        try:
                            next_prod.click()
                            prod_count = 0
                        except:
                            break
                    else:
                        prods[prod_count].find_element(By.TAG_NAME, "a").click()
                else:
                    # con = driver.find_element(By.CLASS_NAME, "p13n-gridRow _cDEzb_grid-row_3Cywl")
                    prods = driver.find_elements(By.CLASS_NAME, "a-link-normal DealCard-module__linkOutlineOffset_2fc037WfeGSjbFp1CAhOUn")
                    if(prod_count > len(prods) - 1):
                        # click next page
                        next_prod = driver.find_element(By.CLASS_NAME, "a-last")
                        try:
                            next_prod.click()
                            prod_count = 0
                        except:
                            break
                    else:
                        # prods[prod_count].find_element(By.TAG_NAME, "a").click()
                        prods[prod_count].click()
                print(prod_count)
                print(len(prods))

                # log the url to track which ones are used
                curr_url = driver.current_url
                with open("urls.txt", "a") as f:
                    f.write(curr_url + "\n\n")

                # first we get the full review page
                driver.find_element(By.XPATH, "//*[@id=\"reviews-medley-footer\"]/div[2]/a").click()

                # we scrape until we get to amazons review limit
                while(True):

                    time.sleep(0.5) # if we move too fast we get a stale element exception

                    # gets review container
                    reviewContainer = driver.find_element(By.XPATH, "//*[@id=\"cm_cr-review_list\"]")

                    # creates list of all reviews on page
                    reviews = reviewContainer.find_elements(By.CLASS_NAME, "a-section.review.aok-relative")

                    for review in reviews:
                        # grab rating, title, and text from review
                        try:
                            # since everything is in a span
                            textReview = review.find_elements(By.TAG_NAME, "span")
                            
                            # the rating scrape is different because the text here isnt actually visible
                            rating = textReview[1].get_attribute("innerText").split(" ")[0]
                            title = textReview[3].text
                            text = textReview[6].text

                            if rating in accepted_ratings:
                                rows.append([rating, title, text])

                            # print(rating, title, text)
                        except exceptions.StaleElementReferenceException as e:
                            print("skipping this review")

                    try:
                        next_page = driver.find_element(By.XPATH, "//*[@id=\"cm_cr-pagination_bar\"]/ul/li[2]/a")
                        next_page.click()
                    except:
                        # print("found end of reviews!")

                        count += 1

                        # writing the data for this product
                        csvwriter.writerows(rows)

                        # go back to main products page
                        driver.get(prod_url)

                        break
                prod_count += 1
            if(not DEAL_PAGE): driver.get(page_url)
            item_count += 1

    print("end of products. total item count: ", count)

# remove punctuation and stop words for preprocessing
    # 4-5 = positive
    # 3 = neutral
    # 1-2 = negative
    # or use stars as score
    # use classification algorithm (NN [likely best], decision tree[?], or another model) 
    # compare model to NLTK and TextBlob