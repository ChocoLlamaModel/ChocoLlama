import requests
from bs4 import BeautifulSoup
import os
import argparse
import sys

# constants
BASE_URL = "https://www.gutenberg.org/"
TOP_100 = BASE_URL + "browse/scores/top"

DEFAULT_DATA_DIR = os.pardir


class Gutenberg_Scraper():
    def __init__(self, **kwargs):
        default_attr = dict(data_directory=DEFAULT_DATA_DIR, check_prev_downloads=True)

        default_attr.update(kwargs)
        for key in default_attr:
            self.__dict__[key] = default_attr.get(key)

        self.downloaded_books = set()
        if (self.check_prev_downloads):
            self.update_prev_downloaded_books()

        self.session = requests.Session()

    def print_vars(self):
        print(vars(self))

    def get_downloaded_books_count(self):
        return len(self.downloaded_books)

    def update_prev_downloaded_books(self):
        self.downloaded_books.update(os.listdir(self.data_directory))

    #         self.downloaded_books.update(os.listdir(self.data_dir_alt))

    def get_downloaded_books(self):
        print(self.downloaded_books)

    def downloaded(self, booktitle):
        return (booktitle + ".txt" in self.downloaded_books or booktitle in self.downloaded_books or booktitle.replace(
            " ", "_") + ".txt" in self.downloaded_books)

    def get_top_100_books(self, limit=100):
        r = self.session.get(TOP_100)
        soup = BeautifulSoup(r.content, features="lxml")
        books = soup.findAll("li")

        top_100_book_links = {}  # dictionary to hold booktitle:link_to_download_book

        for book in books[:limit]:
            booktitle = book.text
            link = book.find("a")["href"]

            if (self.check_prev_downloads):
                if (not self.downloaded(booktitle)):
                    if "ebooks" in str(book.find("a")["href"]):
                        top_100_book_links[booktitle] = link
                else:
                    print("<--- SKIPPING %s; this title is already downloaded --->" % (booktitle))

        for book in top_100_book_links.items():
            booktitle = book[0]
            book_download_link = book[1]
            url = BASE_URL + book_download_link[1:]
            data_link = self.get_data_links(url)
            self.download_book(booktitle, data_link)

    def get_data_links(self, url):
        r = self.session.get(url)
        soup = BeautifulSoup(r.content, features="lxml")
        data_links = soup.findAll("a", {"class": "link"})
        data_link = BASE_URL + self.get_text_or_html_link(data_links)
        return data_link

    def get_book_by_ids(self, bookids):
        for bookid in bookids:
            self.get_book_by_id(bookid)

    def get_book_by_range(self, start, finish):
        for bookid in range(start, finish):
            self.get_book_by_id(str(bookid))

    def get_book_by_id(self, bookid):
        threshold = 0
        url = BASE_URL + "ebooks/" + bookid
        r = self.session.get(url)
        soup = BeautifulSoup(r.content, features="lxml")
        booktitle = soup.find("title").text

        if (self.check_prev_downloads):
            if (not self.downloaded(booktitle)):
                if (self.get_language(soup) != "Dutch"):
                    print("<--- SKIPPING %s; this title is not in Dutch --->" % (booktitle))
                    return
                if (self.get_downloads(soup) < threshold):
                    print("<--- SKIPPING %s; this title has fewer than %d downloads --->" % (booktitle, threshold))
                    return
                #self.get_metadata(soup)
                data_link = self.get_data_links(url)
                self.download_book(booktitle, data_link)

            else:
                print("<--- SKIPPING %s; this title is already downloaded --->" % (booktitle))
        return

    def check_downloaded_by_name(self, book):
        for d in self.downloaded_books:
            if book.lower() in d.lower():
                print(d)
        return

    def get_metadata_only_by_id(self, bookid):
        url = BASE_URL + "ebooks/" + bookid
        r = self.session.get(url)
        soup = BeautifulSoup(r.content)
        self.get_metadata(soup)

    def get_metadata_only_by_range(self, start, finish):
        for bookid in range(start, finish):
            self.get_metadata_only_by_id(str(bookid))

    def get_metadata_only(self, data_link):
        r = self.session.get(data_link)
        soup = BeautifulSoup(r.content)
        self.get_metadata(self, soup)

    def get_language(self, soup):
        bibrec = soup.find("div", {"id": "bibrec"})
        try:
            language = bibrec.find("tr", {"property": "dcterms:language"}).text
            if "Dutch" in language:
                return "Dutch"
            return language
        except:
            print("<--- ERROR GETTING LANGUAGE --->")

    def get_downloads(self, soup):
        bibrec = soup.find("div", {"id": "bibrec"})
        try:
            downloads = bibrec.find("td", {"itemprop": "interactionCount"}).text.split(" ")[0]
            return int(downloads)
        except:
            print("<--- ERROR GETTING DOWNLOADS --->")
            return 0

    def get_metadata(self, soup):
        bibrec = soup.find("div", {"id": "bibrec"})
        try:
            title = bibrec.find("td", {"itemprop": "headline"}).text.replace("\n", "")
            author = bibrec.find("a", {"itemprop": "creator"}).text.replace("\n", "")
            date_published = bibrec.find("td", {"itemprop": "datePublished"}).text.replace("\n", "")

            filename = self.data_directory + "metadata.tsv"

            file = open(filename, "a")
            file.write(title + "\t" + date_published + ",\t" + author + "\n")
            file.close()
        except:
            print("<--- ERROR GETTING METADATA --->")

        return

    def download_book(self, booktitle, data_link):
        try:
            print("<--- DOWNLOADING %s --->" % (booktitle))
            r = self.session.get(data_link)
            filename = self.data_directory + booktitle + ".txt"

            file = open(filename, "w")
            file.write(r.text)
            file.close()
            self.downloaded_books.add(booktitle + ".txt")
        except Exception as e:
            print(e)
            print("<--- ERROR DOWNLOADING %s --->" % (booktitle))

    def get_text_or_html_link(self, links):
        plaintext_link, html_link = "", ""
        for link in links:
            if "Plain Text" in str(link):
                plaintext_link = link["href"]
            elif "HTML" in str(link):
                html_link = link["href"]

        data_link = ""
        if (plaintext_link):
            data_link = plaintext_link
        elif (html_link):
            data_link = html_link
        return data_link


def main(argv):
    parser = argparse.ArgumentParser(
        description="gutenberg-scraper scrapes and downloads books from project gutenberg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        fromfile_prefix_chars='@')

    parser.add_argument('--data-directory', '--data_directory', '-d', default=argparse.SUPPRESS,
                        help='directory where books will be downloaded')
    parser.add_argument('--check-prev-downloads', '--check_prev_downloads', '-c', default=argparse.SUPPRESS,
                        help='check whether book alaready exists in data directory')
    parser.add_argument('--action', '--action', '-a', default=argparse.SUPPRESS,
                        help='to download top 100 books, pass top; to download book by id, pass id of book',
                        required=True)

    args = parser.parse_args(argv)

    scraper = Gutenberg_Scraper(**vars(args))
    if (scraper.action == "top"):
        scraper.get_top_100_books()
    else:
        try:
            scraper.get_book_by_id(scraper.action)
        except Exception as e:
            print(e)

if __name__ == "__main__":
    main(sys.argv[1:])
