import sys
sys.path.append("..")

from src.data_extraction import Hotel_Scraper

scraper = Hotel_Scraper()

scraper.scrape_all()

scraper.merge_csv_files()