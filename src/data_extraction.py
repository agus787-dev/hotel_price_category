import requests
import pandas as pd
import time
import sys
import re

sys.path.append("..")

from src.logger import Logger
from bs4 import BeautifulSoup
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class Hotel_Scraper:

    def __init__(self, log_name="hotel_scraper"):
        # self.BASE_URL = f"https://mybooking.uz/uz/hotels/samarkand?hotel_id=&checkin=10.07.2026&checkout=11.07.2026&adults=3"
        self.BASE_URL = f"https://mybooking.uz/uz/hotels/"
        self.headers = {
            "User-Agent": "Mozilla/5.0"
        }
        self.logger = Logger(log_name)
        self.cities=["samarkand", "tashkent", "bukhara", "fergana", "andijan", "namangan", "khiva", "urgench", "nukus", "navoi", "jizzakh", "termez", "mubarek", "kitab", "mujnak"]
        self.people=[1, 2, 3, 4, 5, 6, 7]

        self.logger.info("Initializing Hotel_Scraper")

        self.output_dir_path = ROOT_DIR / "data"
        self.output_dir_path.mkdir(parents=True, exist_ok=True)

        self.city_dir_path = self.output_dir_path / "cities"
        self.city_dir_path.mkdir(parents=True, exist_ok=True)

        self.merged_dir_path = self.output_dir_path / "merged"
        self.merged_dir_path.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Output directory ready: {self.output_dir_path}")
        self.logger.info(f"City CSV directory ready: {self.city_dir_path}")
        self.logger.info(f"Merged CSV directory ready: {self.merged_dir_path}")

    def hotels(self, city, people):
        url=f"{self.BASE_URL}{city}?hotel_id=&checkin=10.07.2026&checkout=11.07.2026&adults={people}"
        self.logger.info(f"Fetching hotels list: {url}")

        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
        except requests.exceptions.Timeout as e:
            self.logger.error(f"Request timed out: {e}")
            print("Request timed out: {e}")
            return []
        except requests.exceptions.HTTPError as e:
            self.logger.error(f"Request timed out: {e}")
            print(f"HTTP error: {e}")
            return []
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Request failed: {e}")
            print(f"Request failed: {e}")
            return []

        soup = BeautifulSoup(response.content, "html.parser")
        hotels_container = soup.find("div", class_="hotels-container")
        
        if not hotels_container:
            self.logger.warning(f"No result block found for {city} number of people {people}")
            return []

        hotels = hotels_container.find_all("a", class_="hotel-card")
        self.logger.info(f"Found {len(hotels)} listings for {city} number of people {people}")

        hotels_data = []

        for hotel in hotels:

            # title
            title_tag = hotel.find("span", class_="hotel-card__title-name")
            if not title_tag:
                self.logger.warning(f"Skipping hotel entry with missing title link")
            title = title_tag.get_text(strip=True)

            # price
            price_tag = hotel.find("span", class_="prices-value__total")
            price_text = price_tag.get_text(strip=True)
            if not price_tag:
                self.logger.warning(f"Skipping {title} because price was not found")
                continue
            match_p = re.search(r"\d+(?:\.\d+)?", price_text)
            if not match_p:
                continue
            price = float(match_p.group())

            # distance
            distance_tag=hotel.find("span", class_="hotel-card__distanse")
            distance_text=distance_tag.get_text(strip=True)
            if not distance_tag:
                self.logger.warning(f"Skipping {title} because distance was not found")
                continue
            match_d = re.search(r"\d+(?:\.\d+)?", distance_text)
            if not match_d:
                continue
            distance = float(match_d.group())

            # description
            desc_tag=hotel.select_one("span.hotel-card__rooms-name b")
            if not desc_tag:
                self.logger.warning(f"Skipping {title} because description was not found")
                continue
            desc=desc_tag.get_text(strip=True)

            # breakfast
            bf_has_included=hotel.find("div", class_="hotel-card__breakfast") is not None

            hotels_data.append({
                "title": title,
                "price": price,
                "adults": people,
                "city": city,
                "distance_to_center": distance,
                "description": desc,
                "bf_has_included": bf_has_included
            })

        self.logger.info(f"Parsed {len(hotels_data)} hotel records for {city} number of people {people}")
        return hotels_data

    def scrape_all(self):
        self.logger.info("Starting scrape for all hotels")
        for city in self.cities:
            self.logger.info(f"Scraping by city is: {city}")
            all_data = []

            for p in self.people:
                hotels_data = self.hotels(city, p)
                self.logger.info(f"Scraped {len(hotels_data)} hotel records for {city} number of people {p}")
                if not hotels_data:
                    self.logger.warning(f"No data returned for {city} on number of people {p}")
                all_data.extend(hotels_data)
                time.sleep(1)

            csv_path = self.city_dir_path / f"{city}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            if all_data:
                df = pd.DataFrame(all_data)
                df.to_csv(csv_path, index=False)
                self.logger.info(f"Saved {len(all_data)} records to {csv_path}")
            else:
                self.logger.warning(f"No records collected for city {city}; skipping CSV save")

    def merge_csv_files(self):
        self.logger.info("Starting merge of city CSV files")
        dfs = []

        for city in self.cities:
            file_path = self.city_dir_path / f"{city}.csv"
            if file_path.exists():
                self.logger.info(f"Loading {file_path}")
                dfs.append(pd.read_csv(file_path))
            else:
                self.logger.warning(f"CSV file not found: {file_path}")

        if not dfs:
            self.logger.error("No CSV files found to merge")
            return pd.DataFrame()

        merged_df = pd.concat(dfs, ignore_index=True)

        merged_df["price_category"] = merged_df["price"].apply(self.price_category)

        output_file = self.merged_dir_path / "hotels.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(output_file, index=False)

        self.logger.info(f"Merged {len(dfs)} files into {output_file} with {len(merged_df)} rows")
        return merged_df
    
    def price_category(self, price):
        if price < 49.94:
            return "Budget"
        elif price < 83.24:
            return "Standard"
        elif price < 149.83:
            return "Premium"
        else:
            return "Luxury"

    