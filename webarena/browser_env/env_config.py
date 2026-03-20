# websites domain
import os

REDDIT = os.environ.get("REDDIT", "http://143.215.184.110:9999").rstrip("/")
SHOPPING = os.environ.get("SHOPPING", "http://127.0.0.1:7770").rstrip("/")
SHOPPING_ADMIN = os.environ.get("SHOPPING_ADMIN", "http://127.0.0.1:7780/admin").rstrip("/")
GITLAB = os.environ.get("GITLAB", "http://127.0.0.1:8023").rstrip("/")
WIKIPEDIA = os.environ.get("WIKIPEDIA", "http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing").rstrip("/")
MAP = os.environ.get("MAP", "https://www.openstreetmap.org/").rstrip("/")
HOMEPAGE = os.environ.get("HOMEPAGE", "http://127.0.0.1:4399").rstrip("/")

assert (
    REDDIT
    and SHOPPING
    and SHOPPING_ADMIN
    and GITLAB
    and WIKIPEDIA
    and MAP
    and HOMEPAGE
), (
    f"Please setup the URLs to each site. Current: \n"
    + f"Reddit: {REDDIT}\n"
    + f"Shopping: {SHOPPING}\n"
    + f"Shopping Admin: {SHOPPING_ADMIN}\n"
    + f"Gitlab: {GITLAB}\n"
    + f"Wikipedia: {WIKIPEDIA}\n"
    + f"Map: {MAP}\n"
    + f"Homepage: {HOMEPAGE}\n"
)


ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}

URL_MAPPINGS = {
    REDDIT: "http://reddit.com",
    SHOPPING: "http://onestopmarket.com",
    SHOPPING_ADMIN: "http://luma.com/admin",
    GITLAB: "http://gitlab.com",
    WIKIPEDIA: "http://wikipedia.org",
    MAP: "http://openstreetmap.org",
    HOMEPAGE: "http://homepage.com",
}
