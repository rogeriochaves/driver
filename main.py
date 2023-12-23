import sys
from dotenv import load_dotenv

load_dotenv()

from driver.executor import start

start(sys.argv[1])
