from bs4 import BeautifulSoup
import pandas as pd

# Load local HTML file
with open("finance.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse HTML
soup = BeautifulSoup(html_content, 'html.parser')
table = soup.find('table')

# Extract all rows
all_rows = table.find_all('tr')

# Get headers (including the first empty one if it's there)
headers = [th.get_text(strip=True) for th in all_rows[0].find_all('th')]

# Extract data rows
rows = []
for tr in all_rows[1:]:
    cells = [td.get_text(strip=True) for td in tr.find_all('td')]
    rows.append(cells)

# Make sure number of columns in headers matches row length
max_len = max(len(r) for r in rows)
if len(headers) < max_len:
    headers = ['Column ' + str(i) for i in range(max_len)]  # Fallback headers if mismatch

# Create DataFrame
df = pd.DataFrame(rows, columns=headers)

# Show DataFrame
print(df)
