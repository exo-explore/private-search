import requests

url = "https://yahoo-finance15.p.rapidapi.com/api/v1/markets/stock/quotes"

headers = {
	"x-rapidapi-key": "b82ca58dfbmsh11a9804d841eab1p139a4ejsnf106db66e95e",
	"x-rapidapi-host": "yahoo-finance15.p.rapidapi.com"
}

tickers = "A,APPL,GM,MU,TSLA," \
          "ALI=F,CD=F,QM=F,^IXIC,BTC-F," \
          "EURUSD=X,AUDUSD=X," \
          "^DJT,^HSI,^VIX,^TRFK-TC," \
          "SPY,AWSHX,VOO,XAIX.BE," \
          "BTC-USD,ETH-USD," \

querystring = {"ticker": tickers}
response = requests.get(url, headers=headers, params=querystring)

if response.status_code == 200:
    data = response.json()
    
    results = [
        {
            # "symbol": item.get("symbol", "N/A"), # removing the symbol improved the accuracy of retrieval (try to maximize the difference between the search terms)
            "name": item.get("displayName", item.get("shortName", item.get("longName", "N/A"))),
            "currentPrice": item.get("regularMarketPrice", "N/A"),
        }
        for item in data.get("body", [])
    ]
    print(str(results * 3).replace("'", '"'))

else:
    print(f"Failed to fetch data: {response.status_code}")
