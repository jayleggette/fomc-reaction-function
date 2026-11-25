# Data provenance

| File | Source | Retrieved |
|---|---|---|
| `raw/fred_monthly.csv` | FRED: FEDFUNDS, CPIAUCSL, CPILFESL, PCEPILFE, UNRATE, PAYEMS, TB3MS, GS2, GS5, GS10, GS30 | 24 Nov 2026 |
| `raw/fred_quarterly.csv` | FRED: NROU (CBO noncyclical rate of unemployment, Feb 2026 vintage) | 24 Nov 2026 |
| `raw/fred_daily.csv` | FRED: DFF, DFEDTAR, DFEDTARU, DFEDTARL, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10 | 24 Nov 2026 |
| `raw/fomc_statements.csv` | FOMC statements from github.com/vtasca/fed-statement-scraping (scraped from federalreserve.gov) | 24 Nov 2026 |
| `raw/fomc_scheduled_meetings.csv` | Built by `fomc.calendar.build_calendar`: Fed historical pages (1990 to 1999) plus the archive above (2000 on) | 24 Nov 2026 |

The FRED README files that came with each download are kept alongside.

To refresh, download these two files from FRED (12 series per request at most) and replace the CSVs:

* https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS,CPIAUCSL,CPILFESL,PCEPILFE,UNRATE,NROU,PAYEMS,TB3MS,GS2,GS5,GS10,GS30
* https://fred.stlouisfed.org/graph/fredgraph.csv?id=DFF,DFEDTAR,DFEDTARU,DFEDTARL,DGS3MO,DGS6MO,DGS1,DGS2,DGS3,DGS5,DGS7,DGS10

FRED returns a zip when the series have different frequencies; merge the daily files on date.
