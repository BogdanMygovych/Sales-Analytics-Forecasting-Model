-- Retail Sales Analytics Queries
-- Target table: cleaned_retail_data

-- 1) Total sales
SELECT
    SUM(sales) AS total_sales
FROM cleaned_retail_data;

-- 2) Sales by store
SELECT
    store_nbr,
    SUM(sales) AS store_sales
FROM cleaned_retail_data
GROUP BY store_nbr
ORDER BY store_sales DESC;

-- 3) Sales trends over time (daily)
SELECT
    date,
    SUM(sales) AS daily_sales
FROM cleaned_retail_data
GROUP BY date
ORDER BY date;

-- 4) Top product families
SELECT
    family,
    SUM(sales) AS family_sales
FROM cleaned_retail_data
GROUP BY family
ORDER BY family_sales DESC
LIMIT 10;

-- 5) Monthly growth (% change vs previous month)
WITH monthly_sales AS (
    SELECT
        year,
        month,
        SUM(sales) AS monthly_sales
    FROM cleaned_retail_data
    GROUP BY year, month
),
monthly_with_lag AS (
    SELECT
        year,
        month,
        monthly_sales,
        LAG(monthly_sales) OVER (ORDER BY year, month) AS previous_month_sales
    FROM monthly_sales
)
SELECT
    year,
    month,
    monthly_sales,
    previous_month_sales,
    CASE
        WHEN previous_month_sales IS NULL OR previous_month_sales = 0 THEN NULL
        ELSE ROUND(((monthly_sales - previous_month_sales) / previous_month_sales) * 100, 2)
    END AS monthly_growth_pct
FROM monthly_with_lag
ORDER BY year, month;
