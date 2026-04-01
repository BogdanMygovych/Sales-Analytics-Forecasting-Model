-- Retail Sales Analytics Queries
-- Assumes table name: cleaned_retail_data

-- 1. Total sales
SELECT
    SUM(sales) AS total_sales
FROM cleaned_retail_data;

-- 2. Sales by month
SELECT
    year,
    month,
    SUM(sales) AS monthly_sales
FROM cleaned_retail_data
GROUP BY year, month
ORDER BY year, month;

-- 3. Sales by store
SELECT
    store_nbr,
    SUM(sales) AS store_sales
FROM cleaned_retail_data
GROUP BY store_nbr
ORDER BY store_sales DESC;

-- 4. Top products (families)
SELECT
    family,
    SUM(sales) AS family_sales
FROM cleaned_retail_data
GROUP BY family
ORDER BY family_sales DESC
LIMIT 10;

-- 5. Sales trends (daily)
SELECT
    date,
    SUM(sales) AS daily_sales
FROM cleaned_retail_data
GROUP BY date
ORDER BY date;
