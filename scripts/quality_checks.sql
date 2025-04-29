-- Data Quality Checks for Water Quality Dataset

-- 1. Data Completeness Check
SELECT 
    COUNT(*) as total_records,
    SUM(CASE WHEN pH IS NULL THEN 1 ELSE 0 END) as null_pH,
    SUM(CASE WHEN Iron IS NULL THEN 1 ELSE 0 END) as null_Iron,
    SUM(CASE WHEN Nitrate IS NULL THEN 1 ELSE 0 END) as null_Nitrate,
    SUM(CASE WHEN Chloride IS NULL THEN 1 ELSE 0 END) as null_Chloride,
    SUM(CASE WHEN Lead IS NULL THEN 1 ELSE 0 END) as null_Lead,
    SUM(CASE WHEN Zinc IS NULL THEN 1 ELSE 0 END) as null_Zinc,
    SUM(CASE WHEN Turbidity IS NULL THEN 1 ELSE 0 END) as null_Turbidity,
    SUM(CASE WHEN Fluoride IS NULL THEN 1 ELSE 0 END) as null_Fluoride,
    SUM(CASE WHEN Copper IS NULL THEN 1 ELSE 0 END) as null_Copper,
    SUM(CASE WHEN Sulfate IS NULL THEN 1 ELSE 0 END) as null_Sulfate,
    SUM(CASE WHEN Conductivity IS NULL THEN 1 ELSE 0 END) as null_Conductivity,
    SUM(CASE WHEN Chlorine IS NULL THEN 1 ELSE 0 END) as null_Chlorine,
    SUM(CASE WHEN "Total Dissolved Solids" IS NULL THEN 1 ELSE 0 END) as null_TDS,
    SUM(CASE WHEN "Water Temperature" IS NULL THEN 1 ELSE 0 END) as null_WaterTemp,
    SUM(CASE WHEN "Air Temperature" IS NULL THEN 1 ELSE 0 END) as null_AirTemp,
    SUM(CASE WHEN Target IS NULL THEN 1 ELSE 0 END) as null_Target
FROM water_quality_data;

-- 2. Value Range Validation
SELECT 
    COUNT(*) as total_records,
    SUM(CASE WHEN pH < 0 OR pH > 14 THEN 1 ELSE 0 END) as invalid_pH,
    SUM(CASE WHEN Iron < 0 OR Iron > 10 THEN 1 ELSE 0 END) as invalid_Iron,
    SUM(CASE WHEN Nitrate < 0 OR Nitrate > 100 THEN 1 ELSE 0 END) as invalid_Nitrate,
    SUM(CASE WHEN Chloride < 0 OR Chloride > 500 THEN 1 ELSE 0 END) as invalid_Chloride,
    SUM(CASE WHEN Lead < 0 OR Lead > 0.05 THEN 1 ELSE 0 END) as invalid_Lead,
    SUM(CASE WHEN Zinc < 0 OR Zinc > 5 THEN 1 ELSE 0 END) as invalid_Zinc,
    SUM(CASE WHEN Turbidity < 0 OR Turbidity > 10 THEN 1 ELSE 0 END) as invalid_Turbidity,
    SUM(CASE WHEN Fluoride < 0 OR Fluoride > 2 THEN 1 ELSE 0 END) as invalid_Fluoride,
    SUM(CASE WHEN Copper < 0 OR Copper > 2 THEN 1 ELSE 0 END) as invalid_Copper,
    SUM(CASE WHEN Sulfate < 0 OR Sulfate > 500 THEN 1 ELSE 0 END) as invalid_Sulfate,
    SUM(CASE WHEN Conductivity < 0 OR Conductivity > 2000 THEN 1 ELSE 0 END) as invalid_Conductivity,
    SUM(CASE WHEN Chlorine < 0 OR Chlorine > 5 THEN 1 ELSE 0 END) as invalid_Chlorine,
    SUM(CASE WHEN "Total Dissolved Solids" < 0 OR "Total Dissolved Solids" > 2000 THEN 1 ELSE 0 END) as invalid_TDS,
    SUM(CASE WHEN "Water Temperature" < 0 OR "Water Temperature" > 40 THEN 1 ELSE 0 END) as invalid_WaterTemp,
    SUM(CASE WHEN "Air Temperature" < -10 OR "Air Temperature" > 50 THEN 1 ELSE 0 END) as invalid_AirTemp,
    SUM(CASE WHEN Target NOT IN (0, 1) THEN 1 ELSE 0 END) as invalid_Target
FROM water_quality_data;

-- 3. Statistical Distribution Check
SELECT 
    AVG(pH) as avg_pH,
    STDDEV(pH) as std_pH,
    MIN(pH) as min_pH,
    MAX(pH) as max_pH,
    AVG(Iron) as avg_Iron,
    STDDEV(Iron) as std_Iron,
    MIN(Iron) as min_Iron,
    MAX(Iron) as max_Iron,
    AVG(Nitrate) as avg_Nitrate,
    STDDEV(Nitrate) as std_Nitrate,
    MIN(Nitrate) as min_Nitrate,
    MAX(Nitrate) as max_Nitrate
FROM water_quality_data;

-- 4. Correlation Analysis
SELECT 
    CORR(pH, Iron) as pH_Iron_corr,
    CORR(pH, Nitrate) as pH_Nitrate_corr,
    CORR(Iron, Nitrate) as Iron_Nitrate_corr,
    CORR(Conductivity, "Total Dissolved Solids") as Conductivity_TDS_corr
FROM water_quality_data;

-- 5. Target Variable Distribution
SELECT 
    Target,
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM water_quality_data), 2) as percentage
FROM water_quality_data
GROUP BY Target;

-- 6. Temporal Analysis (if time-based data is available)
SELECT 
    EXTRACT(MONTH FROM timestamp) as month,
    COUNT(*) as record_count,
    AVG(pH) as avg_pH,
    AVG(Iron) as avg_Iron,
    AVG(Nitrate) as avg_Nitrate
FROM water_quality_data
GROUP BY EXTRACT(MONTH FROM timestamp)
ORDER BY month; 