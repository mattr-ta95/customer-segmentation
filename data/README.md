# Data Directory

## Required Data File

Place your customer data file in this directory as `CUSTOMERS_CLEAN.csv`

## Expected Format

The CSV file should contain the following columns:

| Column Name | Type | Format | Description |
|-------------|------|--------|-------------|
| Customer ID | String | Any | Unique customer identifier |
| Order ID | String | Any | Unique order identifier |
| Order_Date | Date | DDMMMYYYY | Date of order (e.g., 01JAN2023) |
| Delivery_Date | Date | DDMMMYYYY | Date of delivery |
| Total Revenue | Currency | £1,234.56 or €1,234.56 | Revenue amount with currency symbol |
| Unit Cost | Currency | £12.34 or €12.34 | Cost per unit with currency symbol |
| Customer_BirthDate | Date | DDMMMYYYY | Customer birth date |

## Example Data

```csv
Customer ID,Order ID,Order_Date,Delivery_Date,Total Revenue,Unit Cost,Customer_BirthDate
CUST001,ORD001,01JAN2023,05JAN2023,£150.00,£25.00,15MAR1985
CUST001,ORD002,15FEB2023,20FEB2023,£200.00,£40.00,15MAR1985
CUST002,ORD003,10JAN2023,15JAN2023,€75.50,€15.10,22JUL1990
```

## Data Privacy Notice

⚠️ **IMPORTANT**: This analysis processes customer personally identifiable information (PII) including:
- Customer IDs
- Birth dates
- Purchase history

**Before using real customer data:**
1. Ensure you have proper authorization
2. Anonymize/pseudonymize customer identifiers if required
3. Comply with GDPR, CCPA, or other applicable data privacy regulations
4. Do not commit this data file to version control (it's in .gitignore)
5. Store data securely with appropriate access controls

For demonstration purposes, consider using synthetic or anonymized data.
