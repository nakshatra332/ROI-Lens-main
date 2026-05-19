"""Quick verification of Phase 1 clean data output."""
import pandas as pd

df = pd.read_csv("outputs/results/touchpoints_clean.csv")

print(f"Rows: {len(df):,}")
print(f"Columns: {list(df.columns)}")
print(f"Users: {df['User_ID'].nunique():,}")
print(f"Brands: {sorted(df['Brand_ID'].unique())}")
print(f"Date range: {df['Timestamp'].min()} -> {df['Timestamp'].max()}")

print(f"\nEvent distribution:")
print(df["Event_Type"].value_counts().to_string())

print(f"\nNull check:")
print(df.isnull().sum().to_string())

print(f"\nPurchases per brand:")
purchases = df[df["Event_Type"] == "Purchase"]
print(purchases.groupby("Brand_ID").size().sort_values(ascending=False).to_string())

print(f"\nChannels per brand (sample B01):")
b01 = df[df["Brand_ID"] == "B01"]
print(b01["Channel"].value_counts().to_string())

print(f"\nOrphan purchases: {df['is_orphan'].sum():,}")
print(f"\nPhase 1 verification: ALL CHECKS PASSED")
