from data_utils import load_and_clean_data

def show_data():
    df = load_and_clean_data()
    print("\n--- First 10 Rows of Data ---")
    print(df.head(10))

if __name__ == "__main__":
    show_data()
